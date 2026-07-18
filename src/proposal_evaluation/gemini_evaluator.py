"""
gemini_evaluator.py

Belongs to the ONLINE proposal evaluation pipeline of the
AI-Powered Grant Proposal Evaluation System.

Pipeline position:

    Proposal PDF
            |
    process_proposal.py
            |
    retrieve_context.py
            |
    gemini_evaluator.py   <-- this module
            |
    scoring_engine.py
            |
    generate_html.py
            |
    generate_pdf.py

Responsibility (and ONLY responsibility):
    1. Receive the dictionary produced by retrieve_context.py.
    2. Send its "evaluation_prompt" to Google's Gemini API.
    3. Clean the raw response text (strip markdown fences / stray prose).
    4. Parse the cleaned text as JSON.
    5. Validate that the JSON matches the required evaluation schema.
    6. Return the validated evaluation dictionary.

This module does NOT calculate derived scores, does NOT generate HTML or
PDF reports, and does NOT touch proposal PDFs, embeddings, or the vector
store. Its only job is the AI evaluation call itself.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv # type: ignore
from google import genai    # type: ignore
from google.genai import types  # type: ignore
from google.genai.errors import APIError, ClientError, ServerError  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Centralized here so the model name has exactly one source of truth in this
# module, and is easy to change if the project later moves to a different
# Gemini model tier.
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"

# Gemini calls can legitimately take a while for a long, context-heavy
# evaluation prompt. A generous but finite timeout prevents the pipeline
# from hanging indefinitely if the API stops responding mid-request.
REQUEST_TIMEOUT_SECONDS = 120

# The seven criteria the evaluation JSON's "criteria_scores" object must
# contain. Defined once here so validate_response() and any future caller
# share a single, authoritative list rather than duplicating it.
REQUIRED_CRITERIA_SCORE_FIELDS = (
    "relevance",
    "feasibility",
    "innovation",
    "sustainability",
    "budget_justification",
    "organizational_capacity",
    "expected_impact",
)

# The top-level fields the evaluation JSON must contain, beyond
# "criteria_scores" itself.
REQUIRED_TOP_LEVEL_FIELDS = (
    "proposal_summary",
    "funding_alignment_analysis",
    "criteria_scores",
    "strengths",
    "weaknesses",
    "risk_flags",
    "final_recommendation",
)

MIN_SCORE = 1
MAX_SCORE = 5

LOG_DIR = Path("logs/evaluation_logs")
LOG_FILE = LOG_DIR / "evaluation.log"

# Matches a fenced code block, capturing whatever is inside the fences.
# Handles ```json ... ```, plain ``` ... ```, and is tolerant of leading/
# trailing whitespace or explanatory prose around the fence.
_CODE_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*(?P<body>.*?)\s*```", re.DOTALL | re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GeminiEvaluationError(Exception):
    """Base exception for all failures raised by this module.

    A dedicated exception hierarchy (rather than letting raw ValueError /
    RuntimeError propagate) lets calling code in scoring_engine.py or the
    Gradio layer catch evaluation-specific failures distinctly from
    unrelated bugs elsewhere in the pipeline.
    """


class MissingAPIKeyError(GeminiEvaluationError):
    """Raised when no Gemini API key can be found in the environment."""


class GeminiRequestError(GeminiEvaluationError):
    """Raised when the request to the Gemini API fails outright."""


class InvalidResponseError(GeminiEvaluationError):
    """Raised when Gemini's response is empty, not valid JSON, or otherwise malformed."""


class SchemaValidationError(GeminiEvaluationError):
    """Raised when parsed JSON is valid JSON but does not match the required schema."""


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logger() -> logging.Logger:
    """Configure and return this module's dedicated logger.

    A dedicated logger (rather than the root logger) keeps this module's
    logging behavior independent of logging configuration done elsewhere in
    the pipeline (e.g. process_proposal.py, retrieve_context.py each keep
    their own logger and log file).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("proposal_evaluation.gemini_evaluator")
    logger.setLevel(logging.DEBUG)

    # Guard against duplicate handlers if this module is imported more than
    # once within the same process.
    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = _configure_logger()


# ---------------------------------------------------------------------------
# Module-level Gemini client cache
# ---------------------------------------------------------------------------

# Constructing the client involves reading and validating the API key.
# Caching it at module level means repeated evaluate_proposal() calls within
# the same process (e.g. batch-evaluating several proposals) only do that
# setup once.
_gemini_client: genai.Client | None = None


def _load_api_key() -> str:
    """Load the Gemini API key from the environment (.env file).

    Returns:
        The API key string.

    Raises:
        MissingAPIKeyError: If no key is found, or the value found is blank.
    """
    # load_dotenv() populates os.environ from a local .env file if present.
    # It is safe to call repeatedly; it is a no-op if variables are already
    # loaded or no .env file exists.
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKeyError(
            "GEMINI_API_KEY is missing or empty. Add it to your .env file, "
            "e.g. GEMINI_API_KEY=your_key_here."
        )
    return api_key


def _get_gemini_client() -> genai.Client:
    """Return the cached Gemini client, constructing it on first use.

    Raises:
        MissingAPIKeyError: propagated from _load_api_key().
    """
    global _gemini_client
    if _gemini_client is None:
        api_key = _load_api_key()
        _gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized.")
    return _gemini_client


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_input(retrieval_data: dict[str, Any]) -> str:
    """Validate the dictionary received from retrieve_context.py.

    Args:
        retrieval_data: The dictionary produced by retrieve_context.py,
            expected to contain a non-empty "evaluation_prompt" key.

    Returns:
        The evaluation prompt string, once confirmed present and non-empty.

    Raises:
        ValueError: If retrieval_data is not a dict, is missing the
            "evaluation_prompt" key, or the prompt is blank.
    """
    if not isinstance(retrieval_data, dict):
        raise ValueError(
            f"Expected a dictionary from retrieve_context.py, got {type(retrieval_data).__name__}."
        )

    evaluation_prompt = retrieval_data.get("evaluation_prompt")

    if evaluation_prompt is None:
        raise ValueError(
            "retrieval_data is missing the required 'evaluation_prompt' key. "
            "Was this dictionary produced by retrieve_context.retrieve_context()?"
        )

    if not isinstance(evaluation_prompt, str) or not evaluation_prompt.strip():
        raise ValueError("'evaluation_prompt' is empty or not a string.")

    return evaluation_prompt


# ---------------------------------------------------------------------------
# Gemini request
# ---------------------------------------------------------------------------

def call_gemini(evaluation_prompt: str) -> str:
    """Send the evaluation prompt to Gemini and return the raw response text.

    Args:
        evaluation_prompt: The complete prompt assembled by
            retrieve_context.py.

    Returns:
        The raw text of Gemini's response, exactly as received (not yet
        cleaned or parsed).

    Raises:
        MissingAPIKeyError: If no API key is configured.
        GeminiRequestError: If the request fails for any reason (invalid
            key, connection failure, timeout, server error, or the response
            contains no text at all).
    """
    client = _get_gemini_client()

    # A low temperature keeps Gemini's output close to deterministic and
    # discourages creative embellishment that would make the JSON harder to
    # parse reliably — this is a structured evaluation task, not a creative
    # writing task.
    generation_config = types.GenerateContentConfig(
        temperature=0.2,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000),
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=evaluation_prompt,
            config=generation_config,
        )
    except ClientError as exc:
        # ClientError covers 4xx-class problems, most commonly an invalid
        # or unauthorized API key.
        raise GeminiRequestError(
            f"Gemini rejected the request (client error, e.g. invalid API "
            f"key or malformed request): {exc}"
        ) from exc
    except ServerError as exc:
        raise GeminiRequestError(
            f"Gemini's servers returned an error (5xx). This is usually "
            f"transient; consider retrying: {exc}"
        ) from exc
    except APIError as exc:
        raise GeminiRequestError(f"Gemini API error: {exc}") from exc
    except TimeoutError as exc:
        raise GeminiRequestError(
            f"Gemini request timed out after {REQUEST_TIMEOUT_SECONDS} seconds: {exc}"
        ) from exc
    except Exception as exc:
        # Catches connection failures (e.g. DNS/network issues) and any
        # other SDK-level exception not covered by the specific types above,
        # so the caller never sees an unhandled exception type from a
        # third-party library leaking out of this module.
        raise GeminiRequestError(
            f"Unexpected failure while calling the Gemini API: {exc}"
        ) from exc

    response_text = getattr(response, "text", None)
    if not response_text or not response_text.strip():
        raise GeminiRequestError(
            "Gemini returned an empty response with no usable text. This can "
            "happen if the prompt was blocked by safety filters or the model "
            "returned only non-text content."
        )

    return response_text


# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------

def clean_response(raw_response: str) -> str:
    """Strip markdown code fences and surrounding prose from Gemini's response.

    Gemini sometimes wraps its JSON in a ```json ... ``` fence, and
    sometimes adds a sentence of explanation before or after the JSON
    despite being instructed not to. This function extracts just the JSON
    payload so json.loads() has the best possible chance of succeeding.

    Strategy:
        1. If a fenced code block is present, use its contents.
        2. Otherwise, fall back to slicing from the first '{' to the last
           '}' in the text, which handles the case of stray prose without
           a fence.
        3. Strip leading/trailing whitespace either way.

    Args:
        raw_response: The raw text returned by call_gemini().

    Returns:
        A string that should contain only the JSON payload (parsing is
        still validated by parse_json(), which will raise if this
        extraction did not succeed).
    """
    fence_match = _CODE_FENCE_PATTERN.search(raw_response)
    if fence_match:
        return fence_match.group("body").strip()

    # No fence found — fall back to extracting the outermost {...} span.
    # This tolerates Gemini prefacing or following the JSON with prose
    # (e.g. "Here is the evaluation:") even without a markdown fence.
    first_brace = raw_response.find("{")
    last_brace = raw_response.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return raw_response[first_brace : last_brace + 1].strip()

    # No recognizable JSON structure at all — return as-is and let
    # parse_json() raise a clear, descriptive error rather than silently
    # returning something misleading here.
    return raw_response.strip()


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json(cleaned_response: str) -> dict[str, Any]:
    """Parse the cleaned response text as JSON.

    Args:
        cleaned_response: The output of clean_response().

    Returns:
        The parsed JSON as a Python dictionary.

    Raises:
        InvalidResponseError: If the text is not valid JSON, or parses to
            something other than a JSON object (e.g. a bare list or
            string).
    """
    try:
        parsed = json.loads(cleaned_response)
    except json.JSONDecodeError as exc:
        raise InvalidResponseError(
            f"Gemini's response could not be parsed as JSON after cleaning. "
            f"JSON error: {exc}. "
            f"Cleaned response (truncated to 500 chars): "
            f"{cleaned_response[:500]!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise InvalidResponseError(
            f"Expected Gemini's JSON response to be an object, got "
            f"{type(parsed).__name__} instead."
        )

    return parsed


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_response(evaluation: dict[str, Any]) -> None:
    """Validate that a parsed evaluation dictionary matches the required schema.

    Checks performed:
        1. All required top-level fields are present.
        2. "criteria_scores" is a dict containing all seven required
           criterion fields.
        3. Every criterion score is an integer between 1 and 5 inclusive.
        4. "strengths", "weaknesses", and "risk_flags" are lists.

    Args:
        evaluation: The parsed JSON dictionary from parse_json().

    Raises:
        SchemaValidationError: If any required field is missing or has an
            invalid type/value, with a message describing exactly what was
            wrong.
    """
    missing_top_level = [
        field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in evaluation
    ]
    if missing_top_level:
        raise SchemaValidationError(
            f"Evaluation JSON is missing required top-level field(s): {missing_top_level}"
        )

    criteria_scores = evaluation["criteria_scores"]
    if not isinstance(criteria_scores, dict):
        raise SchemaValidationError(
            f"'criteria_scores' must be an object, got {type(criteria_scores).__name__}."
        )

    missing_criteria = [
        field for field in REQUIRED_CRITERIA_SCORE_FIELDS if field not in criteria_scores
    ]
    if missing_criteria:
        raise SchemaValidationError(
            f"'criteria_scores' is missing required field(s): {missing_criteria}"
        )

    invalid_scores: list[str] = []
    for field in REQUIRED_CRITERIA_SCORE_FIELDS:
        value = criteria_scores[field]
        # bool is technically a subclass of int in Python, so it is
        # explicitly excluded here to avoid True/False silently passing as
        # valid scores of 1/0.
        is_valid_integer = isinstance(value, int) and not isinstance(value, bool)
        if not is_valid_integer or not (MIN_SCORE <= value <= MAX_SCORE):
            invalid_scores.append(f"{field}={value!r}")

    if invalid_scores:
        raise SchemaValidationError(
            f"The following criteria_scores are not valid integers in the "
            f"range {MIN_SCORE}-{MAX_SCORE}: {', '.join(invalid_scores)}"
        )

    for list_field in ("strengths", "weaknesses", "risk_flags"):
        if not isinstance(evaluation[list_field], list):
            raise SchemaValidationError(
                f"'{list_field}' must be a list, got {type(evaluation[list_field]).__name__}."
            )

    for text_field in ("proposal_summary", "funding_alignment_analysis", "final_recommendation"):
        if not isinstance(evaluation[text_field], str) or not evaluation[text_field].strip():
            raise SchemaValidationError(
                f"'{text_field}' must be a non-empty string."
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_proposal(retrieval_data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a grant proposal using Gemini and return a validated result.

    Workflow: validate input -> call Gemini -> clean response -> parse JSON
    -> validate schema -> return dictionary.

    This is the only function scoring_engine.py should call into for the AI
    evaluation stage.

    Args:
        retrieval_data: The dictionary returned by
            retrieve_context.retrieve_context(), containing at minimum an
            "evaluation_prompt" key.

    Returns:
        A validated evaluation dictionary matching the required schema:
            {
                "proposal_summary": str,
                "funding_alignment_analysis": str,
                "criteria_scores": {
                    "relevance": int, "feasibility": int, ...
                },
                "strengths": list[str],
                "weaknesses": list[str],
                "risk_flags": list[str],
                "final_recommendation": str,
            }

    Raises:
        ValueError: If retrieval_data fails input validation.
        MissingAPIKeyError: If no Gemini API key is configured.
        GeminiRequestError: If the Gemini API call itself fails.
        InvalidResponseError: If the response cannot be parsed as JSON.
        SchemaValidationError: If parsed JSON does not match the required
            schema.
        Exception: any other unexpected error is logged with a full
            traceback before being re-raised.
    """
    start_time = time.perf_counter()
    logger.info("=" * 70)
    logger.info("Evaluation started.")
    logger.info("Gemini model: %s", GEMINI_MODEL_NAME)

    try:
        # --- Validate input --------------------------------------------------
        evaluation_prompt = validate_input(retrieval_data)
        logger.info("Prompt length: %d character(s).", len(evaluation_prompt))

        # --- Call Gemini -------------------------------------------------------
        logger.info("Sending request to Gemini...")
        raw_response = call_gemini(evaluation_prompt)
        logger.info("Response received: %d character(s).", len(raw_response))

        # --- Clean + parse -----------------------------------------------------
        cleaned_response = clean_response(raw_response)
        evaluation = parse_json(cleaned_response)
        logger.info("JSON parsed successfully.")

        # --- Validate schema --------------------------------------------------
        validate_response(evaluation)
        logger.info("Schema validation successful.")

        elapsed_seconds = time.perf_counter() - start_time
        logger.info("Evaluation completed successfully in %.2f second(s).", elapsed_seconds)

        return evaluation

    except GeminiEvaluationError:
        # Already a descriptive, module-specific exception — log with
        # traceback for the record, then let it propagate unchanged so the
        # caller can pattern-match on the specific exception type.
        logger.error("Evaluation FAILED.\n%s", traceback.format_exc())
        raise
    except Exception:
        # Anything not already wrapped in our own exception hierarchy is an
        # unexpected failure; log it fully and re-raise as-is so no error is
        # ever silently swallowed.
        logger.error("Evaluation FAILED with an unexpected error.\n%s", traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Standalone development testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A small mock retrieval_data dictionary, matching the shape produced by
    # retrieve_context.py, so this module can be exercised independently.
    # This performs a REAL Gemini API call if a valid GEMINI_API_KEY is
    # present in .env — it is not a mocked/offline test.
    mock_retrieval_data = {
        "proposal_text": (
            "This proposal seeks funding to establish a community-based "
            "digital literacy program for women in rural Ghana, combining "
            "foundational digital skills training with mobile-money "
            "financial literacy to support women's economic inclusion."
        ),
        "retrieved_chunks": [],
        "evaluation_prompt": (
            "You are an expert grant proposal evaluator for the Global "
            "Impact Foundation. Evaluate the following brief proposal "
            "summary against the seven criteria (relevance, feasibility, "
            "innovation, sustainability, budget_justification, "
            "organizational_capacity, expected_impact), each scored 1-5. "
            "Proposal: This proposal seeks funding to establish a "
            "community-based digital literacy program for women in rural "
            "Ghana, combining foundational digital skills training with "
            "mobile-money financial literacy to support women's economic "
            "inclusion. Return ONLY valid JSON matching this schema: "
            '{"proposal_summary": "", "funding_alignment_analysis": "", '
            '"criteria_scores": {"relevance": 0, "feasibility": 0, '
            '"innovation": 0, "sustainability": 0, "budget_justification": 0, '
            '"organizational_capacity": 0, "expected_impact": 0}, '
            '"strengths": [], "weaknesses": [], "risk_flags": [], '
            '"final_recommendation": ""}'
        ),
    }

    print("-" * 50)
    start = time.perf_counter()
    try:
        result = evaluate_proposal(mock_retrieval_data)
        elapsed = time.perf_counter() - start

        print("Gemini Evaluation Summary")
        print("-" * 50)
        print(f"Model Used             : {GEMINI_MODEL_NAME}")
        print(f"Prompt Length          : {len(mock_retrieval_data['evaluation_prompt']):,} characters")
        print(f"Response Length        : {len(json.dumps(result)):,} characters")
        print(f"JSON Validation        : PASSED")
        print(f"Processing Time        : {elapsed:.2f} seconds")
        print(f"Status                 : SUCCESS")
        print("-" * 50)
    except Exception as exc:  # noqa: BLE001 - top-level CLI catch is intentional
        elapsed = time.perf_counter() - start
        print("Gemini Evaluation Summary")
        print("-" * 50)
        print(f"Model Used             : {GEMINI_MODEL_NAME}")
        print(f"Prompt Length          : {len(mock_retrieval_data['evaluation_prompt']):,} characters")
        print(f"Response Length        : N/A")
        print(f"JSON Validation        : FAILED")
        print(f"Processing Time        : {elapsed:.2f} seconds")
        print(f"Status                 : FAILED ({exc})")
        print("-" * 50)