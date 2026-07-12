"""
scoring_engine.py

Belongs to the ONLINE proposal evaluation pipeline of the
AI-Powered Grant Proposal Evaluation System.

Pipeline position:

    Proposal PDF
            |
    process_proposal.py
            |
    retrieve_context.py
            |
    gemini_evaluator.py
            |
    scoring_engine.py   <-- this module
            |
    generate_html.py
            |
    generate_pdf.py

Responsibility (and ONLY responsibility):
    1. Receive the validated evaluation dictionary produced by
       gemini_evaluator.py.
    2. Re-validate the criteria_scores it contains (defense in depth — this
       module does not assume gemini_evaluator.py's validation is the only
       line of defense).
    3. Compute a weighted overall score, normalized to a percentage.
    4. Classify the proposal based on that score.
    5. Append "overall_score" and "classification" to the evaluation
       dictionary and return it.

This module does NOT communicate with Gemini, does NOT touch proposal PDFs,
embeddings, or the vector store, and does NOT generate HTML or PDF reports.
Its only job is deterministic, weighted scoring of criteria that Gemini has
already produced.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Weights are defined as a single source of truth here so the scoring logic,
# the logging output, and any future unit tests all reference the exact same
# values rather than risking drift between duplicated copies.
#
# Weights sum to 1.0 (100%), verified by an assertion below at import time —
# if a future edit accidentally breaks that invariant, the module will fail
# loudly at import rather than silently producing a mis-scaled score.
CRITERIA_WEIGHTS: dict[str, float] = {
    "relevance": 0.25,
    "feasibility": 0.20,
    "expected_impact": 0.20,
    "sustainability": 0.15,
    "budget_justification": 0.10,
    "organizational_capacity": 0.05,
    "innovation": 0.05,
}

_WEIGHT_SUM_TOLERANCE = 1e-9
assert abs(sum(CRITERIA_WEIGHTS.values()) - 1.0) < _WEIGHT_SUM_TOLERANCE, (
    "CRITERIA_WEIGHTS must sum to 1.0 (100%). Check scoring_engine.py constants."
)

MIN_SCORE = 1
MAX_SCORE = 5

# The maximum possible weighted raw score, used to normalize the weighted
# sum onto a 0-100 scale. Since every criterion's max value is MAX_SCORE (5)
# and weights sum to 1.0, the maximum achievable weighted raw score is
# exactly MAX_SCORE (5.0) — but this is computed explicitly, rather than
# hardcoded as 5, so the normalization stays correct even if MAX_SCORE ever
# changes.
_MAX_WEIGHTED_RAW_SCORE = MAX_SCORE * sum(CRITERIA_WEIGHTS.values())

# Classification thresholds, expressed as (minimum_score_inclusive, label).
# Ordered from highest to lowest so classify_proposal() can walk the list
# top-down and return on the first threshold the score meets.
CLASSIFICATION_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (90.0, "Highly Recommended"),
    (80.0, "Recommended"),
    (60.0, "Needs Revision"),
    (0.0, "Not Recommended"),
)

LOG_DIR = Path("logs/evaluation_logs")
LOG_FILE = LOG_DIR / "scoring_engine.log"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ScoringEngineError(Exception):
    """Base exception for all failures raised by this module.

    A dedicated exception type (rather than raw ValueError/KeyError) lets
    calling code in the Gradio layer or main.py catch scoring-specific
    failures distinctly from unrelated bugs elsewhere in the pipeline.
    """


class InvalidEvaluationDataError(ScoringEngineError):
    """Raised when the input evaluation dictionary is missing or malformed."""


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logger() -> logging.Logger:
    """Configure and return this module's dedicated logger.

    A dedicated logger (rather than the root logger) keeps this module's
    logging behavior independent of logging configuration done elsewhere in
    the pipeline (process_proposal.py, retrieve_context.py, and
    gemini_evaluator.py each keep their own logger and log file).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("proposal_evaluation.scoring_engine")
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
# Validation
# ---------------------------------------------------------------------------

def validate_evaluation_data(evaluation: dict[str, Any]) -> dict[str, int]:
    """Validate the evaluation dictionary and return its criteria_scores.

    This module re-validates criteria_scores independently of
    gemini_evaluator.py's own validation. Re-checking here is deliberate
    defense in depth: scoring_engine.py should never trust an upstream
    module's validation as its only safeguard, since a future change to
    gemini_evaluator.py (or a caller that bypasses it entirely, e.g. during
    testing) should not be able to silently corrupt the computed score.

    Args:
        evaluation: The dictionary received from gemini_evaluator.py.

    Returns:
        The validated criteria_scores dictionary, with every required
        criterion present and confirmed to be an integer in [1, 5].

    Raises:
        InvalidEvaluationDataError: If evaluation is not a dict, is missing
            "criteria_scores", is missing any required criterion, or
            contains a score that is not an integer between 1 and 5.
    """
    if not isinstance(evaluation, dict):
        raise InvalidEvaluationDataError(
            f"Expected a dictionary from gemini_evaluator.py, got "
            f"{type(evaluation).__name__}."
        )

    if "criteria_scores" not in evaluation:
        raise InvalidEvaluationDataError(
            "Evaluation dictionary is missing the required 'criteria_scores' key."
        )

    criteria_scores = evaluation["criteria_scores"]
    if not isinstance(criteria_scores, dict):
        raise InvalidEvaluationDataError(
            f"'criteria_scores' must be a dictionary, got "
            f"{type(criteria_scores).__name__}."
        )

    missing_criteria = [
        criterion for criterion in CRITERIA_WEIGHTS if criterion not in criteria_scores
    ]
    if missing_criteria:
        raise InvalidEvaluationDataError(
            f"'criteria_scores' is missing required criterion/criteria: {missing_criteria}"
        )

    invalid_scores: list[str] = []
    for criterion in CRITERIA_WEIGHTS:
        value = criteria_scores[criterion]
        # bool is a subclass of int in Python, so it is explicitly excluded
        # here to prevent True/False from silently passing as a valid
        # score of 1/0.
        is_valid_integer = isinstance(value, int) and not isinstance(value, bool)
        if not is_valid_integer or not (MIN_SCORE <= value <= MAX_SCORE):
            invalid_scores.append(f"{criterion}={value!r}")

    if invalid_scores:
        raise InvalidEvaluationDataError(
            f"The following criteria_scores are not valid integers in the "
            f"range {MIN_SCORE}-{MAX_SCORE}: {', '.join(invalid_scores)}"
        )

    return criteria_scores


# ---------------------------------------------------------------------------
# Weighted score calculation
# ---------------------------------------------------------------------------

def calculate_weighted_score(criteria_scores: dict[str, int]) -> float:
    """Calculate the overall weighted score as a percentage out of 100.

    Each criterion's raw score (1-5) is multiplied by its weight, the
    weighted contributions are summed, and the result is normalized against
    the maximum possible weighted score (5.0, since every criterion's max
    value is 5 and weights sum to 1.0) and scaled to a 0-100 percentage.

    Args:
        criteria_scores: The validated criteria_scores dictionary, as
            returned by validate_evaluation_data().

    Returns:
        The overall score as a percentage, rounded to two decimal places
        (e.g. 86.50).
    """
    weighted_raw_score = sum(
        criteria_scores[criterion] * weight
        for criterion, weight in CRITERIA_WEIGHTS.items()
    )

    overall_score_percentage = (weighted_raw_score / _MAX_WEIGHTED_RAW_SCORE) * 100
    return round(overall_score_percentage, 2)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_proposal(overall_score: float) -> str:
    """Classify a proposal based on its overall weighted score.

    Thresholds:
        90-100   -> "Highly Recommended"
        80-89    -> "Recommended"
        60-79    -> "Needs Revision"
        Below 60 -> "Not Recommended"

    Args:
        overall_score: The overall score as a percentage (0-100).

    Returns:
        The classification label corresponding to the score.

    Raises:
        InvalidEvaluationDataError: If overall_score falls outside the
            valid 0-100 range, which would indicate a bug upstream in
            calculate_weighted_score() rather than a normal evaluation
            outcome.
    """
    if not (0.0 <= overall_score <= 100.0):
        raise InvalidEvaluationDataError(
            f"overall_score {overall_score} is outside the valid 0-100 range. "
            f"This indicates a bug in calculate_weighted_score()."
        )

    # CLASSIFICATION_THRESHOLDS is ordered highest-to-lowest, so the first
    # threshold the score meets or exceeds is always the correct band.
    for minimum_score, label in CLASSIFICATION_THRESHOLDS:
        if overall_score >= minimum_score:
            return label

    # Unreachable in practice, since the last threshold's minimum is 0.0 and
    # overall_score is already confirmed >= 0.0 above — kept only as an
    # explicit safety net rather than allowing a silent fall-through.
    raise InvalidEvaluationDataError(
        f"overall_score {overall_score} did not match any classification threshold."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_proposal(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Compute the overall score and classification for an evaluated proposal.

    Workflow: validate evaluation data -> calculate weighted score ->
    classify -> append results -> return updated dictionary.

    This is the only function the next pipeline stage (generate_html.py /
    generate_pdf.py) should call into for scoring.

    Args:
        evaluation: The validated evaluation dictionary returned by
            gemini_evaluator.evaluate_proposal(), containing at minimum a
            "criteria_scores" key with all seven required criteria.

    Returns:
        The same dictionary, with two additional keys appended:
            - "overall_score": float, the weighted score as a percentage,
              rounded to two decimal places (e.g. 86.50).
            - "classification": str, one of "Highly Recommended",
              "Recommended", "Needs Revision", or "Not Recommended".

    Raises:
        InvalidEvaluationDataError: If the input dictionary fails
            validation (missing/invalid criteria_scores).
        Exception: any other unexpected error is logged with a full
            traceback before being re-raised.
    """
    start_time = time.perf_counter()
    logger.info("=" * 70)
    logger.info("Scoring started.")

    try:
        criteria_scores = validate_evaluation_data(evaluation)

        for criterion, weight in CRITERIA_WEIGHTS.items():
            logger.info(
                "Criterion '%s': score=%d, weight=%.0f%%",
                criterion,
                criteria_scores[criterion],
                weight * 100,
            )

        overall_score = calculate_weighted_score(criteria_scores)
        logger.info("Overall weighted score: %.2f", overall_score)

        classification = classify_proposal(overall_score)
        logger.info("Classification: %s", classification)

        # Append rather than mutate-in-place-only, so the returned
        # dictionary is explicit about what this function adds — though in
        # practice this does update the same dict object the caller passed
        # in, which is intentional: downstream stages (generate_html.py,
        # generate_pdf.py) expect a single, fully-populated evaluation
        # dictionary rather than having to merge two separate objects.
        evaluation["overall_score"] = overall_score
        evaluation["classification"] = classification

        elapsed_seconds = time.perf_counter() - start_time
        logger.info("Scoring completed successfully in %.4f second(s).", elapsed_seconds)

        return evaluation

    except ScoringEngineError:
        logger.error("Scoring FAILED.\n%s", traceback.format_exc())
        raise
    except Exception:
        logger.error("Scoring FAILED with an unexpected error.\n%s", traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Standalone development testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A mock evaluation dictionary matching the shape produced by
    # gemini_evaluator.py, so this module can be exercised independently of
    # a live Gemini call.
    mock_evaluation = {
        "proposal_summary": (
            "A community-based digital literacy program for women in rural "
            "Ghana, combining digital skills training with mobile-money "
            "financial literacy."
        ),
        "funding_alignment_analysis": (
            "Strongly aligned with the Digital Skills Development and "
            "Women's Economic Inclusion priority areas."
        ),
        "criteria_scores": {
            "relevance": 5,
            "feasibility": 4,
            "innovation": 3,
            "sustainability": 4,
            "budget_justification": 4,
            "organizational_capacity": 5,
            "expected_impact": 4,
        },
        "strengths": ["Clear beneficiary targeting", "Realistic budget"],
        "weaknesses": ["Limited detail on trainer recruitment"],
        "risk_flags": ["Currency fluctuation risk not addressed"],
        "final_recommendation": (
            "Recommend funding, subject to clarification of trainer "
            "recruitment plan."
        ),
    }

    print("-" * 50)
    start = time.perf_counter()
    try:
        result = score_proposal(mock_evaluation)
        elapsed = time.perf_counter() - start

        print("Scoring Engine Summary")
        print("-" * 50)
        print(f"Overall Score          : {result['overall_score']:.2f}")
        print(f"Classification         : {result['classification']}")
        print(f"Processing Time        : {elapsed:.4f} seconds")
        print(f"Status                 : SUCCESS")
        print("-" * 50)
    except Exception as exc:  # noqa: BLE001 - top-level CLI catch is intentional
        elapsed = time.perf_counter() - start
        print("Scoring Engine Summary")
        print("-" * 50)
        print(f"Overall Score          : N/A")
        print(f"Classification         : N/A")
        print(f"Processing Time        : {elapsed:.4f} seconds")
        print(f"Status                 : FAILED ({exc})")
        print("-" * 50)