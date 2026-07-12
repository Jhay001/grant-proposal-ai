"""
main.py

Orchestration layer for the AI-Powered Grant Proposal Evaluation System.

This module coordinates the complete proposal evaluation pipeline, from a
proposal PDF on disk to a final, scored evaluation dictionary. It does not
implement any evaluation logic itself — it only imports and calls the
existing functions already implemented in each pipeline stage, in the
correct order, and handles cross-stage error propagation and logging.

Pipeline:

    Proposal PDF
            |
    process_proposal.process_proposal()
            |
    retrieve_context.retrieve_context()
            |
    gemini_evaluator.evaluate_proposal()
            |
    scoring_engine.score_proposal()
            |
    Final evaluation dictionary

NOTE: The reporting stages (generate_html.py and generate_pdf.py) are not
yet implemented and are deliberately NOT called from this module.

NAMING NOTE:
gemini_evaluator.py already exposes a function named `evaluate_proposal`
(signature: evaluate_proposal(retrieval_data: dict) -> dict), which is a
different function from this module's own public entry point, also named
`evaluate_proposal` (signature: evaluate_proposal(pdf_path: str) -> dict).
To avoid a name collision without modifying gemini_evaluator.py, that
import is aliased below as `run_gemini_evaluation`.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

# The four pipeline modules live under src/proposal_evaluation/ and are each
# designed to also run standalone (each has its own `if __name__ ==
# "__main__":` block), meaning they do not assume they are part of an
# installed package. To import them cleanly from main.py at the project
# root without requiring __init__.py files or a package install step, the
# module directory is added to sys.path before importing.
_PROPOSAL_EVALUATION_DIR = Path(__file__).resolve().parent / "src" / "proposal_evaluation"
if str(_PROPOSAL_EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(_PROPOSAL_EVALUATION_DIR))

from process_proposal import process_proposal  # noqa: E402
from retrieve_context import retrieve_context  # noqa: E402
from gemini_evaluator import (  # noqa: E402
    evaluate_proposal as run_gemini_evaluation,
    GeminiEvaluationError,
)
from scoring_engine import (  # noqa: E402
    score_proposal,
    ScoringEngineError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs/system_logs")
LOG_FILE = LOG_DIR / "pipeline.log"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Base exception for all pipeline-orchestration failures raised by main.py.

    Each pipeline stage (process_proposal, retrieve_context,
    gemini_evaluator, scoring_engine) already raises its own descriptive
    exceptions on failure. This module wraps those in stage-specific
    PipelineError subclasses (via `raise ... from exc`) so callers of
    evaluate_proposal() can catch a single, consistent exception hierarchy
    at the orchestration level, while the original underlying exception and
    traceback remain available via `__cause__`.
    """


class ProposalFileNotFoundError(PipelineError):
    """Raised when the supplied proposal PDF path does not exist."""


class ProposalProcessingError(PipelineError):
    """Raised when process_proposal.process_proposal() fails."""


class ContextRetrievalError(PipelineError):
    """Raised when retrieve_context.retrieve_context() fails."""


class GeminiEvaluationPipelineError(PipelineError):
    """Raised when gemini_evaluator.evaluate_proposal() fails."""


class ScoringPipelineError(PipelineError):
    """Raised when scoring_engine.score_proposal() fails."""


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logger() -> logging.Logger:
    """Configure and return this module's dedicated logger.

    A dedicated logger (rather than the root logger) keeps this module's
    logging behavior independent of the logging configuration each pipeline
    stage module already sets up for itself (process_proposal.log,
    retrieval.log, evaluation.log, scoring_engine.log). pipeline.log
    captures the orchestration-level view: which stages ran, how long each
    took, and where (if anywhere) the pipeline stopped.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("proposal_evaluation.main")
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
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_proposal(pdf_path: str) -> dict[str, Any]:
    """Run the complete grant proposal evaluation pipeline for one PDF.

    Coordinates, in strict order:
        1. process_proposal.process_proposal()  — validate/extract/clean/embed.
        2. retrieve_context.retrieve_context()   — retrieve donor context, build prompt.
        3. gemini_evaluator.evaluate_proposal()  — send prompt to Gemini, parse/validate JSON.
        4. scoring_engine.score_proposal()       — compute weighted score and classification.

    If any stage fails, the pipeline stops immediately — later stages are
    never called with a partial or missing result from an earlier one.

    Args:
        pdf_path: Full path (as a string) to the proposal PDF to evaluate.

    Returns:
        The final evaluation dictionary produced by scoring_engine.py,
        containing proposal_summary, funding_alignment_analysis,
        criteria_scores, strengths, weaknesses, risk_flags,
        final_recommendation, overall_score, and classification.

    Raises:
        ProposalFileNotFoundError: If pdf_path does not point to an
            existing file.
        ProposalProcessingError: If process_proposal() fails (invalid PDF,
            extraction failure, embedding failure, etc.).
        ContextRetrievalError: If retrieve_context() fails (vector store
            unavailable, query failure, etc.).
        GeminiEvaluationPipelineError: If the Gemini evaluation stage fails
            (missing API key, request failure, invalid/unparseable
            response, schema validation failure).
        ScoringPipelineError: If the scoring stage fails (malformed
            criteria_scores reaching this stage).
    """
    pipeline_start_time = time.perf_counter()
    pdf_path_obj = Path(pdf_path)

    logger.info("=" * 70)
    logger.info("Pipeline started.")
    logger.info("Proposal path: %s", pdf_path_obj)

    # -----------------------------------------------------------------
    # Preliminary check: does the file exist at all?
    # -----------------------------------------------------------------
    # process_proposal.validate_pdf() performs its own, more thorough
    # validation (file type, size, PyMuPDF openability) inside
    # process_proposal(), but failing fast here — before importing the
    # embedding model or touching any other module — gives a clear,
    # immediate error for the most common mistake (a wrong or mistyped
    # path) without paying the cost of entering the processing stage.
    if not pdf_path_obj.exists():
        message = f"Proposal PDF not found at: {pdf_path_obj}"
        logger.error(message)
        raise ProposalFileNotFoundError(message)

    # -----------------------------------------------------------------
    # Stage 1 — Proposal processing
    # -----------------------------------------------------------------
    try:
        proposal_data = process_proposal(pdf_path_obj)
        logger.info(
            "Proposal processing completed: '%s' (%d page(s)).",
            proposal_data.get("filename"),
            proposal_data.get("page_count"),
        )
    except Exception as exc:
        logger.error(
            "Proposal processing FAILED for '%s'.\n%s",
            pdf_path_obj,
            traceback.format_exc(),
        )
        raise ProposalProcessingError(
            f"Failed to process proposal PDF '{pdf_path_obj.name}': {exc}"
        ) from exc

    # -----------------------------------------------------------------
    # Stage 2 — Context retrieval
    # -----------------------------------------------------------------
    try:
        retrieval_result = retrieve_context(proposal_data)
        logger.info(
            "Context retrieval completed: %d chunk(s) retrieved.",
            len(retrieval_result.get("retrieved_chunks", [])),
        )
    except Exception as exc:
        logger.error("Context retrieval FAILED.\n%s", traceback.format_exc())
        raise ContextRetrievalError(
            f"Failed to retrieve donor context for '{pdf_path_obj.name}': {exc}"
        ) from exc

    # -----------------------------------------------------------------
    # Stage 3 — Gemini evaluation
    # -----------------------------------------------------------------
    try:
        evaluation = run_gemini_evaluation(retrieval_result)
        logger.info("Gemini evaluation completed.")
    except GeminiEvaluationError as exc:
        logger.error("Gemini evaluation FAILED.\n%s", traceback.format_exc())
        raise GeminiEvaluationPipelineError(
            f"Gemini evaluation failed for '{pdf_path_obj.name}': {exc}"
        ) from exc
    except Exception as exc:
        # Catches anything unexpected that escapes gemini_evaluator.py's
        # own exception hierarchy, so the pipeline never propagates a raw,
        # unannotated exception type to the caller.
        logger.error(
            "Gemini evaluation FAILED with an unexpected error.\n%s",
            traceback.format_exc(),
        )
        raise GeminiEvaluationPipelineError(
            f"Unexpected error during Gemini evaluation for "
            f"'{pdf_path_obj.name}': {exc}"
        ) from exc

    # -----------------------------------------------------------------
    # Stage 4 — Scoring
    # -----------------------------------------------------------------
    try:
        final_result = score_proposal(evaluation)
        logger.info(
            "Scoring completed: overall_score=%.2f, classification=%s.",
            final_result.get("overall_score"),
            final_result.get("classification"),
        )
    except ScoringEngineError as exc:
        logger.error("Scoring FAILED.\n%s", traceback.format_exc())
        raise ScoringPipelineError(
            f"Scoring failed for '{pdf_path_obj.name}': {exc}"
        ) from exc
    except Exception as exc:
        logger.error(
            "Scoring FAILED with an unexpected error.\n%s", traceback.format_exc()
        )
        raise ScoringPipelineError(
            f"Unexpected error during scoring for '{pdf_path_obj.name}': {exc}"
        ) from exc

    # -----------------------------------------------------------------
    # Pipeline complete
    # -----------------------------------------------------------------
    total_elapsed_seconds = time.perf_counter() - pipeline_start_time
    logger.info(
        "Pipeline completed successfully in %.2f second(s).", total_elapsed_seconds
    )

    _print_summary(
        filename=proposal_data.get("filename", pdf_path_obj.name),
        overall_score=final_result.get("overall_score"),
        classification=final_result.get("classification"),
        elapsed_seconds=total_elapsed_seconds,
        status="SUCCESS",
    )

    return final_result


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(
    filename: str,
    overall_score: float | None,
    classification: str | None,
    elapsed_seconds: float,
    status: str,
) -> None:
    """Print the final console summary for a pipeline run.

    Kept as a small, separate function (rather than inlined print
    statements at both the success and failure call sites) so the exact
    summary format is defined in exactly one place.

    Args:
        filename: The proposal's filename.
        overall_score: The computed overall score, or None if the pipeline
            failed before scoring completed.
        classification: The computed classification, or None if the
            pipeline failed before scoring completed.
        elapsed_seconds: Total pipeline processing time in seconds.
        status: "SUCCESS" or a failure description.
    """
    score_display = f"{overall_score:.2f}" if overall_score is not None else "N/A"
    classification_display = classification if classification is not None else "N/A"

    print("-" * 50)
    print("Grant Proposal Evaluation Summary")
    print("-" * 50)
    print(f"Proposal Filename      : {filename}")
    print(f"Overall Score          : {score_display}")
    print(f"Classification         : {classification_display}")
    print(f"Processing Time        : {elapsed_seconds:.2f} seconds")
    print(f"Status                 : {status}")
    print("-" * 50)


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    user_input = input("Enter the full path to a proposal PDF: ").strip().strip('"')

    pipeline_start = time.perf_counter()
    try:
        evaluate_proposal(user_input)
        # Note: evaluate_proposal() already prints the success summary
        # internally (see _print_summary() call above), since the elapsed
        # time and result fields it reports are only fully known inside
        # that function. Nothing further to print here on success.
    except PipelineError as exc:
        elapsed = time.perf_counter() - pipeline_start
        _print_summary(
            filename=Path(user_input).name,
            overall_score=None,
            classification=None,
            elapsed_seconds=elapsed,
            status=f"FAILED ({exc})",
        )
    except Exception as exc:  # noqa: BLE001 - top-level CLI catch is intentional
        # Catches any truly unexpected failure that escaped the
        # PipelineError hierarchy entirely, so standalone execution never
        # exits with a raw, unhandled traceback for the developer.
        elapsed = time.perf_counter() - pipeline_start
        logger.error("Pipeline FAILED with an unhandled error.\n%s", traceback.format_exc())
        _print_summary(
            filename=Path(user_input).name,
            overall_score=None,
            classification=None,
            elapsed_seconds=elapsed,
            status=f"FAILED (unexpected error: {exc})",
        )