"""
app.py

Gradio user interface for the AI-Powered Grant Proposal Evaluation System.

This module is a PURE PRESENTATION LAYER. It never performs PDF extraction,
text cleaning, embedding generation, ChromaDB retrieval, Gemini evaluation,
or score calculation itself. Its only job is to:

    1. Accept a proposal PDF upload from the user.
    2. Call main.evaluate_proposal(pdf_path) — the single integration point
       for the entire backend pipeline (proposal processing, retrieval,
       Gemini evaluation, scoring, and both HTML/PDF report generation).
    3. Display the returned results.
    4. Offer the already-generated HTML and PDF reports for download.

No report is regenerated in this module — the HTML and PDF report paths
returned by main.evaluate_proposal() are used directly.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import gradio as gr

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

# app.py lives at src/ui/app.py; main.py lives at the project root. main.py
# adds src/proposal_evaluation/ to sys.path for its own imports, so
# importing main here also makes that backend chain available transitively.
# The project root is added to sys.path (rather than relying on a package
# install step or __init__.py files) so `import main` resolves regardless
# of the working directory app.py is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from main import (  # noqa: E402
    evaluate_proposal as run_evaluation_pipeline,
    PipelineError,
    ProposalFileNotFoundError,
    ProposalProcessingError,
    ContextRetrievalError,
    GeminiEvaluationPipelineError,
    ScoringPipelineError,
    HtmlReportPipelineError,
    PdfReportPipelineError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "AI Powered Grant Proposal Evaluation System"
APP_SUBTITLE = (
    "Evaluate grant proposals against donor funding criteria using "
    "Retrieval Augmented Generation (RAG), ChromaDB, and Google Gemini."
)

# Display order and labels for the Evaluation Scores dataframe, kept as a
# single source of truth here so the table's row order matches the order
# specified for the report (Relevance, Feasibility, Innovation,
# Sustainability, Budget Justification, Organizational Capacity, Expected
# Impact) regardless of key order in the underlying criteria_scores dict.
CRITERIA_DISPLAY_ORDER: tuple[tuple[str, str], ...] = (
    ("relevance", "Relevance"),
    ("feasibility", "Feasibility"),
    ("innovation", "Innovation"),
    ("sustainability", "Sustainability"),
    ("budget_justification", "Budget Justification"),
    ("organizational_capacity", "Organizational Capacity"),
    ("expected_impact", "Expected Impact"),
)

# Colours for the classification badge, matching the palette used in
# generate_html.py so the UI and the downloadable reports present the
# proposal's classification consistently.
CLASSIFICATION_COLOURS: dict[str, dict[str, str]] = {
    "Highly Recommended": {"bg": "#E9F6EC", "text": "#155A26", "border": "#1E7A34"},
    "Recommended": {"bg": "#EAF1FB", "text": "#154A94", "border": "#1D5DBF"},
    "Needs Revision": {"bg": "#FBF1E3", "text": "#8F580A", "border": "#C2760F"},
    "Not Recommended": {"bg": "#FBEAE9", "text": "#8C1D17", "border": "#B3261E"},
}
_DEFAULT_CLASSIFICATION_COLOUR = {"bg": "#EEF0F2", "text": "#3E454E", "border": "#5A6472"}

LOG_DIR = Path("logs/system_logs")
LOG_FILE = LOG_DIR / "ui.log"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logger() -> logging.Logger:
    """Configure and return this module's dedicated logger.

    The UI layer logs its own events (uploads received, pipeline outcomes
    as observed from this layer) separately from main.py's pipeline.log,
    which already captures the detailed stage-by-stage backend log.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("proposal_evaluation.ui")
    logger.setLevel(logging.DEBUG)

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
# Friendly error messages
# ---------------------------------------------------------------------------

# Maps each specific pipeline exception type to a user-facing message. Kept
# as an ordered tuple (rather than a dict keyed by exception type) so
# isinstance() checks against subclasses still work correctly even though
# several of these exception classes share a common PipelineError base.
_FRIENDLY_ERROR_MESSAGES: tuple[tuple[type[Exception], str], ...] = (
    (
        ProposalFileNotFoundError,
        "The uploaded file could not be found. Please try uploading it again.",
    ),
    (
        ProposalProcessingError,
        "The uploaded file could not be processed. Please confirm it is a "
        "valid, readable PDF and try again.",
    ),
    (
        ContextRetrievalError,
        "The system could not retrieve donor context for this proposal. "
        "Please try again shortly, or contact support if the issue persists.",
    ),
    (
        GeminiEvaluationPipelineError,
        "The AI evaluation service could not complete this evaluation. "
        "Please try again shortly, or contact support if the issue persists.",
    ),
    (
        ScoringPipelineError,
        "An error occurred while scoring this proposal. Please try again, "
        "or contact support if the issue persists.",
    ),
    (
        HtmlReportPipelineError,
        "The evaluation completed, but the HTML report could not be "
        "generated. Please try again or contact support.",
    ),
    (
        PdfReportPipelineError,
        "The evaluation completed, but the PDF report could not be "
        "generated. Please try again or contact support.",
    ),
    (
        PipelineError,
        "An error occurred while evaluating this proposal. Please try "
        "again, or contact support if the issue persists.",
    ),
)


def _friendly_error_message(exc: Exception) -> str:
    """Translate a raised exception into a user-friendly status message.

    Deliberately never includes the raw exception text or a traceback in
    the returned message — those are logged (see logger.error calls below)
    but never shown in the UI, per the requirement that Python tracebacks
    must never be exposed to the user.

    Args:
        exc: The exception raised during pipeline execution.

    Returns:
        A short, user-friendly message describing what went wrong.
    """
    for exception_type, message in _FRIENDLY_ERROR_MESSAGES:
        if isinstance(exc, exception_type):
            return message

    # Anything not part of the known PipelineError hierarchy is treated as
    # a fully unexpected failure — still no traceback shown to the user.
    return (
        "An unexpected error occurred while evaluating this proposal. "
        "Please try again, or contact support if the issue persists."
    )


# ---------------------------------------------------------------------------
# Result formatting helpers
# ---------------------------------------------------------------------------

def _format_score_badge(overall_score: float) -> str:
    """Render the overall score as a Markdown heading.

    Args:
        overall_score: The overall weighted score (0-100).

    Returns:
        A Markdown string displaying the score prominently.
    """
    return f"## {overall_score:.2f}%"


def _format_classification_badge(classification: str) -> str:
    """Render the classification as a coloured Markdown/HTML badge.

    Gradio's Markdown component renders embedded HTML, which is used here
    to apply the classification-specific background/text/border colours
    (Highly Recommended = green, Recommended = blue, Needs Revision =
    orange, Not Recommended = red) without requiring a custom component.

    Args:
        classification: The classification label returned by the pipeline.

    Returns:
        An HTML string (to be rendered inside a gr.Markdown component)
        showing the classification as a coloured pill/badge.
    """
    colours = CLASSIFICATION_COLOURS.get(classification, _DEFAULT_CLASSIFICATION_COLOUR)
    return (
        f'<span style="display:inline-block; padding:6px 16px; '
        f'border-radius:999px; font-weight:700; font-size:14px; '
        f'background-color:{colours["bg"]}; color:{colours["text"]}; '
        f'border:1.5px solid {colours["border"]};">'
        f"{classification}</span>"
    )


def _format_list_as_lines(items: list[Any], empty_message: str) -> str:
    """Render a list of strings as one-item-per-line text for a textbox.

    Args:
        items: The list of strings to display (e.g. strengths, risk_flags).
        empty_message: Text to display if items is empty.

    Returns:
        A newline-joined string, or empty_message if items is empty.
    """
    if not items:
        return empty_message
    return "\n".join(str(item) for item in items)


def _build_scores_dataframe(criteria_scores: dict[str, int]) -> list[list[Any]]:
    """Build the row data for the Evaluation Scores gr.Dataframe.

    Args:
        criteria_scores: The criteria_scores dictionary from the
            evaluation result.

    Returns:
        A list of [criterion_label, score] rows, in the fixed display
        order defined by CRITERIA_DISPLAY_ORDER.
    """
    return [
        [label, criteria_scores.get(key, "N/A")]
        for key, label in CRITERIA_DISPLAY_ORDER
    ]


# ---------------------------------------------------------------------------
# Empty / initial UI state
# ---------------------------------------------------------------------------

def _empty_scores_dataframe() -> list[list[Any]]:
    """Return placeholder dataframe rows for the initial, pre-evaluation UI state."""
    return [[label, ""] for _key, label in CRITERIA_DISPLAY_ORDER]


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

def handle_evaluate_click(
    pdf_file: Any,
) -> tuple[Any, ...]:
    """Handle a click of the "Evaluate Proposal" button.

    This is the only place in the UI that calls into the backend, via
    main.evaluate_proposal(). It performs no evaluation logic itself: it
    extracts the uploaded file's path, calls the pipeline, and maps the
    returned dictionary (or a caught exception) onto the UI's output
    components.

    Args:
        pdf_file: The value of the gr.File upload component. Gradio
            provides this as a filepath string (since the upload component
            is configured with type="filepath"), or None if nothing has
            been uploaded.

    Returns:
        A tuple of values for every output component, in the exact order
        they are wired in the Blocks definition below:
            (status, score_badge, classification_badge, proposal_summary,
             funding_alignment, scores_dataframe, strengths, weaknesses,
             risk_flags, html_download, pdf_download, evaluate_button)
    """
    # --- No file uploaded -------------------------------------------------
    if not pdf_file:
        logger.info("Evaluate clicked with no file uploaded.")
        return (
            "Please upload a proposal PDF before evaluating.",
            "## —",
            "",
            "",
            "",
            _empty_scores_dataframe(),
            "",
            "",
            "",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=True),
        )

    pdf_path = str(pdf_file)
    logger.info("Evaluate clicked for uploaded file: %s", pdf_path)

    try:
        result = run_evaluation_pipeline(pdf_path)
    except Exception as exc:  # noqa: BLE001 - UI boundary must catch everything
        # Every exception, expected (PipelineError and its subclasses) or
        # not, is logged here with its full traceback for developer
        # diagnosis, while only a short, friendly message ever reaches the
        # UI itself — per the requirement to never expose tracebacks to
        # the user.
        logger.error(
            "Evaluation failed for '%s'.\n%s", pdf_path, traceback.format_exc()
        )
        friendly_message = _friendly_error_message(exc)
        return (
            f"Error: {friendly_message}",
            "## —",
            "",
            "",
            "",
            _empty_scores_dataframe(),
            "",
            "",
            "",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=True),
        )

    evaluation = result["evaluation"]
    html_report_path = result["html_report"]
    pdf_report_path = result["pdf_report"]

    overall_score = evaluation["overall_score"]
    classification = evaluation["classification"]

    logger.info(
        "Evaluation succeeded for '%s': overall_score=%.2f, classification=%s.",
        pdf_path,
        overall_score,
        classification,
    )

    # --- Missing report files on disk --------------------------------------
    # main.evaluate_proposal() only returns these paths after generate_html.py
    # / generate_pdf.py have already succeeded, so this is a defensive check
    # against the file having been deleted or moved between generation and
    # this point, rather than an expected failure mode.
    html_exists = Path(html_report_path).exists()
    pdf_exists = Path(pdf_report_path).exists()
    if not html_exists or not pdf_exists:
        logger.error(
            "Report file(s) missing after successful evaluation. "
            "HTML exists=%s (%s), PDF exists=%s (%s).",
            html_exists,
            html_report_path,
            pdf_exists,
            pdf_report_path,
        )

    return (
        "Completed.",
        _format_score_badge(overall_score),
        _format_classification_badge(classification),
        evaluation["proposal_summary"],
        evaluation["funding_alignment_analysis"],
        _build_scores_dataframe(evaluation["criteria_scores"]),
        _format_list_as_lines(evaluation["strengths"], "No strengths identified."),
        _format_list_as_lines(evaluation["weaknesses"], "No weaknesses identified."),
        _format_list_as_lines(evaluation["risk_flags"], "No significant risks identified."),
        gr.update(value=html_report_path if html_exists else None, visible=html_exists),
        gr.update(value=pdf_report_path if pdf_exists else None, visible=pdf_exists),
        gr.update(interactive=True),
    )


def handle_evaluate_start() -> tuple[Any, Any]:
    """Handle the immediate UI update when the Evaluate button is first clicked.

    Runs before handle_evaluate_click() (via a two-step .click() chain), so
    the button disables and the status message updates immediately, rather
    than only after the full pipeline (which can take upwards of a minute)
    completes.

    Returns:
        A tuple of (status_message_update, button_update).
    """
    return "Processing proposal...", gr.update(interactive=False)


# ---------------------------------------------------------------------------
# UI construction
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application.

    Returns:
        The fully wired gr.Blocks app, ready to be launched.
    """
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_SUBTITLE)

        with gr.Row():
            # -----------------------------------------------------------
            # Left panel — upload, evaluate, status
            # -----------------------------------------------------------
            with gr.Column(scale=1):
                pdf_upload = gr.File(
                    label="Upload Proposal PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                evaluate_button = gr.Button("Evaluate Proposal", variant="primary")
                status_box = gr.Textbox(
                    label="Evaluation Status",
                    value="Waiting for upload...",
                    interactive=False,
                )

            # -----------------------------------------------------------
            # Right panel — results
            # -----------------------------------------------------------
            with gr.Column(scale=2):
                with gr.Row():
                    score_badge = gr.Markdown("## —", label="Overall Score")
                    classification_badge = gr.Markdown("", label="Classification")

                proposal_summary_box = gr.Textbox(
                    label="Proposal Summary", lines=5, interactive=False
                )
                funding_alignment_box = gr.Textbox(
                    label="Funding Alignment Analysis", lines=5, interactive=False
                )
                scores_dataframe = gr.Dataframe(
                    headers=["Criterion", "Score"],
                    value=_empty_scores_dataframe(),
                    label="Evaluation Scores",
                    interactive=False,
                )
                strengths_box = gr.Textbox(
                    label="Strengths", lines=4, interactive=False
                )
                weaknesses_box = gr.Textbox(
                    label="Weaknesses", lines=4, interactive=False
                )
                risk_flags_box = gr.Textbox(
                    label="Risk Flags", lines=4, interactive=False
                )

                with gr.Row():
                    html_download = gr.DownloadButton(
                        label="Download HTML Report", visible=False
                    )
                    pdf_download = gr.DownloadButton(
                        label="Download PDF Report", visible=False
                    )

        # -----------------------------------------------------------------
        # Wiring: two-step click chain so the UI updates immediately
        # (button disabled, status set to "Processing...") before the
        # potentially long-running pipeline call begins, then the second
        # step runs the actual pipeline and populates every result field.
        # -----------------------------------------------------------------
        evaluate_button.click(
            fn=handle_evaluate_start,
            inputs=None,
            outputs=[status_box, evaluate_button],
        ).then(
            fn=handle_evaluate_click,
            inputs=[pdf_upload],
            outputs=[
                status_box,
                score_badge,
                classification_badge,
                proposal_summary_box,
                funding_alignment_box,
                scores_dataframe,
                strengths_box,
                weaknesses_box,
                risk_flags_box,
                html_download,
                pdf_download,
                evaluate_button,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)