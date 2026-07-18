"""
app.py

Gradio user interface for the AI-Powered Grant Proposal Evaluation System.

This module is a PURE PRESENTATION LAYER. It never performs PDF extraction,
text cleaning, embedding generation, ChromaDB retrieval, Gemini evaluation,
or score calculation itself. Its only job is to:

    1. Accept a proposal PDF upload from the user.
    2. Call main.evaluate_proposal(pdf_path) — the single integration point
       for the entire backend pipeline (proposal processing, retrieval,
       Gemini evaluation, scoring, and PDF report generation).
    3. Display the returned results.
    4. Offer the already-generated PDF report for download.

No report is regenerated in this module — the PDF report path returned by
main.evaluate_proposal() is used directly. The PDF report is the only
report artifact this system produces; there is no HTML report anywhere in
this application, and this module is the application's only other
presentation layer besides that PDF.
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
    PdfReportPipelineError,
)

# ---------------------------------------------------------------------------
# Brand identity
# ---------------------------------------------------------------------------

# The same navy/gold palette used across every other artifact this system
# produces (the GIF donor knowledge base documents and the PDF evaluation
# report), so the UI reads as part of one coherent system rather than a
# visually unrelated demo shell bolted on top of the backend.
NAVY = "#1B3A5C"
NAVY_DARK = "#122A44"
GOLD = "#B8862B"
GOLD_LIGHT = "#E0BC6C"
INK = "#1F2937"
INK_SOFT = "#5A6472"
PAGE_BG = "#F4F6F8"
CARD_BG = "#FFFFFF"
BORDER = "#E2E6EA"

APP_TITLE = "AI Powered Grant Proposal Evaluation System"
APP_SUBTITLE = (
    "Evaluate grant proposals against donor funding criteria using "
    "Retrieval Augmented Generation (RAG), ChromaDB, and Google Gemini."
)

# Display order and labels for the Evaluation Scores dataframe, kept as a
# single source of truth here so the table's row order matches the order
# used in the PDF report (Relevance, Feasibility, Innovation,
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

# Colours for the classification badge, matching the palette used in the
# PDF report (generate_pdf.py) so a proposal's classification reads
# consistently across every artifact a reviewer might see.
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
# Custom theme
# ---------------------------------------------------------------------------

def _build_theme() -> gr.themes.Base:
    """Build the custom navy/gold Gradio theme used by this application.

    Rather than a stock Gradio theme (whose default orange/blue palette
    has no connection to this system's identity), this defines the
    project's own navy/gold brand as first-class theme colours, so
    buttons, focus states, and accents all derive from the same palette
    used throughout the donor documents and the PDF report — not just the
    hand-styled badges layered on top via custom CSS.

    Returns:
        A configured gr.themes.Base instance.
    """
    navy_scale = gr.themes.Color(
        c50="#EAF0F6", c100="#D3E0EC", c200="#A8C2D9", c300="#7CA3C6",
        c400="#4E7EA8", c500="#2C5A85", c600=NAVY, c700="#152E49",
        c800="#102237", c900="#0B1826", c950="#060D15",
    )
    gold_scale = gr.themes.Color(
        c50="#FBF6EC", c100="#F5E9CE", c200="#EBD39D", c300=GOLD_LIGHT,
        c400="#CFA149", c500=GOLD, c600="#93691F", c700="#6E4E17",
        c800="#4A340F", c900="#251A08", c950="#130D04",
    )

    theme = gr.themes.Base(
        primary_hue=navy_scale,
        secondary_hue=gold_scale,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        # Light-mode values.
        body_background_fill=PAGE_BG,
        background_fill_primary=CARD_BG,
        background_fill_secondary=PAGE_BG,
        block_background_fill=CARD_BG,
        block_border_color=BORDER,
        block_border_width="1px",
        block_radius="14px",
        block_shadow="0 1px 3px rgba(16, 24, 40, 0.06)",
        block_label_text_color=INK_SOFT,
        block_title_text_color=NAVY,
        body_text_color=INK,
        body_text_color_subdued=INK_SOFT,
        button_primary_background_fill=NAVY,
        button_primary_background_fill_hover=NAVY_DARK,
        button_primary_text_color="#FFFFFF",
        button_secondary_background_fill="#FFFFFF",
        button_secondary_border_color=GOLD,
        button_secondary_text_color=NAVY,
        input_background_fill="#FFFFFF",
        input_border_color=BORDER,
        input_border_color_focus=NAVY,
        panel_background_fill=PAGE_BG,
        table_even_background_fill=CARD_BG,
        table_odd_background_fill="#F7F9FB",
        table_border_color=BORDER,
        table_text_color=INK,
        # Dark-mode counterparts are pinned to the SAME values as light mode.
        # This system's colours (navy/gold/white cards on a light page) were
        # specifically designed to match the donor documents and PDF report;
        # letting Gradio silently swap to its own dark palette when a
        # reviewer's OS/browser prefers dark mode would break that
        # consistency and — as observed — can also produce illegible
        # text/background colour combinations where custom CSS hardcodes a
        # colour intended for a light background. Pinning every _dark
        # variant to the light value makes the app's appearance
        # deterministic regardless of the viewer's system preference.
        body_background_fill_dark=PAGE_BG,
        background_fill_primary_dark=CARD_BG,
        background_fill_secondary_dark=PAGE_BG,
        block_background_fill_dark=CARD_BG,
        block_border_color_dark=BORDER,
        block_shadow_dark="0 1px 3px rgba(16, 24, 40, 0.06)",
        block_label_text_color_dark=INK_SOFT,
        block_title_text_color_dark=NAVY,
        body_text_color_dark=INK,
        body_text_color_subdued_dark=INK_SOFT,
        button_primary_background_fill_dark=NAVY,
        button_primary_background_fill_hover_dark=NAVY_DARK,
        button_primary_text_color_dark="#FFFFFF",
        button_secondary_background_fill_dark="#FFFFFF",
        button_secondary_border_color_dark=GOLD,
        button_secondary_text_color_dark=NAVY,
        input_background_fill_dark="#FFFFFF",
        input_border_color_dark=BORDER,
        input_border_color_focus_dark=NAVY,
        panel_background_fill_dark=PAGE_BG,
        table_even_background_fill_dark=CARD_BG,
        table_odd_background_fill_dark="#F7F9FB",
        table_border_color_dark=BORDER,
        table_text_color_dark=INK,
    )
    return theme


# Custom CSS layered on top of the theme for the elements Gradio's theming
# system does not directly expose: the header banner, the section-card
# accent rule, and typography for the report-style headings inside the
# results panel. Kept self-contained (no external stylesheet) and scoped
# with specific elem_classes/elem_id selectors to avoid the generic
# type-vs-class specificity collisions the frontend-design guidance warns
# against.
CUSTOM_CSS = f"""
/* ---------------------------------------------------------------------
   DARK-MODE NEUTRALIZATION
   ---------------------------------------------------------------------
   Gradio applies a `.dark` class to the document root when the viewer's
   OS/browser prefers dark mode, and re-scopes its CSS custom properties
   under `:root.dark, :root .dark` (confirmed by inspecting the theme's
   generated CSS directly). Setting dark-mode colours via the Python
   Theme.set(..._dark=...) API did not reliably reach every component
   (verified: card/group panel backgrounds stayed dark while directly
   CSS-targeted elements like the status textbox correctly stayed light
   in both modes). To guarantee a single, deterministic appearance
   regardless of the viewer's system preference, every themeable colour
   variable Gradio exposes is forced here directly, under the same
   selector Gradio itself uses, so there is no dependency on how the
   Python-level theme API maps onto individual components. This is a
   professional reporting tool; its appearance should not vary by
   reviewer's OS setting.
   --------------------------------------------------------------------- */
:root, :root.dark, :root .dark {{
    --body-background-fill: {PAGE_BG};
    --background-fill-primary: {CARD_BG};
    --background-fill-secondary: {PAGE_BG};
    --block-background-fill: {CARD_BG};
    --block-border-color: {BORDER};
    --block-label-background-fill: {CARD_BG};
    --block-label-border-color: {BORDER};
    --block-label-text-color: {INK_SOFT};
    --block-title-background-fill: transparent;
    --block-title-border-color: transparent;
    --block-title-text-color: {NAVY};
    --body-text-color: {INK};
    --body-text-color-subdued: {INK_SOFT};
    --border-color-primary: {BORDER};
    --border-color-accent: {GOLD};
    --button-primary-background-fill: {NAVY};
    --button-primary-background-fill-hover: {NAVY_DARK};
    --button-primary-border-color: {NAVY};
    --button-primary-text-color: #FFFFFF;
    --button-secondary-background-fill: #FFFFFF;
    --button-secondary-border-color: {GOLD};
    --button-secondary-text-color: {NAVY};
    --input-background-fill: #FFFFFF;
    --input-border-color: {BORDER};
    --input-border-color-focus: {NAVY};
    --input-placeholder-color: {INK_SOFT};
    --panel-background-fill: {PAGE_BG};
    --panel-border-color: {BORDER};
    --table-border-color: {BORDER};
    --table-even-background-fill: {CARD_BG};
    --table-odd-background-fill: #F7F9FB;
    --table-text-color: {INK};
    --link-text-color: {NAVY};
    --link-text-color-hover: {NAVY_DARK};
    --link-text-color-visited: {NAVY};
    --link-text-color-active: {NAVY_DARK};
    --checkbox-background-color: #FFFFFF;
    --checkbox-border-color: {BORDER};
    --checkbox-border-color-focus: {NAVY};
    --checkbox-border-color-selected: {NAVY};
    --checkbox-background-color-selected: {NAVY};
    --checkbox-label-background-fill: #FFFFFF;
    --checkbox-label-border-color: {BORDER};
    --checkbox-label-text-color: {INK};
    --code-background-fill: #F7F9FB;
    --error-background-fill: #FBEAE9;
    --error-border-color: #B3261E;
    --error-text-color: #8C1D17;
}}

#gif-header-banner {{
    background: linear-gradient(135deg, {NAVY} 0%, {NAVY_DARK} 100%);
    border-radius: 16px;
    padding: 28px 32px 22px 32px;
    margin-bottom: 18px;
    border-bottom: 3px solid {GOLD};
}}
#gif-header-banner h1 {{
    font-family: 'Source Serif 4', Georgia, serif;
    color: #FFFFFF !important;
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: 0.01em;
}}
#gif-header-banner p {{
    color: #D9E2EC !important;
    font-size: 14.5px;
    margin: 0;
}}

.gif-section-card {{
    border-radius: 14px !important;
}}

/* Gradio renders gr.Markdown content inside a nested `.prose` wrapper with
   its own paragraph-level font-weight/colour rules, which otherwise
   override styles set only on the outer `.gif-section-title` wrapper div.
   Targeting the actual descendant tags directly, with !important, is what
   makes the bold navy heading style actually take effect rather than being
   silently overridden by Gradio's own markdown typography defaults. */
.gif-section-title,
.gif-section-title p,
.gif-section-title span,
.gif-section-title h1,
.gif-section-title h2,
.gif-section-title h3 {{
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: {NAVY} !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    margin: 0 !important;
}}
.gif-section-title {{
    border-bottom: 2px solid {GOLD};
    padding: 2px 4px 8px 6px;
    margin-bottom: 12px !important;
    display: block;
}}

#gif-score-box {{
    text-align: center;
    border-radius: 14px;
    border: 1.5px solid;
    padding: 14px 10px 16px 10px;
}}
#gif-score-box .gif-score-label {{
    font-size: 12.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {INK_SOFT} !important;
    margin-bottom: 4px;
}}
#gif-score-box .gif-score-value {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 46px;
    font-weight: 800;
    line-height: 1.1;
}}

.gif-badge {{
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13.5px;
    border: 1.5px solid;
}}

#gif-status-box textarea,
#gif-status-box textarea:disabled,
#gif-status-box textarea[disabled] {{
    background-color: #FFFFFF !important;
    color: {NAVY} !important;
    font-weight: 600 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: {NAVY} !important;
}}

.gif-footer-note {{
    text-align: center;
    color: {INK_SOFT};
    font-size: 12px;
    margin-top: 6px;
}}
"""


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

def _format_score_badge(overall_score: float, classification: str) -> str:
    """Render the overall score inside a bordered, classification-tinted box.

    Matches the visual treatment already used for the classification badge
    (tinted background, coloured border) rather than plain coloured text,
    so the score reads as an equally prominent focal point rather than
    being visually secondary to the classification badge beside it.

    Args:
        overall_score: The overall weighted score (0-100).
        classification: The classification label, used to colour the
            score box consistently with the classification badge.

    Returns:
        An HTML string (rendered inside a gr.HTML component) showing the
        score prominently inside a coloured box.
    """
    colours = CLASSIFICATION_COLOURS.get(classification, _DEFAULT_CLASSIFICATION_COLOUR)
    return (
        f'<div id="gif-score-box" style="background-color:{colours["bg"]}; '
        f'border-color:{colours["border"]};">'
        f'<div class="gif-score-label">Overall Score</div>'
        f'<div class="gif-score-value" style="color:{colours["border"]};">'
        f"{overall_score:.2f}%</div></div>"
    )


def _format_classification_badge(classification: str) -> str:
    """Render the classification as a coloured badge.

    Args:
        classification: The classification label returned by the pipeline.

    Returns:
        An HTML string (rendered inside a gr.HTML component) showing the
        classification as a coloured pill/badge.
    """
    colours = CLASSIFICATION_COLOURS.get(classification, _DEFAULT_CLASSIFICATION_COLOUR)
    return (
        f'<div style="text-align:center; padding-top:8px;">'
        f'<span class="gif-badge" style="background-color:{colours["bg"]}; '
        f'color:{colours["text"]}; border-color:{colours["border"]};">'
        f"{classification}</span></div>"
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


def _empty_scores_dataframe() -> list[list[Any]]:
    """Return placeholder dataframe rows for the initial, pre-evaluation UI state."""
    return [[label, ""] for _key, label in CRITERIA_DISPLAY_ORDER]


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

def handle_evaluate_click(pdf_file: Any) -> tuple[Any, ...]:
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
             risk_flags, pdf_download, evaluate_button)
    """
    # --- No file uploaded -------------------------------------------------
    if not pdf_file:
        logger.info("Evaluate clicked with no file uploaded.")
        return (
            "Please upload a proposal PDF before evaluating.",
            "",
            "",
            "",
            "",
            _empty_scores_dataframe(),
            "",
            "",
            "",
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
            "",
            "",
            "",
            "",
            _empty_scores_dataframe(),
            "",
            "",
            "",
            gr.update(value=None, visible=False),
            gr.update(interactive=True),
        )

    evaluation = result["evaluation"]
    pdf_report_path = result["pdf_report"]

    overall_score = evaluation["overall_score"]
    classification = evaluation["classification"]

    logger.info(
        "Evaluation succeeded for '%s': overall_score=%.2f, classification=%s.",
        pdf_path,
        overall_score,
        classification,
    )

    # --- Missing report file on disk ----------------------------------------
    # main.evaluate_proposal() only returns this path after
    # generate_pdf.py has already succeeded, so this is a defensive check
    # against the file having been deleted or moved between generation and
    # this point, rather than an expected failure mode.
    pdf_exists = Path(pdf_report_path).exists()
    if not pdf_exists:
        logger.error(
            "PDF report file missing after successful evaluation. Path: %s",
            pdf_report_path,
        )

    return (
        "Completed.",
        _format_score_badge(overall_score, classification),
        _format_classification_badge(classification),
        evaluation["proposal_summary"],
        evaluation["funding_alignment_analysis"],
        _build_scores_dataframe(evaluation["criteria_scores"]),
        _format_list_as_lines(evaluation["strengths"], "No strengths identified."),
        _format_list_as_lines(evaluation["weaknesses"], "No weaknesses identified."),
        _format_list_as_lines(evaluation["risk_flags"], "No significant risks identified."),
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
    theme = _build_theme()

    with gr.Blocks(title=APP_TITLE, theme=theme, css=CUSTOM_CSS) as demo:
        gr.HTML(
            f'<div id="gif-header-banner">'
            f"<h1>{APP_TITLE}</h1>"
            f"<p>{APP_SUBTITLE}</p>"
            f"</div>"
        )

        with gr.Row():
            # -----------------------------------------------------------
            # Left panel — upload, evaluate, status
            # -----------------------------------------------------------
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["gif-section-card"]):
                    gr.Markdown("Submit a Proposal", elem_classes=["gif-section-title"])
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
                        elem_id="gif-status-box",
                    )

            # -----------------------------------------------------------
            # Right panel — results
            # -----------------------------------------------------------
            with gr.Column(scale=2):
                with gr.Group(elem_classes=["gif-section-card"]):
                    with gr.Row():
                        with gr.Column(scale=1):
                            score_badge = gr.HTML("")
                        with gr.Column(scale=1):
                            classification_badge = gr.HTML("")
                    pdf_download = gr.DownloadButton(
                        label="Download PDF Report", visible=False, variant="secondary"
                    )

                with gr.Group(elem_classes=["gif-section-card"]):
                    gr.Markdown("Executive Summary", elem_classes=["gif-section-title"])
                    proposal_summary_box = gr.Textbox(
                        show_label=False, lines=5, interactive=False
                    )

                with gr.Group(elem_classes=["gif-section-card"]):
                    gr.Markdown("Funding Alignment Analysis", elem_classes=["gif-section-title"])
                    funding_alignment_box = gr.Textbox(
                        show_label=False, lines=5, interactive=False
                    )

                with gr.Group(elem_classes=["gif-section-card"]):
                    gr.Markdown("Evaluation Scores", elem_classes=["gif-section-title"])
                    scores_dataframe = gr.Dataframe(
                        headers=["Criterion", "Score"],
                        value=_empty_scores_dataframe(),
                        interactive=False,
                    )

                with gr.Group(elem_classes=["gif-section-card"]):
                    gr.Markdown("Strengths", elem_classes=["gif-section-title"])
                    strengths_box = gr.Textbox(show_label=False, lines=4, interactive=False)

                with gr.Group(elem_classes=["gif-section-card"]):
                    gr.Markdown("Weaknesses", elem_classes=["gif-section-title"])
                    weaknesses_box = gr.Textbox(show_label=False, lines=4, interactive=False)

                with gr.Group(elem_classes=["gif-section-card"]):
                    gr.Markdown("Risk Flags", elem_classes=["gif-section-title"])
                    risk_flags_box = gr.Textbox(show_label=False, lines=4, interactive=False)

        gr.HTML(
            '<div class="gif-footer-note">'
            "Generated by the AI Powered Grant Proposal Evaluation System."
            "</div>"
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