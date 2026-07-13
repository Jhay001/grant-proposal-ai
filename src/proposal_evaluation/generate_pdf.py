"""
generate_pdf.py

Belongs to the reporting stage of the AI-Powered Grant Proposal Evaluation
System.

Pipeline position:

    Proposal PDF
            |
    process_proposal.py
            |
    retrieve_context.py
            |
    gemini_evaluator.py
            |
    scoring_engine.py
            |
    generate_pdf.py   <-- this module (ONLY reporting stage)

Responsibility (and ONLY responsibility):
    Render the final evaluation dictionary (produced by scoring_engine.py),
    together with a small set of processing-metadata fields supplied by the
    caller, directly into a professional PDF evaluation report using
    ReportLab — and save it to reports/pdf/.

This module is a PURE RENDERING module. It does not perform any AI
evaluation, does not calculate scores, and does NOT generate, depend on, or
route through HTML in any form. The PDF is built directly from Python data
using ReportLab's Platypus layout engine. Its only file-system interaction
is writing the single PDF report it produces.

VISUAL DESIGN NOTE:
The layout deliberately mirrors the card-based design of this system's
earlier HTML report: a light grey page background with individual white,
rounded-corner "cards" for each section (each card's heading and a thin
gold divider rule live inside the card itself, exactly as the HTML
version's `.card h2` pattern did), rather than plain section headings
floating between horizontal rules. This is achieved natively in ReportLab
via the ROUNDEDCORNERS table style command and a full-page background fill
drawn in the page-furniture callback — no HTML or WeasyPrint is involved.
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDF_OUTPUT_DIR = Path("reports/pdf")

LOG_DIR = Path("logs/report_logs")
LOG_FILE = LOG_DIR / "pdf_generation.log"

NAVY = colors.HexColor("#1B3A5C")
GOLD = colors.HexColor("#B8862B")
GREY = colors.HexColor("#4A4A4A")

PAGE_BG = colors.HexColor("#F4F6F8")
CARD_BG = colors.white
CARD_BORDER = colors.HexColor("#E2E6EA")
TABLE_ALT_ROW_BG = colors.HexColor("#F7F9FB")

CARD_CORNER_RADIUS = 10
CARD_WIDTH = 6.3 * inch

MIN_SCORE = 1
MAX_SCORE = 5

CRITERIA_DISPLAY_ORDER: tuple[tuple[str, str], ...] = (
    ("relevance", "Relevance"),
    ("feasibility", "Feasibility"),
    ("innovation", "Innovation"),
    ("sustainability", "Sustainability"),
    ("budget_justification", "Budget Justification"),
    ("organizational_capacity", "Organizational Capacity"),
    ("expected_impact", "Expected Impact"),
)

CLASSIFICATION_COLOURS: dict[str, dict[str, colors.Color]] = {
    "Highly Recommended": {
        "base": colors.HexColor("#1E7A34"),
        "bg": colors.HexColor("#E9F6EC"),
        "text": colors.HexColor("#155A26"),
    },
    "Recommended": {
        "base": colors.HexColor("#1D5DBF"),
        "bg": colors.HexColor("#EAF1FB"),
        "text": colors.HexColor("#154A94"),
    },
    "Needs Revision": {
        "base": colors.HexColor("#C2760F"),
        "bg": colors.HexColor("#FBF1E3"),
        "text": colors.HexColor("#8F580A"),
    },
    "Not Recommended": {
        "base": colors.HexColor("#B3261E"),
        "bg": colors.HexColor("#FBEAE9"),
        "text": colors.HexColor("#8C1D17"),
    },
}
_DEFAULT_CLASSIFICATION_COLOUR = {
    "base": colors.HexColor("#5A6472"),
    "bg": colors.HexColor("#EEF0F2"),
    "text": colors.HexColor("#3E454E"),
}


class PdfReportError(Exception):
    """Base exception for all failures raised by this module."""


class InvalidEvaluationDataError(PdfReportError):
    """Raised when the input evaluation dictionary or metadata is missing/malformed."""


class PdfWriteError(PdfReportError):
    """Raised when ReportLab fails to render or write the PDF."""


def _configure_logger() -> logging.Logger:
    """Configure and return this module's dedicated logger."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("proposal_evaluation.generate_pdf")
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


def _build_styles():
    """Build and return the ParagraphStyle set used throughout the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle", fontName="Helvetica-Bold", fontSize=21,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=6, leading=25,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", fontName="Helvetica", fontSize=11,
        textColor=GREY, alignment=TA_CENTER, spaceAfter=4, leading=14,
    ))
    styles.add(ParagraphStyle(
        name="CardTitle", fontName="Helvetica-Bold", fontSize=13,
        textColor=NAVY, spaceAfter=0, leading=16,
    ))
    styles.add(ParagraphStyle(
        name="Body", fontName="Helvetica", fontSize=10, leading=14.5,
        alignment=TA_JUSTIFY, textColor=colors.HexColor("#1A1A1A"),
    ))
    styles.add(ParagraphStyle(
        name="BulletBody", fontName="Helvetica", fontSize=10, leading=14,
        alignment=TA_LEFT, textColor=colors.HexColor("#1A1A1A"),
    ))
    styles.add(ParagraphStyle(
        name="TableHead", fontName="Helvetica-Bold", fontSize=9.5,
        textColor=colors.white, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="TableCell", fontName="Helvetica", fontSize=9.3, leading=12.5,
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        name="CoverLabel", fontName="Helvetica", fontSize=9,
        textColor=GREY, alignment=TA_CENTER, spaceAfter=1,
    ))
    styles.add(ParagraphStyle(
        name="CoverValue", fontName="Helvetica-Bold", fontSize=12.5,
        textColor=colors.HexColor("#1A1A1A"), alignment=TA_CENTER, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ScoreHeroLabel", fontName="Helvetica", fontSize=11,
        textColor=GREY, alignment=TA_CENTER, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="ScoreHeroValue", fontName="Helvetica-Bold", fontSize=34,
        alignment=TA_CENTER, leading=38, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="RecommendationLabel", fontName="Helvetica-Bold", fontSize=9.5,
        textColor=GREY, alignment=TA_CENTER, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="RecommendationHeading", fontName="Helvetica-Bold", fontSize=17,
        alignment=TA_CENTER, leading=21, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="RecommendationBody", fontName="Helvetica", fontSize=10,
        alignment=TA_CENTER, leading=14,
    ))
    styles.add(ParagraphStyle(
        name="MetaLabel", fontName="Helvetica-Bold", fontSize=9.5,
        textColor=NAVY, leading=13,
    ))
    styles.add(ParagraphStyle(
        name="MetaValue", fontName="Helvetica", fontSize=9.5,
        textColor=colors.HexColor("#1A1A1A"), leading=13,
    ))
    styles.add(ParagraphStyle(
        name="Footnote", fontName="Helvetica-Oblique", fontSize=8.5,
        textColor=GREY, alignment=TA_CENTER,
    ))

    return styles


def validate_evaluation(
    evaluation: dict[str, Any],
    proposal_filename: str,
    embedding_model: str,
    llm_model: str,
) -> None:
    """Validate all inputs required to render the PDF report."""
    if not isinstance(evaluation, dict):
        raise InvalidEvaluationDataError(
            f"Expected a dictionary from scoring_engine.py, got "
            f"{type(evaluation).__name__}."
        )

    required_text_fields = (
        "proposal_summary",
        "funding_alignment_analysis",
        "final_recommendation",
        "classification",
    )
    for field in required_text_fields:
        if field not in evaluation:
            raise InvalidEvaluationDataError(
                f"Evaluation dictionary is missing required field: '{field}'."
            )
        if not isinstance(evaluation[field], str) or not evaluation[field].strip():
            raise InvalidEvaluationDataError(
                f"Field '{field}' must be a non-empty string."
            )

    required_list_fields = ("strengths", "weaknesses", "risk_flags")
    for field in required_list_fields:
        if field not in evaluation:
            raise InvalidEvaluationDataError(
                f"Evaluation dictionary is missing required field: '{field}'."
            )
        if not isinstance(evaluation[field], list):
            raise InvalidEvaluationDataError(
                f"Field '{field}' must be a list, got "
                f"{type(evaluation[field]).__name__}."
            )

    if "overall_score" not in evaluation:
        raise InvalidEvaluationDataError(
            "Evaluation dictionary is missing required field: 'overall_score'."
        )
    overall_score = evaluation["overall_score"]
    is_valid_number = isinstance(overall_score, (int, float)) and not isinstance(
        overall_score, bool
    )
    if not is_valid_number or not (0 <= overall_score <= 100):
        raise InvalidEvaluationDataError(
            f"'overall_score' must be a number between 0 and 100, got "
            f"{overall_score!r}."
        )

    if "criteria_scores" not in evaluation:
        raise InvalidEvaluationDataError(
            "Evaluation dictionary is missing required field: 'criteria_scores'."
        )
    criteria_scores = evaluation["criteria_scores"]
    if not isinstance(criteria_scores, dict):
        raise InvalidEvaluationDataError(
            f"'criteria_scores' must be a dictionary, got "
            f"{type(criteria_scores).__name__}."
        )

    missing_criteria = [
        key for key, _label in CRITERIA_DISPLAY_ORDER if key not in criteria_scores
    ]
    if missing_criteria:
        raise InvalidEvaluationDataError(
            f"'criteria_scores' is missing required criterion/criteria: {missing_criteria}"
        )

    invalid_scores: list[str] = []
    for key, _label in CRITERIA_DISPLAY_ORDER:
        value = criteria_scores[key]
        is_valid_integer = isinstance(value, int) and not isinstance(value, bool)
        if not is_valid_integer or not (MIN_SCORE <= value <= MAX_SCORE):
            invalid_scores.append(f"{key}={value!r}")
    if invalid_scores:
        raise InvalidEvaluationDataError(
            f"The following criteria_scores are not valid integers in the "
            f"range {MIN_SCORE}-{MAX_SCORE}: {', '.join(invalid_scores)}"
        )

    for label, value in (
        ("proposal_filename", proposal_filename),
        ("embedding_model", embedding_model),
        ("llm_model", llm_model),
    ):
        if not isinstance(value, str) or not value.strip():
            raise InvalidEvaluationDataError(
                f"'{label}' must be a non-empty string, got {value!r}."
            )


def _card(
    body_flowables: list[Any],
    styles,
    title: str | None = None,
    bg: colors.Color = CARD_BG,
    border: colors.Color = CARD_BORDER,
    rule_color: colors.Color = GOLD,
) -> KeepTogether:
    """Wrap a section's content in a rounded-corner "card" table.

    Replicates the earlier HTML report's `.card` pattern: a white (or
    tinted) rounded-corner box with a border, containing an optional bold
    title, a thin divider rule, and the section's body content — all as a
    single ReportLab Table cell so the rounded border wraps the whole
    section as one visual unit. Wrapped in KeepTogether so ReportLab pushes
    the entire card to the next page rather than splitting it mid-card.
    """
    cell_content: list[Any] = []
    if title:
        cell_content.append(Paragraph(title, styles["CardTitle"]))
        cell_content.append(HRFlowable(
            width="100%", thickness=1.25, color=rule_color,
            spaceBefore=4, spaceAfter=10,
        ))
    cell_content.extend(body_flowables)

    card_table = Table([[cell_content]], colWidths=[CARD_WIDTH])
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 1, border),
        ("ROUNDEDCORNERS", [CARD_CORNER_RADIUS] * 4),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 18),
        ("RIGHTPADDING", (0, 0), (-1, -1), 18),
    ]))

    return KeepTogether([card_table, Spacer(1, 14)])


def _get_classification_colours(classification: str) -> dict[str, colors.Color]:
    """Look up the colour palette for a classification label, with a safe fallback."""
    return CLASSIFICATION_COLOURS.get(classification, _DEFAULT_CLASSIFICATION_COLOUR)


def _bulleted_list(items: list[Any], empty_message: str, styles) -> Any:
    """Render a list of strings as a ReportLab bulleted list, or a fallback message."""
    if not items:
        return Paragraph(f"<i>{empty_message}</i>", styles["Body"])

    return ListFlowable(
        [ListItem(Paragraph(str(item), styles["BulletBody"]), spaceAfter=4) for item in items],
        bulletType="bullet", start="circle", leftIndent=18, bulletFontSize=6,
    )


def _score_progress_bar(overall_score: float, classification: str) -> Table:
    """Build a horizontal, rounded-pill progress bar for the overall score."""
    colours = _get_classification_colours(classification)

    total_width = CARD_WIDTH - (0.36 * inch)
    bar_height = 0.2 * inch

    clamped_score = max(0.0, min(100.0, overall_score))
    filled_width = total_width * (clamped_score / 100.0)
    remainder_width = total_width - filled_width

    filled_width = max(filled_width, 1)
    remainder_width = max(remainder_width, 1)

    bar = Table([["", ""]], colWidths=[filled_width, remainder_width], rowHeights=[bar_height])
    bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colours["base"]),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E9ECEF")),
        ("ROUNDEDCORNERS", [bar_height / 2] * 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return bar


def _criteria_scores_table(criteria_scores: dict[str, int], styles) -> Table:
    """Render the Criteria Scores section as a professional ReportLab table."""
    header = [
        Paragraph("Criterion", styles["TableHead"]),
        Paragraph("Score", styles["TableHead"]),
    ]
    rows = [header]
    for key, label in CRITERIA_DISPLAY_ORDER:
        score = criteria_scores[key]
        rows.append([
            Paragraph(label, styles["TableCell"]),
            Paragraph(f"{score} / {MAX_SCORE}", styles["TableCell"]),
        ])

    inner_width = CARD_WIDTH - (0.5 * inch)
    table = Table(rows, colWidths=[inner_width * 0.68, inner_width * 0.32], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROUNDEDCORNERS", [6, 6, 0, 0]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9CED3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TABLE_ALT_ROW_BG]),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _processing_metadata_table(
    embedding_model: str, llm_model: str, evaluation_timestamp: str, styles
) -> Table:
    """Render the Processing Metadata section as a two-column key/value table."""
    rows = [
        [Paragraph("Embedding Model", styles["MetaLabel"]), Paragraph(embedding_model, styles["MetaValue"])],
        [Paragraph("LLM Used", styles["MetaLabel"]), Paragraph(llm_model, styles["MetaValue"])],
        [Paragraph("Evaluation Timestamp", styles["MetaLabel"]), Paragraph(evaluation_timestamp, styles["MetaValue"])],
    ]
    inner_width = CARD_WIDTH - (0.5 * inch)
    table = Table(rows, colWidths=[inner_width * 0.32, inner_width * 0.68])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9CED3")),
        ("BACKGROUND", (0, 0), (0, -1), TABLE_ALT_ROW_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _build_story(
    evaluation: dict[str, Any],
    proposal_filename: str,
    embedding_model: str,
    llm_model: str,
    generated_at: datetime,
) -> list[Any]:
    """Assemble the full list of ReportLab flowables for the PDF report."""
    styles = _build_styles()

    overall_score = float(evaluation["overall_score"])
    classification = evaluation["classification"]
    colours = _get_classification_colours(classification)
    timestamp_display = generated_at.strftime("%B %d, %Y at %I:%M %p")

    story: list[Any] = []

    story.append(Spacer(1, 6))
    story.append(Paragraph("AI Powered Grant Proposal Evaluation Report", styles["ReportTitle"]))
    story.append(Paragraph(
        "Generated by the AI-Powered Grant Proposal Evaluation System",
        styles["ReportSubtitle"],
    ))
    story.append(Spacer(1, 16))

    cover_grid_rows = [
        [
            Paragraph("PROPOSAL FILENAME", styles["CoverLabel"]),
            Paragraph("EVALUATION DATE", styles["CoverLabel"]),
        ],
        [
            Paragraph(proposal_filename, styles["CoverValue"]),
            Paragraph(timestamp_display, styles["CoverValue"]),
        ],
    ]
    inner_width = CARD_WIDTH - (0.36 * inch)
    cover_grid = Table(cover_grid_rows, colWidths=[inner_width / 2, inner_width / 2])
    cover_grid.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))

    score_hero_flowables = [
        Paragraph("OVERALL SCORE", styles["ScoreHeroLabel"]),
        Paragraph(
            f'<font color="#{colours["base"].hexval()[2:]}">{overall_score:.2f}%</font>',
            styles["ScoreHeroValue"],
        ),
        _score_progress_bar(overall_score, classification),
        Spacer(1, 10),
        Paragraph(
            f'<font color="#{colours["text"].hexval()[2:]}"><b>{classification}</b></font>',
            ParagraphStyle("ClassificationLine", parent=styles["ScoreHeroLabel"], fontSize=12, spaceBefore=4),
        ),
    ]

    report_info_body = [cover_grid, Spacer(1, 14)] + score_hero_flowables
    story.append(_card(report_info_body, styles, title="Report Information"))

    story.append(_card(
        [Paragraph(evaluation["proposal_summary"], styles["Body"])],
        styles, title="Executive Summary",
    ))

    story.append(_card(
        [Paragraph(evaluation["funding_alignment_analysis"], styles["Body"])],
        styles, title="Funding Alignment Analysis",
    ))

    recommendation_body = [
        Paragraph("OVERALL RECOMMENDATION", styles["RecommendationLabel"]),
        Paragraph(
            classification,
            ParagraphStyle("RecommendationHeadingColoured", parent=styles["RecommendationHeading"], textColor=colours["text"]),
        ),
        Paragraph(evaluation["final_recommendation"], styles["RecommendationBody"]),
    ]
    story.append(_card(
        recommendation_body, styles, title=None,
        bg=colours["bg"], border=colours["base"], rule_color=colours["base"],
    ))

    story.append(_card(
        [_criteria_scores_table(evaluation["criteria_scores"], styles)],
        styles, title="Criteria Scores",
    ))

    story.append(_card(
        [_bulleted_list(evaluation["strengths"], "No strengths identified.", styles)],
        styles, title="Strengths",
    ))
    story.append(_card(
        [_bulleted_list(evaluation["weaknesses"], "No weaknesses identified.", styles)],
        styles, title="Weaknesses",
    ))
    story.append(_card(
        [_bulleted_list(evaluation["risk_flags"], "No significant risks identified.", styles)],
        styles, title="Risk Flags",
    ))

    story.append(_card(
        [_processing_metadata_table(embedding_model, llm_model, timestamp_display, styles)],
        styles, title="Processing Metadata",
    ))

    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Generated by the AI Powered Grant Proposal Evaluation System.",
        styles["Footnote"],
    ))

    return story


def _draw_page_background_and_footer(canvas, doc) -> None:
    """Paint the full-page background and draw the per-page footer."""
    canvas.saveState()

    page_width, page_height = letter
    canvas.setFillColor(PAGE_BG)
    canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(0.85 * inch, 0.5 * inch, "AI Powered Grant Proposal Evaluation Report")
    canvas.drawRightString(page_width - 0.85 * inch, 0.5 * inch, f"Page {doc.page}")

    canvas.restoreState()


def generate_pdf_report(
    evaluation: dict[str, Any],
    proposal_filename: str,
    embedding_model: str,
    llm_model: str,
) -> str:
    """Render the evaluation dictionary into a professional PDF report.

    This is the only function main.py should call into for report
    generation. It is a pure rendering function: every piece of
    evaluation content in the output PDF comes directly from the
    `evaluation` argument, and the small set of processing-metadata
    fields are passed in explicitly by the caller. No HTML is generated
    or referenced at any point in this process.

    Args:
        evaluation: The final evaluation dictionary returned by
            scoring_engine.score_proposal().
        proposal_filename: The original proposal PDF's filename.
        embedding_model: The name of the embedding model used.
        llm_model: The name of the LLM used to evaluate the proposal.

    Returns:
        The full file path (as a string) of the generated PDF report.

    Raises:
        InvalidEvaluationDataError: If evaluation or any metadata field
            fails validation.
        PdfWriteError: If ReportLab fails to render or write the PDF.
        Exception: any other unexpected error is logged with a full
            traceback before being re-raised.
    """
    start_time = time.perf_counter()
    logger.info("=" * 70)
    logger.info("PDF generation started.")

    try:
        validate_evaluation(evaluation, proposal_filename, embedding_model, llm_model)

        generated_at = datetime.now()
        timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
        output_filename = f"evaluation_report_{timestamp}.pdf"

        PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PDF_OUTPUT_DIR / output_filename

        story = _build_story(evaluation, proposal_filename, embedding_model, llm_model, generated_at)

        try:
            document = SimpleDocTemplate(
                str(output_path),
                pagesize=letter,
                topMargin=0.85 * inch,
                bottomMargin=0.85 * inch,
                leftMargin=0.85 * inch,
                rightMargin=0.85 * inch,
                title="AI Powered Grant Proposal Evaluation Report",
            )
            document.build(
                story,
                onFirstPage=_draw_page_background_and_footer,
                onLaterPages=_draw_page_background_and_footer,
            )
        except Exception as exc:
            raise PdfWriteError(
                f"ReportLab failed to render/write PDF report to '{output_path}': {exc}"
            ) from exc

        elapsed_seconds = time.perf_counter() - start_time
        logger.info("Output filename: %s", output_filename)
        logger.info("PDF generation completed successfully in %.4f second(s).", elapsed_seconds)

        return str(output_path)

    except PdfReportError:
        logger.error("PDF generation FAILED.\n%s", traceback.format_exc())
        raise
    except Exception:
        logger.error("PDF generation FAILED with an unexpected error.\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    mock_evaluation = {
        "proposal_summary": (
            "A community-based digital literacy program for women in rural "
            "Ghana, combining digital skills training with mobile-money "
            "financial literacy to support women's economic inclusion."
        ),
        "funding_alignment_analysis": (
            "The proposal is strongly aligned with GIF's Digital Skills "
            "Development and Women's Economic Inclusion priority areas, "
            "and demonstrates a credible revenue-generation sustainability "
            "model consistent with the Strategic Priorities Framework."
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
        "strengths": [
            "Clear beneficiary targeting among first-generation digital users.",
            "Realistic, itemized budget proportionate to program scale.",
        ],
        "weaknesses": [
            "Limited detail provided on trainer recruitment and retention.",
        ],
        "risk_flags": [
            "Currency fluctuation risk not explicitly addressed in the budget narrative.",
        ],
        "final_recommendation": (
            "Recommend funding, subject to clarification of the trainer "
            "recruitment and retention plan during due diligence."
        ),
        "overall_score": 86.40,
        "classification": "Recommended",
    }

    print("-" * 50)
    start = time.perf_counter()
    try:
        output_file = generate_pdf_report(
            evaluation=mock_evaluation,
            proposal_filename="Ghana_Digital_Literacy_Proposal.pdf",
            embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            llm_model="gemini-3.5-flash",
        )
        elapsed = time.perf_counter() - start

        print("PDF Report Generation Summary")
        print("-" * 50)
        print(f"Output File           : {output_file}")
        print(f"Processing Time       : {elapsed:.4f} seconds")
        print(f"Status                : SUCCESS")
        print("-" * 50)
    except Exception as exc:  # noqa: BLE001 - top-level CLI catch is intentional
        elapsed = time.perf_counter() - start
        print("PDF Report Generation Summary")
        print("-" * 50)
        print(f"Output File           : N/A")
        print(f"Processing Time       : {elapsed:.4f} seconds")
        print(f"Status                : FAILED ({exc})")
        print("-" * 50)