"""
generate_html.py

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
    generate_html.py   <-- this module
            |
    generate_pdf.py

Responsibility (and ONLY responsibility):
    Render the final evaluation dictionary (produced by scoring_engine.py)
    into a professional, self-contained HTML report and write it to
    reports/html/.

This module is a PURE RENDERING module: it performs no AI evaluation, no
score calculation, and no disk reads of any kind. Every piece of content
that appears in the generated HTML comes directly from the evaluation
dictionary passed in by the caller — nothing is hardcoded and nothing is
read from another file. Its only file-system interaction is writing the
single HTML report it produces.
"""

from __future__ import annotations

import html
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_OUTPUT_DIR = Path("reports/html")

LOG_DIR = Path("logs/report_logs")
LOG_FILE = LOG_DIR / "html_generation.log"

# The seven criteria required in criteria_scores, paired with the display
# label used in the report table. Defined once here as a single source of
# truth, in the exact display order specified for the report.
CRITERIA_DISPLAY_ORDER: tuple[tuple[str, str], ...] = (
    ("relevance", "Relevance"),
    ("feasibility", "Feasibility"),
    ("innovation", "Innovation"),
    ("sustainability", "Sustainability"),
    ("budget_justification", "Budget Justification"),
    ("organizational_capacity", "Organizational Capacity"),
    ("expected_impact", "Expected Impact"),
)

# Maps each possible classification label to the hex colours used for its
# badge and progress bar, per the specified colour scheme (Highly
# Recommended = green, Recommended = blue, Needs Revision = orange, Not
# Recommended = red). A neutral grey fallback is included so an unexpected
# classification value never breaks rendering — it just renders in a
# visibly "unstyled" neutral colour, making the anomaly easy to spot rather
# than silently crashing the report generator.
CLASSIFICATION_COLOURS: dict[str, dict[str, str]] = {
    "Highly Recommended": {"base": "#1E7A34", "bg": "#E9F6EC", "text": "#155A26"},
    "Recommended": {"base": "#1D5DBF", "bg": "#EAF1FB", "text": "#154A94"},
    "Needs Revision": {"base": "#C2760F", "bg": "#FBF1E3", "text": "#8F580A"},
    "Not Recommended": {"base": "#B3261E", "bg": "#FBEAE9", "text": "#8C1D17"},
}
_DEFAULT_CLASSIFICATION_COLOUR = {"base": "#5A6472", "bg": "#EEF0F2", "text": "#3E454E"}

MIN_SCORE = 1
MAX_SCORE = 5


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HtmlReportError(Exception):
    """Base exception for all failures raised by this module."""


class InvalidEvaluationDataError(HtmlReportError):
    """Raised when the input evaluation dictionary is missing or malformed."""


class ReportWriteError(HtmlReportError):
    """Raised when the generated HTML cannot be written to disk."""


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logger() -> logging.Logger:
    """Configure and return this module's dedicated logger.

    A dedicated logger (rather than the root logger) keeps this module's
    logging behavior independent of the logging configuration each other
    pipeline stage module sets up for itself.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("proposal_evaluation.generate_html")
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
# Validation
# ---------------------------------------------------------------------------

def validate_evaluation(evaluation: dict[str, Any]) -> None:
    """Validate that the evaluation dictionary contains everything the report needs.

    This module deliberately re-validates its input rather than trusting
    that scoring_engine.py's output is always well-formed by the time it
    reaches here — the report generator should never silently render a
    broken or partial report from malformed data.

    Args:
        evaluation: The dictionary produced by scoring_engine.py.

    Raises:
        InvalidEvaluationDataError: If evaluation is not a dict, is missing
            any required field, or a field has an invalid type/value.
    """
    if not isinstance(evaluation, dict):
        raise InvalidEvaluationDataError(
            f"Expected a dictionary from scoring_engine.py, got "
            f"{type(evaluation).__name__}."
        )

    required_text_fields = (
        "proposal_summary",
        "funding_alignment_analysis",
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


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------

def _escape(value: Any) -> str:
    """HTML-escape a value before embedding it in the report.

    All dynamic text in this report ultimately originates from an LLM
    (Gemini) response. Even though gemini_evaluator.py validates the JSON
    schema, it does not — and should not have to — sanitize the *content*
    of that text for HTML safety, since that is a rendering concern, not an
    evaluation concern. Escaping happens here, at the point of rendering,
    so this module never emits unescaped LLM-produced text directly into
    the HTML document.

    Args:
        value: Any value to render as text (converted to str first).

    Returns:
        The HTML-escaped string representation of value.
    """
    return html.escape(str(value), quote=True)


def _get_classification_colours(classification: str) -> dict[str, str]:
    """Look up the colour palette for a classification label.

    Falls back to a neutral grey palette for any classification value that
    does not match one of the four expected labels, rather than raising —
    the report should still render (visibly, and identifiably, in a
    "something is off" neutral colour) even if an unexpected classification
    string somehow reaches this module.
    """
    return CLASSIFICATION_COLOURS.get(classification, _DEFAULT_CLASSIFICATION_COLOUR)


def _render_bulleted_list(items: list[Any], empty_message: str) -> str:
    """Render a list of strings as an HTML unordered list.

    Args:
        items: The list of strings to render (e.g. strengths, weaknesses).
        empty_message: Text to display if items is empty (e.g. "No
            significant risks identified.").

    Returns:
        An HTML string: either a <ul> of escaped <li> items, or a single
        <p> containing empty_message.
    """
    if not items:
        return f'<p class="empty-state">{_escape(empty_message)}</p>'

    list_items = "\n".join(f"      <li>{_escape(item)}</li>" for item in items)
    return f"    <ul>\n{list_items}\n    </ul>"


def _render_scores_table(
    criteria_scores: dict[str, int], overall_score: float, classification: str
) -> str:
    """Render the Evaluation Scores section as an HTML table.

    Args:
        criteria_scores: The validated criteria_scores dictionary.
        overall_score: The overall weighted score (0-100).
        classification: The classification label.

    Returns:
        An HTML <table> string, including a summary footer row for overall
        score and classification.
    """
    colours = _get_classification_colours(classification)

    rows = []
    for key, label in CRITERIA_DISPLAY_ORDER:
        score = criteria_scores[key]
        rows.append(
            f"      <tr>\n"
            f"        <td>{_escape(label)}</td>\n"
            f"        <td class=\"score-cell\">{score} / {MAX_SCORE}</td>\n"
            f"      </tr>"
        )
    rows_html = "\n".join(rows)

    return f"""    <table class="scores-table">
      <thead>
        <tr>
          <th>Criterion</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
{rows_html}
      </tbody>
      <tfoot>
        <tr class="summary-row">
          <td>Overall Score</td>
          <td class="score-cell">{overall_score:.2f}%</td>
        </tr>
        <tr class="summary-row">
          <td>Classification</td>
          <td>
            <span class="badge" style="background-color: {colours['bg']}; color: {colours['text']}; border-color: {colours['base']};">
              {_escape(classification)}
            </span>
          </td>
        </tr>
      </tfoot>
    </table>"""


def _render_score_progress_bar(overall_score: float, classification: str) -> str:
    """Render a horizontal progress bar visualizing the overall score.

    This is an enhancement beyond the base specification: a pure HTML/CSS
    progress bar filled proportionally to overall_score (e.g. 86.4% fills
    86.4% of the bar's width), coloured according to the proposal's
    classification, giving reviewers an immediate visual read on proposal
    quality before they reach the detailed scores table.

    Args:
        overall_score: The overall weighted score (0-100).
        classification: The classification label, used to colour the bar
            consistently with the classification badge elsewhere in the
            report.

    Returns:
        An HTML string for the progress bar component.
    """
    colours = _get_classification_colours(classification)

    # Clamp defensively: overall_score is already validated to be within
    # 0-100 by validate_evaluation(), but clamping here means the bar's
    # width can never visually overflow its track even if that invariant
    # were ever violated by a future change upstream.
    clamped_score = max(0.0, min(100.0, overall_score))

    return f"""    <div class="progress-section" role="img" aria-label="Overall score {overall_score:.2f} percent">
      <div class="progress-track">
        <div class="progress-fill" style="width: {clamped_score:.2f}%; background-color: {colours['base']};"></div>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# HTML document assembly
# ---------------------------------------------------------------------------

def _build_html_document(evaluation: dict[str, Any], generated_at: datetime) -> str:
    """Assemble the complete, self-contained HTML report document.

    Args:
        evaluation: The validated evaluation dictionary.
        generated_at: The timestamp to display as the report generation date.

    Returns:
        The complete HTML document as a single string, including an
        embedded <style> block. No external stylesheets, scripts, or fonts
        are referenced.
    """
    proposal_summary = evaluation["proposal_summary"]
    funding_alignment_analysis = evaluation["funding_alignment_analysis"]
    criteria_scores = evaluation["criteria_scores"]
    strengths = evaluation["strengths"]
    weaknesses = evaluation["weaknesses"]
    risk_flags = evaluation["risk_flags"]
    overall_score = float(evaluation["overall_score"])
    classification = evaluation["classification"]

    colours = _get_classification_colours(classification)
    generated_at_display = generated_at.strftime("%B %d, %Y at %I:%M %p")

    scores_table_html = _render_scores_table(criteria_scores, overall_score, classification)
    progress_bar_html = _render_score_progress_bar(overall_score, classification)
    strengths_html = _render_bulleted_list(strengths, "No strengths identified.")
    weaknesses_html = _render_bulleted_list(weaknesses, "No weaknesses identified.")
    risk_flags_html = _render_bulleted_list(risk_flags, "No significant risks identified.")

    # The embedded CSS below is self-contained by design (no Bootstrap, no
    # Tailwind, no external stylesheet or font requests), so the generated
    # HTML file renders identically whether opened locally, emailed as an
    # attachment, or embedded in a PDF conversion step later in the
    # pipeline (generate_pdf.py).
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Powered Grant Proposal Evaluation Report</title>
<style>
  :root {{
    --colour-bg: #F4F6F8;
    --colour-card: #FFFFFF;
    --colour-border: #E2E6EA;
    --colour-text-primary: #1F2937;
    --colour-text-secondary: #5A6472;
    --colour-heading: #1B3A5C;
    --colour-accent: #B8862B;
  }}

  * {{
    box-sizing: border-box;
  }}

  body {{
    margin: 0;
    padding: 32px 16px;
    background-color: var(--colour-bg);
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    color: var(--colour-text-primary);
    line-height: 1.55;
  }}

  .report-container {{
    max-width: 860px;
    margin: 0 auto;
  }}

  .report-header {{
    text-align: center;
    margin-bottom: 28px;
  }}

  .report-header h1 {{
    font-size: 26px;
    color: var(--colour-heading);
    margin: 0 0 6px 0;
    font-weight: 700;
  }}

  .report-header .subtitle {{
    color: var(--colour-text-secondary);
    font-size: 14px;
  }}

  .card {{
    background-color: var(--colour-card);
    border: 1px solid var(--colour-border);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
  }}

  .card h2 {{
    font-size: 16px;
    color: var(--colour-heading);
    margin: 0 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--colour-accent);
    display: inline-block;
  }}

  .card p {{
    margin: 0;
    color: var(--colour-text-primary);
    font-size: 14.5px;
  }}

  .report-info-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    align-items: center;
    justify-content: space-between;
  }}

  .report-info-item {{
    flex: 1;
    min-width: 180px;
  }}

  .report-info-item .label {{
    display: block;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--colour-text-secondary);
    margin-bottom: 4px;
  }}

  .report-info-item .value {{
    font-size: 15px;
    font-weight: 600;
    color: var(--colour-text-primary);
  }}

  .score-hero {{
    text-align: center;
    padding: 8px 0 4px 0;
  }}

  .score-hero .score-label {{
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--colour-text-secondary);
    margin-bottom: 4px;
  }}

  .score-hero .score-value {{
    font-size: 44px;
    font-weight: 800;
    color: {colours['base']};
    line-height: 1.1;
  }}

  .progress-section {{
    margin: 16px 0 4px 0;
  }}

  .progress-track {{
    width: 100%;
    height: 16px;
    background-color: #E9ECEF;
    border-radius: 999px;
    overflow: hidden;
  }}

  .progress-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.3s ease;
  }}

  .badge {{
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    border: 1.5px solid;
  }}

  .badge-hero {{
    margin-top: 10px;
  }}

  .scores-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }}

  .scores-table th {{
    text-align: left;
    padding: 10px 12px;
    background-color: var(--colour-heading);
    color: #FFFFFF;
    font-size: 12.5px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}

  .scores-table th:last-child,
  .scores-table td.score-cell {{
    text-align: right;
  }}

  .scores-table td {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--colour-border);
  }}

  .scores-table tbody tr:nth-child(even) {{
    background-color: #F7F9FB;
  }}

  .scores-table tfoot .summary-row td {{
    border-top: 2px solid var(--colour-heading);
    border-bottom: none;
    font-weight: 700;
    padding-top: 12px;
  }}

  ul {{
    margin: 0;
    padding-left: 22px;
  }}

  ul li {{
    margin-bottom: 8px;
    font-size: 14.5px;
  }}

  ul li:last-child {{
    margin-bottom: 0;
  }}

  .empty-state {{
    color: var(--colour-text-secondary);
    font-style: italic;
    font-size: 14px;
  }}

  .report-footer {{
    text-align: center;
    color: var(--colour-text-secondary);
    font-size: 12.5px;
    margin-top: 24px;
    padding-top: 16px;
  }}

  @media (max-width: 600px) {{
    .card {{
      padding: 18px 18px;
    }}
    .score-hero .score-value {{
      font-size: 34px;
    }}
  }}
</style>
</head>
<body>
  <div class="report-container">

    <header class="report-header">
      <h1>AI Powered Grant Proposal Evaluation Report</h1>
      <div class="subtitle">Generated by the AI-Powered Grant Proposal Evaluation System</div>
    </header>

    <section class="card">
      <h2>Report Information</h2>
      <div class="report-info-grid">
        <div class="report-info-item">
          <span class="label">Report Generation Date</span>
          <span class="value">{_escape(generated_at_display)}</span>
        </div>
        <div class="report-info-item">
          <span class="label">Overall Score</span>
          <span class="value">{overall_score:.2f}%</span>
        </div>
        <div class="report-info-item">
          <span class="label">Classification</span>
          <span class="value">
            <span class="badge" style="background-color: {colours['bg']}; color: {colours['text']}; border-color: {colours['base']};">
              {_escape(classification)}
            </span>
          </span>
        </div>
      </div>

      <div class="score-hero">
        <div class="score-label">Overall Score</div>
        <div class="score-value">{overall_score:.2f}%</div>
{progress_bar_html}
      </div>
    </section>

    <section class="card">
      <h2>Executive Summary</h2>
      <p>{_escape(proposal_summary)}</p>
    </section>

    <section class="card">
      <h2>Funding Alignment Analysis</h2>
      <p>{_escape(funding_alignment_analysis)}</p>
    </section>

    <section class="card">
      <h2>Evaluation Scores</h2>
{scores_table_html}
    </section>

    <section class="card">
      <h2>Strengths</h2>
{strengths_html}
    </section>

    <section class="card">
      <h2>Weaknesses</h2>
{weaknesses_html}
    </section>

    <section class="card">
      <h2>Risk Flags</h2>
{risk_flags_html}
    </section>

    <footer class="report-footer">
      Generated automatically by the AI Powered Grant Proposal Evaluation System.
    </footer>

  </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_html_report(evaluation: dict[str, Any]) -> str:
    """Render the evaluation dictionary into an HTML report and save it.

    This is the only function later pipeline stages (generate_pdf.py, or a
    Gradio handler) should call into for HTML report generation. It is a
    pure rendering function: every piece of content in the output HTML
    comes directly from the `evaluation` argument. Nothing is hardcoded,
    and no other file is read from disk.

    Args:
        evaluation: The final evaluation dictionary returned by
            scoring_engine.score_proposal(), containing proposal_summary,
            funding_alignment_analysis, criteria_scores, strengths,
            weaknesses, risk_flags, overall_score, and classification.

    Returns:
        The full file path (as a string) of the generated HTML report.

    Raises:
        InvalidEvaluationDataError: If evaluation fails validation.
        ReportWriteError: If the HTML report cannot be written to disk.
        Exception: any other unexpected error is logged with a full
            traceback before being re-raised.
    """
    start_time = time.perf_counter()
    logger.info("=" * 70)
    logger.info("HTML generation started.")

    try:
        validate_evaluation(evaluation)

        generated_at = datetime.now()
        timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
        output_filename = f"evaluation_report_{timestamp}.html"

        REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORT_OUTPUT_DIR / output_filename

        html_document = _build_html_document(evaluation, generated_at)

        try:
            output_path.write_text(html_document, encoding="utf-8")
        except OSError as exc:
            raise ReportWriteError(
                f"Failed to write HTML report to '{output_path}': {exc}"
            ) from exc

        elapsed_seconds = time.perf_counter() - start_time
        logger.info("Output filename: %s", output_filename)
        logger.info("HTML generation completed successfully in %.4f second(s).", elapsed_seconds)

        return str(output_path)

    except HtmlReportError:
        logger.error("HTML generation FAILED.\n%s", traceback.format_exc())
        raise
    except Exception:
        logger.error("HTML generation FAILED with an unexpected error.\n%s", traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Standalone development testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A mock evaluation dictionary matching the shape produced by
    # scoring_engine.py, so this module can be exercised independently of
    # the rest of the pipeline.
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
        "overall_score": 86.40,
        "classification": "Recommended",
    }

    print("-" * 50)
    start = time.perf_counter()
    try:
        output_file = generate_html_report(mock_evaluation)
        elapsed = time.perf_counter() - start

        print("HTML Report Generation Summary")
        print("-" * 50)
        print(f"Output File           : {output_file}")
        print(f"Processing Time       : {elapsed:.4f} seconds")
        print(f"Status                : SUCCESS")
        print("-" * 50)
    except Exception as exc:  # noqa: BLE001 - top-level CLI catch is intentional
        elapsed = time.perf_counter() - start
        print("HTML Report Generation Summary")
        print("-" * 50)
        print(f"Output File           : N/A")
        print(f"Processing Time       : {elapsed:.4f} seconds")
        print(f"Status                : FAILED ({exc})")
        print("-" * 50)