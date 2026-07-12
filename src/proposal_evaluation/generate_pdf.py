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
    generate_html.py
            |
    generate_pdf.py   <-- this module

Responsibility (and ONLY responsibility):
    Convert an existing HTML evaluation report (already written to disk by
    generate_html.py) into a PDF version of the same report, saved to
    reports/pdf/.

This module does NOT regenerate the report, does NOT communicate with
Gemini, and does NOT calculate scores. It receives a path to an HTML file
that already exists and produces a PDF rendering of exactly that file,
using WeasyPrint, which preserves the embedded CSS styling already present
in the HTML (fonts, colours, layout, the progress bar, tables, badges, etc.)
without any additional styling logic in this module.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

from weasyprint import HTML

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDF_OUTPUT_DIR = Path("reports/pdf")

LOG_DIR = Path("logs/report_logs")
LOG_FILE = LOG_DIR / "pdf_generation.log"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PdfReportError(Exception):
    """Base exception for all failures raised by this module."""


class HtmlFileNotFoundError(PdfReportError):
    """Raised when the supplied HTML report path does not exist."""


class InvalidHtmlPathError(PdfReportError):
    """Raised when the supplied path exists but is not a usable HTML file."""


class PdfWriteError(PdfReportError):
    """Raised when WeasyPrint fails to render or write the PDF."""


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logger() -> logging.Logger:
    """Configure and return this module's dedicated logger.

    A dedicated logger (rather than the root logger) keeps this module's
    logging behavior independent of the logging configuration each other
    pipeline stage module sets up for itself. It shares the same
    logs/report_logs/ directory as generate_html.py, but writes to its own
    file (pdf_generation.log) so the two reporting stages' logs stay
    cleanly separated.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("proposal_evaluation.generate_pdf")
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

def validate_html_path(html_path: Path) -> None:
    """Validate that html_path points to a usable HTML file.

    Checks performed, in order:
        1. The path exists.
        2. The path is a file (not a directory).
        3. The file extension is ``.html`` (or ``.htm``).
        4. The file is non-empty.

    This module does not parse or otherwise inspect the HTML content
    itself — validating that WeasyPrint can actually render it is left to
    the render_pdf() step, since a lightweight extension/size check here is
    enough to catch the most common mistakes (wrong path, empty file)
    cheaply, before invoking the heavier PDF rendering engine.

    Args:
        html_path: Full path to the HTML report to convert.

    Raises:
        HtmlFileNotFoundError: If the path does not exist.
        InvalidHtmlPathError: If the path is not a file, is not an
            .html/.htm file, or is empty.
    """
    if not html_path.exists():
        raise HtmlFileNotFoundError(f"HTML report not found: {html_path}")

    if not html_path.is_file():
        raise InvalidHtmlPathError(f"Path exists but is not a file: {html_path}")

    if html_path.suffix.lower() not in (".html", ".htm"):
        raise InvalidHtmlPathError(
            f"Expected an .html or .htm file, got extension "
            f"'{html_path.suffix}': {html_path}"
        )

    if html_path.stat().st_size == 0:
        raise InvalidHtmlPathError(f"HTML report is empty (0 bytes): {html_path}")


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------

def resolve_output_path(html_path: Path) -> Path:
    """Derive the output PDF path from the input HTML path.

    The output filename matches the input filename exactly, with the
    extension changed to .pdf (e.g.
    "evaluation_report_20260630_143512.html" ->
    "evaluation_report_20260630_143512.pdf"), and is placed under
    reports/pdf/ regardless of where the source HTML file lives.

    Args:
        html_path: Full path to the source HTML report.

    Returns:
        The full path (not yet created) where the PDF should be written.
    """
    pdf_filename = html_path.stem + ".pdf"
    return PDF_OUTPUT_DIR / pdf_filename


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def render_pdf(html_path: Path, pdf_path: Path) -> None:
    """Render the HTML file at html_path to a PDF at pdf_path using WeasyPrint.

    WeasyPrint parses the HTML file directly from disk (via the `filename`
    argument), which means it resolves the document exactly as a browser
    would from that file's location, including its embedded <style> block.
    Since generate_html.py produces fully self-contained HTML (no external
    stylesheets, fonts, or scripts), no base_url or extra resource
    resolution is required here.

    Args:
        html_path: Full path to the source HTML report.
        pdf_path: Full path where the rendered PDF should be written.

    Raises:
        PdfWriteError: If WeasyPrint fails to parse the HTML or write the
            output PDF for any reason.
    """
    try:
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    except Exception as exc:
        # WeasyPrint can raise a range of exception types depending on the
        # failure (malformed HTML, font resolution issues, filesystem
        # errors writing the output). All are wrapped uniformly here so
        # callers only need to handle one exception type from this module.
        raise PdfWriteError(
            f"WeasyPrint failed to convert '{html_path}' to PDF: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_pdf_report(html_path: str) -> str:
    """Convert an existing HTML evaluation report into a PDF report.

    Workflow: validate HTML path -> resolve output path -> ensure output
    directory exists -> render PDF -> return path.

    This is the only function later pipeline stages (or a Gradio "Download
    PDF" handler) should call into for PDF report generation.

    Args:
        html_path: Full path (as a string) to an HTML report already
            generated by generate_html.generate_html_report().

    Returns:
        The full file path (as a string) of the generated PDF report.

    Raises:
        HtmlFileNotFoundError: If html_path does not point to an existing
            file.
        InvalidHtmlPathError: If html_path exists but is not a usable HTML
            file.
        PdfWriteError: If WeasyPrint fails to render or write the PDF.
        Exception: any other unexpected error is logged with a full
            traceback before being re-raised.
    """
    start_time = time.perf_counter()
    html_path_obj = Path(html_path)

    logger.info("=" * 70)
    logger.info("PDF generation started.")
    logger.info("HTML file used: %s", html_path_obj)

    try:
        validate_html_path(html_path_obj)

        pdf_path_obj = resolve_output_path(html_path_obj)

        # Ensure reports/pdf/ exists before attempting to write into it —
        # WeasyPrint will not create missing intermediate directories on
        # its own.
        PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        render_pdf(html_path_obj, pdf_path_obj)

        elapsed_seconds = time.perf_counter() - start_time
        logger.info("Output filename: %s", pdf_path_obj.name)
        logger.info("PDF generation completed successfully in %.4f second(s).", elapsed_seconds)

        return str(pdf_path_obj)

    except PdfReportError:
        logger.error("PDF generation FAILED.\n%s", traceback.format_exc())
        raise
    except Exception:
        logger.error("PDF generation FAILED with an unexpected error.\n%s", traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Standalone development testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    user_input = input("Enter the full path to the HTML report: ").strip().strip('"')

    print("-" * 50)
    start = time.perf_counter()
    try:
        output_pdf = generate_pdf_report(user_input)
        elapsed = time.perf_counter() - start

        print("PDF Report Generation Summary")
        print("-" * 50)
        print(f"Input HTML File       : {user_input}")
        print(f"Output PDF File       : {output_pdf}")
        print(f"Processing Time       : {elapsed:.4f} seconds")
        print(f"Status                : SUCCESS")
        print("-" * 50)
    except Exception as exc:  # noqa: BLE001 - top-level CLI catch is intentional
        elapsed = time.perf_counter() - start
        print("PDF Report Generation Summary")
        print("-" * 50)
        print(f"Input HTML File       : {user_input}")
        print(f"Output PDF File       : N/A")
        print(f"Processing Time       : {elapsed:.4f} seconds")
        print(f"Status                : FAILED ({exc})")
        print("-" * 50)