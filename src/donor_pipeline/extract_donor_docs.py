"""
Donor Document Extraction Pipeline
------------------------------------------------------------------------------

PURPOSE
-------
Donor knowledge-base documents (Funding Guidelines, Eligibility Requirements,
Strategic Priorities Framework, etc.) are supplied as PDFs. Before they can be
chunked, embedded, and indexed into a vector database, their raw text must be extracted
and saved as plain-text files. This module is that first pipeline stage.

It recursively scans `data/donor_documents/` for PDF files, and
for every PDF found:

    1. Validates the file (exists, is a real .pdf, is not empty).
    2. Extracts text page-by-page using PyMuPDF (fitz).
    3. Saves the extracted text to `data/extracted/<same_name>.txt`.
    4. Records per-file statistics (page count, extracted pages, empty
       pages, status, errors) for an audit-friendly extraction log.

WHY THIS DESIGN
---------------
A grant-evaluation system that silently fails to read a donor document (e.g.
because a PDF is scanned/image-only, corrupted, or empty) would later make
evaluation decisions against an incomplete knowledge base, with no way to
trace why. Every step here therefore favors *visibility over silence*:
nothing is skipped without being recorded in the log and reflected in the
final summary, and a single bad file never crashes the whole batch.

USAGE
-----
    python src/donor_pipeline/extract_donor_docs.py

Run directly from the project root (the directory containing `data/`).
------------------------------------------------------------------------------
"""

# Importing necessary libraries
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# PyMuPDF is imported as `fitz` for historical reasons (it was originally
# built on top of the MuPDF C library under that internal name). 
# We guard the import so that a missing dependency produces a clear, actionable
# error instead of a confusing traceback the moment the script is run.
try:
    import fitz  # type: ignore
except ImportError as exc:  # pragma: no cover - environment guard
    sys.stderr.write(
        "ERROR: PyMuPDF is not installed in this environment.\n"
        "Install it with:  python -m pip install PyMuPDF\n"
    )
    raise SystemExit(1) from exc


# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------
# Centralizing paths as constants (rather than hardcoding strings inside
# functions) means the pipeline's folder layout can be changed in one
# place without hunting through the rest of the code.

DONOR_DOCS_DIR: Path = Path("data/donor_documents")
EXTRACTED_DIR: Path = Path("data/extracted")
LOG_DIR: Path = Path("logs/extraction_logs")
LOG_FILE: Path = LOG_DIR / "extraction_log.txt"


# ------------------------------------------------------------------------
# DATA MODEL
# ------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """
    Holds everything we know about the extraction attempt for a single PDF.

    Using a dataclass (instead of passing around loose dicts or tuples)
    gives every result a fixed, type-checked shape, which makes the log
    writer, the summary printer, and any future caller of
    `process_all_pdfs()` all agree on what fields are guaranteed to exist.

    Attributes:
        filename: The PDF's file name only (e.g. "Funding_Guidelines.pdf").
        filepath: The full path the file was found at, for traceability
            when the same filename exists in multiple sub-folders.
        total_pages: Number of pages PyMuPDF reports for the document.
        extracted_pages: Number of pages that yielded non-empty text.
        empty_pages: Number of pages that yielded no extractable text
            (common for scanned/image-only pages with no OCR layer).
        status: One of "SUCCESS", "PARTIAL", "EMPTY", or "FAILED".
        error_message: Human-readable description of any problems
            encountered; empty string if there were none.
    """
    filename: str
    filepath: str
    total_pages: int = 0
    extracted_pages: int = 0
    empty_pages: int = 0
    status: str = "FAILED"
    error_message: str = ""
    errors: List[str] = field(default_factory=list)


# ------------------------------------------------------------------------
# STEP 1 — DISCOVERY
# ------------------------------------------------------------------------

def find_all_pdfs(root: Path) -> List[Path]:
    """
    Recursively find every PDF file under `root`, including in nested
    sub-folders.

    Why recursive: donor organizations frequently provide their documents
    already split into folders (e.g. "2024/", "Eligibility/", "Archived/").
    A flat scan of `root` only would silently miss those, so we use
    `Path.rglob` to walk the entire directory tree.

    Why a manual suffix check rather than `rglob("*.pdf")` alone: on
    case-sensitive filesystems (Linux/macOS) a pattern of "*.pdf" will not
    match "Report.PDF". Donor-supplied files are not guaranteed to be
    lowercase, so we glob everything and filter on a case-insensitive
    suffix check instead.

    Args:
        root: Directory to scan (expected: data/donor_documents/).

    Returns:
        A sorted list of unique, fully-resolved Paths to PDF files.
        Sorting makes log output and console output deterministic and
        easy to diff between runs.
    """
    pdf_files = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    }
    return sorted(pdf_files)


# ------------------------------------------------------------------------
# STEP 2 — VALIDATION
# ------------------------------------------------------------------------

def validate_pdf(file_path: Path) -> Tuple[bool, str]:
    """
    Run basic sanity checks on a candidate PDF before attempting extraction.

    Why this exists as its own step (rather than just trying to open the
    file and catching whatever exception fitz raises): cheap, specific
    checks let us report a precise, useful reason for rejection ("file is
    0 bytes") rather than a generic library error, and they avoid handing
    obviously-bad input (a missing file, a renamed .docx) to the PDF
    parser at all.

    Checks performed, in order:
        1. The path exists.
        2. The path points to an actual file (not a directory/symlink to one).
        3. The extension is ".pdf" (case-insensitive).
        4. The file size is greater than 0 bytes.

    Args:
        file_path: Path to the candidate PDF file.

    Returns:
        A tuple (is_valid, reason):
            - is_valid: True if all checks passed, False otherwise.
            - reason: "Valid" on success, or a human-readable explanation
              of which check failed.
    """
    if not file_path.exists():
        return False, "File does not exist"

    if not file_path.is_file():
        return False, "Path exists but is not a regular file"

    if file_path.suffix.lower() != ".pdf":
        return False, f"Invalid file extension: '{file_path.suffix}' (expected .pdf)"

    try:
        file_size = file_path.stat().st_size
    except OSError as exc:
        # Covers rare cases like permission errors or a file removed
        # between discovery and validation (race condition on shared drives).
        return False, f"Could not read file size: {exc}"

    if file_size <= 0:
        return False, "File size is 0 bytes (empty file)"

    return True, "Valid"


# ------------------------------------------------------------------------
# STEP 3 — TEXT EXTRACTION
# ------------------------------------------------------------------------

def extract_text_from_pdf(file_path: Path) -> Dict[str, Any]:
    """
    Extract text from every page of a validated PDF using PyMuPDF.

    Why page-by-page rather than one bulk call: tracking which specific
    pages were empty or errored is a hard requirement of this pipeline
    (donor documents sometimes mix text pages with scanned image pages,
    e.g. a signed cover letter inserted into an otherwise text-based
    policy document). Per-page handling also means one bad page cannot
    silently swallow the rest of an otherwise-good document.

    Error handling strategy:
        - If the document itself cannot be opened (corrupted file,
          password-protected, not actually a PDF despite the extension),
          we catch that once and return immediately with total_pages = 0.
        - If an individual page raises an error during extraction, we
          record the error, count that page as empty, and continue to
          the next page rather than aborting the whole document.

    Args:
        file_path: Path to a PDF file that has already passed
            `validate_pdf()`.

    Returns:
        A dictionary with the following keys:
            "page_texts" (List[str]): Extracted text for each page, in
                page order. Pages with no extractable text are
                represented as empty strings (never None), so the list
                length always equals "total_pages".
            "total_pages" (int): Number of pages in the document
                (0 if the document failed to open).
            "extracted_pages" (int): Number of pages with non-empty
                extracted text.
            "empty_pages" (int): Number of pages with no extractable text,
                whether due to a blank/image-only page or a page-level
                extraction error.
            "errors" (List[str]): Human-readable descriptions of any
                document-level or page-level errors encountered.
    """
    page_texts: List[str] = []
    total_pages = 0
    extracted_pages = 0
    empty_pages = 0
    errors: List[str] = []

    # --- Open the document ---------------------------------------------
    try:
        document = fitz.open(file_path)
    except Exception as exc:
        # fitz can raise several different exception types depending on
        # *why* a file fails to open (corrupt structure, encryption, not actually a PDF).
        # We deliberately catch broadly here because
        # from the pipeline's perspective the response is identical either
        # way: log it, count zero pages, move on to the next file.
        errors.append(f"Failed to open document: {exc}")
        return {
            "page_texts": page_texts,
            "total_pages": 0,
            "extracted_pages": 0,
            "empty_pages": 0,
            "errors": errors,
        }

    total_pages = document.page_count

    # --- Extract each page individually ---------------------------------
    try:
        for page_index in range(total_pages):
            try:
                page = document.load_page(page_index)
                text = page.get_text("text")
            except Exception as exc:
                # A single malformed page should not abort extraction of
                # the rest of the document.
                errors.append(f"Page {page_index + 1}: extraction error - {exc}")
                page_texts.append("")
                empty_pages += 1
                continue

            if text and text.strip():
                page_texts.append(text)
                extracted_pages += 1
            else:
                # Most commonly a scanned/image-only page with no OCR text
                # layer, or a genuinely blank page in the source document.
                page_texts.append("")
                empty_pages += 1
    finally:
        # Always release the file handle, even if the loop above raised
        # an unexpected error we did not anticipate, to avoid leaking
        # open file descriptors across a large batch of documents.
        document.close()

    return {
        "page_texts": page_texts,
        "total_pages": total_pages,
        "extracted_pages": extracted_pages,
        "empty_pages": empty_pages,
        "errors": errors,
    }


# ------------------------------------------------------------------------
# STEP 4 — PERSISTENCE
# ------------------------------------------------------------------------

def save_text(filename: str, page_texts: List[str], output_dir: Path) -> Path:
    """
    Save a document's extracted page text to a single .txt file, preserving
    the original document name (only the extension changes).

    Why preserve the name: downstream pipeline stages (chunking, embedding)
    need a predictable way to trace a chunk of text back to its source
    document. Keeping "Funding_Guidelines.pdf" -> "Funding_Guidelines.txt"
    means no separate mapping file is required.

    Why page markers in the output: keeping a visible "--- Page N ---"
    boundary in the saved text lets later pipeline stages (or a human
    reviewer) locate which donor-document page a retrieved chunk came
    from, which matters for citing donor requirements accurately.

    Args:
        filename: Original PDF filename (e.g. "Funding_Guidelines.pdf").
            Only the stem is used; the extension is replaced with ".txt".
        page_texts: List of per-page extracted text, in page order, as
            produced by `extract_text_from_pdf()`.
        output_dir: Directory to save the .txt file into
            (expected: data/extracted/). Created if it does not exist.

    Returns:
        The Path the text file was written to.

    Raises:
        OSError: If the output directory cannot be created or the file
            cannot be written (e.g. disk full, permissions denied). This
            is deliberately re-raised (not swallowed) so the caller can
            mark that document's status as FAILED rather than incorrectly
            reporting success.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_filename = Path(filename).stem + ".txt"
    output_path = output_dir / txt_filename

    try:
        with open(output_path, "w", encoding="utf-8") as out_file:
            for page_number, page_text in enumerate(page_texts, start=1):
                out_file.write(f"\n--- Page {page_number} ---\n")
                if page_text:
                    out_file.write(page_text)
                else:
                    out_file.write("[EMPTY PAGE: no extractable text]")
                out_file.write("\n")
    except OSError as exc:
        raise OSError(f"Failed to write extracted text to '{output_path}': {exc}") from exc

    return output_path


# ------------------------------------------------------------------------
# STEP 5 — LOGGING
# ------------------------------------------------------------------------

def write_log(results: List[ExtractionResult], log_path: Path) -> None:
    """
    Write a human-readable extraction log covering every PDF processed.

    Why a plain-text audit log (separate from console output): console
    output disappears once the terminal is closed, but a final-year
    project (and a real grant-evaluation system) needs a durable record
    of exactly what was extracted from the donor knowledge base and when,
    in case a later evaluation result is questioned and needs to be
    traced back to its source data.

    Args:
        results: List of ExtractionResult records, one per PDF processed.
        log_path: File path to write the log to
            (expected: logs/extraction_logs/extraction_log.txt).

    Returns:
        None. Writes the log file as a side effect. If the log cannot be
        written (e.g. permissions issue), a warning is printed to the
        console rather than crashing the pipeline — losing the audit log
        is bad, but it should never be allowed to take down extraction
        of documents that otherwise succeeded.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("=" * 80 + "\n")
            log_file.write("DONOR DOCUMENT EXTRACTION LOG\n")
            log_file.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            log_file.write(f"Total files processed: {len(results)}\n")
            log_file.write("=" * 80 + "\n\n")

            for result in results:
                log_file.write(f"Filename        : {result.filename}\n")
                log_file.write(f"Path            : {result.filepath}\n")
                log_file.write(f"Total Pages     : {result.total_pages}\n")
                log_file.write(f"Extracted Pages : {result.extracted_pages}\n")
                log_file.write(f"Empty Pages     : {result.empty_pages}\n")
                log_file.write(f"Status          : {result.status}\n")
                if result.error_message:
                    log_file.write(f"Errors          : {result.error_message}\n")
                log_file.write("-" * 80 + "\n")

    except OSError as exc:
        # Deliberately non-fatal: a logging failure should be visible but
        # must not prevent the function from returning normally, since by
        # this point all extraction work has already completed.
        print(f"[WARNING] Could not write extraction log to '{log_path}': {exc}")


# ------------------------------------------------------------------------
# STEP 6 — SUMMARY REPORTING
# ------------------------------------------------------------------------

def print_summary(results: List[ExtractionResult]) -> None:
    """
    Print a concise console summary of the whole extraction run.

    Why this exists separately from write_log(): the full log is detailed
    and file-by-file, which is the right format for later auditing but the
    wrong format for a quick "did this run work?" glance immediately after
    running the script. This function gives that quick glance.

    Args:
        results: List of ExtractionResult records produced by
            `process_all_pdfs()`.

    Returns:
        None. Prints directly to stdout.
    """
    total_pdfs = len(results)
    successful = sum(1 for r in results if r.status == "SUCCESS")
    partial = sum(1 for r in results if r.status == "PARTIAL")
    empty_docs = sum(1 for r in results if r.status == "EMPTY")
    failed = sum(1 for r in results if r.status == "FAILED")
    total_pages = sum(r.total_pages for r in results)
    total_empty_pages = sum(r.empty_pages for r in results)

    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total PDFs processed     : {total_pdfs}")
    print(f"Successful extractions   : {successful}")
    print(f"Partial extractions      : {partial}")
    print(f"Empty-result extractions : {empty_docs}")
    print(f"Failed extractions       : {failed}")
    print(f"Total pages              : {total_pages}")
    print(f"Empty pages              : {total_empty_pages}")
    print("=" * 50)


# ------------------------------------------------------------------------
# ORCHESTRATION
# ------------------------------------------------------------------------

def process_all_pdfs() -> List[ExtractionResult]:
    """
    Run the full donor-document extraction pipeline end to end:

        1. Discover every PDF under DONOR_DOCS_DIR.
        2. Validate each one.
        3. Extract text from valid PDFs.
        4. Save extracted text under EXTRACTED_DIR.
        5. Write a full audit log to LOG_FILE.
        6. Print a console summary.

    This is the single entry point intended to be called by the rest of
    the grant-evaluation system (or directly via the command line) — every
    other function in this module is a building block this one composes.

    Status assigned to each document:
        "FAILED"  - validation failed, the document could not be opened,
                    or the extracted text could not be saved to disk.
        "EMPTY"   - the document opened but yielded zero pages with
                    extractable text (e.g. fully scanned with no OCR).
        "PARTIAL" - some, but not all, pages yielded extractable text.
        "SUCCESS" - every page yielded extractable text.

    Returns:
        The list of ExtractionResult records for every PDF discovered,
        in the same order they were processed. Returning this (rather
        than just printing/logging) lets calling code — or unit tests —
        inspect exactly what happened without re-parsing the log file.
    """
    results: List[ExtractionResult] = []

    print(f"Scanning for donor PDF documents in: {DONOR_DOCS_DIR.resolve()}")

    if not DONOR_DOCS_DIR.exists():
        print(f"[ERROR] Donor documents directory not found: '{DONOR_DOCS_DIR}'")
        print("Create the folder and add donor PDFs before running this script.")
        return results

    pdf_paths = find_all_pdfs(DONOR_DOCS_DIR)

    if not pdf_paths:
        print(f"[WARNING] No PDF files found under '{DONOR_DOCS_DIR}'.")
        return results

    print(f"Found {len(pdf_paths)} PDF file(s). Beginning extraction...\n")

    for pdf_path in pdf_paths:
        result = ExtractionResult(filename=pdf_path.name, filepath=str(pdf_path))

        # --- Validate ----------------------------------------------------
        is_valid, reason = validate_pdf(pdf_path)
        if not is_valid:
            result.status = "FAILED"
            result.error_message = reason
            results.append(result)
            print(f"[FAILED]  {pdf_path.name} - {reason}")
            continue

        # --- Extract -------------------------------------------------------
        # Wrapped in a try/except as a final safety net: extract_text_from_pdf
        # already handles its own internal errors, but this guarantees that
        # even a truly unexpected exception cannot take down the whole batch
        # of remaining documents.
        try:
            extraction = extract_text_from_pdf(pdf_path)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch, see above
            result.status = "FAILED"
            result.error_message = f"Unexpected extraction failure: {exc}"
            results.append(result)
            print(f"[FAILED]  {pdf_path.name} - {result.error_message}")
            continue

        result.total_pages = extraction["total_pages"]
        result.extracted_pages = extraction["extracted_pages"]
        result.empty_pages = extraction["empty_pages"]
        result.errors = extraction["errors"]
        if extraction["errors"]:
            result.error_message = "; ".join(extraction["errors"])

        # --- Determine status --------------------------------------------
        if result.total_pages == 0:
            result.status = "FAILED"
        elif result.extracted_pages == result.total_pages:
            result.status = "SUCCESS"
        elif result.extracted_pages == 0:
            result.status = "EMPTY"
        else:
            result.status = "PARTIAL"

        # --- Save extracted text (for any document that actually opened) --
        if result.total_pages > 0:
            try:
                saved_path = save_text(pdf_path.name, extraction["page_texts"], EXTRACTED_DIR)
                print(f"[{result.status:<7}] {pdf_path.name} "
                      f"- {result.extracted_pages}/{result.total_pages} pages extracted "
                      f"-> {saved_path}")
            except OSError as exc:
                result.status = "FAILED"
                addition = f"Save error: {exc}"
                result.error_message = (
                    f"{result.error_message}; {addition}" if result.error_message else addition
                )
                print(f"[FAILED]  {pdf_path.name} - {addition}")
        else:
            print(f"[FAILED]  {pdf_path.name} - {result.error_message}")

        results.append(result)

    # --- Log and summarize the full run ----------------------------------
    write_log(results, LOG_FILE)
    print(f"\nExtraction log written to: {LOG_FILE.resolve()}")
    print_summary(results)

    return results


# ------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------

if __name__ == "__main__":
    process_all_pdfs()