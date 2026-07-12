"""
process_proposal.py

Belongs to the ONLINE proposal evaluation pipeline of the
AI-Powered Grant Proposal Evaluation System.

Responsibility (and ONLY responsibility):
    1. Validate a single proposal PDF.
    2. Extract text from it using PyMuPDF.
    3. Clean the extracted text.
    4. Generate a single semantic embedding representing the whole proposal.

This module does NOT query ChromaDB/FAISS, retrieve donor context, build
Gemini prompts, call Gemini, score proposals, or generate reports. Those
responsibilities belong to later stages of the pipeline. Keeping this module
narrow means it can be unit-tested and reused independently of whichever UI
(CLI, Gradio) or downstream stage calls it.
"""

from __future__ import annotations

import logging
import re
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any

import fitz  # type: ignore # PyMuPDF
from sentence_transformers import SentenceTransformer # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Centralizing the model name here (rather than scattering the string across
# functions) means there is exactly one place to change it later if the
# project settles on a different embedding model.
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "proposal_processing.log"

# A conservative pattern for standalone page-number lines, e.g. "12", "Page 3",
# "- 4 -". Deliberately narrow: the goal is to strip obvious pagination
# artifacts without risking deletion of real content such as budget figures
# that happen to sit alone on a line.
_PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(page\s+)?[-–—]?\s*\d{1,4}\s*[-–—]?\s*$",
    re.IGNORECASE,
)

# Three or more blank lines collapse to a single paragraph break, and runs of
# horizontal whitespace collapse to a single space, so downstream chunk/LLM
# input isn't padded with wasted tokens.
_MULTI_BLANK_LINES_PATTERN = re.compile(r"\n\s*\n\s*(\n\s*)+")
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logger() -> logging.Logger:
    """Configure and return the module-level logger.

    A dedicated logger (rather than the root logger) is used so that this
    module's logging behavior doesn't interfere with logging configuration
    done elsewhere in the larger pipeline (e.g. the donor_pipeline scripts).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("proposal_evaluation.process_proposal")
    logger.setLevel(logging.DEBUG)

    # Guard against duplicate handlers if this module is imported more than
    # once (e.g. reloaded in a notebook, or imported by multiple pipeline
    # stages within the same process).
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
# Module-level embedding model cache
# ---------------------------------------------------------------------------

# Loading a Sentence Transformers model takes real time (and, on first use,
# memory). A module-level cache means repeated calls to process_proposal()
# within the same process load the model only once, rather than once per
# proposal.
_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Return the cached embedding model, loading it on first use.

    Uses local_files_only=True deliberately: the model is expected to already
    be present in the local Hugging Face cache (downloaded once, ahead of
    time, with internet access). This keeps the evaluation pipeline itself
    fully offline and avoids an unexpected network dependency at evaluation
    time.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, local_files_only=True
        )
        logger.info("Embedding model loaded successfully.")
    return _embedding_model


# ---------------------------------------------------------------------------
# Stage 1 — PDF Validation
# ---------------------------------------------------------------------------

def validate_pdf(pdf_path: Path) -> None:
    """Validate that ``pdf_path`` points to a usable PDF file.

    Checks performed, in order:
        1. The path exists.
        2. The path is a file (not a directory).
        3. The file extension is ``.pdf``.
        4. The file is non-empty.
        5. PyMuPDF can actually open it (catches corrupted/malformed PDFs
           that pass the earlier, cheaper checks).

    Args:
        pdf_path: Full path to the proposal PDF.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a file, is not a .pdf, or is empty.
        RuntimeError: If PyMuPDF cannot open the file as a valid PDF.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Proposal PDF not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Path exists but is not a file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a .pdf file, got extension '{pdf_path.suffix}': {pdf_path}"
        )

    if pdf_path.stat().st_size == 0:
        raise ValueError(f"Proposal PDF is empty (0 bytes): {pdf_path}")

    # Cheap checks above catch obvious problems fast; only attempt the more
    # expensive "does this actually open" check once those pass.
    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count == 0:
                raise ValueError(f"Proposal PDF has no pages: {pdf_path}")
    except Exception as exc:  # PyMuPDF raises its own exception types
        raise RuntimeError(
            f"PyMuPDF could not open '{pdf_path}' as a valid PDF: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Stage 2 — Text Extraction
# ---------------------------------------------------------------------------

def extract_text(pdf_path: Path) -> dict[str, Any]:
    """Extract plain text from every page of the proposal PDF, in order.

    Args:
        pdf_path: Full path to the proposal PDF. Assumed to already have
            passed validate_pdf().

    Returns:
        A dictionary with:
            - "raw_text": the concatenated text of all pages, in reading
              order, separated by form-feed-free newlines.
            - "page_count": total number of pages in the document.
            - "empty_page_count": number of pages whose extracted text was
              blank (useful for flagging scanned/image-only proposals that
              slipped past validation).
    """
    page_texts: list[str] = []
    empty_page_count = 0

    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        # Iterating by index (rather than "for page in doc") keeps reading
        # order explicit and makes it trivial to log which page number, if
        # any, caused a problem.
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            page_text = page.get_text("text")

            if not page_text.strip():
                empty_page_count += 1

            page_texts.append(page_text)

    raw_text = "\n".join(page_texts)

    return {
        "raw_text": raw_text,
        "page_count": page_count,
        "empty_page_count": empty_page_count,
    }


# ---------------------------------------------------------------------------
# Stage 3 — Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(raw_text: str) -> str:
    """Clean extracted proposal text while keeping it human-readable.

    Cleaning performed:
        - Unicode normalization (NFKC), so visually identical characters
          from different PDF encodings compare and embed consistently.
        - Removal of standalone page-number lines.
        - Collapsing runs of horizontal whitespace to a single space.
        - Collapsing 3+ blank lines down to a single paragraph break, while
          preserving genuine paragraph breaks (a single blank line).
        - Stripping leading/trailing whitespace from the whole document.

    Headings, punctuation, numbers, and capitalization are intentionally
    left untouched — aggressive normalization here would destroy signal
    that later stages (and the LLM evaluator) rely on to understand
    document structure.

    Args:
        raw_text: The raw text returned by extract_text().

    Returns:
        The cleaned text, suitable for embedding and for downstream LLM
        evaluation.
    """
    # NFKC normalization folds visually-equivalent Unicode variants (e.g.
    # different dash or quote code points introduced by different PDF
    # producers) into a consistent representation, without altering the
    # visible meaning of the text.
    text = unicodedata.normalize("NFKC", raw_text)

    # Strip obvious standalone page-number lines line-by-line, before
    # collapsing blank lines, so a page-number line doesn't get merged into
    # surrounding paragraph text first.
    lines = text.split("\n")
    kept_lines = [
        line for line in lines if not _PAGE_NUMBER_PATTERN.match(line)
    ]
    text = "\n".join(kept_lines)

    # Collapse horizontal whitespace runs (spaces/tabs) to a single space.
    text = _MULTI_SPACE_PATTERN.sub(" ", text)

    # Collapse 3+ consecutive blank lines to exactly one blank line, which
    # preserves paragraph breaks (a single blank line) while removing large
    # gaps introduced by PDF layout artifacts (e.g. empty table cells).
    text = _MULTI_BLANK_LINES_PATTERN.sub("\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Stage 4 — Embedding Generation
# ---------------------------------------------------------------------------

def generate_embedding(clean_text_value: str) -> dict[str, Any]:
    """Generate a single semantic embedding representing the whole proposal.

    Args:
        clean_text_value: The cleaned proposal text from clean_text().

    Returns:
        A dictionary with:
            - "embedding": the embedding vector as a plain Python list (not
              a numpy array), so the result is directly JSON-serializable
              for logging, caching, or passing to other pipeline stages.
            - "embedding_dimension": the length of the embedding vector.
            - "embedding_model": the model name used, so downstream stages
              (and audit logs) can confirm which model produced a given
              vector.
    """
    model = _get_embedding_model()

    # encode() returns a numpy array by default; convert to a plain list so
    # the returned dictionary is trivially JSON-serializable, matching the
    # "everything stays in memory, nothing is silently pickled" design goal.
    embedding_vector = model.encode(clean_text_value, convert_to_numpy=True)
    embedding_list = embedding_vector.tolist()

    return {
        "embedding": embedding_list,
        "embedding_dimension": len(embedding_list),
        "embedding_model": EMBEDDING_MODEL_NAME,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_proposal(pdf_path: Path) -> dict[str, Any]:
    """Process a single grant proposal PDF end-to-end for evaluation.

    Workflow: validate -> extract -> clean -> embed -> return.

    This is the only function later pipeline stages (retrieval, Gemini
    evaluation, scoring, reporting) should call into for proposal
    preparation. Everything here happens in memory; no intermediate file
    (raw text, cleaned text, or embedding) is written to disk.

    Args:
        pdf_path: Full path to the single proposal PDF to process. This
            function never scans a directory for PDFs — the caller (a CLI
            prompt during development, or a Gradio upload handler in
            production) is responsible for supplying exactly one path.

    Returns:
        A dictionary:
            {
                "filename": str,
                "page_count": int,
                "raw_text": str,
                "clean_text": str,
                "embedding_model": str,
                "embedding_dimension": int,
                "embedding": list[float],
            }

    Raises:
        FileNotFoundError, ValueError, RuntimeError: propagated from
            validate_pdf() if the PDF fails validation.
        Exception: any unexpected error is logged with a full traceback
            before being re-raised, so the caller (e.g. a Gradio handler)
            can decide how to surface it to the user.
    """
    start_time = time.perf_counter()
    logger.info("=" * 70)
    logger.info("Processing started for proposal: %s", pdf_path)

    try:
        # --- Stage 1: Validation ---------------------------------------
        validate_pdf(pdf_path)
        logger.info("Validation passed for: %s", pdf_path.name)

        # --- Stage 2: Extraction -----------------------------------------
        extraction_result = extract_text(pdf_path)
        raw_text = extraction_result["raw_text"]
        page_count = extraction_result["page_count"]
        empty_page_count = extraction_result["empty_page_count"]

        logger.info(
            "Extracted text: %d page(s), %d empty page(s), %d character(s).",
            page_count,
            empty_page_count,
            len(raw_text),
        )

        # --- Stage 3: Cleaning -------------------------------------------
        cleaned_text = clean_text(raw_text)
        logger.info(
            "Cleaned text: %d character(s) (from %d raw).",
            len(cleaned_text),
            len(raw_text),
        )

        # --- Stage 4: Embedding ------------------------------------------
        embedding_result = generate_embedding(cleaned_text)
        logger.info(
            "Generated embedding using '%s' (dimension: %d).",
            embedding_result["embedding_model"],
            embedding_result["embedding_dimension"],
        )

        elapsed_seconds = time.perf_counter() - start_time
        logger.info(
            "Processing completed successfully for '%s' in %.2f second(s).",
            pdf_path.name,
            elapsed_seconds,
        )

        return {
            "filename": pdf_path.name,
            "page_count": page_count,
            "raw_text": raw_text,
            "clean_text": cleaned_text,
            "embedding_model": embedding_result["embedding_model"],
            "embedding_dimension": embedding_result["embedding_dimension"],
            "embedding": embedding_result["embedding"],
        }

    except Exception:
        # Log the full traceback for diagnosis, then re-raise so the caller
        # (CLI test harness, or later a Gradio handler) retains full control
        # over how the failure is surfaced to the end user.
        logger.error(
            "Processing FAILED for '%s'.\n%s", pdf_path, traceback.format_exc()
        )
        raise


# ---------------------------------------------------------------------------
# Standalone development testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    user_input = input("Enter the full path to a proposal PDF: ").strip().strip('"')
    input_path = Path(user_input)

    print("-" * 50)
    try:
        result = process_proposal(input_path)
        print("Proposal Processing Summary")
        print("-" * 50)
        print(f"Filename              : {result['filename']}")
        print(f"Pages                 : {result['page_count']}")
        print(f"Characters Extracted  : {len(result['raw_text']):,}")
        print(f"Characters Cleaned    : {len(result['clean_text']):,}")
        print(f"Embedding Model       : {result['embedding_model']}")
        print(f"Embedding Dimension   : {result['embedding_dimension']}")
        print("Status                : SUCCESS")
        print("-" * 50)
    except Exception as exc:  # noqa: BLE001 - top-level CLI catch is intentional
        print("Proposal Processing Summary")
        print("-" * 50)
        print(f"Status                : FAILED")
        print(f"Error                 : {exc}")
        print("-" * 50)