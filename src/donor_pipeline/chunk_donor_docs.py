"""
chunk_donor_docs.py
===================
Stage 3 of the donor knowledge-base pipeline:

    Donor PDFs → Extraction → Cleaning → [CHUNKING] → Embeddings → FAISS Vector Store

WHY CHUNKING IS NEEDED IN RAG SYSTEMS
--------------------------------------
Retrieval-Augmented Generation (RAG) works by converting text into numerical
vectors (embeddings) and storing them so that, at query time, semantically
similar chunks can be retrieved and passed to the LLM as context.

Embedding models have a maximum input length (typically 256–512 tokens), so
full documents cannot be embedded as a single unit. Chunking splits documents
into pieces that:
  - Fit within the embedding model's token limit
  - Are small enough that retrieved chunks are focused and relevant (not noisy)
  - Are large enough to carry complete semantic meaning

WHY SECTION-AWARE CHUNKING IMPROVES RETRIEVAL
----------------------------------------------
Generic fixed-size chunking (e.g. "split every 1000 characters at a space")
ignores the logical structure of the document. A chunk may span two unrelated
sections, polluting the embedding with mixed signals.

Section-aware chunking uses document headings as natural semantic boundaries.
A chunk that contains only the "Budget Guidelines" section will embed cleanly
as a chunk about budgets; retrieval for budget-related questions will reliably
surface it. This is especially important for GIF donor documents, which are
structured around distinct, named sections (e.g. Eligibility Requirements,
Strategic Priorities, Evaluation Criteria).

WHY OVERLAP IS USED
--------------------
When a large section is split into multiple chunks, important context (a
definition, a condition) may appear at the boundary between two consecutive
chunks. Without overlap, that context disappears from the following chunk.
Overlap replicates the tail of each chunk as the head of the next, so
retrieval is not penalised for a boundary falling mid-sentence.

WHY LARGE SECTIONS ARE SPLIT
-----------------------------
Some sections (e.g. a detailed budget rubric) are much longer than the
embedding model's effective window. Keeping them as a single chunk would
exceed the token limit and cause the embedding to become diffuse. Splitting
ensures every chunk is focused and embeddable at high quality.
"""

import json
import re
import logging
from datetime import datetime
from pathlib import Path

# ── Directory constants ────────────────────────────────────────────────────────
CLEANED_DIR   = Path("data/cleaned")        # Input: cleaned .txt files
CHUNKS_DIR    = Path("data/chunks")         # Output: per-document JSON chunk files
LOG_DIR       = Path("logs/extraction_logs")
LOG_FILE      = LOG_DIR / "chunking_log.txt"

# ── Chunking parameters ────────────────────────────────────────────────────────
MAX_CHUNK_SIZE = 1000   # Maximum characters per chunk before splitting
OVERLAP_SIZE   = 150    # Characters of overlap between consecutive split chunks


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTORY SETUP
# ══════════════════════════════════════════════════════════════════════════════

def ensure_directories() -> None:
    """
    Create required directories if they do not already exist.

    Called once at startup so that subsequent file writes never fail due to a
    missing parent directory. Using Path.mkdir(parents=True, exist_ok=True)
    is idempotent — safe to call even if the directories already exist.
    """
    for directory in (CLEANED_DIR, CHUNKS_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_document(filepath: Path) -> str:
    """
    Read and return the full text content of a cleaned .txt file.

    Args:
        filepath: Path to the cleaned text file.

    Returns:
        The file content as a single string.

    Raises:
        IOError: If the file cannot be read (caught and handled by the caller).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_sections(text: str) -> list[dict]:
    """
    Split document text into sections using lightweight heading heuristics.

    WHY HEURISTICS, NOT A PARSER
    ----------------------------
    Donor documents are plain text extracted from PDFs. They do not have HTML
    tags or XML structure to parse. Instead we rely on observable patterns:

      - ALL CAPS lines  (e.g. "ELIGIBILITY REQUIREMENTS")
      - Title Case lines (e.g. "Budget Guidelines")
      - Short lines (<= 80 chars) followed by a blank line
      - Lines ending with a colon (e.g. "Section 4:")

    This approach is intentionally simple. Over-engineering heading detection
    (e.g. with ML classifiers) adds complexity without meaningful gain for the
    structured documents GIF produces.

    Args:
        text: Full document text as a single string.

    Returns:
        List of dicts, each containing:
          - 'title' (str): The detected heading text, or 'Introduction' for
                           content appearing before the first heading.
          - 'content' (str): The body text belonging to that section.
    """

    lines = text.splitlines()
    sections = []

    # Accumulate lines for the current section
    current_title   = "Introduction"
    current_lines   = []

    def is_heading(line: str, next_line: str) -> bool:
        """
        Return True if `line` looks like a section heading.

        Rules (applied in order; any match returns True):
          1. ALL CAPS line that is not too long (avoids matching full-sentence
             text that happens to be capitalised).
          2. Title Case line short enough to be a heading.
          3. Short line (≤ 80 chars) followed by an empty next line — a
             common PDF-to-text heading pattern.
          4. Line that ends with a colon and is short (e.g. "Criteria:").
        """
        stripped = line.strip()

        # Ignore blank lines and very short fragments (1-2 chars are artifacts)
        if len(stripped) < 3:
            return False

        # Rule 1: ALL CAPS heading
        if stripped.isupper() and len(stripped) <= 80:
            return True

        # Rule 2: Title Case heading (most words capitalised, not a sentence)
        words = stripped.split()
        if (len(words) >= 2
                and sum(1 for w in words if w[0].isupper()) >= len(words) * 0.7
                and len(stripped) <= 80
                and not stripped.endswith(".")):  # sentences end with full stop
            return True

        # Rule 3: Short line followed by blank line
        if len(stripped) <= 80 and next_line.strip() == "":
            return True

        # Rule 4: Short line ending with a colon
        if stripped.endswith(":") and len(stripped) <= 80:
            return True

        return False

    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        if is_heading(line, next_line):
            # Save the section we were building
            content = "\n".join(current_lines).strip()
            if content:  # Don't save empty sections
                sections.append({
                    "title":   current_title,
                    "content": content,
                })
            # Start a new section
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save the final section
    content = "\n".join(current_lines).strip()
    if content:
        sections.append({
            "title":   current_title,
            "content": content,
        })

    # If no headings were found, treat the whole document as one section
    if not sections:
        sections.append({
            "title":   "Full Document",
            "content": text.strip(),
        })

    return sections


# ══════════════════════════════════════════════════════════════════════════════
# LARGE SECTION SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

def split_large_section(text: str,
                         max_size: int = MAX_CHUNK_SIZE,
                         overlap: int  = OVERLAP_SIZE) -> list[str]:
    """
    Split a section that exceeds `max_size` characters into overlapping chunks.

    WHY CHARACTER-BASED (NOT WORD-BASED) SPLITTING
    -----------------------------------------------
    Character counts are deterministic and independent of tokeniser choice.
    Since we are not tied to a specific tokeniser at this stage, character
    counts give a reliable, reproducible result. Each embedding model has a
    token limit; keeping chunks well under 1000 characters keeps them safely
    within that limit for any standard sentence-transformer model.

    OVERLAP MECHANISM
    -----------------
    Given max_size=1000 and overlap=150:
      Chunk 1: characters   0 – 999
      Chunk 2: characters 850 – 1849   (start = 1000 - 150 = 850)
      Chunk 3: characters 1700 – 2699  (start = 1850 - 150 = 1700)

    The last 150 characters of each chunk are repeated as the first 150 of
    the next, preserving context across boundaries.

    Args:
        text:     The section text to split.
        max_size: Maximum characters per chunk.
        overlap:  Characters to repeat between consecutive chunks.

    Returns:
        List of text strings, each at most `max_size` characters.
    """
    chunks = []
    start  = 0

    while start < len(text):
        end = start + max_size
        chunk = text[start:end]
        chunks.append(chunk)

        # If this chunk reaches the end, we're done
        if end >= len(text):
            break

        # Move the start forward by (max_size - overlap) so the next chunk
        # begins overlap characters before the end of the current chunk
        start += max_size - overlap

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK CREATION
# ══════════════════════════════════════════════════════════════════════════════

def create_chunks(sections: list[dict], source_document: str) -> list[dict]:
    """
    Convert detected sections into retrieval-ready chunk objects with metadata.

    Each section becomes one chunk unless its content exceeds MAX_CHUNK_SIZE,
    in which case split_large_section() is called and the section produces
    multiple numbered sub-chunks.

    Chunk ID format:  <stem>_<zero-padded index>
    Example:          funding_guidelines_001

    Args:
        sections:        Output of detect_sections().
        source_document: Filename of the source .txt file (e.g.
                         "funding_guidelines.txt").

    Returns:
        List of chunk dicts, each containing all required metadata fields.
    """
    # Derive a clean base name for chunk IDs (no extension, spaces → underscores)
    doc_stem    = Path(source_document).stem.replace(" ", "_").lower()
    chunks      = []
    chunk_index = 1   # Global index across all sections in this document

    for section in sections:
        title   = section["title"]
        content = section["content"]

        if len(content) <= MAX_CHUNK_SIZE:
            # Section fits in a single chunk — no splitting needed
            chunks.append({
                "chunk_id":       f"{doc_stem}_{chunk_index:03d}",
                "source_document": source_document,
                "section_title":   title,
                "chunk_index":     chunk_index,
                "character_count": len(content),
                "text":            content,
            })
            chunk_index += 1

        else:
            # Section is too large — split it and create sub-chunks
            # Each sub-chunk retains the parent section title so retrieval
            # context is preserved even for deeply split sections.
            sub_texts = split_large_section(content)

            for sub_i, sub_text in enumerate(sub_texts, start=1):
                # Annotate the title to indicate this is a continuation
                sub_title = f"{title} (Part {sub_i} of {len(sub_texts)})"

                chunks.append({
                    "chunk_id":       f"{doc_stem}_{chunk_index:03d}",
                    "source_document": source_document,
                    "section_title":   sub_title,
                    "chunk_index":     chunk_index,
                    "character_count": len(sub_text),
                    "text":            sub_text,
                })
                chunk_index += 1

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK SAVING
# ══════════════════════════════════════════════════════════════════════════════

def save_chunks(chunks: list[dict], source_document: str) -> Path:
    """
    Serialise chunks to a JSON file in the chunks output directory.

    Output filename: data/chunks/<stem>_chunks.json
    Example:         data/chunks/funding_guidelines_chunks.json

    Args:
        chunks:          List of chunk dicts from create_chunks().
        source_document: Source .txt filename (used to derive the output name).

    Returns:
        Path to the saved JSON file.
    """
    stem        = Path(source_document).stem.replace(" ", "_").lower()
    output_path = CHUNKS_DIR / f"{stem}_chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def write_log(filename:       str,
              sections_found: int,
              chunks_created: int,
              avg_chunk_size: float,
              status:         str) -> None:
    """
    Append a one-line log entry to the chunking log file.

    Format:
        funding_guidelines.txt | 8 sections | 12 chunks | 842 chars avg | SUCCESS

    Args:
        filename:       Name of the processed .txt file.
        sections_found: Number of sections detected.
        chunks_created: Number of chunks produced.
        avg_chunk_size: Mean character count across all chunks.
        status:         "SUCCESS" or "FAILED: <reason>".
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line  = (
        f"[{timestamp}] {filename} | "
        f"{sections_found} sections | "
        f"{chunks_created} chunks | "
        f"{avg_chunk_size:.0f} chars avg | "
        f"{status}\n"
    )
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE DOCUMENT PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_document(filepath: Path) -> dict:
    """
    Run the full chunking pipeline for a single cleaned text file.

    Pipeline for this function:
        load_document()
        → detect_sections()
        → create_chunks()
        → save_chunks()
        → write_log()

    A failure at any stage is caught, logged, and reported in the return value
    so that process_all_documents() can continue with the remaining files.

    Args:
        filepath: Path to the cleaned .txt file.

    Returns:
        Dict with keys:
          - 'filename'       (str)
          - 'sections_found' (int)
          - 'chunks_created' (int)
          - 'avg_chunk_size' (float)
          - 'success'        (bool)
          - 'error'          (str | None)
    """
    filename = filepath.name

    try:
        # ── Step 1: Load the document ─────────────────────────────────────────
        text = load_document(filepath)

        if not text.strip():
            raise ValueError("File is empty after loading.")

        # ── Step 2: Detect sections ───────────────────────────────────────────
        sections = detect_sections(text)
        sections_found = len(sections)

        # ── Step 3-5: Create chunks (splitting and overlap happen inside) ─────
        chunks = create_chunks(sections, filename)
        chunks_created = len(chunks)

        # ── Step 6: Calculate average chunk size for logging ──────────────────
        avg_chunk_size = (
            sum(c["character_count"] for c in chunks) / chunks_created
            if chunks_created > 0 else 0.0
        )

        # ── Step 7: Save chunks to JSON ───────────────────────────────────────
        output_path = save_chunks(chunks, filename)

        # ── Step 8: Write log entry ───────────────────────────────────────────
        write_log(filename, sections_found, chunks_created, avg_chunk_size, "SUCCESS")

        print(f"  ✓  {filename} → {chunks_created} chunks saved to {output_path.name}")

        return {
            "filename":       filename,
            "sections_found": sections_found,
            "chunks_created": chunks_created,
            "avg_chunk_size": avg_chunk_size,
            "success":        True,
            "error":          None,
        }

    except Exception as exc:
        error_msg = str(exc)
        write_log(filename, 0, 0, 0.0, f"FAILED: {error_msg}")
        print(f"  ✗  {filename} — FAILED: {error_msg}")

        return {
            "filename":       filename,
            "sections_found": 0,
            "chunks_created": 0,
            "avg_chunk_size": 0.0,
            "success":        False,
            "error":          error_msg,
        }


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_all_documents() -> list[dict]:
    """
    Discover and process every .txt file in the cleaned documents directory.

    Uses Path.glob("*.txt") so new donor documents added to data/cleaned/ are
    picked up automatically without any code changes.

    Returns:
        List of per-document result dicts from process_document().
    """
    # ── Ensure all required directories exist ────────────────────────────────
    ensure_directories()

    # Write a session separator to the log so multiple runs are distinguishable
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"\n{'=' * 70}\n"
            f"Chunking session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 70}\n"
        )

    # ── Discover cleaned text files ───────────────────────────────────────────
    txt_files = sorted(CLEANED_DIR.glob("*.txt"))

    if not txt_files:
        print(f"[WARNING] No .txt files found in '{CLEANED_DIR}'. "
              f"Ensure the cleaning stage has been run first.")
        return []

    print(f"\nChunking {len(txt_files)} document(s) from '{CLEANED_DIR}'...\n")

    # ── Process each document, collecting results ─────────────────────────────
    results = []
    for filepath in txt_files:
        result = process_document(filepath)
        results.append(result)

    # ── Print summary after all documents are processed ───────────────────────
    print_summary(results)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict]) -> None:
    """
    Print a human-readable summary report to stdout after all documents are
    processed.

    Args:
        results: List of per-document result dicts from process_all_documents().
    """
    successful = [r for r in results if r["success"]]
    failed     = [r for r in results if not r["success"]]

    total_sections = sum(r["sections_found"] for r in successful)
    total_chunks   = sum(r["chunks_created"]  for r in successful)
    all_avg_sizes  = [r["avg_chunk_size"] for r in successful if r["chunks_created"] > 0]
    overall_avg    = sum(all_avg_sizes) / len(all_avg_sizes) if all_avg_sizes else 0.0

    print("\n" + "─" * 50)
    print("Chunking Summary")
    print("─" * 50)
    print(f"Documents Processed : {len(successful)}")
    print(f"Sections Found      : {total_sections}")
    print(f"Chunks Created      : {total_chunks}")
    print(f"Average Chunk Size  : {overall_avg:.0f} characters")
    print(f"Failed Files        : {len(failed)}")

    if failed:
        print("\nFailed documents:")
        for r in failed:
            print(f"  - {r['filename']}: {r['error']}")

    print("─" * 50)
    print(f"Chunk files saved to : {CHUNKS_DIR}/")
    print(f"Log written to       : {LOG_FILE}")
    print("─" * 50 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run the full chunking pipeline.
    # Outputs:
    #   data/chunks/<docname>_chunks.json  — one file per input document
    #   logs/extraction_logs/chunking_log.txt — cumulative session log
    process_all_documents()