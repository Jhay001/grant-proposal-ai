"""
clean_donor_docs.py

------------------------------------------------------------------------------
Donor Document Pipeline — Stage 2: Cleaning
------------------------------------------------------------------------------
WHY THIS STAGE EXISTS (beginner-friendly explanation)
------------------------------------------------------
The Extraction stage already pulled real text out of each donor PDF using
PyMuPDF, so the *words* are already correct and trustworthy. However, text
pulled out of a PDF often carries small formatting quirks that have nothing
to do with the actual donor content, for example:

    - Tab characters left over from how the PDF laid out a table.
    - Several spaces in a row, where the PDF renderer used spacing instead
      of real alignment.
    - Long runs of blank lines between sections or pages.
    - Leading/trailing blank space at the very start or end of the file.

None of that affects what the document *means*, but it can hurt the next
pipeline stage. Chunking later splits text into pieces using things like
paragraph breaks and whitespace as signals — messy whitespace can produce
chunks that are too large, too small, or split in odd places. Cleaning it
up first means every donor document is stored in a consistent, predictable
format before anything is chunked or embedded.

IMPORTANT — THIS IS INTENTIONALLY LIGHTWEIGHT
-----------------------------------------------
This stage does NOT rewrite, summarize, correct, or reformat the donor
content in any meaningful way. The donor documents are already high
quality. We are only standardizing whitespace, nothing else. If this
module ever starts doing more than that, it has grown beyond its intended
purpose and that logic belongs in a later, clearly-labeled stage instead.

USAGE
-----
    python src/donor_pipeline/clean_donor_docs.py

Run from the project root (the directory containing `data/` and `logs/`).
------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------
# Keeping all folder paths as constants in one place (instead of typing
# the same string in five different functions) means that if the project's
# folder layout ever changes, there is exactly one place to update it.

EXTRACTED_DIR: Path = Path("data/extracted")
CLEANED_DIR: Path = Path("data/cleaned")
LOG_DIR: Path = Path("logs/extraction_logs")
LOG_FILE: Path = LOG_DIR / "cleaning_log.txt"


# ------------------------------------------------------------------------
# DATA MODEL
# ------------------------------------------------------------------------

@dataclass
class CleaningResult:
    """
    Holds the outcome of cleaning a single file.

    Why use a small data class instead of separate loose variables: it
    gives every result a fixed, named shape (filename, counts, status,
    etc.) so that write_log() and print_summary() can rely on those
    fields always being present, no matter which file produced them.
    This mirrors the ExtractionResult class used in the earlier
    extraction stage, so both stages of the pipeline "look" the same way
    to anyone reading the code.

    Attributes:
        filename: Name of the file that was processed (e.g.
            "funding_guidelines.txt").
        original_chars: Number of characters in the file before cleaning.
        cleaned_chars: Number of characters in the file after cleaning.
        chars_removed: How many characters cleaning removed
            (original_chars - cleaned_chars). Never negative, since
            cleaning only ever removes redundant whitespace.
        status: "SUCCESS" if the file was cleaned and saved without
            problems, otherwise "FAILED".
        error_message: Explanation of what went wrong, only set when
            status is "FAILED".
    """
    filename: str
    original_chars: int = 0
    cleaned_chars: int = 0
    chars_removed: int = 0
    status: str = "FAILED"
    error_message: str = ""


# ------------------------------------------------------------------------
# STEP 1 — MAKE SURE OUR FOLDERS EXIST
# ------------------------------------------------------------------------

def ensure_directories(*directories: Path) -> None:
    """
    Create each given directory (including any missing parent folders)
    if it does not already exist.

    Why this matters: this script might be run on a brand-new clone of
    the project, before anyone has manually created `data/cleaned/` or
    `logs/extraction_logs/`. Without this check, the very first attempt
    to save a cleaned file or write the log would crash with a
    "directory not found" error. Calling this once at the start of the
    pipeline means every later step can safely assume its target folder
    already exists.

    Args:
        *directories: Any number of Path objects to create if missing.

    Returns:
        None. Creates folders on disk as a side effect.
    """
    for directory in directories:
        # parents=True also creates any missing parent folders
        # (e.g. "logs/" if it doesn't exist yet, not just "logs/extraction_logs/").
        # exist_ok=True means "do nothing, no error" if the folder is
        # already there — we only want to *ensure* it exists, not insist
        # on creating it fresh every time.
        directory.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------
# STEP 2 — THE ACTUAL CLEANING LOGIC
# ------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Apply lightweight, content-preserving cleaning to extracted text.

    This is the heart of the cleaning stage. It performs exactly four
    operations, in a deliberate order, and nothing more:

        1. Whitespace Normalization — replace tab characters with a
           single space. Tabs often appear where a PDF used a table or
           an indent, but a tab character is not meaningful once the
           text is just a flat string, so we flatten it to a space.

        2. Multiple Space Reduction — collapse any run of two or more
           spaces into exactly one space. PDFs frequently use repeated
           spaces to visually line text up; once extracted as plain
           text, those repeated spaces serve no purpose and just add
           noise.

        3. Blank Line Reduction — collapse any run of three or more
           newlines down to exactly two newlines (i.e. at most one fully
           blank line between paragraphs/sections). This keeps paragraph
           separation visible (useful for chunking later) without
           leaving large gaps where a PDF had awkward page breaks.

        4. Strip Leading/Trailing Whitespace — remove any blank lines or
           spaces at the very start or end of the whole document, since
           those carry no information at all.

    The order matters: tabs are turned into spaces first so that step 2
    can also catch any new runs of spaces created by that replacement,
    and blank-line reduction happens before the final strip so that a
    document which is *entirely* blank lines at the start still gets
    fully stripped away.

    Args:
        text: The raw text extracted from a donor document.

    Returns:
        The cleaned text, with the same words and meaning, but with
        standardized whitespace.
    """
    
    # --- 1. Whitespace Normalization: tabs -> single space ---------------
    cleaned = text.replace("\t", " ")

    # --- 2. Multiple Space Reduction --------------------------------------
    # The regex ' {2,}' matches two or more literal space characters in a row. 
    # It does NOT match newlines, so this only tidies up spaces
    # within a line and leaves paragraph/line structure untouched.
    cleaned = re.sub(r" {2,}", " ", cleaned)

    # --- 3. Blank Line Reduction -------------------------------------------
    # The regex '\n{3,}' matches three or more newline characters in a row
    # (i.e. two or more fully blank lines). We replace any such run
    # with exactly two newlines, which is one visible blank line between
    # content — enough to preserve paragraph/section breaks for the
    # chunking stage, without leaving large empty gaps in the file.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # --- 4. Strip Leading/Trailing Whitespace ------------------------------
    # .strip() removes any spaces, tabs, or newlines hanging off the very
    # start and end of the document (for example, a blank first page).
    cleaned = cleaned.strip()

    return cleaned


# ------------------------------------------------------------------------
# STEP 3 — SAVE THE CLEANED FILE
# ------------------------------------------------------------------------

def save_cleaned_text(filename: str, text: str, output_dir: Path) -> Path:
    """
    Write cleaned text to disk under the same filename, inside the
    cleaned-files output directory.

    Why keep the same filename: the next pipeline stage (chunking) needs
    a simple, predictable way to find the cleaned version of any given
    document. Keeping "funding_guidelines.txt" as "funding_guidelines.txt"
    (just in a different folder) means no separate lookup table or
    renaming logic is ever needed.

    Args:
        filename: The original file's name, e.g. "funding_guidelines.txt".
        text: The already-cleaned text to write.
        output_dir: Folder to save into (expected: data/cleaned/).

    Returns:
        The Path the cleaned file was written to.

    Raises:
        OSError: If the file cannot be written (e.g. disk full, no
            write permission). This is intentionally allowed to bubble
            up to the caller, so that a save failure is recorded as a
            FAILED result rather than silently ignored.
    """
    output_path = output_dir / filename

    try:
        output_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Failed to write cleaned file '{output_path}': {exc}") from exc

    return output_path


# ------------------------------------------------------------------------
# STEP 4 — LOGGING
# ------------------------------------------------------------------------

def write_log(results: List[CleaningResult], log_path: Path) -> None:
    """
    Write a plain-text audit log recording exactly what happened to every
    file during this cleaning run.

    Why keep a log at all: if a teammate (or an examiner during project
    defense) asks "how much text did cleaning actually remove from the
    Eligibility Requirements document?", this log answers that question
    without anyone needing to re-run the pipeline or compare files by
    hand. Each line is a simple, pipe-separated record matching the
    project's required format:

        filename | original_chars | cleaned_chars | N removed | STATUS

    Args:
        results: List of CleaningResult records, one per file processed.
        log_path: Where to write the log file
            (expected: logs/extraction_logs/cleaning_log.txt).

    Returns:
        None. Writes the log file as a side effect. If the log itself
        cannot be written, a warning is printed instead of raising —
        losing the audit log should never undo the cleaning work that
        already finished successfully.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("=" * 80 + "\n")
            log_file.write("DONOR DOCUMENT CLEANING LOG\n")
            log_file.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            log_file.write(f"Total files processed: {len(results)}\n")
            log_file.write("=" * 80 + "\n\n")

            for result in results:
                if result.status == "SUCCESS":
                    log_file.write(
                        f"{result.filename} | {result.original_chars} | "
                        f"{result.cleaned_chars} | {result.chars_removed} removed | "
                        f"{result.status}\n"
                    )
                else:
                    # Failed files have no meaningful character counts, so
                    # the error message is recorded instead, to explain
                    # why the file is missing from data/cleaned/.
                    log_file.write(
                        f"{result.filename} | FAILED | {result.error_message}\n"
                    )

    except OSError as exc:
        # Non-fatal on purpose: by the time write_log() runs, all the
        # actual cleaning work is already done. A logging problem should
        # be visible to the user but must not be treated as a pipeline
        # failure.
        print(f"[WARNING] Could not write cleaning log to '{log_path}': {exc}")


# ------------------------------------------------------------------------
# STEP 5 — PROCESS ONE FILE (validation + cleaning + saving, with safety net)
# ------------------------------------------------------------------------

def process_file(file_path: Path, output_dir: Path) -> CleaningResult:
    """
    Run the full cleaning process for a single file: read it, clean it,
    save it, and report what happened.

    Why this is its own function (rather than inlined inside a loop in
    process_all_files()): isolating "do one file" from "do all files"
    means a problem with a single document — a permissions issue, a
    strange encoding, a locked file — can be caught and recorded right
    here, without any risk of it crashing the loop that still needs to
    process every other donor document.

    Args:
        file_path: Path to the extracted .txt file to clean
            (expected to live under data/extracted/).
        output_dir: Folder to save the cleaned file into
            (expected: data/cleaned/).

    Returns:
        A CleaningResult describing what happened — on success this
        includes character counts; on failure it includes an error
        message and the file is simply skipped (not written to
        data/cleaned/).
    """
    filename = file_path.name
    result = CleaningResult(filename=filename)

    try:
        # encoding="utf-8" matches how the extraction stage saved these
        # files, so we read them back the same way they were written.
        original_text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        result.status = "FAILED"
        result.error_message = f"Could not read file: {exc}"
        return result
    except UnicodeDecodeError as exc:
        result.status = "FAILED"
        result.error_message = f"File is not valid UTF-8 text: {exc}"
        return result

    result.original_chars = len(original_text)

    try:
        cleaned_text = clean_text(original_text)
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        # clean_text() only does simple string/regex operations, so this
        # branch should be unreachable in practice — but guarding it
        # means even a truly unexpected error here is recorded clearly
        # instead of crashing the whole batch.
        result.status = "FAILED"
        result.error_message = f"Cleaning step failed unexpectedly: {exc}"
        return result

    result.cleaned_chars = len(cleaned_text)
    result.chars_removed = result.original_chars - result.cleaned_chars

    try:
        save_cleaned_text(filename, cleaned_text, output_dir)
    except OSError as exc:
        result.status = "FAILED"
        result.error_message = str(exc)
        return result

    result.status = "SUCCESS"
    return result


# ------------------------------------------------------------------------
# STEP 6 — PROCESS EVERY FILE (the main orchestration function)
# ------------------------------------------------------------------------

def process_all_files() -> List[CleaningResult]:
    """
    Run the full cleaning stage end to end:

        1. Make sure data/cleaned/ and logs/extraction_logs/ exist.
        2. Find every .txt file in data/extracted/.
        3. Clean and save each one (one failure does not stop the rest).
        4. Write the cleaning log.
        5. Print a summary to the console.

    This is the single function the rest of the project (or a terminal
    user) is expected to call — every other function in this file is a
    building block that this one calls in order.

    Returns:
        The list of CleaningResult records for every file processed, in
        the order they were found. Returning this list (instead of just
        printing things) lets calling code, or a future automated test,
        check exactly what happened without needing to re-read the log
        file from disk.
    """
    ensure_directories(CLEANED_DIR, LOG_DIR)

    print(f"Scanning for extracted donor text files in: {EXTRACTED_DIR.resolve()}")

    if not EXTRACTED_DIR.exists():
        print(f"[ERROR] Extracted documents directory not found: '{EXTRACTED_DIR}'")
        print("Run the extraction stage first to create data/extracted/.")
        return []

    # sorted() keeps processing order (and therefore log/console order)
    # consistent and predictable between runs.
    txt_files = sorted(p for p in EXTRACTED_DIR.glob("*.txt") if p.is_file())

    if not txt_files:
        print(f"[WARNING] No .txt files found in '{EXTRACTED_DIR}'.")
        return []

    print(f"Found {len(txt_files)} file(s). Beginning cleaning...\n")

    results: List[CleaningResult] = []

    for file_path in txt_files:
        # process_file() already catches its own errors internally, but
        # this try/except is a final safety net: it guarantees that even
        # a completely unforeseen exception cannot stop the rest of the
        # donor documents from being cleaned.
        try:
            result = process_file(file_path, CLEANED_DIR)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            result = CleaningResult(
                filename=file_path.name,
                status="FAILED",
                error_message=f"Unexpected error: {exc}",
            )

        results.append(result)

        if result.status == "SUCCESS":
            print(
                f"[SUCCESS] {result.filename} "
                f"- {result.original_chars} -> {result.cleaned_chars} chars "
                f"({result.chars_removed} removed)"
            )
        else:
            print(f"[FAILED]  {result.filename} - {result.error_message}")

    write_log(results, LOG_FILE)
    print(f"\nCleaning log written to: {LOG_FILE.resolve()}")

    print_summary(results)

    return results


# ------------------------------------------------------------------------
# STEP 7 — SUMMARY REPORTING
# ------------------------------------------------------------------------

def print_summary(results: List[CleaningResult]) -> None:
    """
    Print a short, easy-to-read summary of the whole cleaning run.

    Why this is separate from write_log(): the log file is meant for
    detailed, file-by-file auditing later on, while this function gives
    an immediate, at-a-glance answer to "did this run go well?" right
    after the script finishes — useful both during development and when
    demonstrating the pipeline.

    Args:
        results: List of CleaningResult records produced by
            process_all_files().

    Returns:
        None. Prints directly to the console.
    """
    total_files = len(results)
    successful = sum(1 for r in results if r.status == "SUCCESS")
    failed = sum(1 for r in results if r.status == "FAILED")
    total_chars_removed = sum(r.chars_removed for r in results if r.status == "SUCCESS")

    print("\nCleaning Summary")
    print("-" * 16)
    print(f"Files processed: {total_files}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Characters removed: {total_chars_removed}")


# ------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------

if __name__ == "__main__":
    process_all_files()