"""
generate_embeddings.py
======================
Stage 4 of the donor knowledge-base pipeline:

    Donor PDFs → Extraction → Cleaning → Chunking → [EMBEDDINGS] → ChromaDB

PURPOSE
-------
This module reads the JSON chunk files produced by chunk_donor_docs.py,
converts each chunk's text into a dense numerical vector (an "embedding")
using a pre-trained sentence-transformer model, and stores those vectors —
along with their metadata — inside a persistent ChromaDB vector database.

The resulting database is the retrieval engine that powers the evaluation
system: when a grant proposal is submitted, its text is embedded with the
same model, and ChromaDB returns the donor-document chunks whose embeddings
are most similar — giving the Gemini evaluation step the relevant GIF context
it needs to score the proposal against funding criteria.

CHUNK SCHEMA (as produced by chunk_donor_docs.py)
-------------------------------------------------
Every chunk loaded from data/chunks/ is expected to contain:

    {
        "chunk_id":        "donor_organization_profile_002",
        "source_document": "Donor_Organization_Profile.txt",
        "section_title":   "Eligibility Requirements (Part 1 of 2)",
        "chunk_index":     2,
        "character_count": 1000,
        "text":            "chunk content here..."
    }

The two fields this module depends on most critically are:
    - "chunk_id"  : used as the unique key in ChromaDB
    - "text"      : the content that gets embedded


EMBEDDING MODEL
---------------
Model: sentence-transformers/multi-qa-mpnet-base-dot-v1

This model is optimised for semantic search (the "multi-qa" prefix indicates
it was fine-tuned on question-answer pairs). It produces 768-dimensional
vectors and clusters related concepts (e.g. "youth employment" ≈ "vocational
training for young people") even when surface wording differs. ChromaDB uses
cosine similarity to rank retrieved chunks by relevance.

INSTALL DEPENDENCIES
--------------------
    pip install sentence-transformers chromadb
"""

import json
import time
from datetime import datetime
from pathlib import Path

# SentenceTransformers: open-source library for producing dense text embeddings.
from sentence_transformers import SentenceTransformer

# ChromaDB: lightweight, persistent vector database with a simple Python API.
import chromadb


# ── Directory constants ────────────────────────────────────────────────────────
CHUNKS_DIR   = Path("data/chunks")      # Input:  JSON chunk files from chunker
VECTORDB_DIR = Path("data/vectordb")    # Output: ChromaDB persistent store
LOG_DIR      = Path("logs/retrieval_logs")
LOG_FILE     = LOG_DIR / "embedding_log.txt"

# ── Model and collection constants ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
COLLECTION_NAME      = "donor_knowledge_base"

# ── Required chunk fields (matching chunk_donor_docs.py output exactly) ────────
# These are the fields that chunk_donor_docs.py writes to every chunk JSON.
# Validation checks for these fields before any embedding work begins.
REQUIRED_CHUNK_FIELDS = {"chunk_id", "source_document", "text"}

# ── Verification query settings ───────────────────────────────────────────────
TEST_QUERY     = "community health intervention"
TEST_TOP_K     = 3
PREVIEW_LENGTH = 150


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTORY SETUP
# ══════════════════════════════════════════════════════════════════════════════

def ensure_directories() -> None:
    """
    Create all required directories if they do not already exist.

    Using Path.mkdir(parents=True, exist_ok=True) is safe to call even when
    the directories are already present — it will not raise an error or
    overwrite existing content.
    """
    for directory in (CHUNKS_DIR, VECTORDB_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def _log(message: str) -> None:
    """
    Write a timestamped message to both the log file and stdout.

    Every log call goes to both places so the operator can monitor progress
    in the terminal while a permanent record is kept for debugging later.

    Args:
        message: The message to record.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line  = f"[{timestamp}] {message}\n"
    print(log_line, end="")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)


def _log_session_separator() -> None:
    """Write a visible header to the log so sessions are easy to distinguish."""
    separator = (
        f"\n{'=' * 70}\n"
        f"Embedding session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 70}\n"
    )
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(separator)
    print(separator, end="")


def _log_skip_warning(filename: str, chunk_id: str, missing_fields: set) -> None:
    """
    Write a structured warning when a chunk is skipped due to missing fields.

    Structured format makes it easy to scan the log and identify exactly
    which file and chunk caused the problem, and what was missing.

    Args:
        filename:       Name of the JSON chunk file being processed.
        chunk_id:       The chunk_id value if present, otherwise '(unknown)'.
        missing_fields: Set of field names that were absent from the chunk.
    """
    warning = (
        f"WARNING:\n"
        f"  File    : {filename}\n"
        f"  Chunk   : {chunk_id}\n"
        f"  Missing : {missing_fields}\n"
    )
    _log(warning)


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_chunks() -> tuple[list[dict], list[str]]:
    """
    Scan data/chunks/ recursively and load all valid JSON chunk files.

    WHY VALIDATION IS DONE HERE
    ---------------------------
    If a chunk is missing its "text" field, model.encode() will receive an
    empty string and produce a meaningless vector. If it is missing "chunk_id",
    ChromaDB will have no unique key to store it under. Catching these problems
    during loading, before any embedding work begins, prevents silent failures
    that would be very difficult to diagnose later.

    WHAT CONSTITUTES A VALID CHUNK
    ------------------------------
    A chunk is valid if it contains all three required fields (chunk_id,
    source_document, text) and the text field is not empty after stripping
    whitespace. Optional fields (section_title, chunk_index, character_count)
    are stored if present and silently defaulted if absent.

    Returns:
        Tuple of:
          - valid_chunks (list[dict]): Chunks that passed all validation checks.
          - load_errors  (list[str]):  Error messages for skipped/failed items.
    """
    valid_chunks = []
    load_errors  = []

    chunk_files = sorted(CHUNKS_DIR.rglob("*.json"))

    if not chunk_files:
        _log(
            f"WARNING: No JSON chunk files found in '{CHUNKS_DIR}'. "
            f"Ensure chunk_donor_docs.py has been run first."
        )
        return [], []

    _log(f"Found {len(chunk_files)} chunk file(s) in '{CHUNKS_DIR}'.")

    for chunk_file in chunk_files:
        file_valid   = 0
        file_skipped = 0

        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # chunk_donor_docs.py writes a list of chunk dicts per file.
            # Guard against malformed files that contain a bare dict or other type.
            if isinstance(data, dict):
                raw_chunks = [data]
            elif isinstance(data, list):
                raw_chunks = data
            else:
                raise ValueError(f"Unexpected top-level JSON type: {type(data).__name__}")

            for chunk in raw_chunks:
                # Identify chunk_id early so it can appear in the skip warning
                # even when it is the only field present.
                chunk_id = chunk.get("chunk_id", "(unknown)")

                # Check all required fields are present
                missing = REQUIRED_CHUNK_FIELDS - set(chunk.keys())
                if missing:
                    _log_skip_warning(chunk_file.name, chunk_id, missing)
                    load_errors.append(
                        f"{chunk_file.name} | chunk '{chunk_id}' | missing: {missing}"
                    )
                    file_skipped += 1
                    continue

                # Reject chunks with empty text — an empty vector is meaningless
                if not chunk["text"].strip():
                    msg = f"{chunk_file.name} | chunk '{chunk_id}' | 'text' is empty"
                    _log(f"WARNING: {msg} — skipped.")
                    load_errors.append(msg)
                    file_skipped += 1
                    continue

                valid_chunks.append(chunk)
                file_valid += 1

            _log(
                f"  {chunk_file.name}: "
                f"{file_valid} valid, {file_skipped} skipped."
            )

        except json.JSONDecodeError as exc:
            msg = f"Could not parse JSON in '{chunk_file.name}': {exc}"
            _log(f"ERROR: {msg}")
            load_errors.append(msg)

        except Exception as exc:
            msg = f"Unexpected error loading '{chunk_file.name}': {exc}"
            _log(f"ERROR: {msg}")
            load_errors.append(msg)

    _log(
        f"Loading complete: {len(valid_chunks)} valid chunk(s) across "
        f"{len(chunk_files)} file(s). ({len(load_errors)} issue(s) logged.)"
    )
    return valid_chunks, load_errors


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformer model used to convert text into vectors.

    WHY LOAD ONCE
    -------------
    Transformer models consist of hundreds of millions of parameters. Loading
    from disk takes several seconds and significant memory. By loading once and
    reusing the same model object for every chunk, we avoid that overhead on
    every iteration. The model is passed as an argument to functions that need
    it rather than loaded globally, keeping the code testable and modular.

    WHY THIS MODEL
    --------------
    multi-qa-mpnet-base-dot-v1 was fine-tuned on multiple question-answering
    datasets. Its vectors cluster semantically related content even when exact
    words differ — so a proposal mentioning "job readiness for young adults"
    will retrieve a donor chunk about "youth vocational training". This semantic
    matching is the core value of the RAG approach over keyword search.

    Returns:
        A fully loaded SentenceTransformer ready for inference.

    Raises:
        RuntimeError: Wraps any model-loading exception with a clear message.
    """
    _log(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        # A trivial test encode confirms the model weights loaded correctly
        # and the runtime (CPU/GPU) is initialised before the main loop starts.
        model.encode("initialisation check", show_progress_bar=False)
        _log("Embedding model loaded successfully.")
        return model
    except Exception as exc:
        msg = f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {exc}"
        _log(f"ERROR: {msg}")
        raise RuntimeError(msg) from exc


# ══════════════════════════════════════════════════════════════════════════════
# CHROMADB INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def initialize_chromadb() -> tuple[chromadb.PersistentClient, object]:
    """
    Create or reopen the ChromaDB persistent client and donor_knowledge_base
    collection.

    WHY PERSISTENT CLIENT
    ---------------------
    ChromaDB's PersistentClient writes the HNSW vector index and all stored
    metadata to disk at data/vectordb/. This means embeddings survive process
    restarts — the evaluation system can query the database without re-embedding
    all donor documents on every run.

    WHY get_or_create_collection
    ----------------------------
    This call is idempotent: if the collection already exists it is reused; if
    not it is created. Using cosine similarity (hnsw:space = "cosine") is
    appropriate for sentence-transformer embeddings because it measures the
    angle between vectors, which captures semantic similarity regardless of
    vector magnitude.

    WHY NOT create_collection
    -------------------------
    Calling create_collection when the collection already exists raises an
    error. get_or_create_collection prevents that, making the script safely
    re-runnable without clearing the database first.

    Returns:
        Tuple of (PersistentClient, Collection).

    Raises:
        RuntimeError: If ChromaDB cannot be initialised.
    """
    _log(f"Initialising ChromaDB at '{VECTORDB_DIR}' ...")
    try:
        client = chromadb.PersistentClient(path=str(VECTORDB_DIR))

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        existing_count = collection.count()
        _log(
            f"ChromaDB ready. Collection '{COLLECTION_NAME}' "
            f"already contains {existing_count} embedding(s)."
        )
        return client, collection

    except Exception as exc:
        msg = f"Failed to initialise ChromaDB: {exc}"
        _log(f"ERROR: {msg}")
        raise RuntimeError(msg) from exc


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING GENERATION AND STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def generate_embeddings(
    chunks:     list[dict],
    model:      SentenceTransformer,
    collection: object,
) -> dict:
    """
    Embed each chunk's text and insert the vector plus metadata into ChromaDB.

    WHAT GETS STORED IN CHROMADB
    ----------------------------
    For each chunk, ChromaDB stores four things:

    1. id (chunk_id)
       WHY: ChromaDB requires a unique string key per entry. chunk_id serves
       as that key and also enables duplicate detection — if we try to re-run
       this script, existing IDs are skipped rather than duplicated.
       LATER USE: retrieval debugging ("which exact chunk was this?"),
       evaluation traceability ("which source supported this score?").

    2. embedding (768-dimensional float vector)
       WHY: This is the searchable representation of the chunk's meaning.
       At query time, a proposal's text is embedded with the same model and
       ChromaDB returns the chunks whose vectors are most similar.

    3. document (chunk["text"])
       WHY: The raw text is stored alongside the vector so that retrieval
       returns the actual content, not just an ID. The Gemini evaluation
       prompt is built from this returned text.

    4. metadata dict — stored fields and their purpose:

       "chunk_id" (str)
           WHY STORED: Preserved in metadata alongside the ChromaDB id field
           so it is always accessible in query results without a separate
           lookup. Supports source attribution in PDF report generation.

       "source_document" (str)  e.g. "Funding_Guidelines.txt"
           WHY STORED: Tells the evaluation system and human reviewer which
           GIF donor document a retrieved chunk came from. Supports source
           citation in the evaluation report and enables metadata filtering
           (e.g. "only retrieve from Eligibility_Requirements.txt").

       "section_title" (str)  e.g. "Budget Guidelines (Part 2 of 3)"
           WHY STORED: Identifies the logical section within the document.
           Supports retrieval debugging ("did we retrieve the right section?")
           and makes generated reports more readable by showing section context
           alongside retrieved text.

       "chunk_index" (int)
           WHY STORED: The sequential position of this chunk within its source
           document. Supports reconstruction of document order when multiple
           chunks from the same document are retrieved, and helps debugging
           chunk boundary issues.

       "character_count" (int)
           WHY STORED: Records how large this chunk was. Useful for monitoring
           chunking quality and for explaining retrieval behaviour — very short
           chunks may embed poorly; very long ones may exceed model token limits.

    DUPLICATE PROTECTION
    --------------------
    All existing chunk IDs are fetched once at the start of the loop into a
    Python set. Each new chunk_id is checked against this set in O(1) time
    before any embedding work is done. This avoids wasted compute and prevents
    ChromaDB errors from attempted duplicate insertions.

    Args:
        chunks:     List of validated chunk dicts from load_chunks().
        model:      Loaded SentenceTransformer from load_embedding_model().
        collection: Initialised ChromaDB collection from initialize_chromadb().

    Returns:
        Dict with integer counts: 'inserted', 'skipped', 'failed'.
    """
    inserted = 0
    skipped  = 0
    failed   = 0

    # ── Fetch all existing IDs once for efficient duplicate checking ───────────
    # Fetching once and using a set is O(1) per lookup. Querying ChromaDB
    # individually for each chunk would be O(n) database calls — much slower.
    try:
        existing_ids = set(collection.get()["ids"])
        _log(f"Duplicate check: {len(existing_ids)} chunk(s) already in collection.")
    except Exception as exc:
        _log(f"WARNING: Could not retrieve existing IDs — duplicate protection "
             f"disabled for this session. Reason: {exc}")
        existing_ids = set()

    total = len(chunks)
    _log(f"Starting embedding generation for {total} chunk(s)...")

    for i, chunk in enumerate(chunks, start=1):

        # ── Extract fields using the actual chunk_donor_docs.py field names ────
        chunk_id        = chunk["chunk_id"]
        text            = chunk["text"]               # field name from chunker
        source_document = chunk["source_document"]    # field name from chunker

        # Optional fields — default gracefully if not present
        section_title   = chunk.get("section_title",   "")
        chunk_index     = chunk.get("chunk_index",      0)
        character_count = chunk.get("character_count",  0)

        # ── Duplicate check ───────────────────────────────────────────────────
        # Skip without embedding if this chunk_id is already in the collection.
        # This makes the script safely re-runnable: only new chunks are added.
        if chunk_id in existing_ids:
            _log(f"  SKIP [{i}/{total}] '{chunk_id}' already in collection.")
            skipped += 1
            continue

        # ── Generate embedding ────────────────────────────────────────────────
        # model.encode() converts the chunk text into a 768-dimensional vector.
        # convert_to_tensor=False returns a numpy array, which we convert to a
        # plain Python list for ChromaDB (.tolist() is required by ChromaDB's API).
        try:
            embedding = model.encode(text, convert_to_tensor=False)

            # ── Store in ChromaDB ─────────────────────────────────────────────
            collection.add(
                ids        = [chunk_id],
                embeddings = [embedding.tolist()],
                documents  = [text],
                metadatas  = [{
                    # See docstring above for why each field is stored.
                    "chunk_id":        chunk_id,
                    "source_document": source_document,
                    "section_title":   section_title,
                    "chunk_index":     chunk_index,
                    "character_count": character_count,
                }],
            )

            # Track this ID locally so subsequent chunks in the same run
            # are also protected against duplication.
            existing_ids.add(chunk_id)
            inserted += 1

            # Log progress every 10 chunks to keep output readable
            if i % 10 == 0 or i == total:
                _log(
                    f"  Progress: {i}/{total} | "
                    f"inserted {inserted} | skipped {skipped} | failed {failed}"
                )

        except Exception as exc:
            _log(f"  ERROR [{i}/{total}] Failed on '{chunk_id}': {exc}")
            failed += 1

    _log(
        f"Embedding complete — "
        f"inserted: {inserted}, skipped: {skipped}, failed: {failed}."
    )
    return {"inserted": inserted, "skipped": skipped, "failed": failed}


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION QUERY
# ══════════════════════════════════════════════════════════════════════════════

def test_retrieval(model: SentenceTransformer, collection: object) -> None:
    """
    Run a test semantic search and display the top results in a readable format.

    WHY THIS STEP EXISTS
    --------------------
    Embedding generation can complete without errors while still producing poor
    retrieval results — for example, if the wrong model was used, if metadata
    was corrupted, or if the text was empty. Running a live retrieval query and
    printing the results gives the developer immediate, concrete confirmation
    that the system is working end-to-end before moving to the next pipeline
    stage.

    WHAT TO LOOK FOR
    ----------------
    The query "youth employment programs" should return chunks from GIF donor
    documents related to youth empowerment or vocational training. If the top
    results are clearly related to that topic, retrieval is working correctly.
    If unrelated chunks appear at the top (e.g. budget guidelines), the
    embeddings or collection may need investigation.

    SIMILARITY SCORE
    ----------------
    ChromaDB returns cosine distance (0 = identical vectors, 2 = opposite
    vectors). We convert this to a similarity score in [0, 1] using:
        similarity = 1 - distance
    A score above 0.5 generally indicates a meaningful semantic match.

    Args:
        model:      The same SentenceTransformer used during embedding.
                    Using the same model is critical — query and document
                    vectors must live in the same vector space.
        collection: The populated ChromaDB collection to query against.
    """
    _log(f'\nRunning verification query: "{TEST_QUERY}"')

    try:
        # Embed the query text using the same model as the indexed chunks
        query_embedding = model.encode(TEST_QUERY, convert_to_tensor=False)

        results = collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results        = TEST_TOP_K,
            include          = ["documents", "metadatas", "distances"],
        )

        ids       = results.get("ids",       [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        print("\n" + "─" * 65)
        print(f'Verification Query: "{TEST_QUERY}"')
        print(f"Top {TEST_TOP_K} Results:")
        print("─" * 65)

        if not ids:
            print("  No results returned — collection may be empty.")
        else:
            for rank, (cid, doc, meta, dist) in enumerate(
                    zip(ids, documents, metadatas, distances), start=1):

                similarity = round(1 - dist, 4)
                preview    = doc[:PREVIEW_LENGTH].replace("\n", " ")

                print(f"\n  Result {rank}:")
                print(f"    Chunk ID        : {cid}")
                print(f"    Source Document : {meta.get('source_document', 'N/A')}")
                print(f"    Section Title   : {meta.get('section_title',   'N/A')}")
                print(f"    Similarity Score: {similarity}")
                print(f"    Preview         : {preview}...")

        print("─" * 65 + "\n")
        _log("Verification query completed.")

    except Exception as exc:
        _log(f"ERROR: Verification query failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(
    chunk_files_processed: int,
    chunks_discovered:     int,
    counts:                dict,
    collection:            object,
    elapsed_seconds:       float,
) -> None:
    """
    Print and log a human-readable summary of the embedding session.

    Includes the live ChromaDB collection size (collection.count()) rather than
    just the number inserted this session, so the operator can confirm the
    database state matches expectations after any combination of insertions,
    skips, and failures.

    Args:
        chunk_files_processed: Number of JSON chunk files read from disk.
        chunks_discovered:     Total valid chunks found across all files.
        counts:                Dict with 'inserted', 'skipped', 'failed'.
        collection:            ChromaDB collection (used for live .count()).
        elapsed_seconds:       Total wall-clock time for the session.
    """
    try:
        collection_size = collection.count()
    except Exception:
        collection_size = "unavailable"

    summary = (
        f"\n{'─' * 52}\n"
        f"Embedding Generation Summary\n"
        f"{'─' * 52}\n"
        f"Chunk Files Processed  : {chunk_files_processed}\n"
        f"Total Chunks Discovered: {chunks_discovered}\n"
        f"Chunks Embedded        : {counts.get('inserted', 0)}\n"
        f"Chunks Skipped         : {counts.get('skipped',  0)}\n"
        f"Chunks Failed          : {counts.get('failed',   0)}\n"
        f"ChromaDB Collection    : {COLLECTION_NAME}\n"
        f"Collection Size (total): {collection_size}\n"
        f"Embedding Model        : {EMBEDDING_MODEL_NAME}\n"
        f"Database Location      : {VECTORDB_DIR}\n"
        f"Processing Time        : {elapsed_seconds:.2f}s\n"
        f"{'─' * 52}\n"
    )
    print(summary)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(summary)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Run the full embedding pipeline from the project root:

        python src/donor_pipeline/generate_embeddings.py

    Pipeline executed in order:

        ensure_directories()
              ↓
        load_chunks()
              ↓
        load_embedding_model()
              ↓
        initialize_chromadb()
              ↓
        generate_embeddings()
              ↓
        test_retrieval()
              ↓
        print_summary()
    """
    session_start = time.time()

    # ── 0. Ensure all required directories exist ──────────────────────────────
    ensure_directories()
    _log_session_separator()

    # ── 1. Load and validate chunks from data/chunks/ ─────────────────────────
    chunks, load_errors = load_chunks()
    if not chunks:
        _log("No valid chunks to embed. Run chunk_donor_docs.py first. Exiting..")
        raise SystemExit(1)

    # ── 2. Load the sentence-transformer embedding model ──────────────────────
    model = load_embedding_model()

    # ── 3. Initialise ChromaDB (create or reopen collection) ──────────────────
    client, collection = initialize_chromadb()

    # ── 4. Generate embeddings and insert into ChromaDB ───────────────────────
    counts = generate_embeddings(chunks, model, collection)

    # ── 5. Run a verification query to confirm retrieval works end-to-end ─────
    test_retrieval(model, collection)

    # ── 6. Print and log the session summary ──────────────────────────────────
    elapsed = time.time() - session_start

    # Count chunk files so the summary can report files processed vs chunks found
    chunk_file_count = len(list(CHUNKS_DIR.rglob("*.json")))

    print_summary(
        chunk_files_processed = chunk_file_count,
        chunks_discovered     = len(chunks),
        counts                = counts,
        collection            = collection,
        elapsed_seconds       = elapsed,
    )   