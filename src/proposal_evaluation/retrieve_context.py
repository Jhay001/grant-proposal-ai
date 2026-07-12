"""
retrieve_context.py

Belongs to the ONLINE proposal evaluation pipeline of the
AI-Powered Grant Proposal Evaluation System.

Responsibility (and ONLY responsibility):
    1. Load the persistent ChromaDB donor knowledge base.
    2. Query it using an already-generated proposal embedding.
    3. Rank the retrieved donor chunks by similarity.
    4. Build the complete Gemini evaluation prompt.

This module does NOT read PDFs, extract or clean text, generate embeddings,
call Gemini, score proposals, or generate reports. It receives the dictionary
produced by process_proposal.py and hands off a ready-to-send prompt to the
next pipeline stage (the Gemini evaluator).
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any

import chromadb # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Centralized here so both this module and any test/CLI harness agree on
# where the persistent database lives and which collection to open.
VECTORDB_DIR = Path("data/vectordb")
COLLECTION_NAME = "donor_knowledge_base"

# How many donor chunks to retrieve per proposal. Kept as a named constant
# (rather than a magic number scattered through the code) so it is easy to
# tune later based on retrieval-quality experiments (see Phase 6 of the
# project roadmap).
TOP_K = 10

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "retrieval.log"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logger() -> logging.Logger:
    """Configure and return this module's dedicated logger.

    A dedicated logger (rather than the root logger) keeps this module's
    logging behavior independent of logging configuration done elsewhere in
    the pipeline (e.g. process_proposal.py's own logger).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("proposal_evaluation.retrieve_context")
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
# Module-level ChromaDB client/collection cache
# ---------------------------------------------------------------------------

# Opening a persistent ChromaDB client has real I/O cost. Caching it at
# module level means repeated calls to retrieve_context() within the same
# process (e.g. batch-evaluating several proposals) only pay that cost once.
_chroma_collection: chromadb.api.models.Collection.Collection | None = None


# ---------------------------------------------------------------------------
# Stage 1 — Load ChromaDB
# ---------------------------------------------------------------------------

def load_vectordb(
    vectordb_dir: Path = VECTORDB_DIR,
    collection_name: str = COLLECTION_NAME,
):
    """Load the persistent ChromaDB donor knowledge base collection.

    Uses a module-level cache so the database is only opened once per
    process, regardless of how many proposals are evaluated in that run.

    Args:
        vectordb_dir: Directory containing the persistent ChromaDB store.
        collection_name: Name of the donor knowledge base collection.

    Returns:
        The opened ChromaDB collection object.

    Raises:
        FileNotFoundError: If the vector store directory does not exist,
            which almost always means the donor indexing pipeline has not
            been run yet.
        RuntimeError: If the client connects but the named collection
            cannot be opened (e.g. it was never created, or the name is
            misspelled).
    """
    global _chroma_collection

    if _chroma_collection is not None:
        return _chroma_collection

    if not vectordb_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at '{vectordb_dir}'. "
            "Has the donor knowledge base indexing pipeline been run yet?"
        )

    try:
        client = chromadb.PersistentClient(path=str(vectordb_dir))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open persistent ChromaDB client at '{vectordb_dir}': {exc}"
        ) from exc

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open collection '{collection_name}'. "
            f"Confirm it was created by the donor indexing pipeline. "
            f"Underlying error: {exc}"
        ) from exc

    _chroma_collection = collection
    logger.info("ChromaDB collection '%s' loaded successfully.", collection_name)
    return collection


# ---------------------------------------------------------------------------
# Stage 2 — Semantic Search
# ---------------------------------------------------------------------------

def retrieve_chunks(
    collection,
    proposal_embedding: list[float],
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Query ChromaDB for the most relevant donor chunks.

    The proposal embedding is expected to already have been generated by
    process_proposal.py; this function never (re)computes an embedding
    itself, since embedding generation is explicitly out of scope here.

    Args:
        collection: The ChromaDB collection returned by load_vectordb().
        proposal_embedding: The proposal's embedding vector.
        top_k: Number of nearest chunks to retrieve.

    Returns:
        The raw ChromaDB query result dictionary, containing (at minimum)
        "documents", "metadatas", and "distances" for the queried chunks.

    Raises:
        RuntimeError: If the query itself fails (e.g. dimension mismatch
            between the proposal embedding and the indexed embeddings).
    """
    try:
        query_result = collection.query(
            query_embeddings=[proposal_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB query failed. This often indicates an embedding "
            f"dimension mismatch between the proposal embedding and the "
            f"indexed donor chunks. Underlying error: {exc}"
        ) from exc

    return query_result


# ---------------------------------------------------------------------------
# Stage 3 — Rank Results
# ---------------------------------------------------------------------------

def rank_chunks(query_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a raw ChromaDB query result into a ranked list of chunks.

    ChromaDB returns cosine *distance* (lower is more similar); this
    function converts each distance into a similarity score
    (similarity = 1 - distance) and sorts chunks from most to least
    similar. Ranking is done explicitly here, rather than trusting
    ChromaDB's return order, so the pipeline's notion of "ranked" does not
    silently depend on an implementation detail of the vector store.

    Args:
        query_result: The raw dictionary returned by retrieve_chunks().

    Returns:
        A list of dictionaries, sorted by descending similarity_score, each
        containing:
            - chunk_id
            - source_document
            - section_title
            - similarity_score
            - text
    """
    documents = query_result["documents"][0]
    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]
    ids = query_result.get("ids", [[None] * len(documents)])[0]

    ranked: list[dict[str, Any]] = []
    for chunk_id, document_text, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        similarity_score = 1 - distance
        ranked.append(
            {
                "chunk_id": chunk_id or metadata.get("chunk_id", "unknown"),
                "source_document": metadata.get("source_document", "unknown"),
                "section_title": metadata.get("section_title", "unknown"),
                "similarity_score": round(similarity_score, 4),
                "text": document_text,
            }
        )

    # Explicit sort rather than relying on ChromaDB's return order — some
    # backends return results already sorted by distance, but that is an
    # implementation detail this pipeline should not silently depend on.
    ranked.sort(key=lambda chunk: chunk["similarity_score"], reverse=True)

    return ranked


# ---------------------------------------------------------------------------
# Stage 4 — Build Evaluation Prompt
# ---------------------------------------------------------------------------

def build_prompt(proposal_text: str, ranked_chunks: list[dict[str, Any]]) -> str:
    """Assemble the complete Gemini evaluation prompt.

    The prompt is built from four clearly separated sections (system role,
    donor context, proposal, evaluation criteria/output format) so that the
    resulting text is easy to visually audit and debug, and so that each
    section's content can be changed independently later (e.g. adjusting
    only the output JSON schema) without touching the others.

    Args:
        proposal_text: The cleaned proposal text (from process_proposal.py).
        ranked_chunks: The ranked donor chunks returned by rank_chunks().

    Returns:
        The full prompt string, ready to send to the Gemini evaluator
        module.
    """
    section_divider = "-" * 50

    # --- Donor context section -------------------------------------------
    donor_context_blocks = []
    for chunk in ranked_chunks:
        donor_context_blocks.append(
            "Source Document: {source}\n"
            "Section Title: {section}\n"
            "Chunk Text:\n{text}".format(
                source=chunk["source_document"],
                section=chunk["section_title"],
                text=chunk["text"],
            )
        )
    donor_context_text = "\n\n".join(donor_context_blocks)

    prompt = f"""{section_divider}
SYSTEM ROLE
{section_divider}
You are an expert grant proposal evaluator acting on behalf of a donor
organization, the Global Impact Foundation (GIF). Your task is to assess a
submitted grant proposal strictly against the donor context provided below,
which includes GIF's funding guidelines, eligibility requirements, strategic
priorities, and examples of previously funded projects. Base your evaluation
only on the proposal text and the donor context provided. Do not assume
information that is not present in either.

{section_divider}
DONOR CONTEXT
{section_divider}
{donor_context_text}

{section_divider}
PROPOSAL
{section_divider}
{proposal_text}

{section_divider}
EVALUATION CRITERIA
{section_divider}
Evaluate the proposal against each of the following criteria, scoring each
on a scale of 1 (very weak) to 5 (excellent), grounded in the donor context
above:

- Relevance
- Feasibility
- Innovation
- Sustainability
- Budget Justification
- Organizational Capacity
- Expected Impact

{section_divider}
OUTPUT FORMAT
{section_divider}
Return ONLY valid JSON. Do not include any explanatory text, commentary, or
Markdown code fences before or after the JSON. The JSON must conform exactly
to the following schema:

{{
    "proposal_summary": "",
    "criteria_scores": {{
        "relevance": 0,
        "feasibility": 0,
        "innovation": 0,
        "sustainability": 0,
        "budget_justification": 0,
        "organizational_capacity": 0,
        "expected_impact": 0
    }},
    "strengths": [],
    "weaknesses": [],
    "risk_flags": [],
    "funding_alignment_analysis": "",
    "final_recommendation": ""
}}
"""
    return prompt


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def retrieve_context(proposal_data: dict[str, Any]) -> dict[str, Any]:
    """Retrieve donor context for a proposal and build the evaluation prompt.

    Workflow: load ChromaDB -> retrieve chunks -> rank chunks -> build
    prompt -> return dictionary.

    This is the only function the next pipeline stage (the Gemini
    evaluator) should call into for prompt assembly.

    Args:
        proposal_data: The dictionary returned by process_proposal.py,
            containing at minimum "clean_text" and "embedding".

    Returns:
        A dictionary:
            {
                "proposal_text": str,
                "retrieved_chunks": list[dict],
                "evaluation_prompt": str,
            }

    Raises:
        KeyError: If proposal_data is missing required keys.
        FileNotFoundError, RuntimeError: propagated from load_vectordb() or
            retrieve_chunks() if the vector store cannot be loaded or
            queried.
        Exception: any unexpected error is logged with a full traceback
            before being re-raised.
    """
    start_time = time.perf_counter()
    logger.info("=" * 70)
    logger.info("Retrieval started for proposal: %s", proposal_data.get("filename", "unknown"))

    try:
        proposal_text = proposal_data["clean_text"]
        proposal_embedding = proposal_data["embedding"]

        # --- Stage 1: Load ChromaDB ---------------------------------------
        collection = load_vectordb()
        indexed_chunk_count = collection.count()
        logger.info(
            "Collection '%s' contains %d indexed chunk(s).",
            COLLECTION_NAME,
            indexed_chunk_count,
        )

        # --- Stage 2: Semantic search --------------------------------------
        query_result = retrieve_chunks(collection, proposal_embedding, top_k=TOP_K)

        # --- Stage 3: Rank results -------------------------------------------
        ranked_chunks = rank_chunks(query_result)
        similarity_scores = [chunk["similarity_score"] for chunk in ranked_chunks]
        logger.info(
            "Retrieved %d chunk(s). Similarity scores: %s",
            len(ranked_chunks),
            similarity_scores,
        )

        # --- Stage 4: Build prompt -------------------------------------------
        evaluation_prompt = build_prompt(proposal_text, ranked_chunks)
        logger.info("Built evaluation prompt: %d character(s).", len(evaluation_prompt))

        elapsed_seconds = time.perf_counter() - start_time
        logger.info("Retrieval completed successfully in %.2f second(s).", elapsed_seconds)

        return {
            "proposal_text": proposal_text,
            "retrieved_chunks": ranked_chunks,
            "evaluation_prompt": evaluation_prompt,
        }

    except Exception:
        logger.error("Retrieval FAILED.\n%s", traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Standalone development testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A small mock proposal_data dictionary, matching the shape produced by
    # process_proposal.py, so this module can be exercised independently of
    # the actual PDF-processing stage. The embedding here is a placeholder;
    # for a real test, paste in an embedding produced by process_proposal.py
    # so its dimension matches the indexed donor chunks.
    EMBEDDING_DIMENSION_PLACEHOLDER = 768

    mock_proposal_data = {
        "filename": "mock_proposal.pdf",
        "clean_text": (
            "This proposal seeks funding to establish a community-based "
            "digital literacy program for women in rural Ghana, combining "
            "foundational digital skills training with mobile-money "
            "financial literacy to support women's economic inclusion."
        ),
        "embedding": [0.0] * EMBEDDING_DIMENSION_PLACEHOLDER,
    }

    print("-" * 50)
    try:
        result = retrieve_context(mock_proposal_data)

        chunks = result["retrieved_chunks"]
        similarity_scores = [chunk["similarity_score"] for chunk in chunks]
        collection = load_vectordb()

        print("Retrieval Summary")
        print("-" * 50)
        print(f"Collection Loaded      : {COLLECTION_NAME}")
        print(f"Indexed Chunks         : {collection.count()}")
        print(f"Retrieved Chunks       : {len(chunks)}")
        print(f"Highest Similarity     : {max(similarity_scores):.2f}")
        print(f"Lowest Similarity      : {min(similarity_scores):.2f}")
        print(f"Prompt Length          : {len(result['evaluation_prompt']):,} characters")
        print("Status                 : SUCCESS")
        print("-" * 50)
    except Exception as exc:  # noqa: BLE001 - top-level CLI catch is intentional
        print("Retrieval Summary")
        print("-" * 50)
        print("Status                 : FAILED")
        print(f"Error                  : {exc}")
        print("-" * 50)