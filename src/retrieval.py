"""Semantic retrieval module.

Queries the ChromaDB vector index for a collection, computes cosine similarity
scores (1.0 - cosine_distance), reconstructs RetrievedChunk objects with preserved
source metadata, and supports normal (top_k) and retry (retry_top_k) search limits.
"""

from pathlib import Path
from typing import Any, Optional

from src.config import AppConfig, load_config
from src.indexing import get_chroma_client, get_embedding_model
from src.models import RetrievedChunk


class RetrievalError(Exception):
    """Base exception for retrieval errors."""
    pass


class IndexNotFoundError(RetrievalError):
    """Raised when the requested collection index does not exist in ChromaDB."""
    pass


class EmptyIndexError(RetrievalError):
    """Raised when the collection index exists but contains 0 chunks."""
    pass


def distance_to_similarity(distance: float) -> float:
    """Convert ChromaDB cosine distance to cosine similarity score.

    ChromaDB space 'cosine' uses cosine distance:
        cosine_distance = 1.0 - cosine_similarity

    Therefore:
        cosine_similarity = 1.0 - cosine_distance

    Range:
        Identical vectors: distance 0.0 -> similarity 1.0
        Orthogonal vectors: distance 1.0 -> similarity 0.0
        Opposite vectors:  distance 2.0 -> similarity -1.0

    Returns:
        float: Similarity score rounded to 4 decimal places, clamped to [-1.0, 1.0].
    """
    similarity = 1.0 - float(distance)
    # Clamp to theoretical cosine similarity bounds [-1.0, 1.0]
    clamped = max(-1.0, min(1.0, similarity))
    return round(clamped, 4)


def retrieve(
    collection_name: str,
    query: str,
    k: Optional[int] = None,
    is_retry: bool = False,
    config: Optional[AppConfig] = None,
    embedding_model: Optional[Any] = None,
    persist_dir: Optional[Path | str] = None,
) -> list[RetrievedChunk]:
    """Retrieve top-k most relevant chunks for a user query from ChromaDB.

    Args:
        collection_name: Target Chroma collection name.
        query: User question or search query string.
        k: Optional explicit number of chunks to retrieve. If None, uses
           config.retry_top_k if is_retry=True, else config.top_k.
        is_retry: If True and k is None, uses retry_top_k.
        config: Optional AppConfig instance. Loaded automatically if None.
        embedding_model: Optional preloaded or mock model with .encode() method.
        persist_dir: Optional explicit directory for ChromaDB storage.

    Returns:
        list[RetrievedChunk]: Ranked list of retrieved chunks sorted descending by similarity.

    Raises:
        IndexNotFoundError: If collection does not exist in ChromaDB.
        EmptyIndexError: If collection exists but has 0 chunks.
    """
    # 1. Handle empty query gracefully
    cleaned_query = query.strip() if query else ""
    if not cleaned_query:
        return []

    cfg = config or load_config()

    # Determine k
    if k is not None:
        target_k = int(k)
    else:
        target_k = cfg.retry_top_k if is_retry else cfg.top_k

    if target_k <= 0:
        return []

    # 2. Connect to ChromaDB
    target_persist = Path(persist_dir or cfg.get_indexes_path()).resolve()
    if not target_persist.exists():
        raise IndexNotFoundError(
            f"Index storage directory not found: {target_persist}. Run build_index.py first."
        )

    client = get_chroma_client(target_persist)
    try:
        col = client.get_collection(name=collection_name)
    except Exception as e:
        raise IndexNotFoundError(
            f"Collection index '{collection_name}' not found in {target_persist}. "
            f"Run 'python scripts/build_index.py --collection {collection_name}' first."
        ) from e

    total_chunks = col.count()
    if total_chunks == 0:
        raise EmptyIndexError(
            f"Collection index '{collection_name}' is empty (0 records). "
            f"Add documents and run build_index.py."
        )

    # Avoid requesting more results than total available in Chroma
    n_results = min(target_k, total_chunks)

    # 3. Generate query embedding
    model = embedding_model or get_embedding_model(cfg.embedding_model)
    query_embeddings = model.encode([cleaned_query], convert_to_numpy=True).tolist()

    # 4. Perform vector similarity search in ChromaDB
    results = col.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]

    retrieved_chunks: list[RetrievedChunk] = []

    # 5. Construct RetrievedChunk objects preserving metadata and similarity
    for chunk_id, dist, meta, doc_text in zip(ids, distances, metadatas, documents):
        score = distance_to_similarity(dist)

        page_val = meta.get("page")
        page = page_val if page_val is not None and page_val != -1 else None

        sec_val = meta.get("section")
        section = sec_val if sec_val else None

        source_file = meta.get("source_file", "")
        collection = meta.get("collection", collection_name)

        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=doc_text,
                source_file=source_file,
                collection=collection,
                page=page,
                section=section,
                score=score,
            )
        )

    # Sort descending by score for deterministic ordering
    retrieved_chunks.sort(key=lambda c: c.score, reverse=True)

    return retrieved_chunks
