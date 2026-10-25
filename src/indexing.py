"""Vector store indexing module using ChromaDB and SentenceTransformer.

Loads documents, chunks them, embeds them in batches with sentence-transformers,
and stores embeddings, texts, and metadata in a persistent ChromaDB index.
Rebuilding a collection deletes and recreates it to avoid duplicate records.
"""

from pathlib import Path
import time
from typing import Any, Optional

import chromadb

from src.chunking import chunk_documents
from src.config import AppConfig, load_config
from src.ingestion import load_collection
from src.models import Chunk

_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """Load and cache the SentenceTransformer model to avoid repeated disk reads."""
    if model_name not in _EMBEDDING_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL_CACHE[model_name]


def get_chroma_client(persist_dir: Path | str) -> chromadb.ClientAPI:
    """Create a persistent ChromaDB client pointing to the given directory."""
    path_str = str(Path(persist_dir).resolve())
    return chromadb.PersistentClient(path=path_str)


def index_exists(
    collection_name: str,
    config: Optional[AppConfig] = None,
    persist_dir: Optional[Path | str] = None,
) -> bool:
    """Check whether a vector index for the collection exists and contains chunks.

    Args:
        collection_name: Name of the collection.
        config: Optional AppConfig for resolving default indexes_dir.
        persist_dir: Optional explicit persistence directory override.

    Returns:
        bool: True if collection exists and has at least one chunk indexed.
    """
    cfg = config or load_config()
    target_dir = Path(persist_dir or cfg.get_indexes_path())
    if not target_dir.exists():
        return False

    client = get_chroma_client(target_dir)
    try:
        col = client.get_collection(name=collection_name)
        return col.count() > 0
    except Exception:
        return False


def index_chunk_count(
    collection_name: str,
    config: Optional[AppConfig] = None,
    persist_dir: Optional[Path | str] = None,
) -> int:
    """Return the total number of chunks indexed for a given collection.

    Args:
        collection_name: Name of the collection.
        config: Optional AppConfig for resolving default indexes_dir.
        persist_dir: Optional explicit persistence directory override.

    Returns:
        int: Number of chunks in the collection index (0 if index does not exist).
    """
    cfg = config or load_config()
    target_dir = Path(persist_dir or cfg.get_indexes_path())
    if not target_dir.exists():
        return 0

    client = get_chroma_client(target_dir)
    try:
        col = client.get_collection(name=collection_name)
        return col.count()
    except Exception:
        return 0


def build_index(
    collection_name: str,
    config: Optional[AppConfig] = None,
    embedding_model: Optional[Any] = None,
    persist_dir: Optional[Path | str] = None,
    collection_dir: Optional[Path | str] = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Ingest, chunk, embed, and index an entire document collection.

    Rebuilds the Chroma collection from scratch (delete and recreate)
    to prevent duplicate chunks.

    Args:
        collection_name: Identifier of the document collection to build.
        config: Optional AppConfig instance. Loaded automatically if None.
        embedding_model: Optional preloaded or mock model with .encode() method.
        persist_dir: Optional explicit directory for ChromaDB storage.
        collection_dir: Optional explicit directory containing the source files.
                        If None, resolves collection_name via config.
        batch_size: Number of chunks to encode and insert in one batch.

    Returns:
        dict[str, Any]: Summary statistics of the build process:
            - collection: collection name
            - files_loaded: number of successful source documents loaded
            - files_failed: number of files that failed ingestion
            - chunks_created: number of text chunks created
            - indexed_records: total records stored in ChromaDB
            - time_seconds: duration of the build process
    """
    start_time = time.time()
    cfg = config or load_config()

    target_persist = Path(persist_dir or cfg.get_indexes_path()).resolve()
    target_persist.mkdir(parents=True, exist_ok=True)

    # 1. Ingest collection documents
    source_target = Path(collection_dir) if collection_dir else collection_name
    documents, failures = load_collection(source_target, cfg)

    # 2. Chunk documents
    chunks: list[Chunk] = chunk_documents(
        documents=documents,
        collection_name=collection_name,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    # 3. Connect to Chroma and recreate collection to avoid duplicate records
    client = get_chroma_client(target_persist)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    chroma_col = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Embed in batches and insert into Chroma
    if chunks:
        model = embedding_model or get_embedding_model(cfg.embedding_model)

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [
                {
                    "source_file": c.source_file,
                    "collection": c.collection,
                    "page": c.page if c.page is not None else -1,
                    "section": c.section or "",
                }
                for c in batch
            ]

            embeddings = model.encode(texts, convert_to_numpy=True).tolist()

            chroma_col.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

    elapsed = time.time() - start_time

    return {
        "collection": collection_name,
        "files_loaded": len(documents),
        "files_failed": len(failures),
        "chunks_created": len(chunks),
        "indexed_records": chroma_col.count(),
        "time_seconds": round(elapsed, 2),
    }
