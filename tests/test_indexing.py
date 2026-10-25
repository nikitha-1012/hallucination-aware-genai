"""Tests for Part 3: F2 Indexing."""

from pathlib import Path
import pytest
import numpy as np

from src.config import AppConfig
from src.indexing import (
    build_index,
    get_chroma_client,
    index_chunk_count,
    index_exists,
)
from src.models import Chunk, DocumentRecord

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class MockEmbeddingModel:
    """Fast deterministic embedding model for isolated unit tests."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True):
        vectors = []
        for text in texts:
            # Deterministic pseudo-embedding from text hash
            seed = sum(ord(c) for c in text) % 1000
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            # Normalize for cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors)


@pytest.fixture
def test_config(tmp_path):
    """Create an AppConfig instance pointing data_dir to fixtures and indexes_dir to tmp_path."""
    return AppConfig(
        chunk_size=300,
        chunk_overlap=50,
        top_k=5,
        retry_top_k=8,
        strong_score=0.45,
        weak_score=0.30,
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama3.2:3b",
        temperature=0.1,
        llm_timeout_seconds=120,
        ollama_base_url="http://localhost:11434",
        data_dir="tests/fixtures",
        indexes_dir="indexes",
        default_collection="",
        root_dir=Path(__file__).resolve().parent.parent,
    )


def test_indexing_creates_collection_and_stores_chunks(tmp_path, test_config):
    """Indexing a fixture directory stores chunks, embeddings, and metadata."""
    mock_model = MockEmbeddingModel(dim=16)

    # Ingest from fixtures directory
    stats = build_index(
        collection_name="fixtures_test",
        config=test_config,
        embedding_model=mock_model,
        persist_dir=tmp_path / "indexes",
        collection_dir=FIXTURES_DIR,
    )

    assert stats["files_loaded"] >= 1
    assert stats["chunks_created"] > 0
    assert stats["indexed_records"] == stats["chunks_created"]

    # Verify index_exists and index_chunk_count
    assert index_exists("fixtures_test", test_config, persist_dir=tmp_path / "indexes")
    assert index_chunk_count("fixtures_test", test_config, persist_dir=tmp_path / "indexes") == stats["chunks_created"]

    # Verify records stored in ChromaDB
    client = get_chroma_client(tmp_path / "indexes")
    col = client.get_collection(name="fixtures_test")
    data = col.get()

    assert len(data["ids"]) == stats["chunks_created"]
    assert len(data["documents"]) == stats["chunks_created"]
    assert len(data["metadatas"]) == stats["chunks_created"]

    # Check metadata integrity
    first_meta = data["metadatas"][0]
    assert "source_file" in first_meta
    assert "collection" in first_meta
    assert "page" in first_meta
    assert "section" in first_meta
    assert first_meta["collection"] == "fixtures_test"


def test_rebuild_does_not_create_duplicates(tmp_path, test_config):
    """Rebuilding the same collection replaces records instead of appending duplicates."""
    mock_model = MockEmbeddingModel(dim=16)
    persist = tmp_path / "indexes"

    # First build
    stats1 = build_index(
        collection_name="rebuild_test",
        config=test_config,
        embedding_model=mock_model,
        persist_dir=persist,
        collection_dir=FIXTURES_DIR,
    )
    initial_count = stats1["indexed_records"]
    assert initial_count > 0

    # Second build (rebuild)
    stats2 = build_index(
        collection_name="rebuild_test",
        config=test_config,
        embedding_model=mock_model,
        persist_dir=persist,
        collection_dir=FIXTURES_DIR,
    )

    assert stats2["indexed_records"] == initial_count
    assert index_chunk_count("rebuild_test", test_config, persist_dir=persist) == initial_count


def test_index_exists_and_count_missing_collection(tmp_path, test_config):
    """Non-existent collections report False for existence and 0 for count."""
    persist = tmp_path / "indexes"
    assert not index_exists("nonexistent", test_config, persist_dir=persist)
    assert index_chunk_count("nonexistent", test_config, persist_dir=persist) == 0


def test_indexing_with_real_embedding_model(tmp_path, test_config):
    """End-to-end integration test with sentence-transformers and cosine distance."""
    from src.indexing import get_embedding_model

    persist = tmp_path / "real_index"
    real_model = get_embedding_model("all-MiniLM-L6-v2")

    stats = build_index(
        collection_name="real_test",
        config=test_config,
        embedding_model=real_model,
        persist_dir=persist,
        collection_dir=FIXTURES_DIR,
    )

    assert stats["indexed_records"] > 0
    client = get_chroma_client(persist)
    col = client.get_collection("real_test")
    # Verify metadata cosine space
    assert col.metadata.get("hnsw:space") == "cosine"
