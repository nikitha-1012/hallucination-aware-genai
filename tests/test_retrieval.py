"""Tests for Part 4: F3 Semantic Retrieval."""

from pathlib import Path
import pytest
import numpy as np

from src.config import AppConfig
from src.indexing import build_index, get_chroma_client
from src.models import RetrievedChunk
from src.retrieval import (
    EmptyIndexError,
    IndexNotFoundError,
    distance_to_similarity,
    retrieve,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class MockDeterministicEmbeddingModel:
    """Mock model that returns controlled embeddings for deterministic distance testing."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True):
        vectors = []
        for text in texts:
            # Simple keyword-based embedding vector
            t = text.lower()
            if "python" in t or "standard library" in t:
                vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            elif "data structure" in t or "dictionary" in t:
                vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            elif "control flow" in t or "while" in t:
                vec = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            else:
                vec = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            vectors.append(vec)
        return np.array(vectors)


@pytest.fixture
def test_config(tmp_path):
    """Create an AppConfig instance for testing."""
    return AppConfig(
        chunk_size=300,
        chunk_overlap=50,
        top_k=2,
        retry_top_k=5,
        strong_score=0.45,
        weak_score=0.30,
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama3.2:3b",
        temperature=0.1,
        llm_timeout_seconds=120,
        ollama_base_url="http://localhost:11434",
        data_dir="tests/fixtures",
        indexes_dir="indexes",
        default_collection="test_col",
        root_dir=Path(__file__).resolve().parent.parent,
    )


@pytest.fixture
def mock_index_env(tmp_path, test_config):
    """Builds a temporary ChromaDB index with MockDeterministicEmbeddingModel."""
    persist_dir = tmp_path / "indexes"
    config = test_config

    mock_model = MockDeterministicEmbeddingModel()
    build_index(
        collection_name="test_col",
        config=config,
        embedding_model=mock_model,
        persist_dir=persist_dir,
        collection_dir=FIXTURES_DIR,
    )
    return config, persist_dir, mock_model


def test_distance_to_similarity_conversion():
    """Verify exact formula: cosine_similarity = 1.0 - cosine_distance."""
    assert distance_to_similarity(0.0) == 1.0
    assert distance_to_similarity(0.25) == 0.75
    assert distance_to_similarity(0.5) == 0.5
    assert distance_to_similarity(1.0) == 0.0
    assert distance_to_similarity(1.35) == -0.35
    assert distance_to_similarity(2.0) == -1.0
    # Clamping test
    assert distance_to_similarity(-0.1) == 1.0
    assert distance_to_similarity(2.5) == -1.0


def test_basic_retrieval_and_ranking(mock_index_env):
    """Querying retrieves relevant chunks sorted descending by similarity score."""
    config, persist_dir, mock_model = mock_index_env

    results = retrieve(
        collection_name="test_col",
        query="Tell me about Python standard library",
        config=config,
        embedding_model=mock_model,
        persist_dir=persist_dir,
    )

    assert len(results) > 0
    assert len(results) <= config.top_k
    assert isinstance(results[0], RetrievedChunk)

    # First result should have the highest similarity score
    assert results[0].score >= results[-1].score

    # Check that scores are properly descending
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_metadata_preservation_in_retrieval(mock_index_env):
    """RetrievedChunk preserves all source document metadata and stable IDs."""
    config, persist_dir, mock_model = mock_index_env

    results = retrieve(
        collection_name="test_col",
        query="data structure dictionary",
        k=10,
        config=config,
        embedding_model=mock_model,
        persist_dir=persist_dir,
    )

    assert len(results) > 0
    top = results[0]
    assert top.chunk_id != ""
    assert top.text != ""
    assert top.source_file in {"sample.md", "sample.txt", "sample.pdf"}
    assert top.collection == "test_col"
    assert isinstance(top.score, float)
    # page is int or None
    assert top.page is None or isinstance(top.page, int)
    # section is str or None
    assert top.section is None or isinstance(top.section, str)


def test_retry_top_k_retrieval(mock_index_env):
    """When is_retry=True, retrieval expands to config.retry_top_k."""
    config, persist_dir, mock_model = mock_index_env

    # Normal retrieval (top_k = 2)
    normal_results = retrieve(
        collection_name="test_col",
        query="Python",
        is_retry=False,
        config=config,
        embedding_model=mock_model,
        persist_dir=persist_dir,
    )
    assert len(normal_results) == config.top_k

    # Retry retrieval (retry_top_k = 5)
    retry_results = retrieve(
        collection_name="test_col",
        query="Python",
        is_retry=True,
        config=config,
        embedding_model=mock_model,
        persist_dir=persist_dir,
    )
    assert len(retry_results) > len(normal_results)
    assert len(retry_results) <= config.retry_top_k


def test_explicit_k_override(mock_index_env):
    """Explicit k parameter overrides both top_k and retry_top_k."""
    config, persist_dir, mock_model = mock_index_env

    results = retrieve(
        collection_name="test_col",
        query="Python",
        k=1,
        config=config,
        embedding_model=mock_model,
        persist_dir=persist_dir,
    )
    assert len(results) == 1


def test_empty_query_returns_empty_list(mock_index_env):
    """Empty or whitespace query returns empty list without error."""
    config, persist_dir, mock_model = mock_index_env

    assert retrieve("test_col", "", config=config, persist_dir=persist_dir) == []
    assert retrieve("test_col", "   ", config=config, persist_dir=persist_dir) == []


def test_nonexistent_collection_raises_error(mock_index_env):
    """Querying a non-existent collection raises IndexNotFoundError."""
    config, persist_dir, _ = mock_index_env

    with pytest.raises(IndexNotFoundError, match="Collection index 'unknown_col' not found"):
        retrieve(
            collection_name="unknown_col",
            query="Python",
            config=config,
            persist_dir=persist_dir,
        )


def test_empty_collection_raises_error(tmp_path, test_config):
    """Querying an empty index with 0 records raises EmptyIndexError."""
    persist_dir = tmp_path / "indexes"
    client = get_chroma_client(persist_dir)
    client.create_collection(name="empty_col")

    with pytest.raises(EmptyIndexError, match="Collection index 'empty_col' is empty"):
        retrieve(
            collection_name="empty_col",
            query="Python",
            config=test_config,
            persist_dir=persist_dir,
        )


def test_fewer_records_than_top_k(tmp_path, test_config):
    """When the index has fewer records than top_k, returns all records without crashing."""
    persist_dir = tmp_path / "indexes"
    client = get_chroma_client(persist_dir)
    col = client.create_collection(name="tiny_col", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=["id1"],
        documents=["Only one document in index."],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        metadatas=[{"source_file": "doc.txt", "collection": "tiny_col", "page": -1, "section": ""}],
    )

    mock_model = MockDeterministicEmbeddingModel()
    results = retrieve(
        collection_name="tiny_col",
        query="document",
        k=10,  # Request 10 when only 1 exists
        config=test_config,
        embedding_model=mock_model,
        persist_dir=persist_dir,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "id1"


def test_real_embedding_model_retrieval(tmp_path):
    """Integration test using the real sentence-transformers model against indexed fixtures."""
    from src.indexing import get_embedding_model

    persist_dir = tmp_path / "real_index"
    real_model = get_embedding_model("all-MiniLM-L6-v2")

    config = AppConfig(
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
        retry_top_k=5,
        strong_score=0.45,
        weak_score=0.30,
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama3.2:3b",
        temperature=0.1,
        llm_timeout_seconds=120,
        ollama_base_url="http://localhost:11434",
        data_dir="tests/fixtures",
        indexes_dir="indexes",
        default_collection="real_col",
        root_dir=Path(__file__).resolve().parent.parent,
    )

    build_index(
        collection_name="real_col",
        config=config,
        embedding_model=real_model,
        persist_dir=persist_dir,
        collection_dir=FIXTURES_DIR,
    )

    # Direct question that matches sample.pdf ("Python standard library includes math and sys modules")
    direct_results = retrieve(
        collection_name="real_col",
        query="What modules are in the Python standard library?",
        config=config,
        embedding_model=real_model,
        persist_dir=persist_dir,
    )
    assert len(direct_results) > 0
    top_chunk = direct_results[0]
    assert "sample.pdf" in top_chunk.source_file
    assert top_chunk.score > 0.40  # Strong semantic match

    # Unrelated question
    unrelated_results = retrieve(
        collection_name="real_col",
        query="How do you bake a triple chocolate cake with icing?",
        config=config,
        embedding_model=real_model,
        persist_dir=persist_dir,
    )
    assert len(unrelated_results) > 0
    # Unrelated match should have a significantly lower score than the direct match
    assert unrelated_results[0].score < top_chunk.score
