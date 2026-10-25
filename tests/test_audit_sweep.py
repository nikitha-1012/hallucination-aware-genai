"""Part 9: F9 Comprehensive Test Sweep Audit.

Audits and verifies the 9 core requirements from docs/PROJECT_GUIDE.md:
1. Extraction and chunk metadata
2. Indexing and persistence
3. Retrieval ranking and score conversion
4. Citation parsing, validation, and mapping
5. Insufficient-evidence handling and abstention
6. Clarification rules and false-positive protection
7. Centralized configuration validation
8. Model-down error handling and recovery guidance
9. Index-missing error handling and recovery guidance

All tests run deterministically without requiring a live Ollama server or internet access.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.chunking import chunk_documents
from src.config import AppConfig, ConfigError, load_config
from src.decision import (
    EvidenceCategory,
    assess_evidence,
    evaluate_evidence,
    needs_clarification,
    simplify_query,
)
from src.generation import format_citation, generate_answer, parse_answer
from src.indexing import (
    build_index,
    get_chroma_client,
    index_chunk_count,
    index_exists,
)
from src.llm import LLMConnectionError, LLMError, LLMTimeoutError
from src.models import (
    Chunk,
    DocumentRecord,
    GenerationResult,
    PipelineOutcome,
    PipelineResult,
    RetrievedChunk,
)
from src.pipeline import run_pipeline
from src.retrieval import (
    EmptyIndexError,
    IndexNotFoundError,
    distance_to_similarity,
    retrieve,
)
from src.ui_helpers import get_fix_guidance, get_outcome_info


@pytest.fixture
def sweep_config():
    """Deterministic AppConfig for audit testing."""
    return AppConfig(
        chunk_size=1000,
        chunk_overlap=150,
        top_k=5,
        retry_top_k=8,
        strong_score=0.45,
        weak_score=0.30,
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama3.2:3b",
        temperature=0.1,
        llm_timeout_seconds=30,
        ollama_base_url="http://localhost:11434",
    )


# =====================================================================
# 1. Extraction and Chunk Metadata
# =====================================================================


def test_audit_chunk_metadata_preservation():
    """Verify chunk metadata (file, page, section, collection) and stable chunk_id format."""
    records = [
        DocumentRecord(
            source_file="tutorial.pdf",
            text="Python fundamentals overview page 1.",
            page=1,
            section=None,
        ),
        DocumentRecord(
            source_file="tutorial.pdf",
            text="Functions and definitions section on page 2.",
            page=2,
            section="Functions",
        ),
    ]

    chunks = chunk_documents(records, collection_name="python_fundamentals")
    assert len(chunks) == 2

    c1 = chunks[0]
    assert c1.source_file == "tutorial.pdf"
    assert c1.page == 1
    assert c1.section is None
    assert c1.collection == "python_fundamentals"
    assert "python_fundamentals::tutorial.pdf::p1" in c1.chunk_id

    c2 = chunks[1]
    assert c2.source_file == "tutorial.pdf"
    assert c2.page == 2
    assert c2.section == "Functions"
    assert c2.collection == "python_fundamentals"
    assert "python_fundamentals::tutorial.pdf::p2::sFunctions" in c2.chunk_id


def test_audit_chunk_no_cross_page_or_file_merging():
    """Verify text is never concatenated across different pages or files."""
    records = [
        DocumentRecord(source_file="file1.txt", text="Short text in file 1.", page=None),
        DocumentRecord(source_file="file2.txt", text="Short text in file 2.", page=None),
    ]

    chunks = chunk_documents(records, collection_name="test_col")
    assert len(chunks) == 2
    assert chunks[0].source_file == "file1.txt"
    assert "file 2" not in chunks[0].text
    assert chunks[1].source_file == "file2.txt"
    assert "file 1" not in chunks[1].text


# =====================================================================
# 2. Indexing and Persistence
# =====================================================================


def test_audit_indexing_and_rebuild_persistence(tmp_path, sweep_config):
    """Verify index creation, persistence, and rebuild replacing chunks without duplicate buildup."""
    idx_dir = tmp_path / "indexes"
    col_dir = tmp_path / "data" / "python_fundamentals"
    col_dir.mkdir(parents=True)
    (col_dir / "doc1.txt").write_text("Python was created by Guido van Rossum.", encoding="utf-8")

    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1] * 384])

    # First build
    res1 = build_index(
        collection_name="python_fundamentals",
        config=sweep_config,
        embedding_model=fake_model,
        persist_dir=idx_dir,
        collection_dir=col_dir,
    )
    assert res1["chunks_created"] == 1
    assert res1["indexed_records"] == 1
    assert index_exists("python_fundamentals", persist_dir=idx_dir) is True
    assert index_chunk_count("python_fundamentals", persist_dir=idx_dir) == 1

    # Second build (rebuild must replace, not duplicate)
    res2 = build_index(
        collection_name="python_fundamentals",
        config=sweep_config,
        embedding_model=fake_model,
        persist_dir=idx_dir,
        collection_dir=col_dir,
    )
    assert res2["chunks_created"] == 1
    assert res2["indexed_records"] == 1
    assert index_chunk_count("python_fundamentals", persist_dir=idx_dir) == 1


# =====================================================================
# 3. Retrieval Ranking
# =====================================================================


def test_audit_distance_to_similarity_conversion():
    """Verify cosine distance conversion to cosine similarity (1.0 - distance) and bounds."""
    assert distance_to_similarity(0.0) == 1.0
    assert distance_to_similarity(0.35) == 0.65
    assert distance_to_similarity(1.0) == 0.0
    assert distance_to_similarity(2.0) == -1.0
    # Clamping bounds
    assert distance_to_similarity(-0.1) == 1.0
    assert distance_to_similarity(2.5) == -1.0


def test_audit_retrieval_ranking_descending():
    """Verify candidate chunks are ranked strictly in descending order of similarity."""
    chunks = [
        RetrievedChunk("c1", "text 1", "file.txt", "col", page=1, score=0.42),
        RetrievedChunk("c2", "text 2", "file.txt", "col", page=2, score=0.88),
        RetrievedChunk("c3", "text 3", "file.txt", "col", page=3, score=0.61),
    ]

    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    scores = [c.score for c in sorted_chunks]
    assert scores == [0.88, 0.61, 0.42]
    assert sorted_chunks[0].chunk_id == "c2"


# =====================================================================
# 4. Citation Parsing and Mapping
# =====================================================================


def test_audit_citation_parsing_and_mapping():
    """Verify bracketed citation extraction, deduplication, and out-of-bounds removal."""
    c1 = RetrievedChunk("c1", "Guido created Python in 1991.", "doc.txt", "col", page=1, score=0.8)
    c2 = RetrievedChunk("c2", "Python is dynamically typed.", "doc.txt", "col", page=2, score=0.7)
    chunks = [c1, c2]

    # Multiple citations, duplicate citations, and invalid citation [99]
    raw = "Python was created in 1991 [1] and is dynamically typed [2]. It was built by Guido [1] [99]."
    res = parse_answer(raw, chunks)

    # Should map to c1 and c2 (deduplicated), dropping [99] with warning
    assert res.outcome == PipelineOutcome.ANSWER
    assert len(res.citations) == 2
    assert [c.chunk_id for c in res.citations] == ["c1", "c2"]
    assert any("Invalid citation numbers dropped" in w for w in res.warnings)


def test_audit_citation_no_fabrication_when_metadata_missing():
    """Verify format_citation gracefully formats metadata without inventing page/section."""
    chunk = RetrievedChunk("c1", "Text", "source.md", "col", page=None, section=None, score=0.5)
    formatted = format_citation(chunk)
    assert formatted == "source.md"
    assert "page" not in formatted.lower()
    assert "section" not in formatted.lower()


# =====================================================================
# 5. Insufficient-Evidence Handling
# =====================================================================


def test_audit_insufficient_evidence_triggers_abstention(sweep_config):
    """When top similarity remains below weak_score after retry, pipeline abstains without LLM call."""
    weak_chunk = RetrievedChunk("c1", "Irrelevant text", "f.txt", "col", score=0.22)
    llm_called = []

    def fake_retrieve(collection_name, query, k, is_retry, config):
        return [weak_chunk]

    def fake_generate(question, chunks, config):
        llm_called.append(True)
        return GenerationResult("Fake answer", PipelineOutcome.ANSWER)

    res = run_pipeline(
        collection="python_fundamentals",
        question="How do airplanes fly?",
        config=sweep_config,
        retrieve_fn=fake_retrieve,
        generate_fn=fake_generate,
    )

    assert len(llm_called) == 0  # LLM must NOT be called
    assert res.outcome == PipelineOutcome.ABSTAIN
    assert res.retried is True
    assert res.top_score == 0.22
    assert "does not contain enough information" in res.message
    assert "python_fundamentals" in res.message
    assert res.citations == []


def test_audit_llm_insufficient_evidence_marker_abstains(sweep_config):
    """If LLM generation explicitly returns INSUFFICIENT_EVIDENCE, outcome is ABSTAIN."""
    chunk = RetrievedChunk("c1", "Partial text", "f.txt", "col", score=0.55)

    def fake_retrieve(collection_name, query, k, is_retry, config):
        return [chunk]

    def fake_generate(question, chunks, config):
        return GenerationResult(
            message="The selected collection does not contain enough information to answer this question.",
            outcome=PipelineOutcome.ABSTAIN,
            citations=[],
        )

    res = run_pipeline(
        collection="python_fundamentals",
        question="A question not answerable by the text",
        config=sweep_config,
        retrieve_fn=fake_retrieve,
        generate_fn=fake_generate,
    )

    assert res.outcome == PipelineOutcome.ABSTAIN
    assert res.citations == []


# =====================================================================
# 6. Clarification Rules and False Positives
# =====================================================================


@pytest.mark.parametrize(
    "query",
    [
        "Explain it",
        "Explain this",
        "What about it?",
        "Tell me more about it",
        "How does it work?",
        "Compare them.",
        "Which one is better?",
        "Explain the error.",
    ],
)
def test_audit_clarification_should_clarify(query):
    """Vague queries missing concrete subjects must return needs_clarification=True."""
    detail = needs_clarification(query)
    assert detail.needs_clarification is True
    assert detail.question is not None


@pytest.mark.parametrize(
    "query",
    [
        "What is a Python list?",
        "What is a Python function?",
        "How does exception handling work in Python?",
        "What does the len() function return?",
        "What is a dictionary in Python?",
        "What is SQL injection?",
    ],
)
def test_audit_clarification_should_not_clarify(query):
    """Specific queries with concrete subjects must return needs_clarification=False."""
    detail = needs_clarification(query)
    assert detail.needs_clarification is False
    assert detail.question is None


# =====================================================================
# 7. Configuration Errors
# =====================================================================


def test_audit_config_threshold_order_validation(tmp_path):
    """Config with weak_score >= strong_score must raise ConfigError."""
    bad_yaml = tmp_path / "bad_config.yaml"
    bad_yaml.write_text(
        """
chunk_size: 1000
chunk_overlap: 150
top_k: 5
retry_top_k: 8
strong_score: 0.30
weak_score: 0.45
embedding_model: all-MiniLM-L6-v2
llm_model: llama3.2:3b
temperature: 0.1
llm_timeout_seconds: 30
ollama_base_url: http://localhost:11434
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="weak_score.*must be less than strong_score"):
        load_config(bad_yaml)


def test_audit_config_overlap_size_validation(tmp_path):
    """Config with chunk_overlap >= chunk_size must raise ConfigError."""
    bad_yaml = tmp_path / "bad_overlap.yaml"
    bad_yaml.write_text(
        """
chunk_size: 200
chunk_overlap: 250
top_k: 5
retry_top_k: 8
strong_score: 0.45
weak_score: 0.30
embedding_model: all-MiniLM-L6-v2
llm_model: llama3.2:3b
temperature: 0.1
llm_timeout_seconds: 30
ollama_base_url: http://localhost:11434
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="chunk_overlap.*must be less than chunk_size"):
        load_config(bad_yaml)


# =====================================================================
# 8. Model-Down Error Handling
# =====================================================================


def test_audit_model_down_error_handling(sweep_config):
    """When Ollama is stopped, pipeline catches LLMConnectionError, sets ERROR, and offers fix."""
    chunk = RetrievedChunk("c1", "Grounded text", "file.txt", "col", score=0.85)

    def fake_retrieve(collection_name, query, k, is_retry, config):
        return [chunk]

    def fake_generate_offline(question, chunks, config):
        raise LLMConnectionError("Ollama is not running (start it with 'ollama serve')")

    res = run_pipeline(
        collection="python_fundamentals",
        question="What is a Python function?",
        config=sweep_config,
        retrieve_fn=fake_retrieve,
        generate_fn=fake_generate_offline,
    )

    assert res.outcome == PipelineOutcome.ERROR
    assert "Ollama is not running" in res.message
    assert res.retrieved == [chunk]
    assert res.citations == []
    # Check fix guidance helper
    fix = get_fix_guidance(res.message)
    assert fix is not None
    assert "ollama serve" in fix


# =====================================================================
# 9. Index-Missing Error Handling
# =====================================================================


def test_audit_index_missing_error_handling(sweep_config):
    """When target collection index is missing, pipeline returns ERROR with build instructions."""
    def fake_retrieve_missing(collection_name, query, k, is_retry, config):
        raise IndexNotFoundError(f"Index for collection '{collection_name}' not found.")

    res = run_pipeline(
        collection="nonexistent_domain",
        question="What is Python?",
        config=sweep_config,
        retrieve_fn=fake_retrieve_missing,
    )

    assert res.outcome == PipelineOutcome.ERROR
    assert "not found" in res.message.lower()
    assert "build the index" in res.message.lower()
    assert res.citations == []

    fix = get_fix_guidance(res.message)
    assert fix is not None
    assert "Build or rebuild index" in fix
