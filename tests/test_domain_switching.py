"""Tests for Part 10: F8 Domain Switching and Independence.

Verifies that the second collection, 'cybersecurity_fundamentals', functions
end-to-end using the same ingestion, chunking, indexing, retrieval, and pipeline
components without any domain-specific logic in application code.
"""

from pathlib import Path
import re
from unittest.mock import MagicMock, patch
import pytest

from src.config import AppConfig, load_config
from src.chunking import chunk_documents
from src.indexing import index_chunk_count, index_exists
from src.ingestion import load_collection
from src.models import GenerationResult, PipelineOutcome
from src.pipeline import run_pipeline
from src.retrieval import retrieve
from src.ui_helpers import discover_collections


@pytest.fixture
def config():
    """Load application configuration."""
    return load_config()


def test_cybersecurity_documents_ingestion(config):
    """Verify that all markdown documents in the cybersecurity collection load cleanly."""
    collection_dir = config.get_collection_path("cybersecurity_fundamentals")
    assert collection_dir.exists(), f"Collection path {collection_dir} does not exist"

    docs, failures = load_collection("cybersecurity_fundamentals", config)
    assert len(failures) == 0, f"Expected 0 failures, got: {failures}"
    assert len(docs) >= 3, f"Expected at least 3 documents, got {len(docs)}"

    records = [rec for doc_group in docs for rec in doc_group]
    source_files = {rec.source_file for rec in records}
    expected_files = {
        "network_security.md",
        "application_security.md",
        "cryptography_and_access_control.md",
    }
    assert expected_files.issubset(source_files), f"Missing expected files: {expected_files - source_files}"

    # Verify sections were parsed from Markdown headers
    sections = {rec.section for rec in records if rec.section}
    assert "Firewalls" in sections
    assert "SQL Injection (SQLi)" in sections
    assert "Symmetric and Asymmetric Encryption" in sections


def test_cybersecurity_chunking(config):
    """Verify that cybersecurity documents are converted into valid Chunk records."""
    docs, _ = load_collection("cybersecurity_fundamentals", config)
    chunks = chunk_documents(
        docs,
        collection_name="cybersecurity_fundamentals",
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.collection == "cybersecurity_fundamentals"
        assert chunk.chunk_id.startswith("cybersecurity_fundamentals::")
        assert len(chunk.text.strip()) > 0
        assert chunk.source_file in {
            "network_security.md",
            "application_security.md",
            "cryptography_and_access_control.md",
        }


def test_dual_domain_indexing_coexistence():
    """Verify that both collections exist in the vector store and have indexed chunks."""
    assert index_exists("python_fundamentals")
    assert index_exists("cybersecurity_fundamentals")

    python_chunks = index_chunk_count("python_fundamentals")
    cyber_chunks = index_chunk_count("cybersecurity_fundamentals")

    assert python_chunks > 0, "python_fundamentals index should contain chunks"
    assert cyber_chunks >= 10, "cybersecurity_fundamentals index should contain at least 10 chunks"


def test_cybersecurity_retrieval_ranking(config):
    """Verify semantic retrieval correctly ranks cybersecurity topics from the index."""
    # 1. Query SQL Injection
    chunks_sqli = retrieve("cybersecurity_fundamentals", "How does SQL injection work and how is it prevented?", k=3, config=config)
    assert len(chunks_sqli) > 0
    top_sqli = chunks_sqli[0]
    assert top_sqli.source_file == "application_security.md"
    assert "sql injection" in top_sqli.text.lower() or "parameterized" in top_sqli.text.lower()
    assert top_sqli.score >= config.weak_score

    # 2. Query Firewalls
    chunks_firewall = retrieve("cybersecurity_fundamentals", "What is a firewall and how does packet filtering work?", k=3, config=config)
    assert len(chunks_firewall) > 0
    top_firewall = chunks_firewall[0]
    assert top_firewall.source_file == "network_security.md"
    assert "firewall" in top_firewall.text.lower()
    assert top_firewall.score >= config.weak_score

    # 3. Query Encryption
    chunks_crypto = retrieve("cybersecurity_fundamentals", "What is the difference between symmetric and asymmetric encryption?", k=3, config=config)
    assert len(chunks_crypto) > 0
    top_crypto = chunks_crypto[0]
    assert top_crypto.source_file == "cryptography_and_access_control.md"
    assert "symmetric" in top_crypto.text.lower()
    assert top_crypto.score >= config.weak_score


def test_cross_domain_abstention_python_on_cyber(config):
    """Verify asking a Python-specific question against the cybersecurity collection triggers abstention."""
    python_query = "What is the difference between a list and a tuple in Python?"
    chunks = retrieve("cybersecurity_fundamentals", python_query, k=config.top_k, config=config)

    # Retrieval scores should be low or not topically relevant
    if chunks:
        top_score = max(c.score for c in chunks)
        # Top score must be below strong threshold
        assert top_score < config.strong_score

    # Run full pipeline with mock LLM to ensure deterministic abstention
    with patch("src.pipeline.generate_answer") as mock_gen:
        result = run_pipeline("cybersecurity_fundamentals", python_query, config=config)
        # Pipeline must abstain due to lack of evidence in cybersecurity collection
        assert result.outcome == PipelineOutcome.ABSTAIN
        assert "cybersecurity_fundamentals" in result.message
        assert mock_gen.call_count == 0  # LLM generation should not be invoked


def test_cross_domain_abstention_cyber_on_python(config):
    """Verify asking a cybersecurity-specific question against the Python collection triggers abstention."""
    cyber_query = "What is Cross-Site Scripting XSS and how does Content Security Policy prevent it?"
    chunks = retrieve("python_fundamentals", cyber_query, k=config.top_k, config=config)

    if chunks:
        top_score = max(c.score for c in chunks)
        assert top_score < config.strong_score

    with patch("src.pipeline.generate_answer") as mock_gen:
        result = run_pipeline("python_fundamentals", cyber_query, config=config)
        assert result.outcome == PipelineOutcome.ABSTAIN
        assert "python_fundamentals" in result.message
        assert mock_gen.call_count == 0


def test_cybersecurity_pipeline_grounded_answer(config):
    """Verify that a cybersecurity question through the pipeline returns an ANSWER outcome with citations."""
    query = "How do firewalls protect a network?"

    def fake_generator(question, chunks, config):
        return GenerationResult(
            message="A firewall monitors and controls incoming and outgoing network traffic based on predetermined security rules [1].",
            outcome=PipelineOutcome.ANSWER,
            citations=[chunks[0]],
            warnings=[],
            raw_response="A firewall monitors and controls incoming and outgoing network traffic based on predetermined security rules [1].",
        )

    result = run_pipeline(
        "cybersecurity_fundamentals",
        query,
        generate_fn=fake_generator,
        config=config,
    )

    assert result.outcome == PipelineOutcome.ANSWER
    assert len(result.citations) > 0
    assert result.citations[0].source_file == "network_security.md"
    assert "[1]" in result.message


def test_codebase_domain_independence():
    """Verify that core pipeline, decision, and generation modules contain no hardcoded domain answers or rules."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    domain_keywords = [
        r"\bpython_basics\b",
        r"\bnetwork_security\b",
        r"\bapplication_security\b",
        r"\bcryptography_and_access_control\b",
        r"\bsqli\b",
        r"\bxss\b",
    ]

    for py_file in src_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        content = py_file.read_text(encoding="utf-8").lower()
        for kw in domain_keywords:
            matches = re.findall(kw, content)
            assert len(matches) == 0, (
                f"Domain-specific keyword '{kw}' found in generic source file {py_file.name}: {matches}"
            )


def test_ui_collection_discovery_contains_both_domains(config):
    """Verify UI collection discovery dynamically lists both collections."""
    collections = discover_collections(config.data_dir)
    assert "python_fundamentals" in collections
    assert "cybersecurity_fundamentals" in collections
    assert collections == sorted(collections)


@pytest.mark.parametrize("collection", ["python_fundamentals", "cybersecurity_fundamentals"])
def test_explain_it_returns_clarify_for_both_collections_without_history(collection, config):
    """Regression test: 'Explain it' must be intercepted BEFORE retrieval and return CLARIFY for both collections."""
    retrieval_called = []

    def mock_retrieve(collection_name, query, k, is_retry, config):
        retrieval_called.append((collection_name, query))
        return []

    result = run_pipeline(
        collection=collection,
        question="Explain it",
        retrieve_fn=mock_retrieve,
        config=config,
    )

    assert result.outcome == PipelineOutcome.CLARIFY
    assert 'What does "it" refer to?' in result.message
    assert len(retrieval_called) == 0, f"Retrieval must not be called for 'Explain it', but was called on {retrieval_called}"


@pytest.mark.parametrize("collection", ["python_fundamentals", "cybersecurity_fundamentals"])
def test_explain_it_returns_clarify_for_both_collections_with_prior_history(collection, config):
    """Regression test: 'Explain it' must return CLARIFY even when prior conversation turns exist in the session."""
    retrieval_called = []

    def mock_retrieve(collection_name, query, k, is_retry, config):
        retrieval_called.append((collection_name, query))
        return []

    history = [
        {"role": "user", "content": "What is SQL injection?" if collection == "cybersecurity_fundamentals" else "What is a list in Python?"},
        {"role": "assistant", "content": "Here is the factual grounded explanation..."},
        {"role": "user", "content": "Explain it"},
    ]

    result = run_pipeline(
        collection=collection,
        question="Explain it",
        history=history,
        retrieve_fn=mock_retrieve,
        config=config,
    )

    assert result.outcome == PipelineOutcome.CLARIFY
    assert 'What does "it" refer to?' in result.message
    assert len(retrieval_called) == 0, f"Retrieval must not be called for 'Explain it' with history, but was called on {retrieval_called}"

