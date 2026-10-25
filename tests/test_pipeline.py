"""Unit tests for deterministic decision logic and pipeline orchestration.

Tests threshold boundary classification, query simplification, retry logic,
abstention, error handling, citation preservation, and latency recording
without requiring a live Ollama or Chroma service.
"""

from pathlib import Path
import pytest

from src.config import AppConfig
from src.decision import (
    EvidenceCategory,
    EvidenceDecision,
    assess_evidence,
    evaluate_evidence,
    needs_clarification,
    simplify_query,
)
from src.models import (
    GenerationResult,
    PipelineOutcome,
    PipelineResult,
    RetrievedChunk,
)
from src.pipeline import run_pipeline
from src.retrieval import EmptyIndexError, IndexNotFoundError, RetrievalError


@pytest.fixture
def mock_config():
    """Config with distinct, deterministic thresholds for testing."""
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


def make_chunk(chunk_id: str, score: float, text: str = "Evidence text.") -> RetrievedChunk:
    """Helper to construct RetrievedChunk with a specific similarity score."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_file="sample.txt",
        collection="python_fundamentals",
        page=1,
        section="Overview",
        score=score,
    )


# ---------------------------------------------------------------------------
# Decision & Threshold Tests
# ---------------------------------------------------------------------------


def test_decision_clearly_above_strong_score(mock_config):
    """Score > strong_score classifies as STRONG and should generate without retry."""
    chunks = [make_chunk("c1", 0.75), make_chunk("c2", 0.40)]
    decision = evaluate_evidence(chunks, is_retry=False, config=mock_config)

    assert decision.category == EvidenceCategory.STRONG
    assert decision.top_score == 0.75
    assert decision.should_generate is True
    assert decision.should_retry is False
    assert decision.should_abstain is False
    assert assess_evidence(chunks, config=mock_config) == "strong"


def test_decision_exactly_at_strong_score(mock_config):
    """Score == strong_score classifies as STRONG (boundary condition)."""
    chunks = [make_chunk("c1", 0.45)]
    decision = evaluate_evidence(chunks, is_retry=False, config=mock_config)

    assert decision.category == EvidenceCategory.STRONG
    assert decision.top_score == 0.45
    assert decision.should_generate is True
    assert decision.should_retry is False
    assert decision.should_abstain is False
    assert assess_evidence(chunks, config=mock_config) == "strong"


def test_decision_between_thresholds(mock_config):
    """weak_score <= Score < strong_score classifies as MODERATE and generates."""
    chunks = [make_chunk("c1", 0.38)]
    decision = evaluate_evidence(chunks, is_retry=False, config=mock_config)

    assert decision.category == EvidenceCategory.MODERATE
    assert decision.top_score == 0.38
    assert decision.should_generate is True
    assert decision.should_retry is False
    assert decision.should_abstain is False
    assert assess_evidence(chunks, config=mock_config) == "moderate"


def test_decision_exactly_at_weak_score(mock_config):
    """Score == weak_score classifies as MODERATE (boundary condition)."""
    chunks = [make_chunk("c1", 0.30)]
    decision = evaluate_evidence(chunks, is_retry=False, config=mock_config)

    assert decision.category == EvidenceCategory.MODERATE
    assert decision.top_score == 0.30
    assert decision.should_generate is True
    assert decision.should_retry is False
    assert decision.should_abstain is False
    assert assess_evidence(chunks, config=mock_config) == "moderate"


def test_decision_below_weak_score_initial(mock_config):
    """Score < weak_score on initial retrieval triggers retry, not immediate generation."""
    chunks = [make_chunk("c1", 0.25)]
    decision = evaluate_evidence(chunks, is_retry=False, config=mock_config)

    assert decision.category == EvidenceCategory.WEAK
    assert decision.top_score == 0.25
    assert decision.should_generate is False
    assert decision.should_retry is True
    assert decision.should_abstain is False
    assert assess_evidence(chunks, config=mock_config) == "weak"


def test_decision_below_weak_score_after_retry(mock_config):
    """Score < weak_score after retry leads to abstention."""
    chunks = [make_chunk("c1", 0.25)]
    decision = evaluate_evidence(chunks, is_retry=True, config=mock_config)

    assert decision.category == EvidenceCategory.WEAK
    assert decision.top_score == 0.25
    assert decision.should_generate is False
    assert decision.should_retry is False
    assert decision.should_abstain is True


def test_decision_empty_evidence_initial(mock_config):
    """Empty chunks list initially requests retry."""
    decision = evaluate_evidence([], is_retry=False, config=mock_config)

    assert decision.category == EvidenceCategory.NONE
    assert decision.top_score == 0.0
    assert decision.should_generate is False
    assert decision.should_retry is True
    assert decision.should_abstain is False
    assert assess_evidence([], config=mock_config) == "weak"


def test_decision_empty_evidence_after_retry(mock_config):
    """Empty chunks list after retry results in abstention."""
    decision = evaluate_evidence([], is_retry=True, config=mock_config)

    assert decision.category == EvidenceCategory.NONE
    assert decision.top_score == 0.0
    assert decision.should_generate is False
    assert decision.should_retry is False
    assert decision.should_abstain is True


def test_simplify_query():
    """Query simplification strips question words and stopwords, keeping content words."""
    q = "What is the difference between a list and a tuple in Python?"
    simplified = simplify_query(q)
    words = simplified.split()
    # Stopwords removed: what, is, the, a, and, in
    assert "what" not in words
    assert "is" not in words
    assert "the" not in words
    assert "difference" in words
    assert "list" in words
    assert "tuple" in words
    assert "python" in words


def test_simplify_query_fallback_on_all_stopwords():
    """If a question contains only stopwords, it falls back to the stripped original."""
    q = "What is it?"
    assert simplify_query(q) == "What is it?"


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------


def test_pipeline_strong_evidence_no_retry(mock_config):
    """Pipeline with strong evidence answers immediately without retry."""
    chunk = make_chunk("c1", 0.85)

    def mock_retrieve(collection_name, query, k, is_retry, config):
        assert is_retry is False
        assert k == mock_config.top_k
        return [chunk]

    def mock_generate(question, chunks, config):
        assert chunks == [chunk]
        return GenerationResult(
            message="Python lists are mutable sequences [1].",
            outcome=PipelineOutcome.ANSWER,
            citations=[chunk],
            warnings=[],
        )

    res = run_pipeline(
        collection="python_fundamentals",
        question="What is a list?",
        config=mock_config,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    assert res.outcome == PipelineOutcome.ANSWER
    assert "Python lists are mutable sequences" in res.message
    assert len(res.citations) == 1
    assert res.citations[0].chunk_id == "c1"
    assert res.top_score == 0.85
    assert res.retried is False
    assert res.latency_seconds >= 0.0


def test_pipeline_weak_evidence_retry_succeeds(mock_config):
    """Pipeline retries on initial weak score and generates if retry evidence is sufficient."""
    weak_chunk = make_chunk("c_weak", 0.20)
    strong_chunk = make_chunk("c_strong", 0.65)
    call_log = []

    def mock_retrieve(collection_name, query, k, is_retry, config):
        call_log.append((query, is_retry, k))
        if not is_retry:
            return [weak_chunk]
        else:
            return [strong_chunk]

    def mock_generate(question, chunks, config):
        assert chunks == [strong_chunk]
        return GenerationResult(
            message="Resolved after retry [1].",
            outcome=PipelineOutcome.ANSWER,
            citations=[strong_chunk],
            warnings=[],
        )

    res = run_pipeline(
        collection="python_fundamentals",
        question="Explain what a list is",
        config=mock_config,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    assert len(call_log) == 2
    assert call_log[0][1] is False  # initial is_retry
    assert call_log[1][1] is True   # retry is_retry
    assert call_log[1][2] == mock_config.retry_top_k
    assert res.outcome == PipelineOutcome.ANSWER
    assert res.retried is True
    assert res.top_score == 0.65


def test_pipeline_weak_evidence_retry_still_insufficient_abstains(mock_config):
    """Pipeline abstains without calling generation if evidence remains below weak_score after retry."""
    weak1 = make_chunk("c1", 0.15)
    weak2 = make_chunk("c2", 0.22)
    gen_called = []

    def mock_retrieve(collection_name, query, k, is_retry, config):
        return [weak2] if is_retry else [weak1]

    def mock_generate(question, chunks, config):
        gen_called.append(True)
        return GenerationResult(message="Should not be called", outcome=PipelineOutcome.ANSWER)

    res = run_pipeline(
        collection="python_fundamentals",
        question="How do you make pizza dough?",
        config=mock_config,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    assert len(gen_called) == 0  # LLM must NOT be called
    assert res.outcome == PipelineOutcome.ABSTAIN
    assert res.retried is True
    assert res.top_score == 0.22
    assert "does not contain enough information" in res.message
    assert "python_fundamentals" in res.message
    assert res.citations == []


def test_pipeline_no_evidence_abstains(mock_config):
    """Empty retrieval after retry causes abstention without calling LLM."""
    gen_called = []

    def mock_retrieve(collection_name, query, k, is_retry, config):
        return []

    def mock_generate(question, chunks, config):
        gen_called.append(True)
        return GenerationResult(message="Should not be called", outcome=PipelineOutcome.ANSWER)

    res = run_pipeline(
        collection="python_fundamentals",
        question="Nonexistent topic",
        config=mock_config,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    assert len(gen_called) == 0
    assert res.outcome == PipelineOutcome.ABSTAIN
    assert res.retried is True
    assert res.top_score == 0.0
    assert "does not contain enough information" in res.message


def test_pipeline_empty_question_returns_error(mock_config):
    """Empty or whitespace question produces ERROR result immediately."""
    res = run_pipeline(
        collection="python_fundamentals",
        question="   ",
        config=mock_config,
    )

    assert res.outcome == PipelineOutcome.ERROR
    assert "cannot be empty" in res.message.lower()
    assert res.latency_seconds >= 0.0


def test_pipeline_index_not_found_returns_error(mock_config):
    """IndexNotFoundError during retrieval produces clean ERROR result."""
    def mock_retrieve(collection_name, query, k, is_retry, config):
        raise IndexNotFoundError("Index does not exist")

    res = run_pipeline(
        collection="missing_collection",
        question="What is Python?",
        config=mock_config,
        retrieve_fn=mock_retrieve,
    )

    assert res.outcome == PipelineOutcome.ERROR
    assert "not found" in res.message.lower()
    assert "missing_collection" in res.message


def test_pipeline_empty_index_returns_error(mock_config):
    """EmptyIndexError during retrieval produces clean ERROR result."""
    def mock_retrieve(collection_name, query, k, is_retry, config):
        raise EmptyIndexError("Collection contains 0 chunks")

    res = run_pipeline(
        collection="empty_collection",
        question="What is Python?",
        config=mock_config,
        retrieve_fn=mock_retrieve,
    )

    assert res.outcome == PipelineOutcome.ERROR
    assert "is empty" in res.message.lower()


def test_pipeline_generation_failure_returns_error(mock_config):
    """Failure in LLM generation produces clean ERROR result without fabricating an answer."""
    chunk = make_chunk("c1", 0.80)

    def mock_retrieve(collection_name, query, k, is_retry, config):
        return [chunk]

    def mock_generate(question, chunks, config):
        raise RuntimeError("Ollama connection crashed")

    res = run_pipeline(
        collection="python_fundamentals",
        question="What is a list?",
        config=mock_config,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    assert res.outcome == PipelineOutcome.ERROR
    assert "Generation error" in res.message
    assert res.retrieved == [chunk]


def test_pipeline_partial_outcome_preserved(mock_config):
    """If generation detects partial coverage, PARTIAL outcome is preserved."""
    chunk = make_chunk("c1", 0.50)

    def mock_retrieve(collection_name, query, k, is_retry, config):
        return [chunk]

    def mock_generate(question, chunks, config):
        return GenerationResult(
            message="Lists are ordered sequences [1].",
            outcome=PipelineOutcome.PARTIAL,
            citations=[chunk],
            not_covered="Performance benchmarks are not addressed.",
        )

    res = run_pipeline(
        collection="python_fundamentals",
        question="What is a list and what is its Big-O performance?",
        config=mock_config,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    assert res.outcome == PipelineOutcome.PARTIAL
    assert "Lists are ordered sequences" in res.message
    assert len(res.citations) == 1


def test_pipeline_llm_abstain_marker_preserved(mock_config):
    """If LLM generation returns ABSTAIN (INSUFFICIENT_EVIDENCE), pipeline returns ABSTAIN."""
    chunk = make_chunk("c1", 0.50)

    def mock_retrieve(collection_name, query, k, is_retry, config):
        return [chunk]

    def mock_generate(question, chunks, config):
        return GenerationResult(
            message="The selected collection does not contain enough information.",
            outcome=PipelineOutcome.ABSTAIN,
            citations=[],
        )

    res = run_pipeline(
        collection="python_fundamentals",
        question="A question not covered by the chunk",
        config=mock_config,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    assert res.outcome == PipelineOutcome.ABSTAIN
    assert "does not contain enough information" in res.message
    assert res.citations == []


# ---------------------------------------------------------------------------
# Part 7 — Clarification Tests (F6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Explain it",
        "Explain it.",
        "Explain this",
        "Explain this.",
        "What is this?",
        "What about it?",
        "Tell me more about it",
        "Tell me about that.",
        "How does it work?",
        "Explain the error.",
        "What does it mean?",
        "Compare them.",
        "How do I fix it?",
        "Why does it happen?",
        "Which one is better?",
        "Explain the first one.",
        "Tell me about the previous one.",
        "Explain the issue.",
    ],
)
def test_needs_clarification_should_clarify_list(question):
    """Vague queries missing necessary details must trigger clarification."""
    detail = needs_clarification(question)
    assert detail.needs_clarification is True
    assert detail.question is not None
    assert len(detail.question.strip()) > 0


@pytest.mark.parametrize(
    "question",
    [
        "What is a Python list?",
        "What is a Python function?",
        "What is a variable in Python?",
        "How does a Python for loop work?",
        "How does exception handling work in Python?",
        "What is phishing?",
        "What is SQL injection?",
        "What does the len() function return?",
        "How does a firewall work?",
        "What is a dictionary in Python?",
        "What is the difference between a list and a tuple in Python?",
        "How do I fix the IndexError in my list?",
        "Why is it important to handle exceptions in Python?",
        "Does this function mutate a Python list?",
    ],
)
def test_needs_clarification_should_not_clarify_list(question):
    """Specific queries with concrete domain subjects must NOT trigger clarification."""
    detail = needs_clarification(question)
    assert detail.needs_clarification is False
    assert detail.question is None


def test_needs_clarification_single_word():
    """Single word queries trigger clarification requesting more detail."""
    detail = needs_clarification("Explain")
    assert detail.needs_clarification is True
    assert "too brief" in detail.question.lower()

    detail_topic = needs_clarification("Python")
    assert detail_topic.needs_clarification is True
    assert "python" in detail_topic.question.lower()


def test_needs_clarification_with_current_turn_in_history():
    """When history only contains the current user turn (as in Streamlit state), clarification still triggers."""
    history = [{"role": "user", "content": "Explain it"}]
    detail = needs_clarification("Explain it", history=history)
    assert detail.needs_clarification is True
    assert 'What does "it" refer to?' in detail.question


def test_needs_clarification_with_history_deferred():
    """When genuine prior conversation history exists, dangling references are deferred to context resolution."""
    history = [{"role": "user", "content": "What is a list in Python?"}]
    detail = needs_clarification("How does it work?", history=history)
    assert detail.needs_clarification is False


def test_pipeline_clarification_triggers_before_retrieval(mock_config):
    """Ambiguous query returns CLARIFY before calling retrieval or LLM generation."""
    retrieval_called = []
    generation_called = []

    def mock_retrieve(collection_name, query, k, is_retry, config):
        retrieval_called.append(query)
        return []

    def mock_generate(question, chunks, config):
        generation_called.append(question)
        return GenerationResult(message="Should not be called", outcome=PipelineOutcome.ANSWER)

    res = run_pipeline(
        collection="python_fundamentals",
        question="Explain this.",
        config=mock_config,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    # Retrieval and generation must NOT be called
    assert len(retrieval_called) == 0
    assert len(generation_called) == 0

    assert res.outcome == PipelineOutcome.CLARIFY
    assert "this" in res.message.lower()
    assert res.citations == []
    assert res.retrieved == []
    assert res.top_score == 0.0
    assert res.retried is False
    assert res.latency_seconds >= 0.0
    assert any("clarification" in w.lower() for w in res.warnings)


def test_pipeline_generic_error_clarifies_before_retrieval(mock_config):
    """Query referencing generic 'the error' triggers CLARIFY without retrieval."""
    retrieval_called = []

    def mock_retrieve(collection_name, query, k, is_retry, config):
        retrieval_called.append(query)
        return []

    res = run_pipeline(
        collection="python_fundamentals",
        question="Explain the error.",
        config=mock_config,
        retrieve_fn=mock_retrieve,
    )

    assert len(retrieval_called) == 0
    assert res.outcome == PipelineOutcome.CLARIFY
    assert "error" in res.message.lower()
    assert res.citations == []


def test_pipeline_specific_question_does_not_clarify(mock_config):
    """Specific question proceeds past clarification check into retrieval and generation."""
    chunk = make_chunk("c1", 0.85)
    retrieval_called = []
    generation_called = []

    def mock_retrieve(collection_name, query, k, is_retry, config):
        retrieval_called.append(query)
        return [chunk]

    def mock_generate(question, chunks, config):
        generation_called.append(question)
        return GenerationResult(
            message="A list is a mutable sequence [1].",
            outcome=PipelineOutcome.ANSWER,
            citations=[chunk],
        )

    res = run_pipeline(
        collection="python_fundamentals",
        question="What is a Python list?",
        config=mock_config,
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
    )

    assert len(retrieval_called) == 1
    assert len(generation_called) == 1
    assert res.outcome == PipelineOutcome.ANSWER
    assert len(res.citations) == 1

