"""Tests for Part 5: F4 Grounded Generation and Ollama Client."""

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.config import AppConfig
from src.generation import (
    build_messages,
    format_citation,
    generate_answer,
    parse_answer,
)
from src.llm import (
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMTimeoutError,
    chat,
)
from src.models import PipelineOutcome, RetrievedChunk


# =====================================================================
# 1. Ollama Client Tests (src/llm.py)
# =====================================================================


def test_llm_chat_success():
    """Verify chat() successfully sends payload and parses message content."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": "Python is a programming language [1]."}
    }

    with patch("src.llm.requests.post", return_value=mock_resp) as mock_post:
        msgs = [{"role": "user", "content": "What is Python?"}]
        result = chat(
            messages=msgs,
            base_url="http://localhost:11434",
            model="llama3.2:3b",
            temperature=0.1,
            timeout=30,
        )

        assert result == "Python is a programming language [1]."
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "llama3.2:3b"
        assert call_kwargs["json"]["options"]["temperature"] == 0.1
        assert call_kwargs["timeout"] == 30


def test_llm_chat_connection_error():
    """Verify ConnectionError is converted to LLMConnectionError."""
    with patch("src.llm.requests.post", side_effect=requests.exceptions.ConnectionError("Refused")):
        with pytest.raises(LLMConnectionError, match="Ollama is not running"):
            chat(messages=[{"role": "user", "content": "Hi"}], base_url="http://localhost:11434")


def test_llm_chat_timeout_error():
    """Verify Timeout is converted to LLMTimeoutError."""
    with patch("src.llm.requests.post", side_effect=requests.exceptions.Timeout("Timed out")):
        with pytest.raises(LLMTimeoutError, match="The model took too long to respond"):
            chat(messages=[{"role": "user", "content": "Hi"}], timeout=5)


def test_llm_chat_model_not_found_404():
    """Verify HTTP 404 is converted to LLMModelNotFoundError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "model not found"

    with patch("src.llm.requests.post", return_value=mock_resp):
        with pytest.raises(LLMModelNotFoundError, match="Model 'nonexistent' not found"):
            chat(messages=[{"role": "user", "content": "Hi"}], model="nonexistent")


def test_llm_chat_http_500_error():
    """Verify other HTTP errors raise LLMError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    with patch("src.llm.requests.post", return_value=mock_resp):
        with pytest.raises(LLMError, match="Ollama returned HTTP error 500"):
            chat(messages=[{"role": "user", "content": "Hi"}])


def test_llm_chat_malformed_response():
    """Verify malformed JSON payload raises LLMError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"bad_key": 123}

    with patch("src.llm.requests.post", return_value=mock_resp):
        with pytest.raises(LLMError, match="Malformed response from Ollama"):
            chat(messages=[{"role": "user", "content": "Hi"}])


# =====================================================================
# 2. Grounded Generation Tests (src/generation.py)
# =====================================================================


@pytest.fixture
def sample_retrieved_chunks():
    return [
        RetrievedChunk(
            chunk_id="c1",
            text="Python was created in 1991 by Guido van Rossum.",
            source_file="python_history.txt",
            collection="python_fundamentals",
            page=None,
            section=None,
            score=0.85,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="The standard library includes sys, math, and os modules.",
            source_file="tutorial.pdf",
            collection="python_fundamentals",
            page=12,
            section="Standard Modules",
            score=0.78,
        ),
    ]


def test_build_messages_structure(sample_retrieved_chunks):
    """Verify prompt messages contain system rules, numbered passages, and the question."""
    messages = build_messages("What modules are in the standard library?", sample_retrieved_chunks)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "INSUFFICIENT_EVIDENCE" in messages[0]["content"]
    assert "untrusted" in messages[0]["content"].lower()

    user_content = messages[1]["content"]
    assert "[1] (Source: python_history.txt)" in user_content
    assert "[2] (Source: tutorial.pdf, Page 12, Section: Standard Modules)" in user_content
    assert "What modules are in the standard library?" in user_content


def test_format_citation():
    """Verify citation formatting with varying metadata presence."""
    c1 = RetrievedChunk(chunk_id="1", text="t", source_file="a.txt", collection="c", page=None, section=None)
    assert format_citation(c1) == "a.txt"

    c2 = RetrievedChunk(chunk_id="2", text="t", source_file="b.pdf", collection="c", page=5, section=None)
    assert format_citation(c2) == "b.pdf, page 5"

    c3 = RetrievedChunk(chunk_id="3", text="t", source_file="c.md", collection="c", page=None, section="Syntax")
    assert format_citation(c3) == "c.md, section 'Syntax'"

    c4 = RetrievedChunk(chunk_id="4", text="t", source_file="d.pdf", collection="c", page=3, section="Intro")
    assert format_citation(c4) == "d.pdf, page 3, section 'Intro'"


def test_parse_answer_success_with_citations(sample_retrieved_chunks):
    """Verify parser extracts citations and sets outcome to ANSWER."""
    raw = "The standard library includes math, sys, and os [2]. Python was made in 1991 [1]."
    result = parse_answer(raw, sample_retrieved_chunks)

    assert result.outcome == PipelineOutcome.ANSWER
    assert len(result.citations) == 2
    assert result.citations[0].chunk_id == "c2"
    assert result.citations[1].chunk_id == "c1"
    assert result.warnings == []
    assert result.not_covered is None


def test_parse_answer_insufficient_evidence(sample_retrieved_chunks):
    """Verify INSUFFICIENT_EVIDENCE triggers ABSTAIN outcome."""
    raw = "INSUFFICIENT_EVIDENCE"
    result = parse_answer(raw, sample_retrieved_chunks)

    assert result.outcome == PipelineOutcome.ABSTAIN
    assert len(result.citations) == 0
    assert "does not contain enough information" in result.message


def test_parse_answer_partial_answer(sample_retrieved_chunks):
    """Verify 'Not covered:' line triggers PARTIAL outcome and isolates missing details."""
    raw = (
        "Python was released in 1991 [1].\n"
        "Not covered: When Python 3 was released."
    )
    result = parse_answer(raw, sample_retrieved_chunks)

    assert result.outcome == PipelineOutcome.PARTIAL
    assert result.not_covered == "When Python 3 was released."
    assert "Not covered" not in result.message
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"


def test_parse_answer_invalid_citations_dropped(sample_retrieved_chunks):
    """Verify out-of-range citations like [99] are dropped and warned."""
    raw = "Python was created in 1991 [1] and updated in 2020 [99]."
    result = parse_answer(raw, sample_retrieved_chunks)

    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"
    assert any("Invalid citation numbers dropped" in w for w in result.warnings)


def test_parse_answer_no_citations_warned(sample_retrieved_chunks):
    """Answers without citations receive a warning."""
    raw = "Python is a dynamic programming language."
    result = parse_answer(raw, sample_retrieved_chunks)

    assert result.citations == []
    assert "No verifiable citations" in result.warnings


def test_generate_answer_empty_chunks():
    """Generating with empty chunks returns ABSTAIN immediately without calling LLM."""
    mock_chat = MagicMock()
    result = generate_answer("What is Python?", chunks=[], chat_fn=mock_chat)

    assert result.outcome == PipelineOutcome.ABSTAIN
    mock_chat.assert_not_called()


def test_generate_answer_retry_for_citations(sample_retrieved_chunks):
    """If initial answer has no citations, regenerate once with a reminder."""
    # First response lacks citations, second response includes citations
    call_counts = {"count": 0}

    def fake_chat(messages):
        call_counts["count"] += 1
        if call_counts["count"] == 1:
            return "Python was made in 1991."
        return "Python was created in 1991 by Guido van Rossum [1]."

    result = generate_answer(
        question="When was Python made?",
        chunks=sample_retrieved_chunks,
        chat_fn=fake_chat,
    )

    assert call_counts["count"] == 2
    assert result.outcome == PipelineOutcome.ANSWER
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"
