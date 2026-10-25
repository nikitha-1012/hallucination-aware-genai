"""Tests for shared data models (Part 1)."""

from src.models import (
    Chunk,
    ClarificationDetail,
    DocumentRecord,
    IngestionFailure,
    PipelineOutcome,
    PipelineResult,
    RetrievedChunk,
)


def test_chunk_creation():
    """Chunk stores text, file, page, section, and stable chunk ID."""
    chunk = Chunk(
        chunk_id="python_fundamentals::intro.pdf::1::0",
        text="Python is an interpreted, high-level programming language.",
        source_file="intro.pdf",
        collection="python_fundamentals",
        page=1,
        section="Overview",
    )
    assert chunk.chunk_id == "python_fundamentals::intro.pdf::1::0"
    assert chunk.page == 1
    assert chunk.section == "Overview"
    assert "interpreted" in chunk.text


def test_retrieved_chunk_score():
    """RetrievedChunk inherits from Chunk and adds cosine similarity score."""
    rc = RetrievedChunk(
        chunk_id="python_fundamentals::intro.pdf::1::0",
        text="Python is interpreted.",
        source_file="intro.pdf",
        collection="python_fundamentals",
        page=1,
        section=None,
        score=0.88,
    )
    assert isinstance(rc, Chunk)
    assert rc.score == 0.88
    assert rc.source_file == "intro.pdf"


def test_pipeline_result_defaults():
    """PipelineResult defaults to empty lists and appropriate types."""
    result = PipelineResult(
        outcome=PipelineOutcome.ANSWER,
        message="Python was created by Guido van Rossum.",
        top_score=0.75,
    )
    assert result.outcome == PipelineOutcome.ANSWER
    assert result.outcome == "ANSWER"
    assert result.citations == []
    assert result.retrieved == []
    assert result.warnings == []
    assert result.retried is False
    assert result.latency_seconds == 0.0


def test_pipeline_outcomes_enumeration():
    """Verify all 5 required outcome states exist."""
    assert PipelineOutcome.ANSWER.value == "ANSWER"
    assert PipelineOutcome.PARTIAL.value == "PARTIAL"
    assert PipelineOutcome.CLARIFY.value == "CLARIFY"
    assert PipelineOutcome.ABSTAIN.value == "ABSTAIN"
    assert PipelineOutcome.ERROR.value == "ERROR"


def test_document_and_failure_records():
    """Verify raw document records and failure reporting records."""
    doc = DocumentRecord(source_file="test.txt", text="Hello world", page=None, section="intro")
    assert doc.text == "Hello world"

    fail = IngestionFailure(file_path="scanned.pdf", reason="no extractable text (possibly scanned)")
    assert "scanned" in fail.reason

    clarify = ClarificationDetail(needs_clarification=True, question="What does 'it' refer to?")
    assert clarify.needs_clarification is True
    assert clarify.question is not None
