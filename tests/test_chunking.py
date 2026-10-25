"""Tests for Part 3: F2 Chunking."""

import pytest

from src.chunking import chunk_documents, split_text_into_chunks
from src.models import Chunk, DocumentRecord


def test_short_document_single_chunk():
    """A document shorter than chunk_size produces exactly one chunk."""
    doc = DocumentRecord(
        source_file="intro.txt",
        text="Python is a dynamically typed programming language.",
        page=None,
        section="Overview",
    )
    chunks = chunk_documents([doc], collection_name="python_fundamentals", chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].text == "Python is a dynamically typed programming language."
    assert chunks[0].source_file == "intro.txt"
    assert chunks[0].collection == "python_fundamentals"
    assert chunks[0].section == "Overview"
    assert chunks[0].page is None
    assert "intro.txt" in chunks[0].chunk_id


def test_long_document_multiple_chunks():
    """A document longer than chunk_size produces multiple overlapping chunks."""
    paragraphs = [
        f"Paragraph {i}: This is detailed content about Python design philosophy and features."
        for i in range(1, 20)
    ]
    long_text = "\n\n".join(paragraphs)

    doc = DocumentRecord(
        source_file="guide.txt",
        text=long_text,
        page=2,
        section="Details",
    )
    chunk_size = 200
    chunk_overlap = 50
    chunks = chunk_documents([doc], collection_name="python_fundamentals", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= chunk_size + 50  # boundaries allow small leeway around natural breaks
        assert len(chunk.text.strip()) > 0
        assert chunk.source_file == "guide.txt"
        assert chunk.page == 2
        assert chunk.section == "Details"


def test_no_empty_chunks():
    """Chunking text with irregular whitespace does not emit empty chunks."""
    text = "   \n\n   First sentence.   \n\n\n\n  Second sentence.   \n\n  "
    chunks = split_text_into_chunks(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.strip()) > 0


def test_unique_deterministic_chunk_ids():
    """Chunks receive deterministic and unique IDs within and across documents."""
    doc1 = DocumentRecord(source_file="a.txt", text="Alpha content. " * 30, page=1)
    doc2 = DocumentRecord(source_file="b.txt", text="Beta content. " * 30, page=1)

    chunks = chunk_documents([doc1, doc2], collection_name="test_col", chunk_size=150, chunk_overlap=30)
    chunk_ids = [c.chunk_id for c in chunks]

    # All IDs must be unique
    assert len(chunk_ids) == len(set(chunk_ids))

    # Re-running chunking produces identical chunk IDs (deterministic)
    chunks_repeat = chunk_documents([doc1, doc2], collection_name="test_col", chunk_size=150, chunk_overlap=30)
    assert [c.chunk_id for c in chunks_repeat] == chunk_ids


def test_multiple_documents_remain_distinguishable():
    """Chunks retain distinct source_file and page metadata across multiple documents."""
    doc_page1 = DocumentRecord(source_file="book.pdf", text="Page one text.", page=1)
    doc_page2 = DocumentRecord(source_file="book.pdf", text="Page two text.", page=2)

    chunks = chunk_documents([doc_page1, doc_page2], collection_name="test_col", chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 2
    assert chunks[0].source_file == "book.pdf"
    assert chunks[0].page == 1
    assert chunks[1].source_file == "book.pdf"
    assert chunks[1].page == 2
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_invalid_chunk_overlap_raises():
    """chunk_overlap >= chunk_size must raise ValueError."""
    doc = DocumentRecord(source_file="test.txt", text="Sample text.")
    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly less than chunk_size"):
        chunk_documents([doc], collection_name="col", chunk_size=100, chunk_overlap=100)
