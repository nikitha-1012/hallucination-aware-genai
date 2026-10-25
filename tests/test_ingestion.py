"""Tests for Part 2: F1 Document Ingestion."""

from pathlib import Path
import pytest

from src.ingestion import (
    clean_text,
    load_collection,
    load_file,
    flatten_records,
)
from src.models import DocumentRecord, IngestionFailure

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_clean_text():
    """Verify clean_text normalizes whitespace, rejoins hyphens, and preserves punctuation."""
    raw = "This is a multi-\nlingual   system.\n\n\n\nCode: x = 42; y = [1, 2, 3]."
    cleaned = clean_text(raw)

    assert "multilingual" in cleaned
    assert "read-\nability" not in cleaned
    assert "x = 42; y = [1, 2, 3]." in cleaned
    assert "\n\n\n" not in cleaned
    assert "   " not in cleaned


def test_load_txt_file():
    """Verify TXT ingestion extracts clean text with source_file metadata."""
    txt_path = FIXTURES_DIR / "sample.txt"
    records, failure = load_file(txt_path)

    assert failure is None
    assert len(records) == 1
    rec = records[0]
    assert rec.source_file == "sample.txt"
    assert rec.page is None
    assert rec.section is None
    assert "readability" in rec.text
    assert "1991" in rec.text


def test_load_markdown_file():
    """Verify Markdown ingestion splits on headings and preserves section titles."""
    md_path = FIXTURES_DIR / "sample.md"
    records, failure = load_file(md_path)

    assert failure is None
    assert len(records) == 3

    sections = [r.section for r in records]
    assert "Python Overview" in sections
    assert "Data Structures" in sections
    assert "Control Flow" in sections

    for r in records:
        assert r.source_file == "sample.md"
        assert r.page is None
        assert len(r.text) > 0


def test_load_pdf_file():
    """Verify PDF ingestion extracts text with 1-indexed page metadata."""
    pdf_path = FIXTURES_DIR / "sample.pdf"
    records, failure = load_file(pdf_path)

    assert failure is None
    assert len(records) == 1
    rec = records[0]
    assert rec.source_file == "sample.pdf"
    assert rec.page == 1
    assert rec.section is None
    assert "standard library" in rec.text


def test_load_empty_file():
    """Empty files must be reported as IngestionFailure."""
    empty_path = FIXTURES_DIR / "empty.txt"
    records, failure = load_file(empty_path)

    assert len(records) == 0
    assert isinstance(failure, IngestionFailure)
    assert "empty file" in failure.reason.lower()


def test_load_scanned_pdf():
    """PDFs with no extractable text must be reported as possibly scanned."""
    scanned_path = FIXTURES_DIR / "scanned_empty.pdf"
    records, failure = load_file(scanned_path)

    assert len(records) == 0
    assert isinstance(failure, IngestionFailure)
    assert "scanned" in failure.reason.lower()


def test_load_unsupported_file():
    """Unsupported file formats must be reported as IngestionFailure."""
    bin_path = FIXTURES_DIR / "unsupported.bin"
    records, failure = load_file(bin_path)

    assert len(records) == 0
    assert isinstance(failure, IngestionFailure)
    assert "unsupported file format" in failure.reason.lower()


def test_load_collection_mixed_folder():
    """Mixed folder must process valid files and report failures without crashing."""
    docs, failures = load_collection(FIXTURES_DIR)

    # Valid files loaded: sample.txt, sample.md, sample.pdf -> 3 document groups
    assert len(docs) == 3

    flat = flatten_records(docs)
    source_names = {r.source_file for r in flat}
    assert "sample.txt" in source_names
    assert "sample.md" in source_names
    assert "sample.pdf" in source_names

    # Failed files reported: empty.txt, scanned_empty.pdf, unsupported.bin
    failed_names = {Path(f.file_path).name for f in failures}
    assert "empty.txt" in failed_names
    assert "scanned_empty.pdf" in failed_names
    assert "unsupported.bin" in failed_names


def test_load_collection_nonexistent_directory(tmp_path):
    """Nonexistent collection directory must return failure gracefully without crash."""
    nonexistent = tmp_path / "does_not_exist"
    docs, failures = load_collection(nonexistent)

    assert len(docs) == 0
    assert len(failures) == 1
    assert "does not exist" in failures[0].reason
