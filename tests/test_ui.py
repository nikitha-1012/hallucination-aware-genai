"""Unit tests for UI helper functions and presentation logic.

Tests collection discovery, outcome label formatting, chunk citation rendering,
and actionable fix guidance without requiring an active browser or Streamlit server.
"""

from pathlib import Path
import pytest

from src.models import PipelineOutcome, RetrievedChunk
from src.ui_helpers import (
    discover_collections,
    format_chunk_citation_header,
    get_fix_guidance,
    get_outcome_info,
)


def test_discover_collections_existing_directory(tmp_path):
    """Subdirectories under data_dir are discovered and sorted."""
    (tmp_path / "python_fundamentals").mkdir()
    (tmp_path / "cybersecurity_fundamentals").mkdir()
    (tmp_path / "notes.txt").write_text("not a folder")

    collections = discover_collections(tmp_path)
    assert collections == ["cybersecurity_fundamentals", "python_fundamentals"]


def test_discover_collections_nonexistent_directory():
    """Non-existent directory returns empty list without error."""
    collections = discover_collections(Path("non/existent/path/for/tests"))
    assert collections == []


def test_get_outcome_info_all_outcomes():
    """All 5 project outcomes map to distinct labels, icons, and variants."""
    expected = {
        PipelineOutcome.ANSWER: ("Answered", "success"),
        PipelineOutcome.PARTIAL: ("Partial", "info"),
        PipelineOutcome.CLARIFY: ("Needs clarification", "warning"),
        PipelineOutcome.ABSTAIN: ("Not enough evidence", "warning"),
        PipelineOutcome.ERROR: ("Error", "error"),
    }

    for outcome, (expected_label, expected_variant) in expected.items():
        info = get_outcome_info(outcome)
        assert info["label"] == expected_label
        assert info["variant"] == expected_variant
        assert "icon" in info
        assert len(info["icon"]) > 0


def test_get_outcome_info_string_fallback():
    """String outcome representation matches enum or returns fallback."""
    info = get_outcome_info("ANSWER")
    assert info["label"] == "Answered"

    unknown_info = get_outcome_info("CUSTOM_STATUS")
    assert unknown_info["label"] == "CUSTOM_STATUS"
    assert unknown_info["variant"] == "info"


def test_format_chunk_citation_header_complete():
    """Header formats source file, page, section, and similarity score."""
    chunk = RetrievedChunk(
        chunk_id="test::1",
        text="Sample text",
        source_file="tutorial.pdf",
        collection="python_fundamentals",
        page=42,
        section="Functions",
        score=0.87654,
    )
    header = format_chunk_citation_header(chunk)
    assert "tutorial.pdf" in header
    assert "42" in header
    assert "Functions" in header
    assert "0.8765" in header


def test_format_chunk_citation_header_missing_optional_fields():
    """Header omits missing page and section without fabricating values."""
    chunk = RetrievedChunk(
        chunk_id="test::2",
        text="Sample text",
        source_file="notes.txt",
        collection="python_fundamentals",
        page=None,
        section=None,
        score=0.4500,
    )
    header = format_chunk_citation_header(chunk)
    assert "notes.txt" in header
    assert "Page:" not in header
    assert "Section:" not in header
    assert "0.4500" in header


def test_get_fix_guidance_known_errors():
    """Troubleshooting guidance provides actionable commands for known failures."""
    ollama_down = "Failed to establish a new connection: [WinError 10061] Connection refused"
    assert "ollama serve" in get_fix_guidance(ollama_down)

    model_missing = "Model 'llama3.2:3b' not found: HTTP 404"
    assert "ollama pull" in get_fix_guidance(model_missing)

    timeout = "The model took too long to respond (timeout)"
    assert "llm_timeout_seconds" in get_fix_guidance(timeout)

    index_missing = "Index for collection 'python' not found"
    assert "Build or rebuild index" in get_fix_guidance(index_missing)

    empty_col = "Collection 'python' is empty. No documents indexed."
    assert "Add document files" in get_fix_guidance(empty_col)


def test_get_fix_guidance_unknown_error():
    """Unrecognized errors return None rather than misleading advice."""
    assert get_fix_guidance("Unrelated arithmetic division error") is None
