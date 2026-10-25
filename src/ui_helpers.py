"""Presentation and utility helpers for the Streamlit user interface.

Contains deterministic functions for collection discovery, outcome badge formatting,
chunk citation rendering, and error troubleshooting suggestions.
"""

from pathlib import Path
from typing import Any, Optional

from src.models import PipelineOutcome, RetrievedChunk

# Mapping from pipeline outcome to display attributes
OUTCOME_DISPLAY_MAP: dict[str, dict[str, str]] = {
    PipelineOutcome.ANSWER.value: {
        "label": "Answered",
        "tag": "ANSWER",
        "description": "Answer grounded in retrieved documentation with source citations.",
        "icon": "✅",
        "variant": "success",
    },
    PipelineOutcome.PARTIAL.value: {
        "label": "Partial",
        "tag": "PARTIAL",
        "description": "Answer covers only part of the question; remaining aspects were not found.",
        "icon": "⚠️",
        "variant": "info",
    },
    PipelineOutcome.CLARIFY.value: {
        "label": "Needs clarification",
        "tag": "CLARIFY",
        "description": "Question is ambiguous or missing necessary details.",
        "icon": "❓",
        "variant": "warning",
    },
    PipelineOutcome.ABSTAIN.value: {
        "label": "Not enough evidence",
        "tag": "ABSTAIN",
        "description": "Retrieved evidence similarity is below confidence thresholds.",
        "icon": "🛑",
        "variant": "warning",
    },
    PipelineOutcome.ERROR.value: {
        "label": "Error",
        "tag": "ERROR",
        "description": "An error occurred during retrieval, indexing, or generation.",
        "icon": "❌",
        "variant": "error",
    },
}


def discover_collections(data_dir: Path | str) -> list[str]:
    """Discover available collection subdirectories within the data directory.

    Args:
        data_dir: Path to directory containing collection folders.

    Returns:
        list[str]: Sorted list of collection names.
    """
    path = Path(data_dir).resolve()
    if not path.exists() or not path.is_dir():
        return []
    return sorted([d.name for d in path.iterdir() if d.is_dir()])


def get_outcome_info(outcome: PipelineOutcome | str) -> dict[str, str]:
    """Map a PipelineOutcome to display attributes (label, icon, variant).

    Args:
        outcome: PipelineOutcome instance or string.

    Returns:
        dict[str, str]: Dictionary containing label, tag, icon, description, variant.
    """
    key = outcome.value if isinstance(outcome, PipelineOutcome) else str(outcome)
    return OUTCOME_DISPLAY_MAP.get(
        key,
        {
            "label": str(key),
            "tag": str(key),
            "description": "",
            "icon": "ℹ️",
            "variant": "info",
        },
    )


def format_chunk_citation_header(chunk: RetrievedChunk) -> str:
    """Format human-readable header for a retrieved chunk citation.

    Args:
        chunk: RetrievedChunk instance with metadata.

    Returns:
        str: Clean markdown string summarizing source file, page/section, and score.
    """
    parts = [f"Source: **{chunk.source_file}**"]
    if chunk.page is not None:
        parts.append(f"Page: **{chunk.page}**")
    if chunk.section:
        parts.append(f"Section: **{chunk.section}**")
    parts.append(f"Similarity Score: **{chunk.score:.4f}**")
    return " | ".join(parts)


def get_fix_guidance(error_message: str) -> Optional[str]:
    """Provide actionable troubleshooting guidance based on error message patterns.

    Args:
        error_message: Raw error message string.

    Returns:
        Optional[str]: Actionable fix command or advice, or None if unknown.
    """
    lower = error_message.lower()
    if (
        "ollama is not running" in lower
        or "connection refused" in lower
        or "failed to establish a new connection" in lower
    ):
        return "Fix: Start the local Ollama service by running `ollama serve` in a terminal."
    if "model not found" in lower or "404" in lower:
        return "Fix: Download the configured model using `ollama pull <model_name>`."
    if "took too long" in lower or "timeout" in lower:
        return (
            "Fix: The model timed out. Try increasing `llm_timeout_seconds` in `config.yaml` "
            "or using a smaller model."
        )
    if "index" in lower and ("not found" in lower or "does not exist" in lower):
        return "Fix: Build the collection index using the 'Build or rebuild index' button in the sidebar."
    if "empty" in lower and ("collection" in lower or "index" in lower):
        return "Fix: Add document files (PDF, TXT, or MD) to the collection directory and click 'Build or rebuild index'."
    return None
