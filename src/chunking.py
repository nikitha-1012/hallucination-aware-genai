"""Document chunking module.

Splits DocumentRecord objects into deterministic, overlapping Chunk objects
using configured chunk_size and chunk_overlap, respecting paragraph, sentence,
and word boundaries.
"""

from pathlib import Path
import re
from typing import Optional

from src.config import AppConfig
from src.models import Chunk, DocumentRecord


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split a string of text into overlapping chunks.

    Prefers splitting at:
    1. Paragraph boundaries (`\n\n`)
    2. Sentence boundaries (`. `, `! `, `? `, `.\n`)
    3. Word / whitespace boundaries (` `, `\n`)
    4. Hard character boundary if no separator exists in the search window.

    Args:
        text: Input text to split.
        chunk_size: Maximum character length for each chunk.
        chunk_overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        list[str]: Cleaned, non-empty chunks.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    text_len = len(cleaned)

    # Search window for finding a natural boundary: look back from end by up to
    # max_backtrack characters (proportional to overlap, but not more than half chunk_size)
    search_window_size = max(chunk_overlap, min(150, chunk_size // 2))

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end == text_len:
            chunk_str = cleaned[start:end].strip()
            if chunk_str:
                chunks.append(chunk_str)
            break

        # Look for the best split point in [end - search_window_size, end]
        window_start = max(start + 1, end - search_window_size)
        window = cleaned[window_start:end]

        split_point = -1

        # 1. Look for paragraph boundary (\n\n)
        para_idx = window.rfind("\n\n")
        if para_idx != -1:
            split_point = window_start + para_idx + 2
        else:
            # 2. Look for sentence boundary (. , ! , ? , .\n)
            sentence_matches = list(re.finditer(r"[.!?](?:\s|\n)", window))
            if sentence_matches:
                last_m = sentence_matches[-1]
                split_point = window_start + last_m.end()
            else:
                # 3. Look for whitespace / word boundary
                space_idx = max(window.rfind(" "), window.rfind("\n"))
                if space_idx != -1:
                    split_point = window_start + space_idx + 1

        # 4. Fallback: hard cut at end
        if split_point == -1 or split_point <= start:
            split_point = end

        chunk_str = cleaned[start:split_point].strip()
        if chunk_str:
            chunks.append(chunk_str)

        # Calculate next start position with overlap
        next_start = split_point - chunk_overlap
        # Guard against zero or negative forward progress
        if next_start <= start:
            next_start = split_point

        start = next_start

    return chunks


def chunk_documents(
    documents: list[list[DocumentRecord]] | list[DocumentRecord],
    collection_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Convert a collection of DocumentRecords into structured, metadata-preserving Chunks.

    Args:
        documents: Either a nested list of DocumentRecords (per file) as returned by
                   load_collection, or a flat list of DocumentRecords.
        collection_name: Identifier of the document collection.
        chunk_size: Maximum characters per chunk (default 1000).
        chunk_overlap: Overlap in characters between adjacent chunks (default 150).

    Returns:
        list[Chunk]: Deterministic Chunk objects with unique IDs and preserved metadata.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    # Flatten input if nested
    records: list[DocumentRecord] = []
    if documents and isinstance(documents[0], list):
        for doc_group in documents:
            records.extend(doc_group)
    else:
        records = documents  # type: ignore

    chunks: list[Chunk] = []

    # Track index per file/page/section unit to produce stable, unique chunk IDs
    doc_chunk_counters: dict[str, int] = {}

    for record in records:
        if not record.text or not record.text.strip():
            continue

        raw_chunks = split_text_into_chunks(
            text=record.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Base key for counter
        page_val = f"p{record.page}" if record.page is not None else "pNone"
        sec_val = f"s{record.section}" if record.section is not None else "sNone"
        scope_key = f"{collection_name}::{record.source_file}::{page_val}::{sec_val}"

        for text_chunk in raw_chunks:
            if not text_chunk:
                continue

            idx = doc_chunk_counters.get(scope_key, 0)
            doc_chunk_counters[scope_key] = idx + 1

            chunk_id = f"{scope_key}::{idx}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text_chunk,
                    source_file=record.source_file,
                    collection=collection_name,
                    page=record.page,
                    section=record.section,
                )
            )

    return chunks
