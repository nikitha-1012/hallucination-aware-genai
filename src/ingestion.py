"""Document ingestion and text extraction module.

Discovers and extracts text from supported document types (PDF, TXT, Markdown),
cleans the extracted text, extracts page/section metadata, and reports any
failures without interrupting overall collection loading.
"""

import os
from pathlib import Path
import re
from typing import Optional

import pypdf

from src.config import AppConfig
from src.models import DocumentRecord, IngestionFailure

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def clean_text(text: str) -> str:
    """Clean extracted document text.

    - Joins words broken by line-end hyphens (e.g., 'multi-\nlingual' -> 'multilingual').
    - Normalizes horizontal whitespace (repeated spaces, tabs -> single space).
    - Removes repeated blank lines (caps at 1 blank line between blocks).
    - Preserves numbers, code, and punctuation.
    """
    if not text:
        return ""

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words broken by line-end hyphens: e.g. "com-\nputer" -> "computer"
    text = re.sub(r"(\b\w+)-\s*\n\s*(\w+\b)", r"\1\2", text)

    # Process line-by-line to normalize horizontal whitespace while preserving structure
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    cleaned_lines: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned_lines.append("")
                prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False

    return "\n".join(cleaned_lines).strip()


def _parse_pdf(file_path: Path) -> tuple[list[DocumentRecord], Optional[IngestionFailure]]:
    """Extract text page-by-page from a PDF using pypdf."""
    try:
        reader = pypdf.PdfReader(str(file_path))
    except Exception as e:
        return [], IngestionFailure(
            file_path=str(file_path),
            reason=f"Failed to open PDF file: {e}",
        )

    records: list[DocumentRecord] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        cleaned = clean_text(page_text)
        if cleaned:
            records.append(
                DocumentRecord(
                    source_file=file_path.name,
                    text=cleaned,
                    page=page_idx,
                    section=None,
                )
            )

    if not records:
        return [], IngestionFailure(
            file_path=str(file_path),
            reason="no extractable text (possibly scanned)",
        )

    return records, None


def _parse_txt(file_path: Path) -> tuple[list[DocumentRecord], Optional[IngestionFailure]]:
    """Extract text from a plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    except Exception as e:
        return [], IngestionFailure(
            file_path=str(file_path),
            reason=f"Failed to read TXT file: {e}",
        )

    cleaned = clean_text(raw_text)
    if not cleaned:
        return [], IngestionFailure(
            file_path=str(file_path),
            reason="empty file",
        )

    return [
        DocumentRecord(
            source_file=file_path.name,
            text=cleaned,
            page=None,
            section=None,
        )
    ], None


def _parse_markdown(file_path: Path) -> tuple[list[DocumentRecord], Optional[IngestionFailure]]:
    """Extract text from Markdown, splitting into records on heading boundaries."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    except Exception as e:
        return [], IngestionFailure(
            file_path=str(file_path),
            reason=f"Failed to read Markdown file: {e}",
        )

    cleaned_full = clean_text(raw_text)
    if not cleaned_full:
        return [], IngestionFailure(
            file_path=str(file_path),
            reason="empty file",
        )

    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[tuple[Optional[str], list[str]]] = []
    current_section: Optional[str] = None
    current_lines: list[str] = []

    heading_re = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        m = heading_re.match(line.strip())
        if m:
            if current_lines:
                sections.append((current_section, current_lines))
                current_lines = []
            current_section = m.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, current_lines))

    records: list[DocumentRecord] = []
    for sec_title, sec_lines in sections:
        sec_text = clean_text("\n".join(sec_lines))
        if sec_text:
            records.append(
                DocumentRecord(
                    source_file=file_path.name,
                    text=sec_text,
                    page=None,
                    section=sec_title,
                )
            )

    if not records:
        return [], IngestionFailure(
            file_path=str(file_path),
            reason="empty file",
        )

    return records, None


def load_file(file_path: Path | str) -> tuple[list[DocumentRecord], Optional[IngestionFailure]]:
    """Load and parse an individual source file.

    Args:
        file_path: Path to the target file.

    Returns:
        tuple[list[DocumentRecord], Optional[IngestionFailure]]:
            - Extracted document records (empty if file failed or unsupported).
            - An IngestionFailure if loading failed, else None.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return [], IngestionFailure(
            file_path=str(path),
            reason="file not found",
        )

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return [], IngestionFailure(
            file_path=str(path),
            reason=f"unsupported file format: '{ext}'",
        )

    if ext == ".pdf":
        return _parse_pdf(path)
    elif ext == ".txt":
        return _parse_txt(path)
    elif ext == ".md":
        return _parse_markdown(path)

    return [], IngestionFailure(file_path=str(path), reason=f"unsupported file format: '{ext}'")


def load_collection(
    collection_name_or_dir: str | Path,
    config: Optional[AppConfig] = None,
) -> tuple[list[list[DocumentRecord]], list[IngestionFailure]]:
    """Discover and ingest all supported documents in a collection directory.

    Args:
        collection_name_or_dir: Either a collection name (e.g., 'python_fundamentals')
                                resolved via config, or an explicit directory Path.
        config: Optional AppConfig instance. If None, loaded via load_config().

    Returns:
        tuple[list[list[DocumentRecord]], list[IngestionFailure]]:
            - A list of documents, where each document is a list of DocumentRecord (pages or sections).
            - A list of IngestionFailure objects for unreadable, empty, or unsupported files.
    """
    target_dir: Path
    if isinstance(collection_name_or_dir, Path):
        target_dir = collection_name_or_dir.resolve()
    else:
        candidate_path = Path(collection_name_or_dir)
        if candidate_path.is_dir():
            target_dir = candidate_path.resolve()
        else:
            if config is None:
                from src.config import load_config
                config = load_config()
            target_dir = config.get_collection_path(collection_name_or_dir)

    if not target_dir.exists() or not target_dir.is_dir():
        return [], [
            IngestionFailure(
                file_path=str(target_dir),
                reason=f"collection directory does not exist: {target_dir}",
            )
        ]

    documents: list[list[DocumentRecord]] = []
    failures: list[IngestionFailure] = []

    # Sort files deterministically
    entries = sorted(target_dir.iterdir(), key=lambda p: p.name.lower())

    for entry in entries:
        # Silently skip hidden files (e.g. .gitkeep, .DS_Store) and subdirectories
        if entry.name.startswith(".") or entry.is_dir():
            continue

        records, failure = load_file(entry)
        if failure:
            failures.append(failure)
        if records:
            documents.append(records)

    return documents, failures


def flatten_records(documents: list[list[DocumentRecord]]) -> list[DocumentRecord]:
    """Helper to flatten a list of documents into a single list of DocumentRecord."""
    return [record for doc in documents for record in doc]
