"""Shared data models and interfaces across modules.

These data classes serve as the contract between ingestion, chunking,
indexing, retrieval, generation, decision, and the pipeline orchestrator.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PipelineOutcome(str, Enum):
    """Possible outcomes for the question answering pipeline."""
    ANSWER = "ANSWER"
    PARTIAL = "PARTIAL"
    CLARIFY = "CLARIFY"
    ABSTAIN = "ABSTAIN"
    ERROR = "ERROR"


@dataclass
class DocumentRecord:
    """Represents raw text extracted from a single page or section of a document."""
    source_file: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None


@dataclass
class IngestionFailure:
    """Records a failure to extract text from a file."""
    file_path: str
    reason: str


@dataclass
class Chunk:
    """Represents a discrete text chunk produced during document splitting."""
    chunk_id: str
    text: str
    source_file: str
    collection: str
    page: Optional[int] = None
    section: Optional[str] = None


@dataclass
class RetrievedChunk(Chunk):
    """A Chunk retrieved from vector storage with an associated cosine similarity score.

    Score is cosine similarity in range [-1.0, 1.0] (higher is better),
    computed as 1.0 minus Chroma cosine distance.
    """
    score: float = 0.0


@dataclass
class ClarificationDetail:
    """Result of rule-based clarification evaluation."""
    needs_clarification: bool
    question: Optional[str] = None


@dataclass
class PipelineResult:
    """Final result produced by the end-to-end question answering pipeline."""
    outcome: PipelineOutcome | str
    message: str
    citations: list[RetrievedChunk] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    top_score: float = 0.0
    retried: bool = False
    warnings: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0


@dataclass
class GenerationResult:
    """Result of grounded generation produced by generation.py."""
    message: str
    outcome: PipelineOutcome | str
    citations: list[RetrievedChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_response: str = ""
    not_covered: Optional[str] = None
