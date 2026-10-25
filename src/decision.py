"""Deterministic evidence assessment and query simplification decision module.

Classifies retrieved evidence against configured strong and weak similarity
thresholds to determine whether evidence is sufficient for generation, requires
retry retrieval with an expanded query, or warrants explicit abstention.
"""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional

from src.config import AppConfig, load_config
from src.models import RetrievedChunk

# Common question words, auxiliary verbs, and stopwords for query simplification
STOPWORDS = {
    "a", "an", "the", "what", "how", "is", "are", "was", "were", "does", "do",
    "did", "explain", "why", "which", "who", "whom", "where", "when", "can",
    "could", "would", "should", "tell", "me", "about", "of", "in", "to", "for",
    "with", "on", "at", "from", "by", "and", "or", "be", "been", "being", "have",
    "has", "had", "please", "give", "show", "i", "you", "my", "your", "it", "its"
}


class EvidenceCategory(str, Enum):
    """Categorization of retrieved evidence quality."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


@dataclass
class EvidenceDecision:
    """Structured decision output evaluating evidence sufficiency."""
    category: EvidenceCategory
    top_score: float
    should_retry: bool
    should_generate: bool
    should_abstain: bool
    reason: str


def simplify_query(query: str) -> str:
    """Simplify query by removing question words and stopwords, retaining content words.

    Args:
        query: Original user question.

    Returns:
        str: Simplified keyword query string for fallback retry retrieval.
    """
    clean_query = query.strip().lower()
    # Replace non-alphanumeric characters (except whitespace) with space
    words = re.findall(r"\b[a-zA-Z0-9_]+\b", clean_query)
    content_words = [w for w in words if w not in STOPWORDS]

    if not content_words:
        # Fallback to stripped original if removing stopwords leaves nothing
        return query.strip()

    return " ".join(content_words)


def evaluate_evidence(
    chunks: list[RetrievedChunk],
    is_retry: bool = False,
    config: Optional[AppConfig] = None,
) -> EvidenceDecision:
    """Deterministically evaluate evidence chunks against configured similarity thresholds.

    Threshold logic:
    - No chunks or top score <= 0.0:
      Category NONE. Retry if not yet retried, else abstain.
    - Top score < weak_score:
      Category WEAK. Retry if not yet retried, else abstain.
    - weak_score <= top_score < strong_score:
      Category MODERATE. Sufficient to generate with strict grounding instructions.
    - top_score >= strong_score:
      Category STRONG. Sufficient to generate.

    Args:
        chunks: List of RetrievedChunk objects from retrieval.
        is_retry: True if this evaluation occurs after a retry retrieval attempt.
        config: Optional AppConfig instance (loads default if None).

    Returns:
        EvidenceDecision: Structured evaluation detailing next pipeline action.
    """
    cfg = config or load_config()

    if not chunks:
        return EvidenceDecision(
            category=EvidenceCategory.NONE,
            top_score=0.0,
            should_retry=not is_retry,
            should_generate=False,
            should_abstain=is_retry,
            reason="No evidence chunks retrieved.",
        )

    top_score = max(chunk.score for chunk in chunks)

    if top_score < cfg.weak_score:
        return EvidenceDecision(
            category=EvidenceCategory.WEAK,
            top_score=top_score,
            should_retry=not is_retry,
            should_generate=False,
            should_abstain=is_retry,
            reason=(
                f"Top score {top_score:.4f} is below weak threshold {cfg.weak_score:.4f}."
            ),
        )

    if top_score < cfg.strong_score:
        return EvidenceDecision(
            category=EvidenceCategory.MODERATE,
            top_score=top_score,
            should_retry=False,
            should_generate=True,
            should_abstain=False,
            reason=(
                f"Top score {top_score:.4f} is moderate "
                f"(between weak {cfg.weak_score:.4f} and strong {cfg.strong_score:.4f})."
            ),
        )

    return EvidenceDecision(
        category=EvidenceCategory.STRONG,
        top_score=top_score,
        should_retry=False,
        should_generate=True,
        should_abstain=False,
        reason=(
            f"Top score {top_score:.4f} meets or exceeds strong threshold {cfg.strong_score:.4f}."
        ),
    )


def assess_evidence(
    chunks: list[RetrievedChunk],
    config: Optional[AppConfig] = None,
) -> str:
    """Classify evidence into 'strong', 'moderate', or 'weak' based on top chunk score.

    Implements the signature specified in docs/PROJECT_GUIDE.md Section 8 (Part 6).

    Args:
        chunks: List of RetrievedChunk objects.
        config: Optional AppConfig instance.

    Returns:
        str: 'strong', 'moderate', or 'weak'.
    """
    decision = evaluate_evidence(chunks, is_retry=False, config=config)
    if decision.category == EvidenceCategory.NONE:
        return "weak"
    return decision.category.value
