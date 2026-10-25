"""Deterministic evidence assessment and query simplification decision module.

Classifies retrieved evidence against configured strong and weak similarity
thresholds to determine whether evidence is sufficient for generation, requires
retry retrieval with an expanded query, or warrants explicit abstention.
"""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Optional

from src.config import AppConfig, load_config
from src.models import ClarificationDetail, RetrievedChunk

# Common question words, auxiliary verbs, and stopwords for query simplification
STOPWORDS = {
    "a", "an", "the", "what", "how", "is", "are", "was", "were", "does", "do",
    "did", "explain", "why", "which", "who", "whom", "where", "when", "can",
    "could", "would", "should", "tell", "me", "about", "of", "in", "to", "for",
    "with", "on", "at", "from", "by", "and", "or", "be", "been", "being", "have",
    "has", "had", "please", "give", "show", "i", "you", "my", "your", "it", "its"
}

DANGLING_PRONOUNS = {
    "it", "its", "this", "that", "these", "those", "they", "them"
}

DANGLING_PHRASES = [
    "the first one",
    "the second one",
    "the previous one",
    "the latter",
    "the former",
]

GENERIC_ERROR_PHRASES = [
    "the error",
    "this error",
    "that error",
    "the issue",
    "the problem",
    "the bug",
    "the failure",
    "the exception",
]

# Words that appear in general inquiries but do not by themselves constitute a specific topic
GENERIC_QUERY_WORDS = {
    # Question words
    "what", "how", "why", "when", "where", "who", "which",
    # Auxiliaries & copulas
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "can", "could", "would", "should", "will", "shall",
    "have", "has", "had",
    # Determiners & articles
    "the", "a", "an",
    # Pronouns & fillers
    "it", "its", "this", "that", "these", "those", "they", "them",
    "i", "me", "my", "you", "your", "we", "us", "our",
    "one", "ones", "thing", "things",
    # Common prepositions & conjunctions
    "to", "for", "in", "on", "at", "of", "with", "from", "by", "about",
    "and", "or", "but", "so", "if",
    # Generic action/inquiry verbs & qualifiers
    "explain", "tell", "describe", "mean", "means", "work", "works",
    "fix", "fixes", "happen", "happens", "do", "does", "done", "doing",
    "use", "uses", "show", "give", "help", "solve", "say", "says",
    "run", "runs", "see", "get", "make", "know", "detail", "details",
    "more", "much", "some", "any", "all", "info", "information", "elaborate",
    "bit", "further",
}

COMPARE_WORDS = {
    "compare", "difference", "differ", "between", "versus", "vs", "choose", "better",
    "prefer", "two", "both", "either", "neither",
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


def _has_prior_history(question: str, history: Optional[list[Any]]) -> bool:
    """Determine whether history contains conversation turns prior to the current question."""
    if not history:
        return False
    clean_target = question.strip().lower()
    prior: list[Any] = []
    for item in history:
        if isinstance(item, dict):
            content = item.get("content", "").strip().lower()
            role = item.get("role", "")
            # Skip if this item is just the current question appended prior to execution
            if content == clean_target and role == "user":
                continue
            prior.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            content = str(item[1]).strip().lower()
            if content == clean_target:
                continue
            prior.append(item)
        else:
            if str(item).strip().lower() == clean_target:
                continue
            prior.append(item)
    return len(prior) > 0


def needs_clarification(
    question: str,
    history: Optional[list[Any]] = None,
) -> ClarificationDetail:
    """Evaluate whether a question requires clarification before retrieval.

    Clarification triggers deterministically without calling the LLM when:
    1. The question is too brief (fewer than 2 words).
    2. The question asks to compare or choose without naming specific subjects.
    3. The question refers to a generic error or problem without context or error names.
    4. When there is no conversation history, a short question (<= 10 words) depends
       on a dangling reference (it, its, this, that, these, those, they, them,
       'the first one', etc.) without naming any concrete topic.

    Args:
        question: User query string.
        history: Optional list of previous conversation turns.

    Returns:
        ClarificationDetail: dataclass indicating whether clarification is needed
                             and containing the user-facing clarification question.
    """
    if not question or not question.strip():
        # Empty questions are handled by question validation in pipeline
        return ClarificationDetail(needs_clarification=False)

    clean_q = question.strip()
    lower_q = clean_q.lower()
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", lower_q)

    if not tokens:
        return ClarificationDetail(needs_clarification=False)

    # 1. Very brief queries (fewer than 2 words)
    if len(tokens) < 2:
        word = tokens[0]
        if word in GENERIC_QUERY_WORDS or word in STOPWORDS:
            return ClarificationDetail(
                needs_clarification=True,
                question="Your question is too brief. Please specify what you are asking about.",
            )
        return ClarificationDetail(
            needs_clarification=True,
            question=f"Your question is too brief. Please ask a specific question about {clean_q.strip('?.!')}.",
        )

    # 2. Ultra-brief dangling queries (<= 2 words with dangling pronoun, e.g. "Explain it", "Explain this")
    # These lack any concrete topic and must ALWAYS trigger clarification, even if conversation history exists.
    if len(tokens) <= 2:
        found_pronouns = [w for w in tokens if w in DANGLING_PRONOUNS]
        if found_pronouns:
            concrete_words = [
                w for w in tokens
                if w not in GENERIC_QUERY_WORDS and w not in DANGLING_PRONOUNS
            ]
            if not concrete_words:
                pronoun = found_pronouns[0]
                return ClarificationDetail(
                    needs_clarification=True,
                    question=f'What does "{pronoun}" refer to? Please name the topic you are asking about.',
                )

    # If conversation history exists prior to this question, multi-word follow-up questions
    # (e.g. "How does it work?") may refer to earlier turns to be resolved in F10.
    if _has_prior_history(clean_q, history):
        return ClarificationDetail(needs_clarification=False)

    # 2. Compare / choose without naming subjects
    has_compare_phrase = any(
        phrase in lower_q
        for phrase in [
            "compare them", "which is better", "which one is better",
            "compare the two", "choose between them", "what is the difference",
            "how do they differ", "which should i choose", "which should i use"
        ]
    )
    if has_compare_phrase or ("compare" in tokens and any(p in tokens for p in ["them", "these", "those", "both", "two"])):
        concrete_words = [
            w for w in tokens
            if w not in GENERIC_QUERY_WORDS and w not in COMPARE_WORDS and w not in DANGLING_PRONOUNS
        ]
        if not concrete_words:
            return ClarificationDetail(
                needs_clarification=True,
                question="What would you like to compare? Please name the specific concepts or items.",
            )

    # 3. Generic error / problem phrases without specific context
    for phrase in GENERIC_ERROR_PHRASES:
        if phrase in lower_q:
            error_words = {"error", "issue", "problem", "bug", "failure", "exception"}
            concrete_words = [
                w for w in tokens
                if w not in GENERIC_QUERY_WORDS and w not in error_words and w not in DANGLING_PRONOUNS
            ]
            if not concrete_words:
                return ClarificationDetail(
                    needs_clarification=True,
                    question="Which error or issue are you referring to? Please provide the error message or context.",
                )

    # 4. Multi-word dangling phrases (e.g. "the first one", "the previous one")
    for phrase in DANGLING_PHRASES:
        if phrase in lower_q:
            phrase_tokens = phrase.split()
            concrete_words = [
                w for w in tokens
                if w not in GENERIC_QUERY_WORDS and w not in phrase_tokens and w not in DANGLING_PRONOUNS
            ]
            if not concrete_words:
                return ClarificationDetail(
                    needs_clarification=True,
                    question=f'What does "{phrase}" refer to? Please name the topic you are asking about.',
                )

    # 5. Short questions (<= 10 words) with dangling pronouns and no concrete topic
    if len(tokens) <= 10:
        found_pronouns = [w for w in tokens if w in DANGLING_PRONOUNS]
        if found_pronouns:
            concrete_words = [
                w for w in tokens
                if w not in GENERIC_QUERY_WORDS and w not in DANGLING_PRONOUNS
            ]
            if not concrete_words:
                pronoun = found_pronouns[0]
                return ClarificationDetail(
                    needs_clarification=True,
                    question=f'What does "{pronoun}" refer to? Please name the topic you are asking about.',
                )

    return ClarificationDetail(needs_clarification=False)

