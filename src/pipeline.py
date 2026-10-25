"""End-to-end question answering pipeline orchestrator.

Coordinates semantic retrieval, deterministic evidence sufficiency assessment,
query simplification and retry, grounded generation, and citation assembly
into a unified PipelineResult.
"""

from collections.abc import Callable
import time
from typing import Any, Optional

from src.config import AppConfig, load_config
from src.decision import evaluate_evidence, needs_clarification, simplify_query
from src.generation import generate_answer
from src.llm import LLMError
from src.models import GenerationResult, PipelineOutcome, PipelineResult, RetrievedChunk
from src.retrieval import EmptyIndexError, IndexNotFoundError, RetrievalError, retrieve


def run_pipeline(
    collection: str,
    question: str,
    history: Optional[list[Any]] = None,
    config: Optional[AppConfig] = None,
    retrieve_fn: Optional[Callable[..., list[RetrievedChunk]]] = None,
    generate_fn: Optional[Callable[..., GenerationResult]] = None,
) -> PipelineResult:
    """Execute the hallucination-aware question answering pipeline.

    Workflow:
    1. Validate query input (non-empty).
    2. Check rule-based clarification before retrieval; return CLARIFY immediately if needed.
    3. Retrieve top_k evidence chunks from vector storage.
    4. Evaluate evidence deterministically against similarity thresholds.
    5. If evidence is weak, retry once with a simplified query and retry_top_k.
    6. Re-evaluate expanded evidence.
    7. If evidence is sufficient (strong or moderate), generate grounded answer.
    8. If evidence remains insufficient, abstain without invoking LLM.
    9. Record full evidence traceability, citations, retry status, and latency.

    Args:
        collection: Target document collection name.
        question: User query string.
        history: Optional conversation turn history (for future context support).
        config: Optional AppConfig instance (loads default if None).
        retrieve_fn: Optional retrieval callable (defaults to src.retrieval.retrieve).
        generate_fn: Optional generation callable (defaults to src.generation.generate_answer).

    Returns:
        PipelineResult: Complete structured pipeline result.
    """
    start_time = time.time()
    cfg = config or load_config()
    retriever = retrieve_fn or retrieve
    generator = generate_fn or generate_answer
    warnings: list[str] = []

    # 1. Validate question input
    if not question or not question.strip():
        elapsed = round(max(0.0, time.time() - start_time), 4)
        return PipelineResult(
            outcome=PipelineOutcome.ERROR,
            message="Question cannot be empty. Please enter a valid question.",
            citations=[],
            retrieved=[],
            top_score=0.0,
            retried=False,
            warnings=["Empty question string provided."],
            latency_seconds=elapsed,
        )

    clean_question = question.strip()

    # 2. Rule-based clarification check (BEFORE retrieval)
    clarification = needs_clarification(clean_question, history=history)
    if clarification.needs_clarification:
        elapsed = round(max(0.0, time.time() - start_time), 4)
        return PipelineResult(
            outcome=PipelineOutcome.CLARIFY,
            message=clarification.question or "Please clarify your question.",
            citations=[],
            retrieved=[],
            top_score=0.0,
            retried=False,
            warnings=["Question requires clarification: missing necessary detail."],
            latency_seconds=elapsed,
        )

    # 3. Perform initial retrieval
    try:
        chunks = retriever(
            collection_name=collection,
            query=clean_question,
            k=cfg.top_k,
            is_retry=False,
            config=cfg,
        )
    except IndexNotFoundError as e:
        elapsed = round(max(0.0, time.time() - start_time), 4)
        return PipelineResult(
            outcome=PipelineOutcome.ERROR,
            message=f"Index for collection '{collection}' not found. Please build the index first.",
            citations=[],
            retrieved=[],
            top_score=0.0,
            retried=False,
            warnings=[str(e)],
            latency_seconds=elapsed,
        )
    except EmptyIndexError as e:
        elapsed = round(max(0.0, time.time() - start_time), 4)
        return PipelineResult(
            outcome=PipelineOutcome.ERROR,
            message=f"Collection '{collection}' is empty. No documents indexed.",
            citations=[],
            retrieved=[],
            top_score=0.0,
            retried=False,
            warnings=[str(e)],
            latency_seconds=elapsed,
        )
    except (RetrievalError, Exception) as e:
        elapsed = round(max(0.0, time.time() - start_time), 4)
        return PipelineResult(
            outcome=PipelineOutcome.ERROR,
            message=f"Retrieval error: {e}",
            citations=[],
            retrieved=[],
            top_score=0.0,
            retried=False,
            warnings=[str(e)],
            latency_seconds=elapsed,
        )

    # 3. Evaluate initial evidence
    decision = evaluate_evidence(chunks, is_retry=False, config=cfg)
    top_score = decision.top_score
    retried = False

    # 4. Optional retry if evidence is weak or none
    if decision.should_retry:
        retried = True
        retry_query = simplify_query(clean_question)
        try:
            retry_chunks = retriever(
                collection_name=collection,
                query=retry_query,
                k=cfg.retry_top_k,
                is_retry=True,
                config=cfg,
            )
            chunks = retry_chunks
            decision = evaluate_evidence(chunks, is_retry=True, config=cfg)
            top_score = decision.top_score
        except Exception as e:
            warnings.append(f"Retry retrieval failed: {e}")
            # Re-evaluate whatever chunks we currently have with is_retry=True
            decision = evaluate_evidence(chunks, is_retry=True, config=cfg)
            top_score = decision.top_score

    # 5. Check for abstention if evidence remains insufficient
    if decision.should_abstain:
        elapsed = round(max(0.0, time.time() - start_time), 4)
        abstain_message = (
            f"The selected collection '{collection}' does not contain enough information "
            f"to answer this question. Try rephrasing your question or selecting another collection."
        )
        return PipelineResult(
            outcome=PipelineOutcome.ABSTAIN,
            message=abstain_message,
            citations=[],
            retrieved=chunks,
            top_score=top_score,
            retried=retried,
            warnings=warnings,
            latency_seconds=elapsed,
        )

    # 6. Generate grounded answer
    try:
        gen_result = generator(
            question=clean_question,
            chunks=chunks,
            config=cfg,
        )
    except (LLMError, Exception) as e:
        elapsed = round(max(0.0, time.time() - start_time), 4)
        return PipelineResult(
            outcome=PipelineOutcome.ERROR,
            message=f"Generation error: {e}",
            citations=[],
            retrieved=chunks,
            top_score=top_score,
            retried=retried,
            warnings=warnings + [str(e)],
            latency_seconds=elapsed,
        )

    elapsed = round(max(0.0, time.time() - start_time), 4)
    combined_warnings = warnings + gen_result.warnings

    # Handle generation outcome types
    if gen_result.outcome == PipelineOutcome.ERROR:
        return PipelineResult(
            outcome=PipelineOutcome.ERROR,
            message=gen_result.message,
            citations=[],
            retrieved=chunks,
            top_score=top_score,
            retried=retried,
            warnings=combined_warnings,
            latency_seconds=elapsed,
        )

    if gen_result.outcome == PipelineOutcome.ABSTAIN:
        abstain_message = (
            f"The selected collection '{collection}' does not contain enough information "
            f"to answer this question. Try rephrasing your question or selecting another collection."
        )
        return PipelineResult(
            outcome=PipelineOutcome.ABSTAIN,
            message=abstain_message,
            citations=[],
            retrieved=chunks,
            top_score=top_score,
            retried=retried,
            warnings=combined_warnings,
            latency_seconds=elapsed,
        )

    return PipelineResult(
        outcome=gen_result.outcome,
        message=gen_result.message,
        citations=gen_result.citations,
        retrieved=chunks,
        top_score=top_score,
        retried=retried,
        warnings=combined_warnings,
        latency_seconds=elapsed,
    )
