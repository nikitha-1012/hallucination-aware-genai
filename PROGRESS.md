# PROGRESS

Last updated: 2026-10-25
Current part: Part 6 - Evidence check, pipeline (F5)

## Environment
- Python version: 3.14.0
- Ollama model chosen: llama3.2:3b (verified available and responding at http://localhost:11434)
- Typical answer time: ~1-3s
- Embedding model: all-MiniLM-L6-v2 (verified downloaded and loaded, 384 dimensions)
- Collections present: python_fundamentals (data/collections/python_fundamentals/)

## Part status
- Part 0 Environment and skeleton: DONE and tested
- Part 1 Config and data classes: DONE and tested (13 tests passing)
- Part 2 Ingestion (F1): DONE and tested (22 tests passing)
- Part 3 Chunking and indexing (F2): DONE and tested (32 tests passing)
- Part 4 Retrieval (F3): DONE and tested (42 tests passing)
- Part 5 Generation (F4): DONE and tested (57 tests passing)
- Part 6 Evidence check, pipeline (F5): DONE and tested (77 tests passing total)
- Part 7 Clarification (F6): TODO
- Part 8 Streamlit UI (F7): TODO
- Part 9 Test sweep (F9): TODO
- Part 10 Second domain (F8): TODO
- Part 11 Conversation context (F10, P1): TODO
- Part 12 Answer checks (F11, P1): TODO
- Part 13 Evaluation (F12, P1): TODO
- Part 14 Report and demo: TODO

## Calibrated values
- weak_score: 0.30 (initial starting value in config.yaml)
- strong_score: 0.45 (initial starting value in config.yaml)

## Known issues
- None

## Decisions made that differ from the guide
- None. Followed all specifications from docs/PROJECT_GUIDE.md:
  - Deterministic decision layer (`src/decision.py`) evaluates evidence against `strong_score` and `weak_score` without calling the LLM:
    - `STRONG` (`top_score >= strong_score`): generate immediately without retry.
    - `MODERATE` (`weak_score <= top_score < strong_score`): generate with strict grounding constraints.
    - `WEAK` (`top_score < weak_score`): retry retrieval once if not yet retried; if still weak, abstain.
    - `NONE` (empty chunks / `top_score <= 0.0`): retry once if not yet retried; if still empty, abstain.
  - Query simplification (`simplify_query`) removes question words, auxiliary verbs, and stopwords (`STOPWORDS`), falling back safely to the original stripped query if no content words remain.
  - Abstention response explicitly identifies the searched collection and suggests rephrasing or switching collections.
  - End-to-end pipeline (`src/pipeline.py`) orchestrates retrieval -> decision -> retry -> generation -> `PipelineResult`.
  - Empty questions, index not found (`IndexNotFoundError`), empty index (`EmptyIndexError`), retrieval errors, and generation errors return structured `PipelineOutcome.ERROR` results with latency and warnings, avoiding synthetic answers.
  - Calibration script (`scripts/calibrate_thresholds.py`) provides CLI testing for domain answerable vs. out-of-scope query scoring.

## Commands
- Check environment: python scripts/check_env.py
- Run tests: pytest
- Build index: python scripts/build_index.py [--collection python_fundamentals]
- Calibrate thresholds: python scripts/calibrate_thresholds.py [--collection python_fundamentals]
- Run app: streamlit run app.py (to be built in Part 8)
- Run evaluation: TODO

## Next step
Implement Part 7: F6 Basic clarification in src/decision.py (needs_clarification rule-based evaluation) and integrate with pipeline.