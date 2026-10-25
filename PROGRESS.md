# PROGRESS

Last updated: 2026-10-25
Current part: Part 7 - Clarification (F6)

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
- Part 6 Evidence check, pipeline (F5): DONE and tested (77 tests passing)
- Part 7 Clarification (F6): DONE and tested (107 tests passing total)
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
  - Deterministic, rule-based clarification (`needs_clarification`) in `src/decision.py` without calling the LLM:
    - Queries fewer than 2 words prompt for more detail.
    - Questions with compare/choice triggers without named concepts (e.g., "compare them") prompt to name the concepts.
    - Generic error references without error codes or context (e.g., "explain the error") prompt for the error message or context.
    - Multi-word dangling phrases (e.g., "the first one", "the previous one") prompt to specify the referenced topic.
    - Short questions (<= 10 words) with dangling pronouns (`it`, `its`, `this`, `that`, `these`, `those`, `they`, `them`) and no concrete domain nouns prompt for the subject.
    - Specific questions (e.g., "What is a Python list?", "What is SQL injection?") do not trigger clarification and proceed to retrieval.
    - When conversation history is present, dangling references are deferred to multi-turn context resolution (Part 11).
  - Clarification is integrated into `src/pipeline.py` immediately after empty question validation and BEFORE retrieval:
    - If clarification is needed, returns `PipelineOutcome.CLARIFY` immediately without invoking vector search or LLM generation.
    - No fabricated citations or retrieved evidence are attached; non-negative latency is recorded.
    - Empty or whitespace questions continue to trigger `PipelineOutcome.ERROR`.

## Commands
- Check environment: python scripts/check_env.py
- Run tests: pytest
- Build index: python scripts/build_index.py [--collection python_fundamentals]
- Calibrate thresholds: python scripts/calibrate_thresholds.py [--collection python_fundamentals]
- Run app: streamlit run app.py (to be built in Part 8)
- Run evaluation: TODO

## Next step
Implement Part 8: F7 Streamlit UI in app.py with sidebar collection selector, index status/rebuild, chat interface, expandable sources, and pipeline integration.