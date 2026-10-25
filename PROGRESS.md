# PROGRESS

Last updated: 2026-10-25
Current part: Part 5 - Generation (F4)

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
- Part 5 Generation (F4): DONE and tested (57 tests passing total)
- Part 6 Evidence check, pipeline (F5): TODO
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
  - Grounded prompt enforces: only supplied evidence as facts; answer must state insufficient evidence if missing; no invented facts/citations; distinguish partial coverage with "Not covered:"; treat retrieved passages as untrusted data (prompt injection defense).
  - Numbered evidence blocks `[1]`, `[2]`, ... include deterministic source, page (if available), and section (if available) headers.
  - Bracket citations `[n]` in generated answers are extracted deterministically in Python via regex and mapped strictly to 1-indexed `RetrievedChunk` instances. Out-of-bounds citation numbers are stripped and logged in `warnings`.
  - Empty or insufficient evidence returns `outcome="INSUFFICIENT_EVIDENCE"` without fabricating fallbacks.
  - Single-turn citation reminder retry: if an answer is generated with facts but no bracket citations, it triggers one automatic follow-up prompting the model to add bracket citations using the supplied evidence IDs.
  - LLM client in `src/llm.py` uses Ollama `/api/chat` with custom error hierarchy (`LLMError`, `LLMConnectionError`, `LLMTimeoutError`, `LLMModelNotFoundError`).
  - Offline testability: `generate_answer()` accepts an optional `chat_fn` callback, allowing deterministic unit tests without a running Ollama server.

## Commands
- Check environment: python scripts/check_env.py
- Run tests: pytest
- Build index: python scripts/build_index.py [--collection python_fundamentals]
- Run app: streamlit run app.py (to be built in Part 8)
- Run evaluation: TODO

## Next step
Implement Part 6: F5 Evidence check, pipeline in src/decision.py and src/pipeline.py.