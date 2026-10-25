# PROGRESS

Last updated: 2026-10-25
Current part: Part 4 - Retrieval (F3)

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
- Part 4 Retrieval (F3): DONE and tested (42 tests passing total)
- Part 5 Generation (F4): TODO
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
  - Exact similarity conversion: `cosine_similarity = round(1.0 - cosine_distance, 4)`, clamped to `[-1.0, 1.0]`.
  - Normal retrieval uses `config.top_k` (default: 5); retry retrieval uses `config.retry_top_k` (default: 8).
  - Empty or whitespace query gracefully returns empty list `[]`.
  - Non-existent index raises `IndexNotFoundError`; empty index raises `EmptyIndexError`.
  - Request count clamped to `min(k, total_chunks)` to avoid database query errors.

## Commands
- Check environment: python scripts/check_env.py
- Run tests: pytest
- Build index: python scripts/build_index.py [--collection python_fundamentals]
- Run app: streamlit run app.py (to be built in Part 8)
- Run evaluation: TODO

## Next step
Implement Part 5: F4 Grounded Generation in src/llm.py (Ollama client with error handling) and src/generation.py (numbered evidence prompt construction, answer parsing, and citation extraction).