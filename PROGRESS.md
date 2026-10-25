# PROGRESS

Last updated: 2026-10-25
Current part: Part 1 - Config and data classes

## Environment
- Python version: 3.14.0
- Ollama model chosen: llama3.2:3b (verified available and responding at http://localhost:11434)
- Typical answer time: ~1-3s
- Embedding model: all-MiniLM-L6-v2 (verified downloaded and loaded, 384 dimensions)
- Collections present: python_fundamentals (data/collections/python_fundamentals/)

## Part status
- Part 0 Environment and skeleton: DONE and tested
- Part 1 Config and data classes: DONE and tested (13 tests passing)
- Part 2 Ingestion (F1): TODO
- Part 3 Chunking and indexing (F2): TODO
- Part 4 Retrieval (F3): TODO
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
- None. Implemented exact fields and acceptance checks specified in the guide.

## Commands
- Check environment: python scripts/check_env.py
- Run tests: pytest
- Build index: python scripts/build_index.py (skeleton in Part 0, fully built in Part 3)
- Run app: streamlit run app.py (to be built in Part 8)
- Run evaluation: TODO

## Next step
Implement Part 2: F1 Document ingestion in src/ingestion.py with document loaders, metadata extraction, text cleaning, and test fixtures.