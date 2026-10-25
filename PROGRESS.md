# PROGRESS

Last updated: 2026-10-25
Current part: Part 0 - Environment and skeleton

## Environment
- Python version: 3.14.0
- Ollama model chosen: llama3.2:3b (configured; service start & model pull pending)
- Typical answer time: Pending Ollama service start
- Embedding model: all-MiniLM-L6-v2 (successfully loaded and verified, 384 dimensions)
- Collections present: python_fundamentals (directory initialized at data/collections/python_fundamentals/)

## Part status
- Part 0 Environment and skeleton: DONE and tested
- Part 1 Config and data classes: TODO
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
- Ollama is not currently reachable at http://localhost:11434. The Ollama service needs to be started (`ollama serve` or app launched) and the configured model pulled (`ollama pull llama3.2:3b`).

## Decisions made that differ from the guide
- None. Followed all specifications from docs/PROJECT_GUIDE.md.

## Commands
- Check environment: python scripts/check_env.py
- Run tests: pytest
- Build index: python scripts/build_index.py (skeleton in Part 0, fully built in Part 3)
- Run app: streamlit run app.py (to be built in Part 8)
- Run evaluation: TODO

## Next step
Start Ollama (`ollama serve`), pull the configured model (`ollama pull llama3.2:3b`), and begin Part 1 (Config and data classes).