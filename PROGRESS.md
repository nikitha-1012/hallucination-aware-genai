# PROGRESS

Last updated: 2026-10-25
Current part: Part 8 - Streamlit UI (F7)

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
- Part 7 Clarification (F6): DONE and tested (107 tests passing)
- Part 8 Streamlit UI (F7): DONE and tested (123 tests passing total)
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
  - Built interactive Streamlit UI in `app.py` with presentation helper functions in `src/ui_helpers.py`:
    - Sidebar:
      - Dynamic collection discovery from `data/collections/` with default collection fallback.
      - Real-time index status (`index_exists`) and chunk counts (`index_chunk_count`).
      - "Build or rebuild index" button with progress spinner invoking `build_index()`, reporting file and chunk counts.
      - Active model and threshold metadata display.
    - Main Area:
      - Clean chat interface via Streamlit's `st.chat_message` and `st.chat_input`.
      - Missing index guard: disables input and displays explicit guidance if the selected collection's index has not been built yet.
      - Outcome visualization with distinct colors and icons:
        - `ANSWER` -> "Answered" (Success)
        - `PARTIAL` -> "Partial" (Info)
        - `CLARIFY` -> "Needs clarification" (Warning)
        - `ABSTAIN` -> "Not enough evidence" (Warning)
        - `ERROR` -> "Error" (Error)
      - Traceable "Sources" expander showing only chunks actually cited (source file, page/section, score, and passage text).
      - Debugging "All retrieved passages" expander showing full retrieval transparency.
      - Actionable error fix guidance (e.g. `ollama serve`, `ollama pull`, index build).
      - Model caching via `@st.cache_resource` for the embedding model to avoid repeated disk reads.

## Commands
- Check environment: python scripts/check_env.py
- Run tests: pytest
- Build index: python scripts/build_index.py [--collection python_fundamentals]
- Calibrate thresholds: python scripts/calibrate_thresholds.py [--collection python_fundamentals]
- Run app: streamlit run app.py (verified running on http://localhost:8501)
- Run evaluation: TODO

## Next step
Implement Part 9: F9 Test sweep across all core capabilities, defect fixes, and manual verification set.