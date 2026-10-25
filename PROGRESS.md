# PROGRESS

Last updated: 2026-10-25
Current part: Part 10 - Second domain (F8)

## Environment
- Python version: 3.14.0
- Ollama model chosen: llama3.2:3b (verified available and responding at http://localhost:11434)
- Typical answer time: ~1-3s
- Embedding model: all-MiniLM-L6-v2 (verified downloaded and loaded, 384 dimensions)
- Collections present:
  - python_fundamentals (data/collections/python_fundamentals/)
  - cybersecurity_fundamentals (data/collections/cybersecurity_fundamentals/)

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
- Part 9 Test sweep (F9): DONE and tested (150 tests passing total; manual UI verification complete in tests/manual_test_results.txt)
- Part 10 Second domain (F8): DONE and tested (163 tests passing total; clarification regression for "Explain it" resolved)
- Part 11 Conversation context (F10, P1): TODO
- Part 12 Answer checks (F11, P1): TODO
- Part 13 Evaluation (F12, P1): TODO
- Part 14 Report and demo: TODO

## Calibrated values
- weak_score: 0.30 (initial starting value in config.yaml)
- strong_score: 0.45 (initial starting value in config.yaml)

## Known issues
- Query "What is a Python function?" currently appears as `PARTIAL` outcome even though the answer is correctly grounded and cited. This occurs because the LLM appends a "Not covered:" clause detailing advanced concepts (e.g. decorators, lambda expressions) not present in retrieved context, which `parse_answer` detects. Grounding and citations remain completely valid. Thresholds and prompt logic were intentionally left intact.
- Local LLM generation latency on CPU/local hardware is ~20–40s for 3B parameter model queries.
- Manual test results tracking file: `tests/manual_test_results.txt` covers 15 test cases across answerable, out-of-scope, partial, clarification, false-premise, model-down error, and index-missing guard scenarios.

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
  - Test Sweep (F9):
    - Added comprehensive audit test suite `tests/test_audit_sweep.py` with 27 unit tests verifying all 9 requirement areas: extraction/chunk metadata, indexing/persistence, retrieval ranking, citation parsing/mapping, insufficient evidence handling, clarification edge cases/false-positive rejection, configuration validation, model-down error recovery, and missing index guard.
    - Added `tests/manual_test_results.txt` containing 15 detailed test cases with actual verified behaviors and notes.
  - Second Domain & Domain Independence (F8 / Part 10):
    - Added `data/collections/cybersecurity_fundamentals/` containing 3 factual reference documents: `network_security.md`, `application_security.md`, and `cryptography_and_access_control.md`.
    - Added `SOURCES.md` tracking licenses (Python PSF, NIST Public Domain, OWASP CC BY-SA 4.0).
    - Built persistent vector index for `cybersecurity_fundamentals` (13 chunks created across 3 documents).
    - Added `tests/test_domain_switching.py` (13 tests total) verifying document ingestion, chunking metadata, dual index coexistence, retrieval ranking, cross-domain abstention (Python on cyber, cyber on Python), pipeline execution with grounded citations, codebase domain independence, UI discovery, and cross-domain clarification interception.
    - Resolved clarification regression: in `src/decision.py`, ensured ultra-brief dangling queries (<= 2 words with dangling pronoun, e.g. "Explain it") lack any concrete subject and deterministically trigger `CLARIFY` before retrieval across both collections regardless of prior session history. In `app.py`, bound `key="selected_collection_sidebar"` and ensured immediate chat reset upon switching collections.
    - Zero domain-specific logic or answer branches added to generic core code.

## Commands
- Check environment: python scripts/check_env.py
- Run tests: pytest
- Build index: python scripts/build_index.py [--collection python_fundamentals | --collection cybersecurity_fundamentals]
- Calibrate thresholds: python scripts/calibrate_thresholds.py [--collection python_fundamentals]
- Run app: streamlit run app.py (verified running on http://localhost:8501)
- Run evaluation: TODO

## Next step
Implement Part 11: F10 Basic conversation context (P1)