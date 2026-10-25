# Domain-Independent, Hallucination-Aware GenAI Assistant

A document-grounded question-answering assistant that uses selected collections of trusted documents, provides inspectable source citations, and abstains or asks for clarification when evidence is insufficient or ambiguous.

## Tech Stack
- **Language**: Python 3.10+
- **Vector Database**: ChromaDB (`PersistentClient`)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Local model via Ollama (configured in `config.yaml`, e.g., `llama3.2:3b`)
- **UI**: Streamlit
- **Testing**: pytest

## Project Structure
```
project-root/
  PROGRESS.md                 # Live progress tracking across parts
  README.md                   # Project overview and quickstart
  requirements.txt            # Python dependencies
  config.yaml                 # Configurable project settings
  docs/
    PROJECT_GUIDE.md          # Complete project specification and build plan
  data/collections/
    python_fundamentals/      # Source document collection 1
  indexes/                    # ChromaDB vector index storage (gitignored)
  src/
    config.py                 # Configuration loader and validator
    models.py                 # Shared data classes and schemas
    ingestion.py              # File loading and text extraction
    chunking.py               # Document chunking logic
    indexing.py               # Chroma indexing operations
    retrieval.py              # Semantic retrieval
    llm.py                    # Ollama client and error handling
    generation.py             # Prompt formatting and citation parsing
    decision.py               # Evidence sufficiency and clarification logic
    pipeline.py               # Orchestrator
  scripts/
    build_index.py            # CLI index builder
    check_env.py              # Environment diagnostic script
  eval/
    questions.jsonl           # Benchmark questions
    results/                  # Evaluation outputs
  tests/                      # pytest test suite
```

## Quickstart

### 1. Check Environment
Run the environment verification script to check Python, Ollama, the LLM model, and the embedding model:
```bash
python scripts/check_env.py
```

### 2. Run Tests
```bash
pytest
```
