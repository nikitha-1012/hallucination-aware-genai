"""Part 0 sanity test: verifies project skeleton and imports."""

from pathlib import Path


def test_project_structure_exists():
    root = Path(__file__).resolve().parent.parent
    expected_paths = [
        root / "requirements.txt",
        root / "config.yaml",
        root / "README.md",
        root / "PROGRESS.md",
        root / "docs" / "PROJECT_GUIDE.md",
        root / "scripts" / "check_env.py",
        root / "scripts" / "build_index.py",
        root / "data" / "collections" / "python_fundamentals",
        root / "indexes",
        root / "eval" / "questions.jsonl",
        root / "eval" / "results",
    ]
    for p in expected_paths:
        assert p.exists(), f"Expected path does not exist: {p}"


def test_src_modules_import():
    import src
    import src.config
    import src.models
    import src.ingestion
    import src.chunking
    import src.indexing
    import src.retrieval
    import src.llm
    import src.generation
    import src.decision
    import src.pipeline

    assert src is not None
