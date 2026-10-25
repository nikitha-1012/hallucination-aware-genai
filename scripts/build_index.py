"""Command-line index builder script.

Builds a persistent ChromaDB vector index for a chosen document collection.
Usage:
    python scripts/build_index.py [--collection COLLECTION_NAME]
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path for direct script execution
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import ConfigError, load_config
from src.indexing import build_index, index_chunk_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a persistent ChromaDB index for a document collection."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Name of the document collection to index (defaults to default_collection in config.yaml).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Hallucination-Aware GenAI Assistant - Index Builder")
    print("=" * 60)

    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"[-] Configuration error: {e}")
        sys.exit(1)

    collection_name = args.collection or cfg.default_collection
    collection_dir = cfg.get_collection_path(collection_name)

    print(f"Target Collection: {collection_name}")
    print(f"Collection Path:   {collection_dir}")
    print(f"Indexes Path:      {cfg.get_indexes_path()}")
    print(f"Embedding Model:   {cfg.embedding_model}")
    print(f"Chunk Size:        {cfg.chunk_size} (overlap: {cfg.chunk_overlap})")
    print("-" * 60)

    if not collection_dir.exists() or not collection_dir.is_dir():
        print(f"[X] FAIL: Collection directory does not exist: {collection_dir}")
        print("    Instruction: Create the directory under data/collections/ and place source documents in it.")
        sys.exit(1)

    # Check if collection contains any non-hidden files
    source_files = [
        f for f in collection_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ]
    if not source_files:
        print(f"[!] Warning: No source documents found in '{collection_dir}'.")
        print("    Add PDF, TXT, or Markdown documents to this folder before building the index.")
        print("    An empty index will be initialized.")

    print(f"Found {len(source_files)} source files. Building index...")
    try:
        stats = build_index(collection_name, cfg)
    except Exception as e:
        print(f"[X] FAIL: Error building index: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Build Summary:")
    print("=" * 60)
    print(f"  Collection:        {stats['collection']}")
    print(f"  Files Loaded:      {stats['files_loaded']}")
    print(f"  Files Failed:      {stats['files_failed']}")
    print(f"  Chunks Created:    {stats['chunks_created']}")
    print(f"  Indexed Records:   {stats['indexed_records']}")
    print(f"  Elapsed Time:      {stats['time_seconds']}s")
    print("=" * 60)
    print("[+] Index build complete.")


if __name__ == "__main__":
    main()
