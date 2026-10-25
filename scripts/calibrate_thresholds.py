"""Threshold calibration script for scoring sample questions against a collection.

Reads a list of test questions, executes retrieval against the specified collection,
prints the top retrieval score and evidence assessment category for each question,
and helps developers calibrate `strong_score` and `weak_score` in config.yaml.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.decision import assess_evidence, evaluate_evidence
from src.retrieval import retrieve

DEFAULT_CALIBRATION_QUESTIONS = [
    # Domain answerable questions (expected high/moderate scores)
    "What is a list in Python?",
    "How does slicing work with strings and lists?",
    "What is the difference between a list and a tuple?",
    "How do dictionaries store key-value pairs?",
    "What does the len() function return?",
    "How do for loops iterate over sequences in Python?",
    "What is a list comprehension?",
    "How does error handling work with try and except blocks?",
    "What are function arguments and return values in Python?",
    "How do you define a class in Python?",
    # Domain out-of-scope / unrelated questions (expected weak scores)
    "How do you bake a sourdough bread loaf?",
    "What is the recipe for chocolate chip cookies?",
    "What is the capital city of France?",
    "How does quantum superposition work in quantum mechanics?",
    "What are the rules of basketball?",
    "How do you change the engine oil in a Toyota Camry?",
    "What causes volcanic eruptions on ocean ridges?",
    "What is the Pythagorean theorem in Euclidean geometry?",
    "How do airplanes generate aerodynamic lift?",
    "What is the fastest land animal on Earth?",
]


def calibrate(collection_name: str, questions_file: str | None = None) -> None:
    """Run calibration questions and print similarity scores and classifications."""
    config = load_config()

    if questions_file and Path(questions_file).exists():
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        questions = DEFAULT_CALIBRATION_QUESTIONS

    print(f"\nCalibrating collection: {collection_name}")
    print(f"Current config: strong_score={config.strong_score}, weak_score={config.weak_score}\n")
    print(f"{'#':<3} | {'Top Score':<10} | {'Assessment':<10} | {'Question'}")
    print("-" * 75)

    scores: list[float] = []
    for idx, q in enumerate(questions, start=1):
        try:
            chunks = retrieve(collection_name=collection_name, query=q, k=config.top_k, config=config)
            decision = evaluate_evidence(chunks, is_retry=False, config=config)
            top_score = decision.top_score
            cat = decision.category.value
            scores.append(top_score)
            print(f"{idx:<3} | {top_score:<10.4f} | {cat:<10} | {q}")
        except Exception as e:
            print(f"{idx:<3} | {'ERROR':<10} | {'ERROR':<10} | {q} ({e})")

    if scores:
        print("\nSummary:")
        print(f"  Min score: {min(scores):.4f}")
        print(f"  Max score: {max(scores):.4f}")
        print(f"  Avg score: {sum(scores) / len(scores):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate similarity thresholds for a collection")
    parser.add_argument("--collection", default="python_fundamentals", help="Collection name")
    parser.add_argument("--questions-file", default=None, help="Path to text file with questions")
    args = parser.parse_args()

    calibrate(args.collection, args.questions_file)


if __name__ == "__main__":
    main()
