"""Evaluate a checkpoint saved by scripts/train.py on the test split.

Usage: python -m scripts.evaluate --checkpoint outputs/<run>/checkpoints/best
Writes outputs/<run>/evaluation_best/{metrics.json, predictions.csv}.
"""

import argparse
from pathlib import Path

from src.engine.evaluate import run_evaluation
from src.utils.run import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint on the test split"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory, usually outputs/<run>/checkpoints/best",
    )
    args = parser.parse_args()

    setup_logging(None)
    run_evaluation(args.checkpoint)


if __name__ == "__main__":
    main()
