"""Evaluate a checkpoint saved by scripts/train.py.

Usage: python -m scripts.evaluate --checkpoint outputs/<run>/checkpoints/epoch_<i> --split test
"""

import argparse
import logging
from pathlib import Path

from src.engine.evaluate import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("src").setLevel(logging.INFO)
    run_evaluation(args.checkpoint, args.split)


if __name__ == "__main__":
    main()
