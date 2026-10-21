"""Train a model registered in src/models/registry.py.

Usage:
    python -m scripts.train --config configs/train.yaml
    python -m scripts.train --config configs/train.yaml --resume outputs/<run>/checkpoints/latest
"""

import argparse
import logging
from pathlib import Path

from src.engine.config import TrainConfig
from src.engine.trainer import run_training
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NAML")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Checkpoint directory to continue from")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger("src").setLevel(logging.INFO)
    config = TrainConfig.model_validate(load_config(args.config))
    run_training(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
