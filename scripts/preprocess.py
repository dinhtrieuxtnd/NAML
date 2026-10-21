"""Preprocess raw behaviors/news CSV into training artifacts.

Usage: python -m scripts.preprocess --config configs/preprocess.yaml
"""

import argparse
from pathlib import Path

from src.data.config import PreprocessConfig
from src.data.pipeline import run_preprocessing
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess real news data for NAML")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = PreprocessConfig.model_validate(load_config(args.config))
    run_preprocessing(config)


if __name__ == "__main__":
    main()
