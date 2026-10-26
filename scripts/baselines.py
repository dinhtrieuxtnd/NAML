"""Score the val/test impressions with non-learned baselines (random, popularity).

Usage: python -m scripts.baselines --config configs/baselines.yaml
Writes outputs/<run>_<timestamp>/metrics.json.
"""

import argparse
from pathlib import Path

from src.engine.baselines import run_baselines
from src.engine.config import BaselineConfig
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate random and popularity baselines")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = BaselineConfig.model_validate(load_config(args.config))
    run_baselines(config)


if __name__ == "__main__":
    main()
