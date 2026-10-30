"""Non-learned rankers on the val/test impressions, as reference points for the model.

- random: a uniformly random score per candidate.
- popularity: clicks on the candidate during the popularity_window_minutes before the
  impression, counted from the cleaned event log at the real click time (only earlier clicks).
"""

import logging
from pathlib import Path

import numpy as np
import torch

from src.data.constants import (
    CANDIDATES_COLUMN,
    IMPRESSION_ID_COLUMN,
    LABELS_COLUMN,
    TIME_COLUMN,
)
from src.data.dataset import load_split_samples
from src.data.parser import read_clicks, read_impression_log
from src.data.popularity import ClickCounter
from src.engine.config import BaselineConfig
from src.engine.metrics import RankingMetrics
from src.utils.io import save_json
from src.utils.run import METRICS_FILE, new_run_dir, start_run

logger = logging.getLogger(__name__)

STAGE = "baselines"
EVAL_SPLITS = ("val", "test")


def run_baselines(config: BaselineConfig) -> None:
    run_dir = new_run_dir(config.run)
    with start_run(STAGE, run_dir, config, resume=False):
        _run(config, run_dir)


def _run(config: BaselineConfig, run_dir: Path) -> None:
    data_cfg = config.data
    logs = {
        "val": read_impression_log(data_cfg.val_behaviors),
        "test": read_impression_log(data_cfg.test_behaviors),
    }
    clicks = ClickCounter(read_clicks(data_cfg.click_log))
    logger.info(
        "Clicks: %d from %s, popularity window=%dmin",
        clicks.num_clicks,
        data_cfg.click_log,
        config.popularity_window_minutes,
    )

    rng = np.random.default_rng(config.seed)
    results = {}
    for split in EVAL_SPLITS:
        samples = load_split_samples(data_cfg.processed_dir, split)
        impression_times = logs[split].set_index(
            IMPRESSION_ID_COLUMN, verify_integrity=True
        )[TIME_COLUMN]
        # Mỗi candidate một phần tử: [num_candidates].
        news_ids = np.array(
            [news for sample in samples for news in sample[CANDIDATES_COLUMN]], dtype=np.int64
        )
        labels = torch.tensor(
            [label for sample in samples for label in sample[LABELS_COLUMN]]
        )
        indexes = torch.tensor(
            [i for i, sample in enumerate(samples) for _ in sample[CANDIDATES_COLUMN]]
        )
        times = impression_times.loc[
            [sample[IMPRESSION_ID_COLUMN] for sample in samples]
        ].to_numpy()[indexes.numpy()]
        popularity = clicks.count(news_ids, times, config.popularity_window_minutes * 60)
        logger.info(
            "%s: %d impressions, %d candidates, %.1f%% with no click in window",
            split,
            len(samples),
            len(news_ids),
            100 * (popularity == 0).mean(),
        )

        scores = {
            "random": rng.random(len(news_ids)),
            # Cộng nhiễu [0, 1) để phá hoà ngẫu nhiên mà không đổi thứ tự số click.
            "popularity": popularity + rng.random(len(news_ids)),
        }
        results[split] = {}
        for name, score in scores.items():
            metrics = RankingMetrics(torch.device("cpu"))
            metrics.update(torch.from_numpy(score), labels, indexes)
            results[split][name] = metrics.compute()
            logger.info(
                "%s %s: %s",
                split,
                name,
                " ".join(f"{metric}={value:.4f}" for metric, value in results[split][name].items()),
            )
    save_json(results, run_dir / METRICS_FILE)
