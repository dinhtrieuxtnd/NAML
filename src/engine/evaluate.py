import logging
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import SAFE_WEIGHTS_NAME, gather_object, load_checkpoint_in_model
from einops import repeat
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.collator import EvalCollator, NewsCollator
from src.data.constants import (
    CANDIDATE_MASK_KEY,
    CANDIDATES_COLUMN,
    HISTORY_COLUMN,
    HISTORY_MASK_KEY,
    IMPRESSION_ID_COLUMN,
    INDEXES_KEY,
    LABELS_COLUMN,
    NEWS_CATEGORY_KEY,
    NEWS_ROWS_KEY,
    NEWS_TOKENS_KEY,
    PREDICTION_LABEL_COLUMN,
    PREDICTION_NEWS_ID_COLUMN,
    RANK_COLUMN,
    SCORE_COLUMN,
    USER_ID_COLUMN,
)
from src.data.dataset import EvalDataset, NewsDataset, load_split_samples
from src.data.news import NewsFeatures
from src.engine.config import TrainConfig
from src.engine.metrics import RankingMetrics
from src.models.naml import NAML
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.io import save_json
from src.utils.run import (
    CONFIG_FILE,
    EVALUATION_DIR_TEMPLATE,
    METRICS_FILE,
    PREDICTIONS_FILE,
)

logger = logging.getLogger(__name__)


# Cột tạm của predictions thô: vị trí impression trong dataset và vị trí candidate trong impression.
_SAMPLE_INDEX = "sample_index"
_POSITION = "position"


@torch.no_grad()
def evaluate(
    model: NAML,
    accelerator: Accelerator,
    dataset: EvalDataset,
    news: NewsFeatures,
    batch_size: int,
    num_workers: int,
    collect_predictions: bool = False,
) -> tuple[dict[str, float], pd.DataFrame | None]:
    """Encodes every news the split needs once, then ranks the candidates of each impression.

    Returns the metrics and, with collect_predictions, one row per candidate: its impression's
    index in `dataset`, its position among the impression's candidates and its score.
    """
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    disable_progress = None if accelerator.is_main_process else True

    news_loader = accelerator.prepare(
        DataLoader(
            NewsDataset(dataset.news_rows()),
            batch_size=batch_size,
            collate_fn=NewsCollator(news),
            num_workers=num_workers,
        )
    )
    news_table = None
    for batch in tqdm(
        news_loader, desc="encode news", leave=False, disable=disable_progress
    ):
        with accelerator.autocast():
            # vectors: [U, D]
            vectors = unwrapped.encode_news(
                batch[NEWS_TOKENS_KEY], batch[NEWS_CATEGORY_KEY]
            )
        rows, vectors = accelerator.gather_for_metrics(
            (batch[NEWS_ROWS_KEY], vectors.float())
        )
        if news_table is None:
            # news_table: [num_rows, D]; rows not needed by this split (incl. padding) stay 0.
            news_table = torch.zeros(
                news.num_rows, vectors.shape[-1], device=vectors.device
            )
        news_table[rows] = vectors

    ranking_metrics = RankingMetrics(accelerator.device)
    eval_loader = accelerator.prepare(
        DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=EvalCollator(),
            num_workers=num_workers,
        )
    )
    predictions = []
    for batch in tqdm(
        eval_loader, desc="score impressions", leave=False, disable=disable_progress
    ):
        # history: [B, N] -> [B, N, D]; user: [B, D]
        user = unwrapped.encode_user(
            news_table[batch[HISTORY_COLUMN]], batch[HISTORY_MASK_KEY]
        )
        # candidates: [B, C_max] -> [B, C_max, D]; scores: [B, C_max]
        scores = unwrapped.score(user, news_table[batch[CANDIDATES_COLUMN]])
        mask = batch[CANDIDATE_MASK_KEY]
        # indexes: [B] -> [B, C_max]
        indexes = repeat(batch[INDEXES_KEY], "b -> b c", c=scores.shape[1])
        # [B, C_max] -> [num_valid_candidates]
        preds, labels, valid_indexes = (
            scores[mask],
            batch[LABELS_COLUMN][mask],
            indexes[mask],
        )
        ranking_metrics.update(preds, labels, valid_indexes)
        if collect_predictions:
            # positions: [C_max] -> [B, C_max] -> [num_valid_candidates]
            positions = repeat(
                torch.arange(scores.shape[1], device=scores.device),
                "c -> b c",
                b=scores.shape[0],
            )[mask]
            predictions.append(
                pd.DataFrame(
                    {
                        _SAMPLE_INDEX: valid_indexes.cpu().numpy(),
                        _POSITION: positions.cpu().numpy(),
                        SCORE_COLUMN: preds.float().cpu().numpy(),
                    }
                )
            )

    metrics = ranking_metrics.compute()
    if not collect_predictions:
        return metrics, None
    # Nhiều process: gom về một chỗ; impression bị lặp do chia đều batch giữa các process thì bỏ.
    predictions = pd.concat(gather_object(predictions), ignore_index=True)
    return metrics, predictions.drop_duplicates(
        [_SAMPLE_INDEX, _POSITION], ignore_index=True
    )


def _to_prediction_table(raw: pd.DataFrame, samples: list[dict]) -> pd.DataFrame:
    """Raw predictions of evaluate() -> predictions.csv rows, sorted by impression then rank."""
    raw = raw.sort_values([_SAMPLE_INDEX, _POSITION], ignore_index=True)
    rows = [samples[i] for i in raw[_SAMPLE_INDEX]]
    positions = raw[_POSITION].tolist()
    table = pd.DataFrame(
        {
            IMPRESSION_ID_COLUMN: [sample[IMPRESSION_ID_COLUMN] for sample in rows],
            USER_ID_COLUMN: [sample[USER_ID_COLUMN] for sample in rows],
            PREDICTION_NEWS_ID_COLUMN: [
                sample[CANDIDATES_COLUMN][position]
                for sample, position in zip(rows, positions)
            ],
            PREDICTION_LABEL_COLUMN: [
                sample[LABELS_COLUMN][position]
                for sample, position in zip(rows, positions)
            ],
            SCORE_COLUMN: raw[SCORE_COLUMN],
        }
    )
    # rank 1 = điểm cao nhất trong impression; hoà điểm thì candidate đứng trước xếp trước.
    table[RANK_COLUMN] = (
        table.groupby(raw[_SAMPLE_INDEX])[SCORE_COLUMN]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    table[_SAMPLE_INDEX] = raw[_SAMPLE_INDEX]
    table = table.sort_values([_SAMPLE_INDEX, RANK_COLUMN], ignore_index=True)
    return table.drop(columns=_SAMPLE_INDEX)


def run_evaluation(checkpoint_dir: Path) -> dict[str, float]:
    """Evaluates a checkpoint saved by run_training on the test split.

    The config is read from <checkpoint_dir>/config.yaml. metrics.json and predictions.csv go
    to <run_dir>/evaluation_<checkpoint name>/ (checkpoints live in <run_dir>/checkpoints/).
    """
    config = TrainConfig.model_validate(load_config(checkpoint_dir / CONFIG_FILE))
    data_cfg = config.data
    output_dir = checkpoint_dir.parent.parent / EVALUATION_DIR_TEMPLATE.format(
        checkpoint=checkpoint_dir.name
    )
    accelerator = Accelerator(mixed_precision=config.trainer.mixed_precision)

    news = NewsFeatures(data_cfg.processed_dir, data_cfg.text_views)
    samples = load_split_samples(data_cfg.processed_dir, "test")[
        : data_cfg.max_eval_impressions
    ]
    dataset = EvalDataset(samples, news, data_cfg.history_len)
    model = build_model(
        config.model, news.num_categories, data_cfg.text_views, news.embedding
    )
    # Chỉ cần trọng số model (accelerator.save_state lưu ở SAFE_WEIGHTS_NAME), không nạp
    # optimizer/scheduler/RNG của checkpoint.
    load_checkpoint_in_model(
        model, str(checkpoint_dir / SAFE_WEIGHTS_NAME), strict=True
    )
    model = accelerator.prepare(model)
    logger.info("Loaded %s, test impressions=%d", checkpoint_dir, len(dataset))

    metrics, raw_predictions = evaluate(
        model,
        accelerator,
        dataset,
        news,
        data_cfg.eval_batch_size,
        data_cfg.num_workers,
        collect_predictions=True,
    )
    logger.info(
        "test: %s", ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    )
    if accelerator.is_main_process:
        output_dir.mkdir(exist_ok=True)
        save_json(metrics, output_dir / METRICS_FILE)
        predictions = _to_prediction_table(raw_predictions, samples)
        predictions.to_csv(output_dir / PREDICTIONS_FILE, index=False)
        logger.info(
            "Saved %s and %s (%d rows) to %s",
            METRICS_FILE,
            PREDICTIONS_FILE,
            len(predictions),
            output_dir,
        )
    return metrics
