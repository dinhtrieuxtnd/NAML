import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from einops import repeat
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC
from torchmetrics.retrieval import RetrievalAUROC, RetrievalMRR, RetrievalNormalizedDCG

from src.data.collator import EvalCollator, NewsCollator
from src.data.constants import (
    CANDIDATE_MASK_KEY,
    CANDIDATES_COLUMN,
    CONFIG_FILE,
    HISTORY_COLUMN,
    HISTORY_MASK_KEY,
    INDEXES_KEY,
    LABELS_COLUMN,
    NEWS_CATEGORY_KEY,
    NEWS_ROWS_KEY,
    NEWS_TOKENS_KEY,
)
from src.data.dataset import EvalDataset, NewsDataset, load_split_samples
from src.data.news import NewsFeatures
from src.engine.config import TrainConfig
from src.models.naml import NAML
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.io import save_json

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: NAML,
    accelerator: Accelerator,
    dataset: EvalDataset,
    news: NewsFeatures,
    batch_size: int,
    num_workers: int,
) -> dict[str, float]:
    """Encodes every news the split needs once, then ranks the candidates of each impression."""
    model.eval()
    unwrapped = accelerator.unwrap_model(model)

    news_loader = accelerator.prepare(DataLoader(
        NewsDataset(dataset.news_rows()), batch_size=batch_size,
        collate_fn=NewsCollator(news), num_workers=num_workers))
    news_table = None
    for batch in news_loader:
        with accelerator.autocast():
            # vectors: [U, D]
            vectors = unwrapped.encode_news(batch[NEWS_TOKENS_KEY], batch[NEWS_CATEGORY_KEY])
        rows, vectors = accelerator.gather_for_metrics((batch[NEWS_ROWS_KEY], vectors.float()))
        if news_table is None:
            # news_table: [num_rows, D]; rows not needed by this split (incl. padding) stay 0.
            news_table = torch.zeros(news.num_rows, vectors.shape[-1], device=vectors.device)
        news_table[rows] = vectors

    auroc = BinaryAUROC().to(accelerator.device)
    retrieval = MetricCollection({
        "impression_auc": RetrievalAUROC(empty_target_action="skip"),
        "mrr": RetrievalMRR(empty_target_action="skip"),
        "ndcg@5": RetrievalNormalizedDCG(top_k=5, empty_target_action="skip"),
        "ndcg@10": RetrievalNormalizedDCG(top_k=10, empty_target_action="skip"),
    }).to(accelerator.device)

    eval_loader = accelerator.prepare(DataLoader(
        dataset, batch_size=batch_size, collate_fn=EvalCollator(), num_workers=num_workers))
    for batch in eval_loader:
        # history: [B, N] -> [B, N, D]; user: [B, D]
        user = unwrapped.encode_user(news_table[batch[HISTORY_COLUMN]], batch[HISTORY_MASK_KEY])
        # candidates: [B, C_max] -> [B, C_max, D]; scores: [B, C_max]
        scores = unwrapped.score(user, news_table[batch[CANDIDATES_COLUMN]])
        mask = batch[CANDIDATE_MASK_KEY]
        # indexes: [B] -> [B, C_max]
        indexes = repeat(batch[INDEXES_KEY], "b -> b c", c=scores.shape[1])
        # [B, C_max] -> [num_valid_candidates]
        preds, labels, indexes = scores[mask], batch[LABELS_COLUMN][mask], indexes[mask]
        auroc.update(preds, labels)
        retrieval.update(preds, labels, indexes=indexes)

    metrics = {"auc": auroc.compute(), **retrieval.compute()}
    return {name: value.item() for name, value in metrics.items()}


def run_evaluation(checkpoint_dir: Path, split: str) -> dict[str, float]:
    """Evaluates a checkpoint saved by run_training on one split.

    Checkpoints live in <run_dir>/checkpoints/<name>, so the run's resolved config is read
    from <run_dir>/config.yaml. Metrics go to <checkpoint_dir>/metrics_<split>.json.
    """
    run_dir = checkpoint_dir.parent.parent
    config = TrainConfig.model_validate(load_config(run_dir / CONFIG_FILE))
    data_cfg = config.data
    accelerator = Accelerator(mixed_precision=config.trainer.mixed_precision)

    news = NewsFeatures(data_cfg.processed_dir, data_cfg.text_views, config.model.pretrained_name)
    dataset = EvalDataset(
        load_split_samples(data_cfg.processed_dir, split)[:data_cfg.max_eval_impressions],
        news, data_cfg.history_len)
    model = accelerator.prepare(
        build_model(config.model, news.num_categories, data_cfg.text_views))
    accelerator.load_state(str(checkpoint_dir))
    logger.info("Loaded %s, %s impressions=%d", checkpoint_dir, split, len(dataset))

    metrics = evaluate(model, accelerator, dataset, news,
                       data_cfg.eval_batch_size, data_cfg.num_workers)
    if accelerator.is_main_process:
        save_json(metrics, checkpoint_dir / f"metrics_{split}.json")
    logger.info("%s: %s", split, ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))
    return metrics
