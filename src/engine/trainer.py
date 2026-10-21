import datetime
import logging

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader

from src.data.collator import TrainCollator
from src.data.constants import (
    CANDIDATES_COLUMN,
    CONFIG_FILE,
    HISTORY_COLUMN,
    HISTORY_MASK_KEY,
    NEWS_CATEGORY_KEY,
    NEWS_TOKENS_KEY,
)
from src.data.dataset import EvalDataset, TrainDataset, load_split_samples
from src.data.news import NewsFeatures
from src.engine.config import TrainConfig
from src.engine.evaluate import evaluate
from src.models.registry import build_loss, build_model
from src.utils.config import save_config

logger = logging.getLogger(__name__)


def _build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    encoder_ids = {id(p) for p in model.news_encoder.pretrained_parameters()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    encoder_params = [p for p in trainable if id(p) in encoder_ids]
    head_params = [p for p in trainable if id(p) not in encoder_ids]
    groups = [{"params": head_params, "lr": config.optimizer.lr}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": config.optimizer.encoder_lr})
    logger.info("Trainable params: encoder=%d, head=%d",
                sum(p.numel() for p in encoder_params), sum(p.numel() for p in head_params))
    return torch.optim.AdamW(groups)


def run_training(config: TrainConfig) -> None:
    trainer_cfg, data_cfg = config.trainer, config.data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = trainer_cfg.output_root / f"{trainer_cfg.run_name}_{timestamp}"
    accelerator = Accelerator(
        mixed_precision=trainer_cfg.mixed_precision,
        gradient_accumulation_steps=trainer_cfg.gradient_accumulation_steps,
        log_with="tensorboard",
        project_config=ProjectConfiguration(
            project_dir=str(run_dir),
            automatic_checkpoint_naming=True,
            total_limit=trainer_cfg.checkpoint_total_limit,
        ),
    )
    set_seed(trainer_cfg.seed)
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True)
        save_config(config.model_dump(mode="json"), run_dir / CONFIG_FILE)
    accelerator.init_trackers("tensorboard")
    logger.info("Run directory: %s", run_dir)

    news = NewsFeatures(data_cfg.processed_dir, data_cfg.text_views, config.model.pretrained_name)
    train_dataset = TrainDataset(
        load_split_samples(data_cfg.processed_dir, "train"), news,
        data_cfg.history_len, data_cfg.num_negatives)
    val_dataset = EvalDataset(
        load_split_samples(data_cfg.processed_dir, "val")[:data_cfg.max_eval_impressions],
        news, data_cfg.history_len)
    logger.info("Samples: train=%d, val=%d", len(train_dataset), len(val_dataset))

    model = build_model(config.model, news.num_categories, data_cfg.text_views)
    loss_fn = build_loss(config.loss.name)
    optimizer = _build_optimizer(model, config)
    train_loader = DataLoader(
        train_dataset, batch_size=data_cfg.train_batch_size, shuffle=True,
        collate_fn=TrainCollator(news), num_workers=data_cfg.num_workers)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    global_step = 0
    reached_max_steps = False
    for epoch in range(trainer_cfg.epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                # scores: [B, K + 1], the positive is at position 0
                scores = model(
                    news_tokens=batch[NEWS_TOKENS_KEY],
                    news_category=batch[NEWS_CATEGORY_KEY],
                    history=batch[HISTORY_COLUMN],
                    history_mask=batch[HISTORY_MASK_KEY],
                    candidates=batch[CANDIDATES_COLUMN],
                )
                target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
                loss = loss_fn(scores, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss {loss.item()} at step {global_step}")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % trainer_cfg.log_every == 0:
                    accelerator.log({"train/loss": loss.item(), "train/epoch": epoch}, step=global_step)
                    logger.info("epoch %d step %d loss %.4f", epoch, global_step, loss.item())
                if global_step == trainer_cfg.max_train_steps:
                    reached_max_steps = True
                    break

        # Chỉ đánh giá val để theo dõi; test chạy riêng bằng scripts/evaluate.py trên checkpoint đã chọn.
        metrics = evaluate(model, accelerator, val_dataset, news,
                           data_cfg.eval_batch_size, data_cfg.num_workers)
        accelerator.log({f"val/{name}": value for name, value in metrics.items()}, step=global_step)
        checkpoint_dir = accelerator.save_state()
        logger.info("epoch %d step %d val: %s | saved %s", epoch, global_step,
                    ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()),
                    checkpoint_dir)
        if reached_max_steps:
            break

    accelerator.end_training()
