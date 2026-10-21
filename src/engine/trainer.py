import datetime
import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from torch.utils.data import DataLoader

from src.data.collator import TrainCollator
from src.data.constants import (
    CANDIDATES_COLUMN,
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
from src.utils.config import load_config
from src.utils.io import load_json, save_json
from src.utils.run import (
    CHECKPOINTS_DIR,
    CONFIG_FILE,
    EPOCH_CHECKPOINT_TEMPLATE,
    LATEST_CHECKPOINT,
    TRAINING_STATE_FILE,
    start_run,
)

logger = logging.getLogger(__name__)

# Các trường phải giữ nguyên khi resume, nếu không checkpoint không còn khớp với run.
_RESUME_FIXED_FIELDS = {
    "model": True,
    "loss": True,
    "optimizer": True,
    "data": {"processed_dir", "text_views", "history_len", "num_negatives", "train_batch_size"},
    "trainer": {"seed", "gradient_accumulation_steps"},
}


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


def _check_resume_config(config: TrainConfig, run_dir: Path) -> None:
    saved = TrainConfig.model_validate(load_config(run_dir / CONFIG_FILE))
    new_dump, saved_dump = config.model_dump(mode="json"), saved.model_dump(mode="json")
    mismatches = [
        f"{section}.{field}" if field else section
        for section, fields in _RESUME_FIXED_FIELDS.items()
        for field in ([None] if fields is True else sorted(fields))
        if (new_dump[section] if field is None else new_dump[section][field])
        != (saved_dump[section] if field is None else saved_dump[section][field])
    ]
    if mismatches:
        raise ValueError(f"Cannot resume {run_dir}: config differs in {', '.join(mismatches)}")


def run_training(config: TrainConfig, resume_from: Path | None = None) -> None:
    """Trains from scratch, or continues from a checkpoint saved by a previous call.

    A new run goes to run.output_root/<run.name>_<timestamp>; a resumed run keeps writing to
    the checkpoint's run directory.
    """
    if resume_from is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = config.run.output_root / f"{config.run.name}_{timestamp}"
    else:
        run_dir = resume_from.parent.parent
    with start_run("train", run_dir, config, resume=resume_from is not None):
        _train(config, run_dir, resume_from)


def _train(config: TrainConfig, run_dir: Path, resume_from: Path | None) -> None:
    trainer_cfg, data_cfg = config.trainer, config.data
    if resume_from is not None:
        _check_resume_config(config, run_dir)
    checkpoints_dir = run_dir / CHECKPOINTS_DIR

    accelerator = Accelerator(
        mixed_precision=trainer_cfg.mixed_precision,
        gradient_accumulation_steps=trainer_cfg.gradient_accumulation_steps,
        # Thứ tự shuffle chỉ phụ thuộc (seed, epoch) để resume giữa epoch bỏ qua đúng các batch đã train.
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True, data_seed=trainer_cfg.seed),
    )
    set_seed(trainer_cfg.seed)

    news = NewsFeatures(data_cfg.processed_dir, data_cfg.text_views, config.model.pretrained_name)
    train_dataset = TrainDataset(
        load_split_samples(data_cfg.processed_dir, "train"), news,
        data_cfg.history_len, data_cfg.num_negatives, seed=trainer_cfg.seed)
    val_dataset = EvalDataset(
        load_split_samples(data_cfg.processed_dir, "val")[:data_cfg.max_eval_impressions],
        news, data_cfg.history_len)
    logger.info("Samples: train=%d, val=%d", len(train_dataset), len(val_dataset))

    model = build_model(config.model, news.num_categories, data_cfg.text_views)
    loss_fn = build_loss(config.loss.name)
    optimizer = _build_optimizer(model, config)
    # generator riêng: DataLoader không lấy seed từ RNG toàn cục của torch (giữ dropout khớp khi resume).
    train_loader = DataLoader(
        train_dataset, batch_size=data_cfg.train_batch_size, shuffle=True,
        collate_fn=TrainCollator(news), num_workers=data_cfg.num_workers,
        generator=torch.Generator().manual_seed(trainer_cfg.seed))
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Vị trí train tiếp theo: epoch, số batch đã train trong epoch đó, số optimizer step.
    state = {"epoch": 0, "batches_in_epoch": 0, "global_step": 0}
    if resume_from is not None:
        accelerator.load_state(str(resume_from))
        state = load_json(resume_from / TRAINING_STATE_FILE)
        logger.info("Resumed from %s: %s", resume_from, state)

    def save_checkpoint(name: str, epoch: int, batches_in_epoch: int, global_step: int) -> None:
        checkpoint_dir = checkpoints_dir / name
        accelerator.save_state(str(checkpoint_dir))
        if accelerator.is_main_process:
            save_json({"epoch": epoch, "batches_in_epoch": batches_in_epoch,
                       "global_step": global_step}, checkpoint_dir / TRAINING_STATE_FILE)
        logger.info("Saved %s (epoch %d, batch %d, step %d)",
                    checkpoint_dir, epoch, batches_in_epoch, global_step)

    global_step = state["global_step"]
    if trainer_cfg.max_train_steps is not None and global_step >= trainer_cfg.max_train_steps:
        logger.info("Already at step %d >= max_train_steps, nothing to train", global_step)
        accelerator.end_training()
        return

    for epoch in range(state["epoch"], trainer_cfg.epochs):
        model.train()
        skipped = state["batches_in_epoch"] if epoch == state["epoch"] else 0
        epoch_loader = accelerator.skip_first_batches(train_loader, skipped)
        epoch_loader.set_epoch(epoch)
        train_dataset.set_epoch(epoch)
        batches_in_epoch = skipped
        reached_max_steps = False
        for batch in epoch_loader:
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
            batches_in_epoch += 1

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % trainer_cfg.log_every == 0:
                    logger.info("epoch %d step %d loss %.4f", epoch, global_step, loss.item())
                if global_step == trainer_cfg.max_train_steps:
                    reached_max_steps = True
                    break
                if (trainer_cfg.save_every_steps is not None
                        and global_step % trainer_cfg.save_every_steps == 0):
                    save_checkpoint(LATEST_CHECKPOINT, epoch, batches_in_epoch, global_step)

        # Chỉ đánh giá val để theo dõi; test chạy riêng bằng scripts/evaluate.py trên checkpoint đã chọn.
        # fork_rng: eval không làm lệch RNG của train, nên resume cho kết quả giống chạy liên tục.
        cuda_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            metrics = evaluate(model, accelerator, val_dataset, news,
                               data_cfg.eval_batch_size, data_cfg.num_workers)
        logger.info("epoch %d step %d val: %s", epoch, global_step,
                    ", ".join(f"{name}={value:.4f}" for name, value in metrics.items()))
        if reached_max_steps:
            save_checkpoint(LATEST_CHECKPOINT, epoch, batches_in_epoch, global_step)
            break
        save_checkpoint(EPOCH_CHECKPOINT_TEMPLATE.format(epoch=epoch), epoch + 1, 0, global_step)

    accelerator.end_training()
