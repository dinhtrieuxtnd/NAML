import logging
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.data.collator import TrainCollator
from src.data.constants import (
    CANDIDATES_COLUMN,
    HISTORY_COLUMN,
    HISTORY_MASK_KEY,
    HOUR_KEY,
    METRIC_NAMES,
    NEWS_CATEGORY_KEY,
    NEWS_TOKENS_KEY,
    POPULARITY_KEY,
    STATISTICS_FILE,
)
from src.data.dataset import EvalDataset, TrainDataset, load_split_samples
from src.data.news import NewsFeatures
from src.engine.config import TrainConfig
from src.engine.evaluate import evaluate
from src.engine.plots import plot_history
from src.models.registry import build_loss, build_model
from src.utils.config import load_config, save_config
from src.utils.io import load_json, save_json
from src.utils.run import (
    BEST_CHECKPOINT,
    CHECKPOINTS_DIR,
    CONFIG_FILE,
    HISTORY_FILE,
    LATEST_CHECKPOINT,
    METRICS_FILE,
    PLOTS_DIR,
    TRAINING_STATE_FILE,
    new_run_dir,
    start_run,
)

logger = logging.getLogger(__name__)

# Các trường phải giữ nguyên khi resume, nếu không checkpoint không còn khớp với run.
# scheduler cố định vì load_state nạp lại factor/patience/min_lr đã lưu.
_RESUME_FIXED_FIELDS = {
    "model": True,
    "loss": True,
    "optimizer": True,
    "scheduler": True,
    "data": {
        "processed_dir",
        "text_views",
        "history_len",
        "num_negatives",
        "train_batch_size",
        "max_train_samples",
    },
    "trainer": {"seed", "gradient_accumulation_steps", "monitor"},
}

# Tên viết tắt của metric trong dòng log mỗi epoch (metric không có ở đây giữ nguyên tên).
_LOG_ABBREVIATIONS = {"impression_auc": "imp_auc"}


def _build_optimizer(
    model: torch.nn.Module, config: TrainConfig
) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        "Params: total=%d, trainable=%d",
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in trainable),
    )
    return torch.optim.AdamW(
        trainable, lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay
    )


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def _initial_state() -> dict:
    """Everything a resumed run needs besides model/optimizer/scheduler/RNG states.

    epoch / batches_in_epoch: next batch to train; loss_sum / loss_count: running loss of the
    current epoch; best_value / best_epoch / bad_epochs: early stopping; history: one row per
    finished epoch (as in history.json). best_epoch and history epochs are 1-based.
    """
    return {
        "epoch": 0,
        "batches_in_epoch": 0,
        "global_step": 0,
        "loss_sum": 0.0,
        "loss_count": 0,
        "best_value": None,
        "best_epoch": None,
        "bad_epochs": 0,
        "history": [],
    }


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
        raise ValueError(
            f"Cannot resume {run_dir}: config differs in {', '.join(mismatches)}"
        )


def run_training(config: TrainConfig, resume_from: Path | None = None) -> None:
    """Trains from scratch, or continues from a checkpoint saved by a previous call.

    A new run goes to run.output_root/<run.name>_<timestamp>; a resumed run keeps writing to
    the checkpoint's run directory.
    """
    if resume_from is None:
        run_dir = new_run_dir(config.run)
    else:
        run_dir = resume_from.parent.parent
    with start_run("train", run_dir, config, resume=resume_from is not None):
        _train(config, run_dir, resume_from)


def _train(config: TrainConfig, run_dir: Path, resume_from: Path | None) -> None:
    trainer_cfg, data_cfg, monitor = config.trainer, config.data, config.trainer.monitor
    if resume_from is not None:
        _check_resume_config(config, run_dir)
    start_time = time.perf_counter()
    checkpoints_dir = run_dir / CHECKPOINTS_DIR

    accelerator = Accelerator(
        mixed_precision=trainer_cfg.mixed_precision,
        gradient_accumulation_steps=trainer_cfg.gradient_accumulation_steps,
        # Thứ tự shuffle chỉ phụ thuộc (seed, epoch) để resume giữa epoch bỏ qua đúng các batch đã train.
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True, data_seed=trainer_cfg.seed
        ),
    )
    set_seed(trainer_cfg.seed)
    logger.info(
        "Run %s: device=%s, processes=%d, mixed_precision=%s",
        config.run.name,
        accelerator.device,
        accelerator.num_processes,
        trainer_cfg.mixed_precision,
    )

    statistics = load_json(data_cfg.processed_dir / STATISTICS_FILE)
    logger.info(
        "Embedding: vocab=%d, token coverage=%.4f",
        statistics["vocab_size"],
        statistics["embedding"]["token_coverage"],
    )
    news = NewsFeatures(data_cfg.processed_dir, data_cfg.text_views)
    train_dataset = TrainDataset(
        load_split_samples(data_cfg.processed_dir, "train")[
            : data_cfg.max_train_samples
        ],
        news,
        data_cfg.history_len,
        data_cfg.num_negatives,
        seed=trainer_cfg.seed,
    )
    val_dataset = EvalDataset(
        load_split_samples(data_cfg.processed_dir, "val")[
            : data_cfg.max_eval_impressions
        ],
        news,
        data_cfg.history_len,
    )
    logger.info("Samples: train=%d, val=%d", len(train_dataset), len(val_dataset))

    model = build_model(
        config.model, news.num_categories, data_cfg.text_views, news.embedding
    )
    loss_fn = build_loss(config.loss.name)
    optimizer = _build_optimizer(model, config)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler.factor,
        patience=config.scheduler.patience,
        min_lr=config.scheduler.min_lr,
    )
    # generator riêng: DataLoader không lấy seed từ RNG toàn cục của torch (giữ dropout khớp khi resume).
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.train_batch_size,
        shuffle=True,
        collate_fn=TrainCollator(news),
        num_workers=data_cfg.num_workers,
        generator=torch.Generator().manual_seed(trainer_cfg.seed),
    )
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    # Scheduler step theo epoch nên không qua prepare, chỉ cần lưu/nạp cùng checkpoint.
    accelerator.register_for_checkpointing(scheduler)

    state = _initial_state()
    if resume_from is not None:
        accelerator.load_state(str(resume_from))
        state = load_json(resume_from / TRAINING_STATE_FILE)
        logger.info(
            "Resumed from %s: epoch=%d, batch=%d, step=%d",
            resume_from,
            state["epoch"] + 1,
            state["batches_in_epoch"],
            state["global_step"],
        )

    def save_checkpoint(name: str, metrics: dict[str, float] | None = None) -> None:
        checkpoint_dir = checkpoints_dir / name
        accelerator.save_state(str(checkpoint_dir))
        if accelerator.is_main_process:
            save_json(state, checkpoint_dir / TRAINING_STATE_FILE)
            save_config(config.model_dump(mode="json"), checkpoint_dir / CONFIG_FILE)
            if metrics is not None:
                save_json(metrics, checkpoint_dir / METRICS_FILE)

    stop_training = False
    if (
        trainer_cfg.max_train_steps is not None
        and state["global_step"] >= trainer_cfg.max_train_steps
    ):
        logger.info(
            "Already at step %d >= max_train_steps, nothing to train",
            state["global_step"],
        )
        stop_training = True
    elif state["bad_epochs"] >= config.early_stopping.patience:
        logger.info("Early stopping was already triggered, nothing to train")
        stop_training = True

    cuda_devices = [accelerator.device] if accelerator.device.type == "cuda" else []
    with logging_redirect_tqdm():
        for epoch in range(state["epoch"], trainer_cfg.epochs):
            if stop_training:
                break
            epoch_start = time.perf_counter()
            model.train()
            epoch_loader = accelerator.skip_first_batches(
                train_loader, state["batches_in_epoch"]
            )
            epoch_loader.set_epoch(epoch)
            train_dataset.set_epoch(epoch)
            progress = tqdm(
                epoch_loader,
                desc=f"epoch {epoch + 1}",
                total=len(train_loader),
                initial=state["batches_in_epoch"],
                leave=False,
                disable=None if accelerator.is_main_process else True,
            )
            for batch in progress:
                with accelerator.accumulate(model):
                    # scores: [B, K + 1], the positive is at position 0
                    scores = model(
                        news_tokens=batch[NEWS_TOKENS_KEY],
                        news_category=batch[NEWS_CATEGORY_KEY],
                        history=batch[HISTORY_COLUMN],
                        history_mask=batch[HISTORY_MASK_KEY],
                        candidates=batch[CANDIDATES_COLUMN],
                        popularity=batch[POPULARITY_KEY],
                        hour=batch[HOUR_KEY],
                    )
                    target = torch.zeros(
                        scores.shape[0], dtype=torch.long, device=scores.device
                    )
                    loss = loss_fn(scores, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss {loss.item()} at step {state['global_step']}"
                        )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(), trainer_cfg.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                # Loss trung bình trên mọi process.
                batch_loss = accelerator.gather(loss.detach()).mean().item()
                state["loss_sum"] += batch_loss
                state["loss_count"] += 1
                state["batches_in_epoch"] += 1
                progress.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    lr=f"{_current_lr(optimizer):.2e}",
                    refresh=False,
                )
                if accelerator.sync_gradients:
                    state["global_step"] += 1
                    if state["global_step"] == trainer_cfg.max_train_steps:
                        # Dừng giữa epoch, không evaluate: resume train tiếp rồi mới đóng epoch.
                        save_checkpoint(LATEST_CHECKPOINT)
                        logger.info(
                            "Reached max_train_steps=%d at epoch %d, batch %d",
                            state["global_step"],
                            epoch + 1,
                            state["batches_in_epoch"],
                        )
                        stop_training = True
                        break
                    if (
                        trainer_cfg.save_every_steps is not None
                        and state["global_step"] % trainer_cfg.save_every_steps == 0
                    ):
                        save_checkpoint(LATEST_CHECKPOINT)
            if stop_training:
                break

            # Chỉ đánh giá val để theo dõi; test chạy riêng bằng scripts/evaluate.py trên checkpoint đã chọn.
            # fork_rng: eval không làm lệch RNG của train, nên resume cho kết quả giống chạy liên tục.
            with torch.random.fork_rng(devices=cuda_devices):
                metrics, _ = evaluate(
                    model,
                    accelerator,
                    val_dataset,
                    news,
                    data_cfg.eval_batch_size,
                    data_cfg.num_workers,
                )
            lr = _current_lr(optimizer)
            value = metrics[monitor]
            is_best = (
                state["best_value"] is None
                or value > state["best_value"] + config.early_stopping.min_delta
            )
            if is_best:
                state.update(best_value=value, best_epoch=epoch + 1, bad_epochs=0)
            else:
                state["bad_epochs"] += 1
            scheduler.step(value)

            epoch_time = time.perf_counter() - epoch_start
            train_loss = state["loss_sum"] / state["loss_count"]
            state["history"].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "lr": lr,
                    "time": epoch_time,
                    **metrics,
                }
            )
            state.update(
                epoch=epoch + 1, batches_in_epoch=0, loss_sum=0.0, loss_count=0
            )
            if is_best:
                save_checkpoint(
                    BEST_CHECKPOINT, metrics={"epoch": epoch + 1, **metrics}
                )
            save_checkpoint(LATEST_CHECKPOINT)
            if accelerator.is_main_process:
                save_json(state["history"], run_dir / HISTORY_FILE)
                plot_history(state["history"], state["best_epoch"], run_dir / PLOTS_DIR)

            logger.info(
                "epoch=%d loss=%.4f %s lr=%.2e time=%.1fs%s",
                epoch + 1,
                train_loss,
                " ".join(
                    f"{_LOG_ABBREVIATIONS.get(name, name)}={metrics[name]:.4f}"
                    for name in METRIC_NAMES
                ),
                lr,
                epoch_time,
                " best" if is_best else "",
            )
            if state["bad_epochs"] >= config.early_stopping.patience:
                logger.info(
                    "Early stopping: %s has not improved by more than %g for %d epochs",
                    monitor,
                    config.early_stopping.min_delta,
                    state["bad_epochs"],
                )
                break

    if state["best_epoch"] is not None:
        logger.info(
            "Best epoch %d: %s=%.4f", state["best_epoch"], monitor, state["best_value"]
        )
    logger.info("Total time: %.1fs", time.perf_counter() - start_time)
    accelerator.end_training()
