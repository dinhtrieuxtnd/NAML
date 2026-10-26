from typing import Literal

from pydantic import (
    DirectoryPath,
    FilePath,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    confloat,
)

from src.data.config import DataConfig
from src.data.constants import METRIC_NAMES
from src.models.config import NAMLConfig
from src.utils.config import RunConfig, StrictModel


class LossConfig(StrictModel):
    name: str


class OptimizerConfig(StrictModel):
    # AdamW, cùng một lr cho mọi tham số (kể cả word embedding).
    lr: PositiveFloat
    weight_decay: NonNegativeFloat


class TrainerConfig(StrictModel):
    seed: NonNegativeInt
    epochs: PositiveInt
    gradient_accumulation_steps: PositiveInt
    mixed_precision: Literal["no", "fp16", "bf16"]
    max_grad_norm: PositiveFloat
    # Metric val dùng chung cho chọn checkpoint best, scheduler và early stopping.
    monitor: Literal[METRIC_NAMES]
    # Dừng sau N optimizer step (dùng cho smoke test); None = chạy đủ epochs.
    max_train_steps: PositiveInt | None
    # Ghi đè checkpoints/latest mỗi N optimizer step để resume giữa epoch; None = chỉ lưu cuối epoch.
    save_every_steps: PositiveInt | None


class SchedulerConfig(StrictModel):
    # ReduceLROnPlateau theo trainer.monitor, step một lần mỗi epoch.
    factor: confloat(gt=0.0, lt=1.0)
    patience: NonNegativeInt
    min_lr: NonNegativeFloat


class EarlyStoppingConfig(StrictModel):
    # Dừng sau `patience` epoch liên tiếp monitor không vượt best + min_delta.
    patience: PositiveInt
    min_delta: NonNegativeFloat


class BaselineDataConfig(StrictModel):
    processed_dir: DirectoryPath
    # Behaviors thô đã dùng để tạo processed_dir; chỉ đọc thời gian impression và click.
    train_behaviors: FilePath
    val_behaviors: FilePath
    test_behaviors: FilePath


class BaselineConfig(StrictModel):
    run: RunConfig
    data: BaselineDataConfig
    seed: NonNegativeInt
    # popularity: số click vào tin trong N phút ngay trước thời điểm impression.
    popularity_window_minutes: PositiveInt


class TrainConfig(StrictModel):
    run: RunConfig
    data: DataConfig
    model: NAMLConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    early_stopping: EarlyStoppingConfig
    trainer: TrainerConfig
