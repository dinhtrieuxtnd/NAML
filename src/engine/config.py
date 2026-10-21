from typing import Literal

from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from src.data.config import DataConfig
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
    log_every: PositiveInt
    # Dừng sau N optimizer step (dùng cho smoke test); None = chạy đủ epochs.
    max_train_steps: PositiveInt | None
    # Ghi đè checkpoints/latest mỗi N optimizer step để resume giữa epoch; None = chỉ lưu cuối epoch.
    save_every_steps: PositiveInt | None


class TrainConfig(StrictModel):
    run: RunConfig
    data: DataConfig
    model: NAMLConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
