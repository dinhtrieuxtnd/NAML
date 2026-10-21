from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt

from src.data.config import DataConfig
from src.models.config import NAMLCNNConfig, NAMLConfig
from src.utils.config import StrictModel


class LossConfig(StrictModel):
    name: str


class OptimizerConfig(StrictModel):
    # lr cho phần head; encoder_lr cho phần pretrained được train (layer PhoBERT hoặc word embedding).
    lr: PositiveFloat
    encoder_lr: PositiveFloat


class TrainerConfig(StrictModel):
    seed: NonNegativeInt
    epochs: PositiveInt
    gradient_accumulation_steps: PositiveInt
    mixed_precision: Literal["no", "fp16", "bf16"]
    log_every: PositiveInt
    # Dừng sau N optimizer step (dùng cho smoke test); None = chạy đủ epochs.
    max_train_steps: PositiveInt | None
    checkpoint_total_limit: PositiveInt
    output_root: Path
    run_name: str


class TrainConfig(StrictModel):
    data: DataConfig
    model: Annotated[NAMLConfig | NAMLCNNConfig, Field(discriminator="name")]
    loss: LossConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
