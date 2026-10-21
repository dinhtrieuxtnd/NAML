from typing import Literal

from pydantic import NonNegativeInt, PositiveInt, confloat

from src.utils.config import StrictModel


class NAMLConfig(StrictModel):
    name: Literal["naml"]
    pretrained_name: str
    # 0 = đóng băng toàn bộ PhoBERT; k = chỉ train k layer cuối;
    # k = số layer của PhoBERT = finetune toàn bộ (kể cả embeddings).
    num_trainable_layers: NonNegativeInt
    category_dim: PositiveInt
    query_dim: PositiveInt
    dropout: confloat(ge=0.0, lt=1.0)


class NAMLCNNConfig(StrictModel):
    name: Literal["naml_cnn"]
    # Lấy ma trận input embedding của model này làm word embedding (không chạy transformer).
    pretrained_name: str
    freeze_embedding: bool
    num_filters: PositiveInt
    window_size: PositiveInt
    category_dim: PositiveInt
    query_dim: PositiveInt
    dropout: confloat(ge=0.0, lt=1.0)
