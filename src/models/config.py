from typing import Literal

from pydantic import PositiveInt, confloat

from src.utils.config import StrictModel


class NAMLConfig(StrictModel):
    name: Literal["naml"]
    # Word embedding khởi tạo từ embedding.npy của processed dataset (số chiều lấy từ ma trận).
    freeze_embedding: bool
    num_filters: PositiveInt
    window_size: PositiveInt
    category_dim: PositiveInt
    query_dim: PositiveInt
    dropout: confloat(ge=0.0, lt=1.0)
