from typing import Annotated, Literal

from pydantic import Field, PositiveInt, confloat

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


class NAMLPopularityConfig(NAMLConfig):
    """NAML + popularity head: s_i = u.n_i + MLP([t, Linear(u), popularity_i])."""

    name: Literal["naml_popularity"]
    # Số chiều của user sau khi chiếu, trước khi ghép với thời điểm và popularity.
    popularity_user_dim: PositiveInt
    popularity_hidden_dim: PositiveInt


ModelConfig = Annotated[NAMLConfig | NAMLPopularityConfig, Field(discriminator="name")]
