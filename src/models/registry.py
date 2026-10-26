"""The project's only registry: maps config names to model / loss builders."""

from collections.abc import Callable

import torch
from torch import nn

from src.models.config import NAMLConfig
from src.models.naml import NAML, NewsEncoder


def build_naml(
    config: NAMLConfig,
    num_categories: int,
    text_views: list[str],
    word_embedding: torch.Tensor,
) -> NAML:
    """word_embedding: [V, E]."""
    news_encoder = NewsEncoder(
        word_embedding=word_embedding,
        freeze_embedding=config.freeze_embedding,
        text_views=text_views,
        num_categories=num_categories,
        num_filters=config.num_filters,
        window_size=config.window_size,
        category_dim=config.category_dim,
        query_dim=config.query_dim,
        dropout=config.dropout,
    )
    return NAML(news_encoder, news_dim=config.num_filters, query_dim=config.query_dim)


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "naml": build_naml,
}

LOSS_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
}


def build_model(
    config: NAMLConfig,
    num_categories: int,
    text_views: list[str],
    word_embedding: torch.Tensor,
) -> nn.Module:
    """word_embedding: [V, E]."""
    return MODEL_REGISTRY[config.name](
        config, num_categories, text_views, word_embedding
    )


def build_loss(name: str) -> nn.Module:
    return LOSS_REGISTRY[name]()
