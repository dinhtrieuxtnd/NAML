"""The project's only registry: maps config names to model / loss builders."""

from collections.abc import Callable

from torch import nn
from transformers import AutoModel

from src.models.config import NAMLCNNConfig, NAMLConfig
from src.models.naml import NAML, NewsEncoder, freeze_encoder
from src.models.naml_cnn import CNNNewsEncoder


def build_naml(config: NAMLConfig, num_categories: int, text_views: list[str]) -> NAML:
    encoder = AutoModel.from_pretrained(config.pretrained_name, add_pooling_layer=False)
    freeze_encoder(encoder, config.num_trainable_layers)
    news_encoder = NewsEncoder(
        encoder=encoder,
        text_views=text_views,
        num_categories=num_categories,
        category_dim=config.category_dim,
        query_dim=config.query_dim,
        dropout=config.dropout,
    )
    return NAML(news_encoder, news_dim=encoder.config.hidden_size, query_dim=config.query_dim)


def build_naml_cnn(config: NAMLCNNConfig, num_categories: int, text_views: list[str]) -> NAML:
    # Chỉ giữ ma trận input embedding [V, E] của PhoBERT, bỏ phần transformer.
    pretrained = AutoModel.from_pretrained(config.pretrained_name, add_pooling_layer=False)
    word_embedding = pretrained.get_input_embeddings().weight.detach().clone()
    pad_token_id = pretrained.config.pad_token_id
    del pretrained
    news_encoder = CNNNewsEncoder(
        word_embedding=word_embedding,
        freeze_embedding=config.freeze_embedding,
        pad_token_id=pad_token_id,
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
    "naml_cnn": build_naml_cnn,
}

LOSS_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "cross_entropy": nn.CrossEntropyLoss,
}


def build_model(
    config: NAMLConfig | NAMLCNNConfig, num_categories: int, text_views: list[str]
) -> nn.Module:
    return MODEL_REGISTRY[config.name](config, num_categories, text_views)


def build_loss(name: str) -> nn.Module:
    return LOSS_REGISTRY[name]()
