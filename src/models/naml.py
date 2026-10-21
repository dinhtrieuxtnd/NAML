import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn
from transformers import PreTrainedModel


class AdditiveAttention(nn.Module):
    """a_i = q^T tanh(V x_i + v), alpha = softmax(a) over unmasked positions, out = sum alpha_i x_i."""

    def __init__(self, in_dim: int, query_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_dim, query_dim)
        self.query = nn.Linear(query_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, S, D], mask: [B, S] (True = keep) -> [B, D]."""
        # scores: [B, S, D] -> [B, S, 1] -> [B, S]
        scores = rearrange(self.query(torch.tanh(self.projection(x))), "b s 1 -> b s")
        if mask is not None:
            # finfo.min thay vì -inf để hàng bị mask toàn bộ không sinh NaN.
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        # x: [B, S, D] -> [B, D]
        return einsum(weights, x, "b s, b s d -> b d")


class CategoryEncoder(nn.Module):
    """Category view: embedding -> dropout -> dense + ReLU."""

    def __init__(self, num_categories: int, category_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_categories, category_dim)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Sequential(nn.Linear(category_dim, out_dim), nn.ReLU())

    def forward(self, category: torch.Tensor) -> torch.Tensor:
        """category: [U] -> [U, out_dim]."""
        # [U] -> [U, category_dim] -> [U, out_dim]
        return self.dense(self.dropout(self.embedding(category)))


class NewsEncoder(nn.Module):
    """Text views: PhoBERT (shared) + word attention per view; category view: embedding + dense.

    All views are combined by view-level attention into one vector of size hidden_size.
    """

    def __init__(
        self,
        encoder: PreTrainedModel,
        text_views: list[str],
        num_categories: int,
        category_dim: int,
        query_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_size = encoder.config.hidden_size
        self.encoder = encoder
        self.pad_token_id = encoder.config.pad_token_id
        self.dropout = nn.Dropout(dropout)
        self.word_attention = nn.ModuleDict(
            {view: AdditiveAttention(hidden_size, query_dim) for view in text_views})
        self.category_encoder = CategoryEncoder(num_categories, category_dim, hidden_size, dropout)
        self.view_attention = AdditiveAttention(hidden_size, query_dim)

    def forward(self, tokens: dict[str, torch.Tensor], category: torch.Tensor) -> torch.Tensor:
        """tokens: {view: [U, L_view]}, category: [U] -> [U, D]."""
        view_vectors = []
        for view, attention in self.word_attention.items():
            input_ids = tokens[view]
            mask = input_ids != self.pad_token_id
            # hidden: [U, L_view, D]
            hidden = self.encoder(input_ids=input_ids, attention_mask=mask).last_hidden_state
            # [U, L_view, D] -> [U, D]
            view_vectors.append(attention(self.dropout(hidden), mask))
        # category: [U] -> [U, D]
        view_vectors.append(self.category_encoder(category))
        # views: V x [U, D] -> [U, V, D]
        views = torch.stack(view_vectors, dim=1)
        # [U, V, D] -> [U, D]
        return self.view_attention(views)

    def pretrained_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())


class NAML(nn.Module):
    """news_encoder: any module mapping ({view: [U, L_view]}, category [U]) -> [U, news_dim]."""

    def __init__(self, news_encoder: nn.Module, news_dim: int, query_dim: int) -> None:
        super().__init__()
        self.news_encoder = news_encoder
        self.user_attention = AdditiveAttention(news_dim, query_dim)

    def encode_news(self, tokens: dict[str, torch.Tensor], category: torch.Tensor) -> torch.Tensor:
        """tokens: {view: [U, L_view]}, category: [U] -> [U, D]."""
        return self.news_encoder(tokens, category)

    def encode_user(self, history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """history: [B, N, D], history_mask: [B, N] -> [B, D]."""
        return self.user_attention(history, history_mask)

    @staticmethod
    def score(user: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """user: [B, D], candidates: [B, C, D] -> [B, C]."""
        return einsum(user, candidates, "b d, b c d -> b c")

    def forward(
        self,
        news_tokens: dict[str, torch.Tensor],
        news_category: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Scores candidates against the user built from history.

        news_tokens: {view: [U, L_view]}, news_category: [U] are the batch's distinct news.
        history: [B, N], candidates: [B, C] index them as 1 + i (0 = padding).
        history_mask: [B, N]. Returns scores [B, C].
        """
        # news_table: [U, D] -> [U + 1, D], row 0 = padding
        news_table = F.pad(self.encode_news(news_tokens, news_category), (0, 0, 1, 0))
        # history: [B, N] -> [B, N, D]; user: [B, D]
        user = self.encode_user(news_table[history], history_mask)
        # candidates: [B, C] -> [B, C, D]
        return self.score(user, news_table[candidates])


def freeze_encoder(encoder: PreTrainedModel, num_trainable_layers: int) -> None:
    """Trains only the last `num_trainable_layers` transformer layers; all layers = full finetune."""
    layers = encoder.encoder.layer
    if num_trainable_layers > len(layers):
        raise ValueError(
            f"num_trainable_layers={num_trainable_layers} > {len(layers)} layers of the encoder")
    full_finetune = num_trainable_layers == len(layers)
    encoder.embeddings.requires_grad_(full_finetune)
    for i, layer in enumerate(layers):
        layer.requires_grad_(i >= len(layers) - num_trainable_layers)
