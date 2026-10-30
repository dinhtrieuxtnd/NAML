import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import nn

from src.data.constants import PAD_TOKEN_IDX


class AdditiveAttention(nn.Module):
    """a_i = q^T tanh(V x_i + v), alpha = softmax(a) over unmasked positions, out = sum alpha_i x_i."""

    def __init__(self, in_dim: int, query_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_dim, query_dim)
        self.query = nn.Linear(query_dim, 1, bias=False)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def __init__(
        self, num_categories: int, category_dim: int, out_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_categories, category_dim)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Sequential(nn.Linear(category_dim, out_dim), nn.ReLU())

    def forward(self, category: torch.Tensor) -> torch.Tensor:
        """category: [U] -> [U, out_dim]."""
        # [U] -> [U, category_dim] -> [U, out_dim]
        return self.dense(self.dropout(self.embedding(category)))


class TextCNNEncoder(nn.Module):
    """One text view as in the NAML paper: CNN + ReLU -> dropout -> word attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_filters: int,
        window_size: int,
        query_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cnn = nn.Conv1d(
            embedding_dim, num_filters, window_size, padding=window_size // 2
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = AdditiveAttention(num_filters, query_dim)

    def forward(self, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """embedded: [U, L, E], mask: [U, L] -> [U, num_filters]."""
        # [U, L, E] -> [U, E, L]
        embedded = rearrange(embedded, "u l e -> u e l")
        # [U, E, L] -> [U, F, L] -> [U, L, F]
        context = rearrange(torch.relu(self.cnn(embedded)), "u f l -> u l f")
        # [U, L, F] -> [U, F]
        return self.attention(self.dropout(context), mask)


class NewsEncoder(nn.Module):
    """NAML news encoder: word embedding (shared by all text views) + CNN per view.

    All views are combined by view-level attention into one vector of size num_filters.
    """

    def __init__(
        self,
        word_embedding: torch.Tensor,
        freeze_embedding: bool,
        text_views: list[str],
        num_categories: int,
        num_filters: int,
        window_size: int,
        category_dim: int,
        query_dim: int,
        dropout: float,
    ) -> None:
        """word_embedding: [V, E]."""
        super().__init__()
        self.word_embedding = nn.Embedding.from_pretrained(
            word_embedding, freeze=freeze_embedding, padding_idx=PAD_TOKEN_IDX
        )
        self.dropout = nn.Dropout(dropout)
        self.text_encoders = nn.ModuleDict(
            {
                view: TextCNNEncoder(
                    word_embedding.shape[1],
                    num_filters,
                    window_size,
                    query_dim,
                    dropout,
                )
                for view in text_views
            }
        )
        self.category_encoder = CategoryEncoder(
            num_categories, category_dim, num_filters, dropout
        )
        self.view_attention = AdditiveAttention(num_filters, query_dim)

    def forward(
        self, tokens: dict[str, torch.Tensor], category: torch.Tensor
    ) -> torch.Tensor:
        """tokens: {view: [U, L_view]}, category: [U] -> [U, F]."""
        view_vectors = []
        for view, text_encoder in self.text_encoders.items():
            input_ids = tokens[view]
            # [U, L_view] -> [U, L_view, E]
            embedded = self.dropout(self.word_embedding(input_ids))
            # [U, L_view, E] -> [U, F]
            view_vectors.append(text_encoder(embedded, input_ids != PAD_TOKEN_IDX))
        # category: [U] -> [U, F]
        view_vectors.append(self.category_encoder(category))
        # views: V x [U, F] -> [U, V, F]
        views = torch.stack(view_vectors, dim=1)
        # [U, V, F] -> [U, F]
        return self.view_attention(views)


class PopularityHead(nn.Module):
    """q_i = MLP([t, Linear(u), popularity_i]), t = [sin, cos] of the hour of day.

    Time, user and candidate popularity are combined in one MLP, so each user / time of day
    gets its own way of using popularity. Within an impression only popularity_i varies, so
    this term ranks candidates by popularity only.
    """

    def __init__(self, news_dim: int, user_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.user_projection = nn.Linear(news_dim, user_dim)
        # input: 2 (hour sin/cos) + user_dim + 1 (popularity)
        self.mlp = nn.Sequential(
            nn.Linear(2 + user_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, user: torch.Tensor, popularity: torch.Tensor, hour: torch.Tensor
    ) -> torch.Tensor:
        """user: [B, D], popularity: [B, C], hour: [B] -> [B, C]."""
        angle = 2 * math.pi * hour / 24
        # time: [B] -> [B, 2]
        time = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        # user: [B, D] -> [B, P]
        user = self.user_projection(user)
        num_candidates = popularity.shape[1]
        # [B, 2] + [B, P] + [B, C] -> [B, C, 2 + P + 1]
        features = torch.cat(
            [
                repeat(time, "b t -> b c t", c=num_candidates),
                repeat(user, "b p -> b c p", c=num_candidates),
                rearrange(popularity, "b c -> b c 1"),
            ],
            dim=-1,
        )
        # [B, C, 2 + P + 1] -> [B, C, 1] -> [B, C]
        return rearrange(self.mlp(features), "b c 1 -> b c")


class NAML(nn.Module):
    """news_encoder: any module mapping ({view: [U, L_view]}, category [U]) -> [U, news_dim].

    popularity_head: None scores candidates by u.n only (plain NAML); otherwise any module
    mapping (user [B, D], popularity [B, C], hour [B]) -> [B, C], added to u.n.
    """

    def __init__(
        self,
        news_encoder: nn.Module,
        news_dim: int,
        query_dim: int,
        popularity_head: nn.Module | None,
    ) -> None:
        super().__init__()
        self.news_encoder = news_encoder
        self.user_attention = AdditiveAttention(news_dim, query_dim)
        self.popularity_head = popularity_head

    def encode_news(
        self, tokens: dict[str, torch.Tensor], category: torch.Tensor
    ) -> torch.Tensor:
        """tokens: {view: [U, L_view]}, category: [U] -> [U, D]."""
        return self.news_encoder(tokens, category)

    def encode_user(
        self, history: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        """history: [B, N, D], history_mask: [B, N] -> [B, D]."""
        return self.user_attention(history, history_mask)

    def score(
        self,
        user: torch.Tensor,
        candidates: torch.Tensor,
        popularity: torch.Tensor,
        hour: torch.Tensor,
    ) -> torch.Tensor:
        """user: [B, D], candidates: [B, C, D], popularity: [B, C], hour: [B] -> [B, C]."""
        scores = einsum(user, candidates, "b d, b c d -> b c")
        if self.popularity_head is None:
            return scores
        return scores + self.popularity_head(user, popularity, hour)

    def forward(
        self,
        news_tokens: dict[str, torch.Tensor],
        news_category: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
        popularity: torch.Tensor,
        hour: torch.Tensor,
    ) -> torch.Tensor:
        """Scores candidates against the user built from history.

        news_tokens: {view: [U, L_view]}, news_category: [U] are the batch's distinct news.
        history: [B, N], candidates: [B, C] index them as 1 + i (0 = padding).
        history_mask: [B, N], popularity: [B, C], hour: [B]. Returns scores [B, C].
        """
        # news_table: [U, D] -> [U + 1, D], row 0 = padding
        news_table = F.pad(self.encode_news(news_tokens, news_category), (0, 0, 1, 0))
        # history: [B, N] -> [B, N, D]; user: [B, D]
        user = self.encode_user(news_table[history], history_mask)
        # candidates: [B, C] -> [B, C, D]
        return self.score(user, news_table[candidates], popularity, hour)
