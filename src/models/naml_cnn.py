import torch
from einops import rearrange
from torch import nn

from src.models.naml import AdditiveAttention, CategoryEncoder


class TextCNNEncoder(nn.Module):
    """One text view as in the NAML paper: CNN + ReLU -> dropout -> word attention."""

    def __init__(
        self, embedding_dim: int, num_filters: int, window_size: int, query_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.cnn = nn.Conv1d(embedding_dim, num_filters, window_size, padding=window_size // 2)
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


class CNNNewsEncoder(nn.Module):
    """NAML news encoder with static word embeddings (shared by all text views) + CNN per view.

    All views are combined by view-level attention into one vector of size num_filters.
    """

    def __init__(
        self,
        word_embedding: torch.Tensor,
        freeze_embedding: bool,
        pad_token_id: int,
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
        self.pad_token_id = pad_token_id
        self.word_embedding = nn.Embedding.from_pretrained(
            word_embedding, freeze=freeze_embedding, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)
        self.text_encoders = nn.ModuleDict({
            view: TextCNNEncoder(
                word_embedding.shape[1], num_filters, window_size, query_dim, dropout)
            for view in text_views
        })
        self.category_encoder = CategoryEncoder(num_categories, category_dim, num_filters, dropout)
        self.view_attention = AdditiveAttention(num_filters, query_dim)

    def forward(self, tokens: dict[str, torch.Tensor], category: torch.Tensor) -> torch.Tensor:
        """tokens: {view: [U, L_view]}, category: [U] -> [U, F]."""
        view_vectors = []
        for view, text_encoder in self.text_encoders.items():
            input_ids = tokens[view]
            # [U, L_view] -> [U, L_view, E]
            embedded = self.dropout(self.word_embedding(input_ids))
            # [U, L_view, E] -> [U, F]
            view_vectors.append(text_encoder(embedded, input_ids != self.pad_token_id))
        # category: [U] -> [U, F]
        view_vectors.append(self.category_encoder(category))
        # views: V x [U, F] -> [U, V, F]
        views = torch.stack(view_vectors, dim=1)
        # [U, V, F] -> [U, F]
        return self.view_attention(views)

    def pretrained_parameters(self) -> list[nn.Parameter]:
        return list(self.word_embedding.parameters())
