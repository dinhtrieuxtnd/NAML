import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.constants import (
    CANDIDATE_MASK_KEY,
    CANDIDATES_COLUMN,
    HISTORY_COLUMN,
    HISTORY_MASK_KEY,
    INDEXES_KEY,
    LABELS_COLUMN,
    NEWS_CATEGORY_KEY,
    NEWS_PAD_IDX,
    NEWS_ROWS_KEY,
    NEWS_TOKENS_KEY,
)
from src.data.news import NewsFeatures


class TrainCollator:
    """Encodes each distinct news of the batch once.

    history / candidates are re-indexed into the batch's distinct news: 0 = padding,
    1 + i = i-th entry of news_tokens / news_category.
    """

    def __init__(self, news: NewsFeatures) -> None:
        self.news = news

    def __call__(self, items: list[dict[str, torch.Tensor]]) -> dict:
        # history: [B, N], candidates: [B, K + 1]
        history = torch.stack([item[HISTORY_COLUMN] for item in items])
        candidates = torch.stack([item[CANDIDATES_COLUMN] for item in items])

        all_rows = torch.cat([history.flatten(), candidates.flatten()])
        # unique_rows: [U], sorted
        unique_rows = torch.unique(all_rows[all_rows != NEWS_PAD_IDX])
        tokens, category = self.news.lookup(unique_rows)

        def to_local(rows: torch.Tensor) -> torch.Tensor:
            local = torch.searchsorted(unique_rows, rows) + 1
            return local.masked_fill(rows == NEWS_PAD_IDX, 0)

        return {
            NEWS_TOKENS_KEY: tokens,
            NEWS_CATEGORY_KEY: category,
            HISTORY_COLUMN: to_local(history),
            HISTORY_MASK_KEY: history != NEWS_PAD_IDX,
            CANDIDATES_COLUMN: to_local(candidates),
        }


class EvalCollator:
    """Keeps global news rows; candidates are right-padded to the longest impression."""

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        # history: [B, N]
        history = torch.stack([item[HISTORY_COLUMN] for item in items])
        # candidates, labels: [B, C_max]
        candidates = pad_sequence(
            [item[CANDIDATES_COLUMN] for item in items], batch_first=True,
            padding_value=NEWS_PAD_IDX)
        labels = pad_sequence([item[LABELS_COLUMN] for item in items], batch_first=True)
        return {
            HISTORY_COLUMN: history,
            HISTORY_MASK_KEY: history != NEWS_PAD_IDX,
            CANDIDATES_COLUMN: candidates,
            CANDIDATE_MASK_KEY: candidates != NEWS_PAD_IDX,
            LABELS_COLUMN: labels,
            INDEXES_KEY: torch.tensor([item[INDEXES_KEY] for item in items], dtype=torch.long),
        }


class NewsCollator:
    def __init__(self, news: NewsFeatures) -> None:
        self.news = news

    def __call__(self, rows: list[int]) -> dict:
        # rows: [U]
        rows_tensor = torch.tensor(rows, dtype=torch.long)
        tokens, category = self.news.lookup(rows_tensor)
        return {
            NEWS_ROWS_KEY: rows_tensor,
            NEWS_TOKENS_KEY: tokens,
            NEWS_CATEGORY_KEY: category,
        }
