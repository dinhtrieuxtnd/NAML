import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.constants import (
    ARTIFACTS_DIR,
    CANDIDATES_COLUMN,
    HISTORY_COLUMN,
    INDEXES_KEY,
    LABELS_COLUMN,
    NEGATIVE_POOL_KEY,
    NEWS_PAD_IDX,
    POSITIVE_ITEM_KEY,
    SAMPLES_FILE_TEMPLATE,
)
from src.data.news import NewsFeatures
from src.utils.io import load_pickle


def load_split_samples(processed_dir: Path, split: str) -> list[dict]:
    return load_pickle(processed_dir / ARTIFACTS_DIR / SAMPLES_FILE_TEMPLATE.format(split=split))


def _history_rows(news: NewsFeatures, history: list[int], history_len: int) -> torch.Tensor:
    """Keeps the last `history_len` clicks and right-pads with NEWS_PAD_IDX -> [history_len]."""
    rows = news.to_rows(history[-history_len:])
    return torch.tensor(rows + [NEWS_PAD_IDX] * (history_len - len(rows)), dtype=torch.long)


class TrainDataset(Dataset):
    """Each item: history [N] and candidates [K + 1] as news rows, the positive at position 0."""

    def __init__(
        self, samples: list[dict], news: NewsFeatures, history_len: int, num_negatives: int
    ) -> None:
        self.samples = samples
        self.news = news
        self.history_len = history_len
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        pool = sample[NEGATIVE_POOL_KEY]
        # Impression có ít hơn K tin âm thì lấy lặp lại.
        if len(pool) >= self.num_negatives:
            negatives = random.sample(pool, self.num_negatives)
        else:
            negatives = random.choices(pool, k=self.num_negatives)
        candidates = self.news.to_rows([sample[POSITIVE_ITEM_KEY], *negatives])
        return {
            HISTORY_COLUMN: _history_rows(self.news, sample[HISTORY_COLUMN], self.history_len),
            CANDIDATES_COLUMN: torch.tensor(candidates, dtype=torch.long),
        }


class EvalDataset(Dataset):
    """Each item: one impression with history [N], candidates [C] and labels [C]."""

    def __init__(self, samples: list[dict], news: NewsFeatures, history_len: int) -> None:
        self.samples = samples
        self.news = news
        self.history_len = history_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        sample = self.samples[idx]
        return {
            HISTORY_COLUMN: _history_rows(self.news, sample[HISTORY_COLUMN], self.history_len),
            CANDIDATES_COLUMN: torch.tensor(
                self.news.to_rows(sample[CANDIDATES_COLUMN]), dtype=torch.long),
            LABELS_COLUMN: torch.tensor(sample[LABELS_COLUMN], dtype=torch.long),
            INDEXES_KEY: idx,
        }

    def news_rows(self) -> list[int]:
        """Sorted rows of every news this split needs (history + candidates), without padding."""
        rows = set()
        for sample in self.samples:
            rows.update(self.news.to_rows(sample[HISTORY_COLUMN][-self.history_len:]))
            rows.update(self.news.to_rows(sample[CANDIDATES_COLUMN]))
        return sorted(rows)


class NewsDataset(Dataset):
    def __init__(self, rows: list[int]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> int:
        return self.rows[idx]
