import numpy as np
import pandas as pd

from src.data.constants import (
    CANDIDATES_COLUMN,
    HISTORY_COLUMN,
    IMPRESSION_ID_COLUMN,
    LABELS_COLUMN,
    NEGATIVE_POOL_KEY,
    NEWS_CATEGORY_COLUMN,
    NEWS_ID_COLUMN,
    NEWS_PAD_IDX,
    PAD_TOKEN_IDX,
    POSITIVE_ITEM_KEY,
    USER_ID_COLUMN,
)
from src.data.text import encode


def _dedup_history(history: list[int]) -> list[int]:
    """Keeps only the latest click of each news, in click order."""
    seen = set()
    latest_first = []
    for news_id in reversed(history):
        if news_id not in seen:
            seen.add(news_id)
            latest_first.append(news_id)
    return latest_first[::-1]


def build_train_samples(
    behaviors_df: pd.DataFrame,
) -> tuple[list[dict], dict[str, int]]:
    train_samples = []
    train_counter = {
        "input_impressions": len(behaviors_df),
        "output_samples": 0,
        "empty_history": 0,
        "duplicate_history_clicks": 0,
    }
    for _, behavior in behaviors_df.iterrows():
        if behavior[HISTORY_COLUMN] == []:
            train_counter["empty_history"] += 1
            continue
        history = _dedup_history(behavior[HISTORY_COLUMN])
        train_counter["duplicate_history_clicks"] += len(
            behavior[HISTORY_COLUMN]
        ) - len(history)
        candidates, labels = behavior[CANDIDATES_COLUMN], behavior[LABELS_COLUMN]
        positive_impressions = [
            news for news, label in zip(candidates, labels) if label == 1
        ]
        negative_impressions = [
            news for news, label in zip(candidates, labels) if label == 0
        ]

        for pos_item in positive_impressions:
            train_samples.append(
                {
                    IMPRESSION_ID_COLUMN: behavior[IMPRESSION_ID_COLUMN],
                    USER_ID_COLUMN: behavior[USER_ID_COLUMN],
                    HISTORY_COLUMN: history,
                    NEGATIVE_POOL_KEY: negative_impressions,
                    POSITIVE_ITEM_KEY: pos_item,
                }
            )
            train_counter["output_samples"] += 1
    return train_samples, train_counter


def build_eval_samples(behaviors_df: pd.DataFrame) -> tuple[list[dict], dict[str, int]]:
    eval_samples = []
    eval_counter = {
        "input_impressions": len(behaviors_df),
        "output_samples": 0,
        "empty_history": 0,
        "duplicate_history_clicks": 0,
    }
    for _, behavior in behaviors_df.iterrows():
        if behavior[HISTORY_COLUMN] == []:
            eval_counter["empty_history"] += 1
            continue
        history = _dedup_history(behavior[HISTORY_COLUMN])
        eval_counter["duplicate_history_clicks"] += len(behavior[HISTORY_COLUMN]) - len(
            history
        )
        eval_samples.append(
            {
                IMPRESSION_ID_COLUMN: behavior[IMPRESSION_ID_COLUMN],
                USER_ID_COLUMN: behavior[USER_ID_COLUMN],
                HISTORY_COLUMN: history,
                CANDIDATES_COLUMN: behavior[CANDIDATES_COLUMN],
                LABELS_COLUMN: behavior[LABELS_COLUMN],
            }
        )
        eval_counter["output_samples"] += 1
    return eval_samples, eval_counter


def build_token_matrix(
    news_df: pd.DataFrame,
    tokenized: dict[str, list[list[str]]],
    vocab: dict[str, int],
    max_length: dict[str, int],
) -> tuple[dict[str, np.ndarray], dict[int, int], dict[str, dict]]:
    """Returns one int32 matrix [N + 1, L] per text column, rows in news_df order after row 0.

    tokenized[column] holds the tokens of each news. Row NEWS_PAD_IDX is filled entirely with
    PAD_TOKEN_IDX, so the mask of any row is `matrix != PAD_TOKEN_IDX`.
    """
    matrixes = {}
    statistics_token_length = {}
    for column, length in max_length.items():
        # pad_row: [1, L]
        pad_row = np.full((1, length), PAD_TOKEN_IDX, dtype=np.int32)
        # encoded: N x [L] -> [N, L]
        encoded = np.stack(
            [encode(tokens, vocab, length) for tokens in tokenized[column]]
        )
        # pad_row: [1, L], encoded: [N, L] -> [N + 1, L]
        matrixes[column] = np.concatenate([pad_row, encoded], axis=0)

        token_length = pd.Series([len(tokens) for tokens in tokenized[column]])
        statistics_token_length[f"{column}_length"] = {
            "min": token_length.min(),
            "max": token_length.max(),
            "mean": token_length.mean(),
            "std": token_length.std(),
            "p95": token_length.quantile(0.95),
            "max_length": length,
            "truncated_ratio": (token_length > length).mean(),
        }
    first_idx = NEWS_PAD_IDX + 1
    index_matrix = dict(
        zip(
            news_df[NEWS_ID_COLUMN],
            np.arange(first_idx, len(news_df) + first_idx, dtype=np.int32),
        )
    )
    return matrixes, index_matrix, statistics_token_length


def build_category_dict(news_df: pd.DataFrame) -> dict[str, int]:
    categories = sorted(news_df[NEWS_CATEGORY_COLUMN].unique().tolist())
    return {category: i for i, category in enumerate(categories)}


def build_newsId_cateId_mapping(
    news_df: pd.DataFrame, category_dict: dict[str, int]
) -> dict[int, int]:
    return dict(
        zip(news_df[NEWS_ID_COLUMN], news_df[NEWS_CATEGORY_COLUMN].map(category_dict))
    )
