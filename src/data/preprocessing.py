import numpy as np
import pandas as pd

from src.data.constants import (
    CANDIDATES_COLUMN,
    HISTORY_COLUMN,
    HOUR_KEY,
    IMPRESSION_ID_COLUMN,
    LABELS_COLUMN,
    NEGATIVE_POOL_KEY,
    NEGATIVE_POOL_POPULARITY_KEY,
    NEWS_CATEGORY_COLUMN,
    NEWS_ID_COLUMN,
    NEWS_PAD_IDX,
    PAD_TOKEN_IDX,
    POPULARITY_KEY,
    POSITIVE_ITEM_KEY,
    POSITIVE_POPULARITY_KEY,
    TIME_COLUMN,
    USER_ID_COLUMN,
)
from src.data.popularity import ClickCounter, hour_of_day, relative_popularity
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


def _candidate_popularity(
    behavior: pd.Series, clicks: ClickCounter, window_seconds: int
) -> tuple[list[float], int]:
    """Relative popularity of every candidate of the impression (mean over all of them, before
    negative sampling), and how many candidates had no click in the window."""
    candidates = np.asarray(behavior[CANDIDATES_COLUMN], dtype=np.int64)
    times = np.full(len(candidates), behavior[TIME_COLUMN].to_datetime64())
    counts = clicks.count(candidates, times, window_seconds)
    return relative_popularity(counts).tolist(), int((counts == 0).sum())


def _new_counter(behaviors_df: pd.DataFrame) -> dict[str, int]:
    return {
        "input_impressions": len(behaviors_df),
        "output_samples": 0,
        "empty_history": 0,
        "duplicate_history_clicks": 0,
        "candidates": 0,
        "zero_click_candidates": 0,
    }


def build_train_samples(
    behaviors_df: pd.DataFrame, clicks: ClickCounter, window_seconds: int
) -> tuple[list[dict], dict[str, int]]:
    train_samples = []
    train_counter = _new_counter(behaviors_df)
    for _, behavior in behaviors_df.iterrows():
        if behavior[HISTORY_COLUMN] == []:
            train_counter["empty_history"] += 1
            continue
        history = _dedup_history(behavior[HISTORY_COLUMN])
        train_counter["duplicate_history_clicks"] += len(
            behavior[HISTORY_COLUMN]
        ) - len(history)
        candidates, labels = behavior[CANDIDATES_COLUMN], behavior[LABELS_COLUMN]
        popularity, num_zero = _candidate_popularity(behavior, clicks, window_seconds)
        train_counter["candidates"] += len(candidates)
        train_counter["zero_click_candidates"] += num_zero
        positives = [
            (news, pop) for news, label, pop in zip(candidates, labels, popularity) if label == 1
        ]
        negatives = [
            (news, pop) for news, label, pop in zip(candidates, labels, popularity) if label == 0
        ]
        negative_pool = [news for news, _ in negatives]
        negative_pool_popularity = [pop for _, pop in negatives]
        hour = hour_of_day(behavior[TIME_COLUMN])

        for pos_item, pos_popularity in positives:
            train_samples.append(
                {
                    IMPRESSION_ID_COLUMN: behavior[IMPRESSION_ID_COLUMN],
                    USER_ID_COLUMN: behavior[USER_ID_COLUMN],
                    HISTORY_COLUMN: history,
                    NEGATIVE_POOL_KEY: negative_pool,
                    NEGATIVE_POOL_POPULARITY_KEY: negative_pool_popularity,
                    POSITIVE_ITEM_KEY: pos_item,
                    POSITIVE_POPULARITY_KEY: pos_popularity,
                    HOUR_KEY: hour,
                }
            )
            train_counter["output_samples"] += 1
    return train_samples, train_counter


def build_eval_samples(
    behaviors_df: pd.DataFrame, clicks: ClickCounter, window_seconds: int
) -> tuple[list[dict], dict[str, int]]:
    eval_samples = []
    eval_counter = _new_counter(behaviors_df)
    for _, behavior in behaviors_df.iterrows():
        if behavior[HISTORY_COLUMN] == []:
            eval_counter["empty_history"] += 1
            continue
        history = _dedup_history(behavior[HISTORY_COLUMN])
        eval_counter["duplicate_history_clicks"] += len(behavior[HISTORY_COLUMN]) - len(
            history
        )
        popularity, num_zero = _candidate_popularity(behavior, clicks, window_seconds)
        eval_counter["candidates"] += len(popularity)
        eval_counter["zero_click_candidates"] += num_zero
        eval_samples.append(
            {
                IMPRESSION_ID_COLUMN: behavior[IMPRESSION_ID_COLUMN],
                USER_ID_COLUMN: behavior[USER_ID_COLUMN],
                HISTORY_COLUMN: history,
                CANDIDATES_COLUMN: behavior[CANDIDATES_COLUMN],
                LABELS_COLUMN: behavior[LABELS_COLUMN],
                POPULARITY_KEY: popularity,
                HOUR_KEY: hour_of_day(behavior[TIME_COLUMN]),
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
