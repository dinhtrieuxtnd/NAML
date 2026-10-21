import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from transformers import AutoTokenizer

from src.data.config import TextConfig
from src.data.constants import (
    CANDIDATES_COLUMN,
    HISTORY_COLUMN,
    IMPRESSION_ID_COLUMN,
    LABELS_COLUMN,
    NEGATIVE_POOL_KEY,
    NEWS_CATEGORY_COLUMN,
    NEWS_ID_COLUMN,
    NEWS_PAD_IDX,
    POSITIVE_ITEM_KEY,
    USER_ID_COLUMN,
)


def build_train_samples(behaviors_df: pd.DataFrame) -> tuple[list[dict], dict[str, int]]:
    train_samples = []
    train_counter = {
        "input_impressions": len(behaviors_df),
        "output_samples": 0,
        "empty_history": 0,
    }
    for _, behavior in behaviors_df.iterrows():
        if behavior[HISTORY_COLUMN] == []:
            train_counter["empty_history"] += 1
            continue
        candidates, labels = behavior[CANDIDATES_COLUMN], behavior[LABELS_COLUMN]
        positive_impressions = [news for news, label in zip(candidates, labels) if label == 1]
        negative_impressions = [news for news, label in zip(candidates, labels) if label == 0]

        for pos_item in positive_impressions:
            train_samples.append(
                {
                    IMPRESSION_ID_COLUMN: behavior[IMPRESSION_ID_COLUMN],
                    USER_ID_COLUMN: behavior[USER_ID_COLUMN],
                    HISTORY_COLUMN: behavior[HISTORY_COLUMN],
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
    }
    for _, behavior in behaviors_df.iterrows():
        if behavior[HISTORY_COLUMN] == []:
            eval_counter["empty_history"] += 1
            continue
        eval_samples.append(
            {
                IMPRESSION_ID_COLUMN: behavior[IMPRESSION_ID_COLUMN],
                USER_ID_COLUMN: behavior[USER_ID_COLUMN],
                HISTORY_COLUMN: behavior[HISTORY_COLUMN],
                CANDIDATES_COLUMN: behavior[CANDIDATES_COLUMN],
                LABELS_COLUMN: behavior[LABELS_COLUMN],
            }
        )
        eval_counter["output_samples"] += 1
    return eval_samples, eval_counter


def build_token_matrix(
    news_df: pd.DataFrame, text_config: TextConfig, tokenizer_name: str
) -> tuple[dict[str, np.ndarray], dict[int, int], dict[str, dict]]:
    """Returns one int32 matrix [N + 1, L] per text column.

    Row NEWS_PAD_IDX is filled entirely with pad_token_id, so the attention mask of any
    row is `matrix != pad_token_id` and the padding news has no real token.
    """
    matrixes = {}
    statistics_token_length = {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for column, max_length in text_config.max_length.model_dump().items():
        texts = news_df[column]
        if text_config.lowercase:
            texts = texts.str.lower()
        texts = texts.apply(ViTokenizer.tokenize).tolist()

        encoded = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        # pad_row: [1, L]
        pad_row = np.full((1, max_length), tokenizer.pad_token_id)
        # pad_row: [1, L], encoded: [N, L] -> [N + 1, L]
        matrixes[column] = np.concatenate(
            [pad_row, encoded["input_ids"]], axis=0
        ).astype(np.int32)

        token_length = pd.Series(
            [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
        )
        statistics_token_length[f"{column}_length"] = {
            "min": token_length.min(),
            "max": token_length.max(),
            "mean": token_length.mean(),
            "std": token_length.std(),
            "p95": token_length.quantile(0.95),
            "max_length": max_length,
            "truncated_ratio": (token_length > max_length).mean(),
        }
    first_idx = NEWS_PAD_IDX + 1
    index_matrix = dict(
        zip(news_df[NEWS_ID_COLUMN], np.arange(first_idx, len(news_df) + first_idx, dtype=np.int32))
    )
    return matrixes, index_matrix, statistics_token_length


def build_category_dict(news_df: pd.DataFrame) -> dict[str, int]:
    categories = sorted(news_df[NEWS_CATEGORY_COLUMN].unique().tolist())
    return {category: i for i, category in enumerate(categories)}


def build_newsId_cateId_mapping(
    news_df: pd.DataFrame, category_dict: dict[str, int]
) -> dict[int, int]:
    return dict(zip(news_df[NEWS_ID_COLUMN], news_df[NEWS_CATEGORY_COLUMN].map(category_dict)))
