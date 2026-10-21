from pathlib import Path

import pandas as pd

from src.data.constants import (
    BEHAVIORS_COLUMNS,
    CANDIDATES_COLUMN,
    HISTORY_COLUMN,
    IMPRESSIONS_COLUMN,
    LABELS_COLUMN,
    NEWS_CATEGORY_COLUMN,
    NEWS_ID_COLUMN,
    TIME_COLUMN,
)


def read_behaviors(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path, header=None, names=BEHAVIORS_COLUMNS,
        dtype={HISTORY_COLUMN: str, IMPRESSIONS_COLUMN: str})
    num_missing = df[IMPRESSIONS_COLUMN].isna().sum()
    if num_missing:
        raise ValueError(f"{file_path}: {num_missing} rows have empty {IMPRESSIONS_COLUMN}")
    df[HISTORY_COLUMN] = df[HISTORY_COLUMN].fillna("").apply(
        lambda x: [int(item) for item in x.split()])
    df[CANDIDATES_COLUMN] = df[IMPRESSIONS_COLUMN].apply(
        lambda x: [int(item.split("-")[0]) for item in x.split()])
    df[LABELS_COLUMN] = df[IMPRESSIONS_COLUMN].apply(
        lambda x: [int(item.split("-")[1]) for item in x.split()])
    return df.drop(columns=[TIME_COLUMN, IMPRESSIONS_COLUMN])


def read_news(file_path: Path, text_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    num_missing = df[NEWS_ID_COLUMN].isna().sum()
    if num_missing:
        raise ValueError(f"{file_path}: {num_missing} rows have empty {NEWS_ID_COLUMN}")
    duplicated = df[NEWS_ID_COLUMN].duplicated()
    if duplicated.any():
        raise ValueError(f"{file_path}: {duplicated.sum()} duplicated {NEWS_ID_COLUMN}")
    # Empty category is kept as its own "" category.
    fill_columns = [*text_columns, NEWS_CATEGORY_COLUMN]
    df[fill_columns] = df[fill_columns].fillna("")
    return df
