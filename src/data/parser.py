from pathlib import Path

import pandas as pd

from src.data.constants import (
    BEHAVIORS_COLUMNS,
    CANDIDATES_COLUMN,
    CLICK_EVENT_ID,
    HISTORY_COLUMN,
    IMPRESSION_ID_COLUMN,
    IMPRESSIONS_COLUMN,
    LABELS_COLUMN,
    LOG_EVENT_ID_COLUMN,
    LOG_TIME_COLUMN,
    LOG_TIME_ZONE,
    NEWS_CATEGORY_COLUMN,
    NEWS_ID_COLUMN,
    TIME_COLUMN,
    TIME_FORMAT,
)


def _split_impressions(df: pd.DataFrame, file_path: Path) -> None:
    """Replaces IMPRESSIONS_COLUMN with CANDIDATES_COLUMN and LABELS_COLUMN, in place."""
    num_missing = df[IMPRESSIONS_COLUMN].isna().sum()
    if num_missing:
        raise ValueError(
            f"{file_path}: {num_missing} rows have empty {IMPRESSIONS_COLUMN}"
        )
    df[CANDIDATES_COLUMN] = df[IMPRESSIONS_COLUMN].apply(
        lambda x: [int(item.split("-")[0]) for item in x.split()]
    )
    df[LABELS_COLUMN] = df[IMPRESSIONS_COLUMN].apply(
        lambda x: [int(item.split("-")[1]) for item in x.split()]
    )
    df.drop(columns=IMPRESSIONS_COLUMN, inplace=True)


def read_behaviors(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        header=None,
        names=BEHAVIORS_COLUMNS,
        dtype={HISTORY_COLUMN: str, IMPRESSIONS_COLUMN: str},
    )
    _split_impressions(df, file_path)
    df[HISTORY_COLUMN] = (
        df[HISTORY_COLUMN].fillna("").apply(lambda x: [int(item) for item in x.split()])
    )
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], format=TIME_FORMAT)
    return df


def read_impression_log(file_path: Path) -> pd.DataFrame:
    """impression_id, time (datetime), candidates, labels of each row; history is not read."""
    df = pd.read_csv(
        file_path,
        header=None,
        names=BEHAVIORS_COLUMNS,
        usecols=[IMPRESSION_ID_COLUMN, TIME_COLUMN, IMPRESSIONS_COLUMN],
        dtype={IMPRESSIONS_COLUMN: str},
    )
    _split_impressions(df, file_path)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], format=TIME_FORMAT)
    return df


def read_clicks(log_dir: Path) -> pd.DataFrame:
    """newsId (int64) and time (local datetime) of every click in the cleaned event log."""
    df = pd.read_parquet(
        log_dir,
        columns=[NEWS_ID_COLUMN, LOG_TIME_COLUMN],
        filters=[(LOG_EVENT_ID_COLUMN, "==", CLICK_EVENT_ID)],
    )
    # News ids of the behaviors files are integers; other ids can never be a candidate.
    df = df[df[NEWS_ID_COLUMN].str.fullmatch(r"\d+")]
    return pd.DataFrame(
        {
            NEWS_ID_COLUMN: df[NEWS_ID_COLUMN].astype("int64").to_numpy(),
            TIME_COLUMN: pd.to_datetime(df[LOG_TIME_COLUMN], unit="ms", utc=True)
            .dt.tz_convert(LOG_TIME_ZONE)
            .dt.tz_localize(None)
            .to_numpy(),
        }
    )


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
