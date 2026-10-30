"""Click counts of news around an impression time, from read_clicks output."""

import numpy as np
import pandas as pd

from src.data.constants import NEWS_ID_COLUMN, TIME_COLUMN

# Khoảng cách giữa hai tin trong khoá (tin, giây) của ClickCounter; lớn hơn mọi timestamp giây.
_SECONDS_SPAN = 2**40


def _to_seconds(times: np.ndarray) -> np.ndarray:
    return times.astype("datetime64[s]").astype(np.int64)


class ClickCounter:
    """Counts the clicks on each news inside a time window, from read_clicks output."""

    def __init__(self, clicks: pd.DataFrame) -> None:
        news_ids = clicks[NEWS_ID_COLUMN].to_numpy(dtype=np.int64)
        self.news_index = pd.Index(np.unique(news_ids))
        codes = self.news_index.get_indexer(news_ids)
        # keys: [num_clicks], sorted by news then click time.
        self.keys = np.sort(codes * _SECONDS_SPAN + _to_seconds(clicks[TIME_COLUMN].values))
        self.num_clicks = len(clicks)

    def count(
        self, news_ids: np.ndarray, times: np.ndarray, window_seconds: int
    ) -> np.ndarray:
        """news_ids, times: [Q] -> clicks on news_ids[i] in [times[i] - window, times[i])."""
        codes = self.news_index.get_indexer(news_ids)
        end = codes * _SECONDS_SPAN + _to_seconds(times)
        counts = np.searchsorted(self.keys, end) - np.searchsorted(
            self.keys, end - window_seconds
        )
        # Tin chưa từng được click (code = -1) không có click nào.
        return np.where(codes >= 0, counts, 0)


def relative_popularity(counts: np.ndarray) -> np.ndarray:
    """counts: [C] clicks of one impression's candidates -> log1p(c) - mean(log1p(c)), [C]."""
    log_counts = np.log1p(counts)
    return log_counts - log_counts.mean()


def hour_of_day(time: pd.Timestamp) -> float:
    """Hour of day with minutes as fraction, in [0, 24)."""
    return time.hour + time.minute / 60
