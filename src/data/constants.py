IMPRESSION_ID_COLUMN = "impression_id"
USER_ID_COLUMN = "user_id"
TIME_COLUMN = "time"
HISTORY_COLUMN = "history"
IMPRESSIONS_COLUMN = "impressions"
BEHAVIORS_COLUMNS = [
    IMPRESSION_ID_COLUMN,
    USER_ID_COLUMN,
    TIME_COLUMN,
    HISTORY_COLUMN,
    IMPRESSIONS_COLUMN,
]

# Parsed from IMPRESSIONS_COLUMN ("<newsId>-<label> ...").
CANDIDATES_COLUMN = "candidates"
LABELS_COLUMN = "labels"

# Train sample keys, besides impression_id / user_id / history.
NEGATIVE_POOL_KEY = "negative_pool"
POSITIVE_ITEM_KEY = "positive_item"

NEWS_ID_COLUMN = "newsId"
NEWS_CATEGORY_COLUMN = "category"

# Row 0 of every news token matrix is the padding news.
NEWS_PAD_IDX = 0

# Batch keys, besides HISTORY_COLUMN / CANDIDATES_COLUMN / LABELS_COLUMN.
NEWS_ROWS_KEY = "news_rows"
NEWS_TOKENS_KEY = "news_tokens"
NEWS_CATEGORY_KEY = "news_category"
HISTORY_MASK_KEY = "history_mask"
CANDIDATE_MASK_KEY = "candidate_mask"
INDEXES_KEY = "indexes"

# Data files of a preprocessing output directory (run files are in src/utils/run.py).
STATISTICS_FILE = "statistics.json"
ARTIFACTS_DIR = "artifacts"
SPLITS = ("train", "val", "test")
SAMPLES_FILE_TEMPLATE = "{split}_samples.pkl"
INDEX_MATRIX_FILE = "index_matrix.pkl"
CATEGORY_DICT_FILE = "category_dict.pkl"
NEWS_CATEGORY_MAPPING_FILE = "newsId_cateId_mapping.pkl"
# Token matrices are saved as "<text column>.npy".
TOKEN_MATRIX_FILE_TEMPLATE = "{column}.npy"

