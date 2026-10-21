from pathlib import Path

import numpy as np
import torch

from src.data.constants import (
    ARTIFACTS_DIR,
    CATEGORY_DICT_FILE,
    INDEX_MATRIX_FILE,
    NEWS_CATEGORY_MAPPING_FILE,
    NEWS_PAD_IDX,
    TOKEN_MATRIX_FILE_TEMPLATE,
)
from src.utils.config import load_config
from src.utils.io import load_pickle
from src.utils.run import CONFIG_FILE


class NewsFeatures:
    """Token matrices and category ids of every news, indexed by row (row NEWS_PAD_IDX = padding)."""

    def __init__(self, processed_dir: Path, text_views: list[str], pretrained_name: str) -> None:
        tokenizer_name = load_config(processed_dir / CONFIG_FILE)["tokenizer"]["name"]
        if tokenizer_name != pretrained_name:
            raise ValueError(
                f"{processed_dir} was tokenized with {tokenizer_name}, "
                f"but model.pretrained_name is {pretrained_name}")

        artifacts_dir = processed_dir / ARTIFACTS_DIR
        # tokens[view]: [N + 1, L_view]
        self.tokens = {
            view: torch.from_numpy(
                np.load(artifacts_dir / TOKEN_MATRIX_FILE_TEMPLATE.format(column=view)))
            for view in text_views
        }
        self.index_matrix: dict[int, int] = load_pickle(artifacts_dir / INDEX_MATRIX_FILE)
        self.num_categories = len(load_pickle(artifacts_dir / CATEGORY_DICT_FILE))
        news_category: dict[int, int] = load_pickle(artifacts_dir / NEWS_CATEGORY_MAPPING_FILE)

        num_rows = len(self.index_matrix) + 1
        # category: [N + 1]; the padding row is never encoded, so its value does not matter.
        self.category = torch.full((num_rows,), NEWS_PAD_IDX, dtype=torch.long)
        for news_id, row in self.index_matrix.items():
            self.category[row] = news_category[news_id]

    @property
    def num_rows(self) -> int:
        return len(self.category)

    def to_rows(self, news_ids: list[int]) -> list[int]:
        return [self.index_matrix[news_id] for news_id in news_ids]

    def lookup(self, rows: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """rows: [U] -> ({view: [U, L_view]}, category [U])."""
        tokens = {view: matrix[rows].long() for view, matrix in self.tokens.items()}
        return tokens, self.category[rows]
