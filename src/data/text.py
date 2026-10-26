import re
import unicodedata
from collections import Counter

import numpy as np

from src.data.constants import PAD_TOKEN, PAD_TOKEN_IDX, UNK_TOKEN, UNK_TOKEN_IDX

# Một âm tiết (chuỗi chữ/số) hoặc một dấu câu là một token.
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """The project's only tokenizer: NFKC -> lowercase -> syllables and punctuation."""
    return _TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).lower())


def build_vocab(token_counts: Counter[str]) -> dict[str, int]:
    """PAD and UNK first, then tokens by decreasing count, ties broken alphabetically."""
    tokens = sorted(token_counts, key=lambda token: (-token_counts[token], token))
    special = {PAD_TOKEN: PAD_TOKEN_IDX, UNK_TOKEN: UNK_TOKEN_IDX}
    return {
        **special,
        **{token: i for i, token in enumerate(tokens, start=len(special))},
    }


def encode(tokens: list[str], vocab: dict[str, int], max_length: int) -> np.ndarray:
    """Truncates to max_length and right-pads with PAD_TOKEN_IDX -> int32 [max_length]."""
    ids = np.full(max_length, PAD_TOKEN_IDX, dtype=np.int32)
    truncated = tokens[:max_length]
    ids[: len(truncated)] = [vocab.get(token, UNK_TOKEN_IDX) for token in truncated]
    return ids
