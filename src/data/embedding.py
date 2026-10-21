from collections import Counter
from pathlib import Path

import numpy as np

from src.data.constants import PAD_TOKEN, PAD_TOKEN_IDX, UNK_TOKEN


def build_embedding_matrix(
    vectors_path: Path,
    vocab: dict[str, int],
    token_counts: Counter[str],
    dim: int,
    seed: int,
    unmatched_init_std: float,
) -> tuple[np.ndarray, dict]:
    """Looks up every vocab token in a fastText .vec file, streamed line by line.

    An exact match wins over a case variant (e.g. "Hà" for "hà"); tokens found in neither form
    are drawn from N(0, unmatched_init_std). The PAD row is 0. Returns the float32 matrix
    [V, dim] and its coverage, by distinct token and by token occurrence (token_counts).
    """
    matrix = np.random.default_rng(seed).normal(
        0.0, unmatched_init_std, size=(len(vocab), dim)).astype(np.float32)
    matrix[PAD_TOKEN_IDX] = 0.0

    wanted = set(vocab) - {PAD_TOKEN, UNK_TOKEN}
    exact, case_variant = set(), set()
    with vectors_path.open("rb") as file:
        header = file.readline().split()
        if len(header) != 2 or int(header[1]) != dim:
            raise ValueError(f"{vectors_path}: header {header} does not declare dimension {dim}")
        for line_number, line in enumerate(file, start=2):
            # Chỉ decode phần token; vector chỉ parse khi token cần dùng.
            raw_token, _, values = line.rstrip(b"\r\n ").partition(b" ")
            token = raw_token.decode("utf-8")
            if token in wanted:
                if token in exact:
                    continue
                exact.add(token)
                case_variant.discard(token)
            else:
                token = token.lower()
                if token not in wanted or token in exact or token in case_variant:
                    continue
                case_variant.add(token)
            vector = np.array(values.split(), dtype=np.float32)
            if vector.shape != (dim,):
                raise ValueError(
                    f"{vectors_path}:{line_number}: expected {dim} values, got {vector.shape[0]}")
            matrix[vocab[token]] = vector

    matched = exact | case_variant
    total_count = sum(token_counts.values())
    statistics = {
        "vocab_tokens": len(wanted),
        "exact_matched": len(exact),
        "case_variant_matched": len(case_variant),
        "unmatched": len(wanted) - len(matched),
        "type_coverage": len(matched) / len(wanted),
        "token_coverage": sum(token_counts[token] for token in matched) / total_count,
    }
    return matrix, statistics
