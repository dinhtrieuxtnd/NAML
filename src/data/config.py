from pathlib import Path

from pydantic import DirectoryPath, FilePath, NonNegativeInt, PositiveInt

from src.utils.config import StrictModel


class PreprocessingConfig(StrictModel):
    name: str


class InputConfig(StrictModel):
    train_behaviors: FilePath
    val_behaviors: FilePath
    test_behaviors: FilePath
    news: FilePath


class TokenizerConfig(StrictModel):
    name: str


class OutputConfig(StrictModel):
    directory: Path
    overwrite: bool


class TextMaxLengthConfig(StrictModel):
    title: PositiveInt
    sapo: PositiveInt
    tags: PositiveInt


class TextConfig(StrictModel):
    lowercase: bool
    max_length: TextMaxLengthConfig


class PreprocessConfig(StrictModel):
    preprocessing: PreprocessingConfig
    input: InputConfig
    tokenizer: TokenizerConfig
    output: OutputConfig
    text: TextConfig


class DataConfig(StrictModel):
    processed_dir: DirectoryPath
    text_views: list[str]
    history_len: PositiveInt
    num_negatives: PositiveInt
    train_batch_size: PositiveInt
    eval_batch_size: PositiveInt
    num_workers: NonNegativeInt
    # Chỉ lấy N impression đầu của val/test (dùng cho smoke test); None = toàn bộ.
    max_eval_impressions: PositiveInt | None
