from pydantic import DirectoryPath, FilePath, NonNegativeInt, PositiveFloat, PositiveInt

from src.utils.config import RunConfig, StrictModel


class PreprocessRunConfig(RunConfig):
    # Thư mục output_root/name đã tồn tại thì báo lỗi, trừ khi overwrite: true.
    overwrite: bool


class InputConfig(StrictModel):
    train_behaviors: FilePath
    val_behaviors: FilePath
    test_behaviors: FilePath
    news: FilePath
    # fastText .vec (cc.vi.300.vec); chỉ đọc lúc preprocess.
    word_vectors: FilePath


class EmbeddingConfig(StrictModel):
    dim: PositiveInt
    # Token không có trong word_vectors được khởi tạo N(0, unmatched_init_std).
    unmatched_init_std: PositiveFloat
    seed: NonNegativeInt


class TextMaxLengthConfig(StrictModel):
    title: PositiveInt
    sapo: PositiveInt
    tags: PositiveInt


class TextConfig(StrictModel):
    max_length: TextMaxLengthConfig


class PreprocessConfig(StrictModel):
    run: PreprocessRunConfig
    input: InputConfig
    embedding: EmbeddingConfig
    text: TextConfig


class DataConfig(StrictModel):
    processed_dir: DirectoryPath
    text_views: list[str]
    history_len: PositiveInt
    num_negatives: PositiveInt
    train_batch_size: PositiveInt
    eval_batch_size: PositiveInt
    num_workers: NonNegativeInt
    # Chỉ lấy N sample train đầu (dùng cho smoke test: nhiều epoch rất ngắn); None = toàn bộ.
    max_train_samples: PositiveInt | None
    # Chỉ lấy N impression đầu của val/test (dùng cho smoke test); None = toàn bộ.
    max_eval_impressions: PositiveInt | None
