import logging
from pathlib import Path

import numpy as np

from src.data.config import PreprocessConfig
from src.data.constants import (
    ARTIFACTS_DIR,
    CATEGORY_DICT_FILE,
    INDEX_MATRIX_FILE,
    NEWS_CATEGORY_COLUMN,
    NEWS_CATEGORY_MAPPING_FILE,
    SAMPLES_FILE_TEMPLATE,
    SPLITS,
    STATISTICS_FILE,
    TOKEN_MATRIX_FILE_TEMPLATE,
)
from src.data.parser import read_behaviors, read_news
from src.data.preprocessing import (
    build_category_dict,
    build_eval_samples,
    build_newsId_cateId_mapping,
    build_token_matrix,
    build_train_samples,
)
from src.utils.io import save_json, save_pickle
from src.utils.run import CONFIG_FILE, LOG_FILE_TEMPLATE, RUN_INFO_FILE, start_run

logger = logging.getLogger(__name__)

STAGE = "preprocess"
MANAGED_ROOT_FILES = (
    CONFIG_FILE, LOG_FILE_TEMPLATE.format(stage=STAGE), RUN_INFO_FILE, STATISTICS_FILE)
MANAGED_ARTIFACT_FILES = (
    *(SAMPLES_FILE_TEMPLATE.format(split=split) for split in SPLITS),
    INDEX_MATRIX_FILE,
    CATEGORY_DICT_FILE,
    NEWS_CATEGORY_MAPPING_FILE,
)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already exists. Set run.overwrite: true to replace it.")
    artifacts_dir = output_dir / ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for filename in MANAGED_ROOT_FILES:
            (output_dir / filename).unlink(missing_ok=True)
        for filename in MANAGED_ARTIFACT_FILES:
            (artifacts_dir / filename).unlink(missing_ok=True)
        for file in artifacts_dir.glob(TOKEN_MATRIX_FILE_TEMPLATE.format(column="*")):
            file.unlink()
    return output_dir


def run_preprocessing(config: PreprocessConfig) -> None:
    output_dir = prepare_output_dir(
        config.run.output_root / config.run.name, overwrite=config.run.overwrite)
    with start_run(STAGE, output_dir, config, resume=False):
        _preprocess(config, output_dir)


def _preprocess(config: PreprocessConfig, output_dir: Path) -> None:
    for name, path in config.input:
        logger.info("Input %s: %s", name, path)

    behavior_paths = dict(zip(SPLITS, (
        config.input.train_behaviors,
        config.input.val_behaviors,
        config.input.test_behaviors,
    )))
    behaviors = {split: read_behaviors(path) for split, path in behavior_paths.items()}
    text_columns = list(config.text.max_length.model_dump())
    news = read_news(config.input.news, text_columns)
    logger.info(
        "Loaded data: %s, news=%d",
        ", ".join(f"{split}={len(df)}" for split, df in behaviors.items()), len(news))
    num_empty_category = (news[NEWS_CATEGORY_COLUMN] == "").sum()
    if num_empty_category > 0:
        logger.warning("News with empty category: %d", num_empty_category)

    samples, counters = {}, {}
    for split, behaviors_df in behaviors.items():
        build = build_train_samples if split == "train" else build_eval_samples
        samples[split], counters[split] = build(behaviors_df)
        logger.info("%s samples: %d", split, counters[split]["output_samples"])
        if counters[split]["empty_history"] > 0:
            logger.warning("%s samples with empty history: %d",
                           split, counters[split]["empty_history"])

    matrixes, index_matrix, statistics_token_length = build_token_matrix(
        news, config.text, config.tokenizer.name)
    logger.info("Built token matrix for news articles")
    category_dict = build_category_dict(news)
    newsId_cateId_mapping = build_newsId_cateId_mapping(news, category_dict)

    artifacts_dir = output_dir / ARTIFACTS_DIR
    pickles = {
        SAMPLES_FILE_TEMPLATE.format(split=split): split_samples
        for split, split_samples in samples.items()
    }
    pickles.update({
        INDEX_MATRIX_FILE: index_matrix,
        NEWS_CATEGORY_MAPPING_FILE: newsId_cateId_mapping,
        CATEGORY_DICT_FILE: category_dict,
    })
    for name, obj in pickles.items():
        save_pickle(obj, artifacts_dir / name)
    for column, matrix in matrixes.items():
        np.save(artifacts_dir / TOKEN_MATRIX_FILE_TEMPLATE.format(column=column), matrix)
    logger.info("Saved artifacts to %s", artifacts_dir)

    statistics = {
        "source_rows": {
            **{f"{split}_behaviors": len(df) for split, df in behaviors.items()},
            "news": len(news),
        },
        "samples": counters,
        "text_config": config.text.model_dump(),
        "tokenizer": config.tokenizer.name,
        "token_length": statistics_token_length,
        "num_categories": len(category_dict),
    }
    save_json(statistics, output_dir / STATISTICS_FILE)
