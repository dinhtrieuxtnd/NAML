import contextlib
import datetime
import logging
import platform
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import torch
from accelerate import PartialState
from pydantic import BaseModel

from src.utils.config import save_config
from src.utils.io import load_json, save_json

logger = logging.getLogger(__name__)

# Layout of a run directory (preprocess, train), shared by every stage.
CONFIG_FILE = "config.yaml"
RUN_INFO_FILE = "run_info.json"
LOG_FILE_TEMPLATE = "{stage}.log"
HISTORY_FILE = "history.json"
PLOTS_DIR = "plots"
LOSS_PLOT_FILE = "loss.png"
METRIC_PLOT_FILE_TEMPLATE = "val_{metric}.png"
LR_PLOT_FILE = "lr.png"
# Training checkpoints: checkpoints/best (best val monitor, with metrics.json) and
# checkpoints/latest (overwritten every save_every_steps and after each epoch, for resuming).
CHECKPOINTS_DIR = "checkpoints"
BEST_CHECKPOINT = "best"
LATEST_CHECKPOINT = "latest"
TRAINING_STATE_FILE = "training_state.json"
# Evaluation of a checkpoint goes to <run_dir>/evaluation_<checkpoint name>/.
EVALUATION_DIR_TEMPLATE = "evaluation_{checkpoint}"
METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "predictions.csv"


def setup_logging(log_file: Path | None) -> None:
    """The only place that configures logging: console, plus log_file when given."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    # Thư viện bên ngoài chỉ log từ WARNING, logger của project ("src.*") log từ INFO.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("src").setLevel(logging.INFO)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _environment() -> dict:
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "command": " ".join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpus": gpus,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }


@contextlib.contextmanager
def start_run(stage: str, run_dir: Path, config: BaseModel, resume: bool) -> Iterator[None]:
    """Sets up a run directory and records the run in run_info.json.

    A new run writes config.yaml (never overwritten afterwards); a resumed run keeps it.
    Every call appends one entry to run_info.json, marked completed or failed (the error is
    re-raised). Only the main process writes files.
    """
    is_main = PartialState().is_main_process
    config_path, run_info_path = run_dir / CONFIG_FILE, run_dir / RUN_INFO_FILE
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        if not resume:
            if config_path.exists():
                raise FileExistsError(f"{config_path} already exists, refusing to overwrite it")
            save_config(config.model_dump(mode="json"), config_path)
    setup_logging(run_dir / LOG_FILE_TEMPLATE.format(stage=stage) if is_main else None)
    logger.info("Run directory: %s", run_dir)

    runs = load_json(run_info_path) if is_main and run_info_path.exists() else []
    start_time = datetime.datetime.now()
    entry = {"stage": stage, "status": "running", "resume": resume,
             "start_time": start_time.isoformat(), **_environment()}
    runs.append(entry)

    def save_entry(**fields) -> None:
        entry.update(fields)
        if fields["status"] != "running":
            end_time = datetime.datetime.now()
            entry.update(end_time=end_time.isoformat(),
                         duration_seconds=(end_time - start_time).total_seconds())
        if is_main:
            save_json(runs, run_info_path)

    save_entry(status="running")
    try:
        yield
    except BaseException as e:
        logger.exception("%s failed: %s", stage, e)
        save_entry(status="failed", error_type=type(e).__name__, error_message=str(e))
        raise
    save_entry(status="completed")
    logger.info("%s completed", stage)
