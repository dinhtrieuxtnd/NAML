from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from src.data.constants import METRIC_NAMES  # noqa: E402
from src.utils.run import LOSS_PLOT_FILE, LR_PLOT_FILE, METRIC_PLOT_FILE_TEMPLATE  # noqa: E402


def _save(fig: plt.Figure, path: Path) -> None:
    for ax in fig.axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history: list[dict], best_epoch: int | None, plots_dir: Path) -> None:
    """Redraws loss.png, val_<metric>.png per metric (best epoch marked) and lr.png from
    history.json rows."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [row["train_loss"] for row in history], marker="o")
    ax.set(xlabel="epoch", ylabel="train loss", title="Train loss")
    ax.grid(alpha=0.3)
    _save(fig, plots_dir / LOSS_PLOT_FILE)

    # Mỗi metric một hình riêng với trục y riêng, để thay đổi nhỏ của từng metric vẫn nhìn rõ.
    for metric in METRIC_NAMES:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, [row[metric] for row in history], marker="o")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best (epoch {best_epoch})")
            ax.legend()
        ax.set(xlabel="epoch", ylabel=metric, title=f"Validation {metric}")
        ax.grid(alpha=0.3)
        _save(fig, plots_dir / METRIC_PLOT_FILE_TEMPLATE.format(metric=metric))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [row["lr"] for row in history], marker="o")
    ax.set(xlabel="epoch", ylabel="learning rate", title="Learning rate", yscale="log")
    ax.grid(alpha=0.3)
    _save(fig, plots_dir / LR_PLOT_FILE)
