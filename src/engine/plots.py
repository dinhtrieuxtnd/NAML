from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

METRICS = ("auc", "impression_auc", "mrr", "ndcg@5", "ndcg@10")


def _save(fig: plt.Figure, path: Path) -> None:
    for ax in fig.axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(history: list[dict], best_epoch: int | None, plots_dir: Path) -> None:
    """Redraws loss.png, metrics.png (best epoch marked) and lr.png from history.json rows."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [row["train_loss"] for row in history], marker="o")
    ax.set(xlabel="epoch", ylabel="train loss", title="Train loss")
    ax.grid(alpha=0.3)
    _save(fig, plots_dir / "loss.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in METRICS:
        ax.plot(epochs, [row[metric] for row in history], marker="o", label=metric)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best (epoch {best_epoch})")
    ax.set(xlabel="epoch", ylabel="value", title="Validation metrics")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, plots_dir / "metrics.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [row["lr"] for row in history], marker="o")
    ax.set(xlabel="epoch", ylabel="learning rate", title="Learning rate", yscale="log")
    ax.grid(alpha=0.3)
    _save(fig, plots_dir / "lr.png")
