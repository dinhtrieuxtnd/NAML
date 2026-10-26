import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC
from torchmetrics.retrieval import RetrievalAUROC, RetrievalMRR, RetrievalNormalizedDCG

from src.data.constants import METRIC_NAMES


class RankingMetrics:
    """METRIC_NAMES over candidates grouped by impression (shared by model and baselines)."""

    def __init__(self, device: torch.device) -> None:
        self.auroc = BinaryAUROC().to(device)
        # Retrieval metrics để trên CPU: compute() lặp Python qua từng impression, trên GPU mỗi
        # vòng tốn vài kernel launch + sync nên chậm hơn CPU khoảng 5 lần.
        self.retrieval = MetricCollection(
            {
                "impression_auc": RetrievalAUROC(empty_target_action="skip"),
                "mrr": RetrievalMRR(empty_target_action="skip"),
                "ndcg@5": RetrievalNormalizedDCG(top_k=5, empty_target_action="skip"),
                "ndcg@10": RetrievalNormalizedDCG(top_k=10, empty_target_action="skip"),
            }
        )

    def update(
        self, preds: torch.Tensor, labels: torch.Tensor, indexes: torch.Tensor
    ) -> None:
        """preds, labels, indexes: [num_candidates]; indexes = impression of each candidate."""
        self.auroc.update(preds, labels)
        self.retrieval.update(preds.cpu(), labels.cpu(), indexes=indexes.cpu())

    def compute(self) -> dict[str, float]:
        metrics = {"auc": self.auroc.compute(), **self.retrieval.compute()}
        return {name: metrics[name].item() for name in METRIC_NAMES}
