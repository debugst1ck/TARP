from typing import Optional, Callable
from torch import Tensor
import torch
import sklearn.metrics
from typing import Union


class MultilabelMetrics:
    """
    Computes multiple metrics for multilabel classification in one call.
    """

    def __init__(self, threshold: float = 0.5, logits: bool = True):
        self.threshold = threshold
        self.logits = logits

        # Registry of metrics (name → function)
        self._metrics: dict[str, Callable[[Tensor, Tensor], float]] = {
            "precision": self._precision,
            "recall": self._recall,
            "f1": self._f1,
            "subset_accuracy": self._subset_accuracy,
            "roc_auc": self._roc_auc,
        }

    def _predict_probability(self, logits: Tensor) -> Tensor:
        """Convert logits to probabilities via sigmoid."""
        return torch.sigmoid(logits).detach()

    def _predict(self, logits: Tensor) -> Tensor:
        """Binarize probabilities at threshold."""
        if not self.logits:
            return logits
        probs = self._predict_probability(logits)
        return (probs > self.threshold).float()

    # --- individual metric implementations ---
    def _precision(self, logits: Tensor, targets: Tensor) -> float:
        preds = self._predict(logits)
        return sklearn.metrics.precision_score(
            targets.cpu(), preds.cpu(), average="micro"
        )

    def _recall(self, logits: Tensor, targets: Tensor) -> float:
        preds = self._predict(logits)
        return sklearn.metrics.recall_score(targets.cpu(), preds.cpu(), average="micro")

    def _f1(self, logits: Tensor, targets: Tensor) -> float:
        preds = self._predict(logits)
        return sklearn.metrics.f1_score(targets.cpu(), preds.cpu(), average="micro")

    def _subset_accuracy(self, logits: Tensor, targets: Tensor) -> float:
        preds = self._predict(logits)
        return sklearn.metrics.accuracy_score(targets.cpu(), preds.cpu())

    def _roc_auc(self, logits: Tensor, targets: Tensor) -> Optional[float]:
        if not self.logits:
            raise ValueError("ROC AUC requires logits, but logits=False was set.")
        probs = self._predict_probability(logits)
        try:
            return sklearn.metrics.roc_auc_score(
                targets.cpu(), probs.cpu(), average="macro", multi_class="ovr"
            )
        except ValueError:
            # Handles edge case: ROC AUC cannot be computed if only one class is present
            return None

    # --- public interface ---
    def compute(
        self, logits: Union[Tensor, list[Tensor]], targets: Union[Tensor, list[Tensor]]
    ) -> dict[str, Optional[float]]:
        """
        Compute all metrics at once and return as dict.
        """
        # Handle list of tensors from batches
        if isinstance(logits, list):
            logits = torch.cat(logits, dim=0)
        if isinstance(targets, list):
            targets = torch.cat(targets, dim=0)

        return {name: fn(logits, targets) for name, fn in self._metrics.items()}

    def add_metric(self, name: str, fn: Callable[[Tensor, Tensor], float]) -> None:
        """
        Allow user to register a custom metric.
        """
        self._metrics[name] = fn

    def remove_metric(self, name: str) -> None:
        """
        Allow user to remove a registered metric.
        """
        self._metrics.pop(name, None)