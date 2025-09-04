from typing import Optional
from torch import Tensor
import torch
import sklearn.metrics

# Not importing accuracy_score because it is not suitable for multilabel classification


# Multilabel classification metrices
class MultilabelMetrics:
    """
    Computes various metrics for multilabel classification.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def _predict(self, logits: Tensor) -> Tensor:
        """
        Apply the threshold to the logits to obtain binary predictions.
        
        :param Tensor logits: The raw logits from the model.
        :return: The binary predictions.
        :rtype: Tensor
        """
        # In binary multilabel classification, logits are log-odds
        # We need to convert it to probabilities
        probabilities = self._predict_probability(logits)
        return (probabilities > self.threshold).float()

    def _predict_probability(self, logits: Tensor) -> Tensor:
        """
        Apply the sigmoid function to the logits to obtain probabilities.
        
        :param Tensor logits: The raw logits from the model.
        :return: The predicted probabilities.
        :rtype: Tensor
        """
        return torch.sigmoid(logits).detach()

    def precision(self, logits: Tensor, targets: Tensor) -> float:
        preds = self._predict(logits)
        return sklearn.metrics.precision_score(
            targets.cpu(), preds.cpu(), average="micro"
        )

    def recall(self, logits: Tensor, targets: Tensor) -> float:
        preds = self._predict(logits)
        return sklearn.metrics.recall_score(targets.cpu(), preds.cpu(), average="micro")

    def f1(self, logits: Tensor, targets: Tensor) -> float:
        preds = self._predict(logits)
        return sklearn.metrics.f1_score(targets.cpu(), preds.cpu(), average="micro")

    def subset_accuracy(self, logits: Tensor, targets: Tensor) -> float:
        preds = self._predict(logits)
        return sklearn.metrics.accuracy_score(targets.cpu(), preds.cpu())

    def roc_auc(self, logits: Tensor, targets: Tensor) -> Optional[float]:
        preds = self._predict_probability(logits)
        return sklearn.metrics.roc_auc_score(
            targets.cpu(), preds.cpu(), average="macro", multi_class="ovr"
        ) # Macro-averaged ROC AUC for multilabel classification
