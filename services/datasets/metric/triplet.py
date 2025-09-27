import torch
from torch import Tensor
from torch.utils.data import Dataset
from services.datasets.classification.multilabel import MultiLabelClassificationDataset
from services.loggers.colored import ColoredLogger


class MultilabelOfflineTripletDataset(Dataset):
    """
    Dataset for generating triplets (anchor, positive, negative) from a multi-label classification dataset.
    
    - Hard positive: closest sample that shares at least one label with the anchor.
    - Hard negative: farthest sample that shares no labels with the anchor.
    
    If no positive or negative is found, falls back gracefully.
    """

    def __init__(self, base_dataset: MultiLabelClassificationDataset):
        self.base_dataset = base_dataset
        if len(self.base_dataset) < 2:
            raise ValueError("Base dataset must contain at least two samples.")

        # Precompute labels for efficiency
        self.labels = torch.stack([self.base_dataset[i]["labels"] for i in range(len(self.base_dataset))])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        anchor_sample = self.base_dataset[index]
        anchor_labels = self.labels[index].unsqueeze(0)  # shape (1, num_labels)

        # Compute distances to all samples
        distances = torch.cdist(anchor_labels, self.labels, p=2).squeeze(0)  # shape (N,)

        # Mask out anchor itself
        distances[index] = float("inf")

        # Positive mask: at least one shared label
        overlap = (self.labels * anchor_labels).sum(dim=1) > 0

        # Negative mask: no shared labels
        no_overlap = ~overlap

        # Hardest positive = min distance among positives
        if overlap.any():
            # Find the hardest positive sample
            pos_idx = distances.masked_fill(~overlap, float("inf")).argmin().item()
            positive_sample = self.base_dataset[pos_idx]
        else:
            ColoredLogger.warning(f"No positive sample found for anchor index {index}")
            positive_sample = anchor_sample

        # Hardest negative = max distance among negatives
        if no_overlap.any():
            neg_idx = distances.masked_fill(~no_overlap, float("-inf")).argmax().item()
            negative_sample = self.base_dataset[neg_idx]
        else:
            ColoredLogger.warning(f"No negative sample found for anchor index {index}")
            # fallback: pick a random different sample
            neg_idx = (index + 1) % len(self.base_dataset)
            negative_sample = self.base_dataset[neg_idx]

        return {
            "anchor": anchor_sample,
            "positive": positive_sample,
            "negative": negative_sample,
        }
