import torch
import random
from torch import Tensor
from torch.utils.data import Dataset
from services.datasets.classification.multilabel import MultiLabelClassificationDataset
from services.loggers.colored import ColoredLogger


class MultiLabelTripletDataset(Dataset):
    def __init__(
        self, base_dataset: MultiLabelClassificationDataset, min_pos_overlap: int = 1
    ):
        self.base_dataset = base_dataset
        self.min_pos_overlap = min_pos_overlap

        # Precompute labels
        self.labels = []
        for i in range(len(base_dataset)):
            row = base_dataset.data_source.retrieve(i)
            labels = torch.tensor(
                [row.get(col, 0) for col in base_dataset.label_columns],
                dtype=torch.float,
            )
            self.labels.append(labels)

        # Filter out samples with no possible positives
        self.valid_indices = []
        for i, lbl in enumerate(self.labels):
            if self._has_positive(lbl, i):
                self.valid_indices.append(i)
            else:
                ColoredLogger.warning(f"Sample {i} has no positive samples, skipping.")
                # TODO: In the future, instead of skipping, we augment the dataset with synthetic positives
                pass

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        anchor_idx = self.valid_indices[idx]
        anchor_item = self.base_dataset[anchor_idx]
        anchor_labels = self.labels[anchor_idx]

        # Sample positive
        pos_index = self._sample_positive(anchor_labels, anchor_idx)
        positive_item = self.base_dataset[pos_index]

        # Sample negative
        neg_index = self._sample_negative(anchor_labels, anchor_idx)
        negative_item = self.base_dataset[neg_index]

        return {
            "anchor": anchor_item,
            "positive": positive_item,
            "negative": negative_item,
        }

    def _has_positive(self, anchor_labels: Tensor, anchor_idx: int) -> bool:
        for j, labels in enumerate(self.labels):
            if (
                j != anchor_idx
                and torch.sum((anchor_labels > 0) & (labels > 0)).item()
                >= self.min_pos_overlap
            ):
                return True
        return False

    def _sample_positive(self, anchor_labels: Tensor, anchor_idx: int) -> int:
        candidates = [
            j
            for j, labels in enumerate(self.labels)
            if j != anchor_idx
            and torch.sum((anchor_labels > 0) & (labels > 0)).item()
            >= self.min_pos_overlap
        ]
        return random.choice(candidates)

    def _sample_negative(self, anchor_labels: Tensor, anchor_idx: int) -> int:
        candidates = [
            j
            for j, labels in enumerate(self.labels)
            if j != anchor_idx
            and torch.sum((anchor_labels > 0) & (labels > 0)).item() == 0
        ]
        return random.choice(candidates)
