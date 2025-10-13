from functools import lru_cache
import torch
from torch.utils.data import Dataset

from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)
from tarp.cli.logging.colored import ColoredLogger

from pathlib import Path

import polars as pl


class MultilabelOfflineTripletDataset(Dataset):
    """
    Dataset for generating triplets (anchor, positive, negative) from a multi-label classification dataset.

    - Hard positive: closest sample that shares at least one label with the anchor.
    - Hard negative: farthest sample that shares no labels with the anchor.

    If no positive or negative is found, falls back gracefully.
    """

    def __init__(
        self, base_dataset: MultiLabelClassificationDataset, label_cache: Path = None
    ):
        self.base_dataset = base_dataset
        if len(self.base_dataset) < 2:
            raise ValueError("Base dataset must contain at least two samples.")

        expected_columns = base_dataset.label_columns
        
        self.labels = None
        
        if label_cache:
            print(f"Checking for label cache at: {label_cache}")
            df = pl.read_parquet(label_cache)
            cached_columns = df.columns

            # ✅ Ensure column order and names match expected
            if set(cached_columns) == set(expected_columns) and df.shape[0] == len(self.base_dataset):
                # Reorder columns to match expected order
                df = df.select(expected_columns)
                self.labels = torch.tensor(df.to_numpy(), dtype=torch.float32)
                ColoredLogger.info(f"Loaded labels from cache (aligned to label_columns): {label_cache}")
            else:
                ColoredLogger.warning(
                    f"Label cache mismatch — columns or size differ. "
                    f"(cache columns: {cached_columns}, expected: {expected_columns})"
                )

        # If cache missing or mismatched, recompute and save
        if self.labels is None:
            ColoredLogger.warning(
                "Computing labels from base dataset. This may take a while"
            )
            self.labels = torch.stack(
                [self.base_dataset[i]["labels"] for i in range(len(self.base_dataset))]
            )

            # Convert to Polars DataFrame for saving
            df = pl.DataFrame(self.labels.numpy()).write_parquet(
                label_cache
                if label_cache
                else Path("temp/data/interim/labels_cache.parquet")
            )
            ColoredLogger.info(f"Saved labels to cache: {label_cache}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        anchor_sample = self.base_dataset[index]
        anchor_labels = self.labels[index].unsqueeze(0)  # shape (1, num_labels)

        # Compute distances to all samples
        distances = torch.cdist(anchor_labels, self.labels, p=2).squeeze(
            0
        )  # shape (N,)

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
