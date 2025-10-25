from functools import lru_cache
import torch

from tarp.services.datasets import SequenceDataset
from tarp.services.datasets.classification.multilabel import (
    MultiLabelClassificationDataset,
)
from tarp.cli.logging.colored import ColoredLogger

from pathlib import Path

import polars as pl
from torch import Tensor

from typing import override


class MultiLabelOfflineTripletDataset(SequenceDataset):
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
        self.data_source = base_dataset.data_source
        if self.data_source.height < 2:
            raise ValueError("Base dataset must contain at least two samples.")

        expected_columns = base_dataset.label_columns

        self.labels = None

        if label_cache and label_cache.exists():
            ColoredLogger.debug(f"Checking for label cache at: {label_cache}")
            
            df = pl.read_parquet(label_cache)
            cached_columns = df.columns

            # Ensure column order and names match expected
            if set(cached_columns) == set(expected_columns) and df.shape[0] == self.data_source.height:
                # Reorder columns to match expected order
                df = df.select(expected_columns)
                self.labels = torch.tensor(df.to_numpy(), dtype=torch.float32)
                ColoredLogger.info(
                    f"Loaded labels from cache (aligned to label_columns): {label_cache}"
                )
            else:
                ColoredLogger.warning(
                    f"Label cache mismatch — columns or size differ. "
                    f"Expected shape ({self.data_source.height}, {len(expected_columns)}), "
                    f"found shape {df.shape}, column difference: {set(cached_columns) - set(expected_columns)}"
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
            df = pl.DataFrame(
                self.labels.numpy(), schema=expected_columns
            ).write_parquet(
                label_cache
                if label_cache
                else Path("temp/data/interim/labels_cache.parquet")
            )
            ColoredLogger.info(f"Saved labels to cache: {label_cache}")
   
    @override
    def process_row(self, index: int, row: dict) -> dict[str, dict[str, Tensor]]:
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

        if overlap.any():
            # Find the hardest positive sample
            positive_sample_index = distances.masked_fill(~overlap, float("inf")).argmin().item()
            positive_sample = self.base_dataset[positive_sample_index]
        else:
            # fallback: use anchor as positive
            positive_sample = anchor_sample

        if no_overlap.any():
            negative_sample_index = distances.masked_fill(~no_overlap, float("-inf")).argmax().item()
            negative_sample = self.base_dataset[negative_sample_index]
        else:
            ColoredLogger.warning(f"No negative sample found for anchor index {index}")
            # fallback: pick a random different sample
            negative_sample_index = (index + 1) % len(self.base_dataset)
            negative_sample = self.base_dataset[negative_sample_index]

        return {
            "anchor": anchor_sample,
            "positive": positive_sample,
            "negative": negative_sample,
        }
