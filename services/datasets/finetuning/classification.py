from services.datasets import SequenceDataset
from services.datasource.sequence import SequenceDataSource
from services.tokenizers import Tokenizer
from torch import Tensor
import torch

from services.preprocessing.augumentation import NoAugmentation, AugmentationTechnique


class MultiLabelClassificationDataset(SequenceDataset):
    def __init__(
        self,
        data_source: SequenceDataSource,
        tokenizer: Tokenizer,
        sequence_column: str,
        label_columns: list[str],
        maximum_sequence_length: int,
        augumentation: AugmentationTechnique = NoAugmentation(),
    ):
        super().__init__(
            data_source,
            tokenizer,
            sequence_column,
            maximum_sequence_length,
            augumentation,
        )
        self.label_columns = label_columns

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = super().__getitem__(index)
        row = self.data_source.retrieve(index)
        # Extract labels for multi-source multi-label classification
        labels = [row.get(col, 0) for col in self.label_columns]
        item["labels"] = torch.tensor(labels, dtype=torch.float)
        return item

    def __len__(self) -> int:
        return self.data_source.height
