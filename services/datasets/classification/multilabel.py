from services.datasets.classification import ClassificationDataset

from services.datasets import SequenceDataset
from services.datasource.sequence import SequenceDataSource
from services.tokenizers import Tokenizer
from torch import Tensor
import torch
from services.preprocessing.augumentation import NoAugmentation, AugmentationTechnique

class MultiLabelClassificationDataset(ClassificationDataset):
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
            label_columns,
            maximum_sequence_length,
            augumentation,
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()
