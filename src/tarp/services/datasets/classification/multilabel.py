from torch import Tensor

from tarp.services.datasource.sequence import SequenceDataSource
from tarp.services.tokenizers import Tokenizer
from tarp.services.datasets.classification import ClassificationDataset
from tarp.services.preprocessing.augmentation import NoAugmentation, AugmentationTechnique

class MultiLabelClassificationDataset(ClassificationDataset):
    def __init__(
        self,
        data_source: SequenceDataSource,
        tokenizer: Tokenizer,
        sequence_column: str,
        label_columns: list[str],
        maximum_sequence_length: int,
        augmentation: AugmentationTechnique = NoAugmentation(),
    ):
        super().__init__(
            data_source,
            tokenizer,
            sequence_column,
            label_columns,
            maximum_sequence_length,
            augmentation,
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()
