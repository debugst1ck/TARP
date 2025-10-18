from torch import Tensor
from torch.utils.data import Dataset
import torch

from tarp.services.datasource.sequence import SequenceDataSource
from tarp.services.tokenizers import Tokenizer
from tarp.services.preprocessing.augmentation import NoAugmentation, AugmentationTechnique


class SequenceDataset(Dataset):
    def __init__(
        self,
        data_source: SequenceDataSource,
        tokenizer: Tokenizer,
        sequence_column: str,
        maximum_sequence_length: int,
        augumentation: AugmentationTechnique = NoAugmentation(),
    ):
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.sequence_column = sequence_column
        self.maximum_sequence_length = maximum_sequence_length
        self.padding_value = tokenizer.pad_token_id
        self.augumentation = augumentation

    def __len__(self):
        return self.data_source.height

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.data_source.retrieve(index)
        sequence = row[self.sequence_column]

        sequence = self.augumentation.apply(sequence)

        tokenized = self.tokenizer.tokenize(sequence)

        # Get sequence length from tensor
        sequence_length = tokenized.size(0)

        # Pad tokenized to maximum sequence length
        padded = torch.full(
            (self.maximum_sequence_length,), self.padding_value, dtype=torch.long
        )
        length = min(sequence_length, self.maximum_sequence_length)
        padded[:length] = tokenized[:length]

        # Attention mask
        attention_mask = padded != self.padding_value

        return {"sequence": padded, "attention_mask": attention_mask}
