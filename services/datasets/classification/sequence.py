from torch import Tensor
from torch.utils.data import Dataset
import torch

from services.datasets.source.sequence import SequenceDataSource
from services.tokenizers.base import Tokenizer


class SequenceClassificationDataset(Dataset):
    def __init__(
        self,
        data_source: SequenceDataSource,
        tokenizer: Tokenizer,
        sequence_column: str,
        maximum_sequence_length: int,
    ):
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.sequence_column = sequence_column
        self.maximum_sequence_length = maximum_sequence_length
        self.padding_value = tokenizer.pad_token_id

    def __len__(self):
        return self.data_source.height

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.data_source.retrieve(index)
        sequence = row[self.sequence_column]

        tokenized = self.tokenizer.tokenize(sequence)

        # Get sequence length from tensor
        sequence_length = tokenized.size(dim=1)

        # Pad tokenized to maximum sequence length
        padded = torch.full(
            (self.maximum_sequence_length,), self.padding_value, dtype=torch.long
        )
        padded[:sequence_length] = tokenized

        # Attention mask
        attention_mask = padded != self.padding_value

        return {"sequence": padded, "attention_mask": attention_mask}
