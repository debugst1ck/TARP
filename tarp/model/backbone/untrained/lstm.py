import torch
from torch import nn, Tensor
from typing import Optional

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tarp.model.backbone import Encoder


class LstmEncoder(Encoder):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        hidden_dimension: int,
        number_of_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        padding_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
            padding_idx=padding_id,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dimension,
            hidden_size=hidden_dimension,
            num_layers=number_of_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if number_of_layers > 1 else 0.0,
        )
        self.output_dimension = hidden_dimension * (2 if bidirectional else 1)

    def encode(
        self, sequence: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        embedded = self.embedding(sequence)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed)
        else:
            output, (hidden, cell) = self.lstm(embedded)

        if self.lstm.bidirectional:
            pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            pooled = hidden[-1]
        return pooled  # (batch_size, output_dimension)

    @property
    def encoding_size(self) -> int:
        return self.output_dimension
