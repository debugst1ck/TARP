import torch
from torch import nn

from tarp.model.backbone import Encoder


class LanguageModel(nn.Module):
    """
    A simple language modeling model.
    """

    def __init__(self, encoder: Encoder, vocabulary_size: int):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.encoder = encoder
        self.language_head = nn.Linear(
            self.encoder.encoding_size, vocabulary_size, bias=False
        )

    def forward(
        self,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        return_sequence: bool = True,
    ) -> torch.Tensor:
        """
        :param Tensor sequence: The input sequence for the encoder.
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :return: The language modeling logits.
        :rtype: Tensor
        """
        encoded_representation = self.encoder.encode(
            sequence, attention_mask, return_sequence=return_sequence
        )
        return self.language_head(encoded_representation)
