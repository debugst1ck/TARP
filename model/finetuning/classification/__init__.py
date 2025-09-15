import torch
from torch import nn

from abc import ABC, abstractmethod


class ClassificationModel(nn.Module, ABC):
    """
    A simple classification model.
    """

    def __init__(self, number_of_classes: int, hidden_dimension: int):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.embedding_dimension = hidden_dimension

        self.classifier = nn.Linear(hidden_dimension, number_of_classes)

    @abstractmethod
    def encode(
        self, sequence: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode the input sequence to get the embeddings.

        :param Tensor sequence: The input sequence for the encoder.
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :return: The encoded embeddings.
        :rtype: Tensor
        """
        raise NotImplementedError

    def forward(
        self, sequence: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        :param Tensor sequence: The input sequence for the encoder.
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :return: The classification logits.
        :rtype: Tensor
        """
        pooled_representation = self.encode(sequence, attention_mask)
        return self.classifier(pooled_representation)
