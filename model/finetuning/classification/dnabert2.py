# Using DNABERT for classification
from torch import nn, Tensor
from transformers import AutoModel
from model.layers.pooling.trainable import QueryAttentionPooling

from model.finetuning.classification import ClassificationModel


class Dnabert2ClassificationModel(ClassificationModel):
    """
    A simple classifier using DNABERT for sequence classification.
    """

    def __init__(
        self,
        number_of_classes: int,
        hidden_dimension: int,
        name: str = "zhihan1996/DNABERT-2-117M",
    ):
        super().__init__(number_of_classes, hidden_dimension)
        self.encoder = AutoModel.from_pretrained(name, trust_remote_code=True)
        self.pooling = QueryAttentionPooling(hidden_dimension)

    def encode(self, sequence: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Encode the input sequence using DNABERT and apply attention pooling.

        :param Tensor sequence: The input sequence for the encoder.
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :return: The encoded embeddings.
        :rtype: Tensor
        """
        outputs = self.encoder(input_ids=sequence, attention_mask=attention_mask)[0]
        pooled_representation = self.pooling(outputs)
        return pooled_representation
