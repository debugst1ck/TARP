# Using DNABERT for classification
from torch import nn, Tensor
from transformers import AutoModel
from model.layers.pooling.trainable import QueryAttentionPooling


class SimpleClassifier(nn.Module):
    """
    A simple classifier using DNABERT for sequence classification.
    """

    def __init__(
        self,
        number_of_classes: int,
        embedding_dimension: int,
        model_name: str = "zhihan1996/DNABERT-2-117M",
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.pooling_layer = QueryAttentionPooling(hidden_size=embedding_dimension)
        self.classifier = nn.Linear(embedding_dimension, number_of_classes)

    def forward(self, sequence: Tensor, attention_mask: Tensor) -> Tensor:
        """
        :param Tensor sequence: The input sequence for the encoder.
        :param Tensor attention_mask: Optional attention mask for padding tokens. (0 = pad)
        :return: The classification logits.
        :rtype: Tensor
        """
        model_outputs = self.encoder(input_ids=sequence, attention_mask=attention_mask)[0]
        pooled_representation = self.pooling_layer(model_outputs)
        return self.classifier(pooled_representation)
