# Transformer model for sequence classification
from torch import nn, Tensor
import torch.nn.functional as F

from tarp.model.backbone import Encoder
from typing import Optional

from tarp.model.layers.pooling.trainable import QueryAttentionPooling


class TransformerEncoder(Encoder):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        hidden_dimension: int,
        number_of_layers: int = 2,
        number_of_heads: int = 4,
        dropout: float = 0.1,
        padding_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
            padding_idx=padding_id,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=number_of_heads,
            dim_feedforward=hidden_dimension,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=number_of_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.embedding_dimension = embedding_dimension
        self.pooling = QueryAttentionPooling(hidden_dimension)
        self.output_dimension = hidden_dimension

    def encode(
        self, sequence: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        embeddings = self.embedding(sequence)  # (batch, seq_len, embedding_dim)
        if attention_mask is not None:
            # Invert the attention mask for transformer (1 -> keep, 0 -> ignore)
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None
        encoded = self.transformer_encoder(
            embeddings, src_key_padding_mask=src_key_padding_mask
        )
        return self.pooling(self.dropout(encoded))

    @property
    def encoding_size(self) -> int:
        return self.output_dimension
