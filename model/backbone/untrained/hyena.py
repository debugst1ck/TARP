from torch import nn, Tensor
import torch
from typing import Optional
from model.backbone import Encoder


class HyenaBlock(nn.Module):
    """
    A simplified Hyena block:
    - Uses gated convolutions with long filters.
    - Captures long-range dependencies without quadratic attention.
    """

    def __init__(
        self,
        feature_dimension: int,
        filter_size: int = 64,
        kernel_size: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv_filter = nn.Conv1d(
            feature_dimension,
            feature_dimension,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=feature_dimension,
        )
        self.gate = nn.Conv1d(
            feature_dimension, feature_dimension, kernel_size=1
        )  # gating mechanism
        self.proj = nn.Linear(feature_dimension, feature_dimension)
        self.dropout = nn.Dropout(dropout)
        self.filter_size = filter_size

    def forward(self, input: Tensor) -> Tensor:
        # x: (batch, seq_len, dim)
        input = input.transpose(1, 2)  # (batch, dim, seq_len)
        conv_out = self.conv_filter(input)
        gate_out = torch.sigmoid(self.gate(input))
        input = conv_out * gate_out
        input = input.transpose(1, 2)  # back to (batch, seq_len, dim)
        input = self.proj(input)
        return self.dropout(input)


class HyenaEncoder(Encoder):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        hidden_dimension: int,
        number_of_layers: int = 1,
        dropout: float = 0.1,
        padding_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
            padding_idx=padding_id,
        )
        self.hyena_blocks = nn.ModuleList(
            [
                HyenaBlock(
                    dim=embedding_dimension,
                    filter_size=hidden_dimension,
                    dropout=dropout,
                )
                for _ in range(number_of_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dimension)
        self.output_dimension = embedding_dimension

    def encode(
        self, sequence: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        embedded = self.embedding(
            sequence
        )  # (batch_size, seq_len, embedding_dimension)

        for block in self.hyena_blocks:
            embedded = embedded + block(embedded)  # Residual connection

        embedded = self.norm(embedded)

        # Normalize
        normalized: Tensor = self.norm(embedded)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            masked = (normalized * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            masked = normalized.mean(dim=1)
        return masked  # (batch_size, output_dimension)

    @property
    def encoding_size(self) -> int:
        return self.output_dimension
