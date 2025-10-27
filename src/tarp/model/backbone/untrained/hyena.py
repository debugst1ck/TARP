from torch import nn, Tensor
import torch
from typing import Optional
from tarp.model.backbone import Encoder


class HyenaBlock(nn.Module):
    def __init__(
        self,
        feature_dimension: int,
        filter_size: int = 64,
        kernel_size: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        dilation = max(1, filter_size // kernel_size)
        padding = ((kernel_size - 1) * dilation) // 2

        self.conv_filter = nn.Conv1d(
            feature_dimension,
            feature_dimension,
            kernel_size=kernel_size,
            padding=padding,
            groups=feature_dimension,
            dilation=dilation,
        )
        self.gate = nn.Conv1d(feature_dimension, feature_dimension, kernel_size=1)
        self.proj = nn.Linear(feature_dimension, feature_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x_in = x.transpose(1, 2)
        conv_out: Tensor = self.conv_filter(x_in)
        gate_out = torch.sigmoid(self.gate(x_in))

        # Safety: trim to minimum length
        min_len = min(conv_out.size(2), gate_out.size(2))
        conv_out = conv_out[:, :, :min_len]
        gate_out = gate_out[:, :, :min_len]

        x_out = conv_out * gate_out
        x_out = x_out.transpose(1, 2)
        x_out = self.proj(x_out)
        return self.dropout(x_out)


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
                    feature_dimension=embedding_dimension,
                    filter_size=hidden_dimension,
                    dropout=dropout,
                )
                for _ in range(number_of_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dimension)
        self.output_dimension = embedding_dimension

    def encode(
        self,
        sequence: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_sequence: bool = False,
    ) -> Tensor:
        x = self.embedding(sequence)  # (batch, seq_len, embedding_dim)

        for block in self.hyena_blocks:
            x = x + block(x)  # Residual connection

        x = self.norm(x)

        if return_sequence:
            return x  # (batch_size, seq_len, output_dimension)

        # Sequence level pooling, (mean or masked mean)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            masked = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            masked = x.mean(dim=1)

        return masked  # (batch_size, output_dimension)

    @property
    def encoding_size(self) -> int:
        return self.output_dimension
