"""Encoder module for Seq2Seq Neural Machine Translation."""

from typing import Tuple
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Standard LSTM Encoder for Sequence-to-Sequence translation.

    Encodes input source sequences into final hidden and cell states.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            input_dim,
            embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Args:
            src: Tensor of shape [src_len, batch_size] containing source token IDs.

        Returns:
            Tuple of (hidden, cell) states of shape [num_layers, batch_size, hidden_dim].
        """
        outputs, (hidden, cell) = self.lstm(self.embedding(src))
        return hidden, cell

    def encode_with_outputs(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return encoder outputs together with the final hidden and cell states."""
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell
