"""Decoder module for Seq2Seq Neural Machine Translation."""

from typing import Tuple
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Step-by-step LSTM Decoder with linear projection output layer.

    Generates target token predictions one step at a time given current input
    and recurrent hidden and cell states.
    """

    def __init__(
        self,
        output_dim: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            output_dim,
            embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step forward pass of the decoder.

        Args:
            input_token: Tensor of shape [batch_size] containing current target token IDs.
            hidden: Tensor of shape [num_layers, batch_size, hidden_dim].
            cell: Tensor of shape [num_layers, batch_size, hidden_dim].

        Returns:
            Tuple of:
              prediction: [batch_size, output_dim] unnormalized logits over vocabulary.
              hidden: [num_layers, batch_size, hidden_dim] updated hidden state.
              cell: [num_layers, batch_size, hidden_dim] updated cell state.
        """
        # Add sequence length dimension: [1, batch_size]
        input_expanded = input_token.unsqueeze(0)

        # Embedding: [1, batch_size, embedding_dim]
        embedded = self.embedding(input_expanded)

        # LSTM step
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # Remove sequence dimension: [batch_size, hidden_dim]
        output = output.squeeze(0)

        # Linear projection to vocabulary logits: [batch_size, output_dim]
        prediction = self.fc_out(output)

        return prediction, hidden, cell
