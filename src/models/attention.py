"""Attention-based Seq2Seq model for English-to-Amharic translation."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.encoder import Encoder


class BahdanauAttention(nn.Module):
    """Bahdanau additive attention layer."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute context vector and attention weights."""
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        batch_size, src_len, _ = encoder_outputs.shape
        hidden_expanded = hidden.unsqueeze(1).expand(-1, src_len, -1)
        energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1), attention_weights


class AttentionDecoder(nn.Module):
    """LSTM decoder with additive attention."""

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
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = BahdanauAttention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step decoder forward pass."""
        embedded = self.embedding(input_token.unsqueeze(0))
        hidden_last = hidden[-1]
        context, attn_weights = self.attention(hidden_last, encoder_outputs)
        context = context.unsqueeze(0)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(0)

        combined = torch.cat((output, context.squeeze(0)), dim=1)
        prediction = self.fc_out(combined)
        return prediction, hidden, cell, attn_weights


class AttentionSeq2Seq(nn.Module):
    """Attention-based sequence-to-sequence model."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: AttentionDecoder,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device if device is not None else torch.device("cpu")

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass."""
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        output_dim = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, output_dim, device=self.device)

        encoder_outputs, hidden, cell = self.encoder.encode_with_outputs(src)
        input_token = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = trg[t] if teacher_force else output.argmax(1)

        return outputs
