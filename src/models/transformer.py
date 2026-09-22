"""Transformer-based sequence-to-sequence model for English-to-Amharic translation."""

from typing import Optional, Union

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for a transformer sequence."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        x = x + self.pe[:seq_len].to(x.device)
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    """Encoder-decoder transformer for sequence-to-sequence translation."""

    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        pad_idx: int = 0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.device = device if device is not None else torch.device("cpu")

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )

        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass for the full sequence."""
        src_emb = self.positional_encoding(self.src_embedding(src))
        decoder_input = trg[:-1]
        trg_emb = self.positional_encoding(self.trg_embedding(decoder_input))
        src_padding_mask = src.transpose(0, 1).eq(self.pad_idx)
        trg_padding_mask = decoder_input.transpose(0, 1).eq(self.pad_idx)
        trg_mask = nn.Transformer.generate_square_subsequent_mask(
            decoder_input.size(0), device=trg.device
        )
        decoded = self.transformer(
            src_emb,
            trg_emb,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=trg_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        logits = self.fc_out(decoded)
        first_step = torch.zeros(
            1, logits.size(1), logits.size(2), device=logits.device, dtype=logits.dtype
        )
        return torch.cat([first_step, logits], dim=0)

    def generate(self, src: torch.Tensor, max_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Greedy decoding for a single batch input text."""
        dev = device if device is not None else self.device
        src = src.to(dev)
        src_emb = self.positional_encoding(self.src_embedding(src))
        src_padding_mask = src.transpose(0, 1).eq(self.pad_idx)

        batch_size = src.size(1)
        generated = torch.full((1, batch_size), 2, dtype=torch.long, device=dev)

        for _ in range(max_len - 1):
            trg_emb = self.positional_encoding(self.trg_embedding(generated))
            trg_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.size(0), device=dev
            )
            output = self.fc_out(
                self.transformer(
                    src_emb,
                    trg_emb,
                    tgt_mask=trg_mask,
                    src_key_padding_mask=src_padding_mask,
                    memory_key_padding_mask=src_padding_mask,
                )
            )
            next_tokens = output[-1].argmax(dim=-1)
            generated = torch.cat([generated, next_tokens.unsqueeze(0)], dim=0)
            if torch.all(next_tokens == 3):
                break

        return generated
