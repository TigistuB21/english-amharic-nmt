"""Seq2Seq wrapper model combining Encoder and Decoder."""

from typing import Optional, Union
import torch
import torch.nn as nn

from src.models.encoder import Encoder
from src.models.decoder import Decoder


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model combining Encoder and Decoder with teacher forcing.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
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
        """
        Forward pass for complete sequence translation.

        Args:
            src: Source tensor of shape [src_len, batch_size].
            trg: Target tensor of shape [trg_len, batch_size].
            teacher_forcing_ratio: Probability of using ground-truth token
                                   as next decoder input instead of prediction.

        Returns:
            outputs: Tensor of shape [trg_len, batch_size, output_dim].
        """
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        output_dim = self.decoder.output_dim

        # Tensor to store decoder outputs at each time step
        outputs = torch.zeros(trg_len, batch_size, output_dim, device=self.device)

        # Encode the source sequence
        hidden, cell = self.encoder(src)

        # First input to the decoder is always the <SOS> tokens
        input_token = trg[0, :]

        for t in range(1, trg_len):
            # Single step prediction
            output, hidden, cell = self.decoder(input_token, hidden, cell)

            # Store predictions
            outputs[t] = output

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            # Top predicted token ID
            top1 = output.argmax(1)

            # Next input is either teacher-forced or model's own prediction
            input_token = trg[t] if teacher_force else top1

        return outputs


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
