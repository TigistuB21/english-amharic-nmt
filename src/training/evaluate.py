"""Evaluation loop utilities for calculating validation/test loss."""

from typing import Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    Evaluate model over an entire dataset split (teacher_forcing_ratio=0).

    Args:
        model: Seq2Seq model.
        loader: DataLoader.
        criterion: CrossEntropyLoss criterion.
        device: Device to run evaluation on.

    Returns:
        Average evaluation loss across batches.
    """
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for source, target in loader:
            if device is not None:
                source = source.to(device)
                target = target.to(device)

            # [batch_size, seq_len] -> [seq_len, batch_size]
            source = source.transpose(0, 1)
            target = target.transpose(0, 1)

            # In evaluation, always use model predictions (no teacher forcing)
            output = model(
                source,
                target,
                teacher_forcing_ratio=0.0,
            )

            output_dim = output.shape[-1]

            # Ignore <SOS> position
            output = output[1:].reshape(-1, output_dim)
            target = target[1:].reshape(-1)

            loss = criterion(output, target)
            epoch_loss += loss.item()

    return epoch_loss / len(loader) if len(loader) > 0 else 0.0
