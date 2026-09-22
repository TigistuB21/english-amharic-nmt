"""Training loop and sanity training functions."""

import time
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float = 1.0,
    teacher_forcing_ratio: float = 0.5,
    device: Optional[Union[str, torch.device]] = None,
    max_batches: Optional[int] = None,
) -> float:
    """
    Train model for one full epoch.

    Args:
        model: Seq2Seq model instance.
        loader: PyTorch DataLoader yielding (source, target) batches.
        optimizer: Optimizer instance.
        criterion: CrossEntropyLoss criterion.
        clip: Gradient norm clipping threshold.
        teacher_forcing_ratio: Probability of using teacher forcing.
        device: Device to run computations on.

    Returns:
        Average epoch loss.
    """
    model.train()
    epoch_loss = 0.0

    for batch_idx, (source, target) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if device is not None:
            source = source.to(device)
            target = target.to(device)

        # DataLoader shape: [batch_size, seq_len] -> [seq_len, batch_size]
        source = source.transpose(0, 1)
        target = target.transpose(0, 1)

        optimizer.zero_grad()

        output = model(
            source,
            target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        output_dim = output.shape[-1]

        # Ignore <SOS> position at index 0 for loss calculation
        output = output[1:].reshape(-1, output_dim)
        target = target[1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    batches_seen = min(len(loader), max_batches) if max_batches is not None else len(loader)
    return epoch_loss / batches_seen if batches_seen > 0 else 0.0


def run_sanity_training(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_batches: int = 100,
    clip: float = 1.0,
    teacher_forcing_ratio: float = 0.5,
    device: Optional[Union[str, torch.device]] = None,
    print_interval: int = 10,
) -> Tuple[List[float], float]:
    """
    Run lightweight sanity training on a fixed number of batches.

    Args:
        model: Seq2Seq model.
        loader: DataLoader.
        optimizer: Optimizer.
        criterion: Loss criterion.
        num_batches: Number of batches to train on (default 100).
        clip: Gradient clipping norm threshold (default 1.0).
        teacher_forcing_ratio: Teacher forcing ratio (default 0.5).
        device: Target device.
        print_interval: Logging frequency in batches.

    Returns:
        Tuple of (losses_list, total_elapsed_seconds).
    """
    model.train()
    start_time = time.time()
    losses = []

    for batch_idx, (source, target) in enumerate(loader):
        if batch_idx >= num_batches:
            break

        if device is not None:
            source = source.to(device)
            target = target.to(device)

        # [batch_size, seq_len] -> [seq_len, batch_size]
        source = source.transpose(0, 1)
        target = target.transpose(0, 1)

        optimizer.zero_grad()

        output = model(
            source,
            target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        output_dim = output.shape[-1]

        output = output[1:].reshape(-1, output_dim)
        target = target[1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        losses.append(loss.item())

        if (batch_idx + 1) % print_interval == 0:
            print(f"Batch {batch_idx + 1}/{num_batches} | Loss: {loss.item():.4f}")

    elapsed_time = time.time() - start_time
    return losses, elapsed_time
