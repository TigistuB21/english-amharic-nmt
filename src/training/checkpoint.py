"""Model checkpoint saving and loading utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    payload: Dict[str, Any],
    filepath: Union[str, Path],
) -> None:
    """
    Save model weights, optimizer state, and training metadata to a file.

    Args:
        payload: Dictionary containing model_state_dict, optimizer_state_dict, config, etc.
        filepath: Target destination file path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint file and optionally restore model and optimizer states.

    Args:
        filepath: Path to the .pt checkpoint file.
        model: Optional PyTorch model into which state_dict will be loaded.
        optimizer: Optional PyTorch optimizer into which state_dict will be loaded.
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary loaded from file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
