"""Centralized path and directory configuration for English-Amharic NMT."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class ProjectPaths:
    """Project directory and file paths container."""
    data_root: Path
    raw_dir: Path
    processed_dir: Path
    vocab_dir: Path
    models_dir: Path

    # Unfiltered datasets
    train_path: Path
    val_path: Path
    test_path: Path

    # Filtered datasets (MAX_LEN = 70)
    train_filtered_path: Path
    val_filtered_path: Path
    test_filtered_path: Path

    # Vocabularies
    eng_vocab_path: Path
    amh_vocab_path: Path

    def ensure_directories(self) -> None:
        """Create all required subdirectories if they do not exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


def get_repo_root() -> Path:
    """Return the absolute path to the repository root directory."""
    # src/utils/paths.py -> parents[2] is repo root
    return Path(__file__).resolve().parents[2]


def get_data_paths(data_root: Optional[Union[str, Path]] = None) -> ProjectPaths:
    """
    Resolve and return all project data paths.

    Args:
        data_root: Path to data directory. If None, checks DATA_ROOT
                   environment variable, falling back to '<repo_root>/data'.

    Returns:
        ProjectPaths instance with resolved Path objects.
    """
    if data_root is not None:
        root = Path(data_root)
    elif "DATA_ROOT" in os.environ:
        root = Path(os.environ["DATA_ROOT"])
    else:
        root = get_repo_root() / "data"

    raw_dir = root / "raw"
    processed_dir = root / "processed"
    vocab_dir = processed_dir / "vocab"
    models_dir = root / "models"

    return ProjectPaths(
        data_root=root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        vocab_dir=vocab_dir,
        models_dir=models_dir,
        train_path=processed_dir / "train.csv",
        val_path=processed_dir / "validation.csv",
        test_path=processed_dir / "test.csv",
        train_filtered_path=processed_dir / "train_filtered.csv",
        val_filtered_path=processed_dir / "validation_filtered.csv",
        test_filtered_path=processed_dir / "test_filtered.csv",
        eng_vocab_path=vocab_dir / "eng_vocab.json",
        amh_vocab_path=vocab_dir / "amh_vocab.json",
    )
