"""PyTorch Dataset and DataLoader implementations for English-Amharic translation."""

from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.vocabulary import (
    DEFAULT_MAX_LEN,
    pad_sequence_ids,
    sentence_to_ids,
)

HUGGINGFACE_DATASET_NAME = "michsethowusu/english-amharic_sentence-pairs_mt560"


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for English-to-Amharic parallel text.

    Converts each sentence pair to token IDs with <SOS> and <EOS>,
    applies padding/truncation to max_len, and returns torch.LongTensor pair.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        src_vocab: Dict[str, int],
        trg_vocab: Dict[str, int],
        max_len: int = DEFAULT_MAX_LEN,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        src_sentence = row["eng"]
        trg_sentence = row["amh"]

        src_ids = sentence_to_ids(src_sentence, self.src_vocab)
        trg_ids = sentence_to_ids(trg_sentence, self.trg_vocab)

        src_padded = pad_sequence_ids(src_ids, self.max_len)
        trg_padded = pad_sequence_ids(trg_ids, self.max_len)

        src_tensor = torch.tensor(src_padded, dtype=torch.long)
        trg_tensor = torch.tensor(trg_padded, dtype=torch.long)

        return src_tensor, trg_tensor


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a standard PyTorch DataLoader for a TranslationDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def load_raw_dataset_from_hf(
    dataset_name: str = HUGGINGFACE_DATASET_NAME,
    cache_dir: Optional[Union[str, Path]] = None,
    raw_csv_name: str = "raw_dataset.csv",
) -> pd.DataFrame:
    """
    Load raw parallel dataset from local cache if available; otherwise from Hugging Face Hub.

    Args:
        dataset_name: Hugging Face dataset identifier.
        cache_dir: Optional local directory to cache downloaded dataset.
        raw_csv_name: Filename for the local raw CSV cache.

    Returns:
        pandas DataFrame containing 'eng' and 'amh' columns.
    """
    if cache_dir is not None:
        local_cache_file = Path(cache_dir) / raw_csv_name
        if local_cache_file.exists():
            print(f"Loading raw dataset from local cache: {local_cache_file}")
            return pd.read_csv(local_cache_file)

    from datasets import load_dataset

    cache_path = str(cache_dir) if cache_dir else None
    print(f"Downloading/loading dataset '{dataset_name}' from Hugging Face...")
    hf_dataset = load_dataset(dataset_name, cache_dir=cache_path)
    train_split = hf_dataset["train"]
    df = train_split.to_pandas()

    if cache_dir is not None:
        local_cache_file = Path(cache_dir) / raw_csv_name
        local_cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_cache_file, index=False)
        print(f"Cached raw dataset locally to: {local_cache_file}")

    return df
