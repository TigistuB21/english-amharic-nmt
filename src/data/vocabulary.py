"""Vocabulary construction, serialization, and numericalization utilities."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Union

# Special Token Specifications
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

DEFAULT_MAX_LEN = 70


def build_vocabulary_dict(texts: Iterable[str], min_freq: int = 5) -> Dict[str, int]:
    """
    Build token-to-ID vocabulary dictionary from an iterable of sentences.

    Tokens with frequency >= min_freq are kept and sorted alphabetically
    for deterministic IDs. Special tokens are placed first at IDs 0-3.

    Args:
        texts: Iterable of sentence strings.
        min_freq: Minimum frequency threshold (default 5).

    Returns:
        Dict mapping token string to unique integer ID.
    """
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())

    filtered_tokens = sorted([
        token for token, count in counter.items()
        if count >= min_freq
    ])

    vocab = {
        token: idx
        for idx, token in enumerate(SPECIAL_TOKENS + filtered_tokens)
    }
    return vocab


def sentence_to_ids(
    sentence: str,
    vocab: Dict[str, int],
    unk_idx: int = UNK_IDX,
    sos_idx: int = SOS_IDX,
    eos_idx: int = EOS_IDX,
) -> List[int]:
    """
    Convert a whitespace-separated sentence into a list of token IDs.
    Prepends <SOS> and appends <EOS>. Out-of-vocabulary tokens map to <UNK>.

    Args:
        sentence: Raw sentence string.
        vocab: Dictionary mapping tokens to IDs.
        unk_idx: ID for unknown token (default 1).
        sos_idx: ID for start of sequence token (default 2).
        eos_idx: ID for end of sequence token (default 3).

    Returns:
        List of integer IDs: [SOS, id_1, id_2, ..., EOS]
    """
    tokens = str(sentence).split()
    ids = [sos_idx]
    for token in tokens:
        ids.append(vocab.get(token, unk_idx))
    ids.append(eos_idx)
    return ids


def pad_sequence_ids(
    ids: List[int],
    max_len: int = DEFAULT_MAX_LEN,
    pad_idx: int = PAD_IDX,
    eos_idx: int = EOS_IDX,
) -> List[int]:
    """
    Pad or truncate an ID sequence to exact max_len length.
    If truncated, ensures the final token remains <EOS>.

    Args:
        ids: List of token IDs.
        max_len: Desired target length (default 70).
        pad_idx: ID for padding token (default 0).
        eos_idx: ID for end of sequence token (default 3).

    Returns:
        Padded or truncated list of length max_len.
    """
    if len(ids) > max_len:
        ids = list(ids[:max_len])
        ids[-1] = eos_idx
    elif len(ids) < max_len:
        padding = [pad_idx] * (max_len - len(ids))
        ids = list(ids) + padding
    return ids


def save_vocab(vocab: Dict[str, int], path: Union[str, Path]) -> None:
    """Save vocabulary dictionary to JSON file."""
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: Union[str, Path]) -> Dict[str, int]:
    """Load vocabulary dictionary from JSON file."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Vocabulary file not found at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
