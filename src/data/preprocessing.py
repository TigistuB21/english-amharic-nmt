"""Data cleaning, splitting, and filtering utilities for English-Amharic corpus."""

import re
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_whitespace(text: str) -> str:
    """Normalize repeated whitespace characters into a single space and strip edges."""
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_parallel_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the parallel English-Amharic corpus:
    1. Select ['eng', 'amh'] columns.
    2. Drop missing rows.
    3. Normalize whitespace in English and Amharic sentences.
    4. Remove exact duplicate translation pairs.

    Args:
        df: Raw pandas DataFrame with 'eng' and 'amh' columns.

    Returns:
        Cleaned pandas DataFrame with reset index.
    """
    clean_df = df[["eng", "amh"]].copy()

    # Drop missing values
    clean_df = clean_df.dropna(subset=["eng", "amh"])

    # Normalize whitespace
    clean_df["eng"] = clean_df["eng"].apply(normalize_whitespace)
    clean_df["amh"] = clean_df["amh"].apply(normalize_whitespace)

    # Remove exact duplicate translation pairs
    clean_df = clean_df.drop_duplicates(
        subset=["eng", "amh"]
    ).reset_index(drop=True)

    return clean_df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_ratio_of_temp: float = 0.50,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split clean corpus into train (80%), validation (10%), and test (10%) sets.

    Args:
        df: Cleaned pandas DataFrame.
        test_size: Proportion for temporary test+val split (default 0.20).
        val_ratio_of_temp: Proportion of temp split allocated to validation (default 0.50).
        random_state: Seed for reproducible random shuffling (default 42).

    Returns:
        (train_df, val_df, test_df) tuple.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_ratio_of_temp,
        random_state=random_state,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def token_length(text: str) -> int:
    """Return whitespace token count for a given text."""
    return len(str(text).split())


def filter_by_max_length(df: pd.DataFrame, max_len: int = 70) -> pd.DataFrame:
    """
    Filter sentence pairs where both English and Amharic sequences fit within max_len.
    Note: +2 accounts for <SOS> and <EOS> added during numericalization.

    Args:
        df: DataFrame containing 'eng' and 'amh' columns.
        max_len: Maximum allowed sequence length including SOS and EOS (default 70).

    Returns:
        Filtered DataFrame with reset index.
    """
    eng_len = df["eng"].apply(token_length)
    amh_len = df["amh"].apply(token_length)

    mask = (eng_len + 2 <= max_len) & (amh_len + 2 <= max_len)
    return df.loc[mask].reset_index(drop=True)
