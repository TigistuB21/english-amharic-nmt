"""Create the dataset splits and vocabularies needed for translation training."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_raw_dataset_from_hf
from src.data.preprocessing import clean_parallel_corpus, filter_by_max_length, split_data
from src.data.vocabulary import build_vocabulary_dict, save_vocab
from src.utils.paths import get_data_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))
paths = get_data_paths(DATA_ROOT)
paths.ensure_directories()

raw_df = load_raw_dataset_from_hf(cache_dir=paths.raw_dir)
clean_df = clean_parallel_corpus(raw_df)
train_df, val_df, test_df = split_data(clean_df, test_size=0.20, val_ratio_of_temp=0.50, random_state=42)

train_df.to_csv(paths.train_path, index=False)
val_df.to_csv(paths.val_path, index=False)
test_df.to_csv(paths.test_path, index=False)

MAX_LEN = 70
train_filtered = filter_by_max_length(train_df, max_len=MAX_LEN)
val_filtered = filter_by_max_length(val_df, max_len=MAX_LEN)
test_filtered = filter_by_max_length(test_df, max_len=MAX_LEN)

train_filtered.to_csv(paths.train_filtered_path, index=False)
val_filtered.to_csv(paths.val_filtered_path, index=False)
test_filtered.to_csv(paths.test_filtered_path, index=False)

eng_vocab = build_vocabulary_dict(train_filtered["eng"].tolist(), min_freq=5)
amh_vocab = build_vocabulary_dict(train_filtered["amh"].tolist(), min_freq=5)
save_vocab(eng_vocab, paths.eng_vocab_path)
save_vocab(amh_vocab, paths.amh_vocab_path)

print(f"Clean rows: {len(clean_df):,}")
print(f"Train rows: {len(train_filtered):,}")
print(f"Validation rows: {len(val_filtered):,}")
print(f"Test rows: {len(test_filtered):,}")
print(f"English vocab entries: {len(eng_vocab):,}")
print(f"Amharic vocab entries: {len(amh_vocab):,}")
print(f"Saved vocab to: {paths.vocab_dir}")
