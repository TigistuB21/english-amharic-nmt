"""Train a simple baseline seq2seq model on the prepared English-Amharic corpus."""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import TranslationDataset, get_dataloader
from src.data.vocabulary import load_vocab
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq, count_parameters
from src.training.train import train_epoch
from src.training.evaluate import evaluate_epoch
from src.utils.paths import get_data_paths


DATA_ROOT = os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))
paths = get_data_paths(DATA_ROOT)

eng_vocab = load_vocab(paths.eng_vocab_path)
amh_vocab = load_vocab(paths.amh_vocab_path)

train_df = __import__("pandas").read_csv(paths.train_filtered_path)
val_df = __import__("pandas").read_csv(paths.val_filtered_path)

train_dataset = TranslationDataset(train_df, eng_vocab, amh_vocab, max_len=70)
val_dataset = TranslationDataset(val_df, eng_vocab, amh_vocab, max_len=70)

batch_size = int(os.getenv("BATCH_SIZE", "32"))
train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

epochs = int(os.getenv("EPOCHS", "3"))
max_train_batches = int(os.getenv("MAX_TRAIN_BATCHES", "0")) or None
max_val_batches = int(os.getenv("MAX_VAL_BATCHES", "0")) or None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

encoder = Encoder(input_dim=len(eng_vocab), embedding_dim=256, hidden_dim=512, num_layers=1, dropout=0.2)
decoder = Decoder(output_dim=len(amh_vocab), embedding_dim=256, hidden_dim=512, num_layers=1, dropout=0.2)
model = Seq2Seq(encoder, decoder, device=device).to(device)

checkpoint_path = PROJECT_ROOT / "models" / "baseline_seq2seq.pt"
resume_training = os.getenv("RESUME_TRAINING", "1") == "1"
if resume_training and checkpoint_path.exists():
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Resuming from checkpoint: {checkpoint_path}")

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

start = time.time()
for epoch in range(1, epochs + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, clip=1.0, teacher_forcing_ratio=0.5, device=device, max_batches=max_train_batches)
    val_loss = evaluate_epoch(model, val_loader, criterion, device=device, max_batches=max_val_batches)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

elapsed = time.time() - start
print(f"Training finished in {elapsed:.1f}s")
print(f"Number of parameters: {count_parameters(model):,}")

checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), checkpoint_path)
print(f"Saved baseline model to {checkpoint_path}")
