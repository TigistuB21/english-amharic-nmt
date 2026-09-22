"""Evaluate the trained seq2seq model and save a compact report."""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import sacrebleu
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import TranslationDataset, get_dataloader
from src.data.vocabulary import load_vocab
from src.inference.translator import translate_sentence
from src.models.attention import AttentionDecoder, AttentionSeq2Seq
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.seq2seq import Seq2Seq, count_parameters
from src.models.transformer import TransformerSeq2Seq
from src.training.evaluate import evaluate_epoch
from src.utils.paths import get_data_paths


DATA_ROOT = os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))
paths = get_data_paths(DATA_ROOT)
max_samples = int(os.getenv("MAX_EVAL_SAMPLES", "20"))
max_len = int(os.getenv("MAX_LEN", "70"))
model_type = os.getenv("MODEL_TYPE", "transformer").lower()

eng_vocab = load_vocab(paths.eng_vocab_path)
amh_vocab = load_vocab(paths.amh_vocab_path)
amh_vocab_inv = {index: token for token, index in amh_vocab.items()}
test_df = pd.read_csv(paths.test_filtered_path).head(max_samples)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_type == "attention":
    encoder = Encoder(len(eng_vocab), embedding_dim=256, hidden_dim=512, num_layers=1, dropout=0.2)
    decoder = AttentionDecoder(len(amh_vocab), embedding_dim=256, hidden_dim=512, num_layers=1, dropout=0.2)
    model = AttentionSeq2Seq(encoder, decoder, device=device).to(device)
    checkpoint_path = PROJECT_ROOT / "models" / "attention_seq2seq.pt"
elif model_type == "transformer":
    model = TransformerSeq2Seq(
        src_vocab_size=len(eng_vocab),
        trg_vocab_size=len(amh_vocab),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=200,
        device=device,
    ).to(device)
    checkpoint_path = PROJECT_ROOT / "models" / "transformer_seq2seq.pt"
else:
    encoder = Encoder(len(eng_vocab), embedding_dim=256, hidden_dim=512, num_layers=1, dropout=0.2)
    decoder = Decoder(len(amh_vocab), embedding_dim=256, hidden_dim=512, num_layers=1, dropout=0.2)
    model = Seq2Seq(encoder, decoder, device=device).to(device)
    checkpoint_path = PROJECT_ROOT / "models" / "baseline_seq2seq.pt"

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss(ignore_index=0)
dataset = TranslationDataset(test_df, eng_vocab, amh_vocab, max_len=max_len)
loader = get_dataloader(dataset, batch_size=min(4, max_samples), shuffle=False)
test_loss = evaluate_epoch(model, loader, criterion, device=device)

references = []
predictions = []
examples = []
start = time.perf_counter()
for row in test_df.itertuples(index=False):
    source = str(row.eng)
    reference = str(row.amh)
    tokens, _ = translate_sentence(source, model, eng_vocab, amh_vocab_inv, max_len=max_len, device=device)
    prediction = " ".join(tokens)
    references.append(reference)
    predictions.append(prediction)
    if len(examples) < min(5, max_samples):
        examples.append({"english": source, "reference": reference, "prediction": prediction})
inference_seconds = time.perf_counter() - start

report = {
    "checkpoint": str(checkpoint_path),
    "device": str(device),
    "samples": len(test_df),
    "test_loss": test_loss,
    "bleu": sacrebleu.corpus_bleu(predictions, [references]).score,
    "chrf": sacrebleu.corpus_chrf(predictions, [references]).score,
    "inference_seconds": inference_seconds,
    "seconds_per_sentence": inference_seconds / len(test_df) if len(test_df) else 0.0,
    "trainable_parameters": count_parameters(model),
    "examples": examples,
}

output_path = PROJECT_ROOT / "evaluation" / "results" / "baseline_report.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"Saved report to {output_path}")
