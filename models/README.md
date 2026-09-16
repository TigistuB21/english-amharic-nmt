# Saved Model Checkpoints

This directory stores trained model weights, checkpoints, and serialization artifacts for the English-Amharic translation models.

## Expected Contents

- **`baseline_seq2seq.pt`**: Trained Basic Seq2Seq + LSTM checkpoint.
- **`attention_seq2seq.pt`**: Trained Attention-Based Seq2Seq + LSTM checkpoint.
- **`vocab_en.pt` / `vocab_am.pt`**: Serialized vocabulary mappings.

## Note
- Model weights (`*.pt`, `*.pth`, `*.bin`) are ignored by Git (configured in `.gitignore`).
- Store training run parameters alongside checkpoints for reproducibility.
