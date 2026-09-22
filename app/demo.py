"""Simple Gradio demo UI for the English-to-Amharic translation project."""

import os
import sys
from pathlib import Path

import gradio as gr
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.vocabulary import (
    EOS_IDX,
    PAD_IDX,
    SOS_IDX,
    UNK_IDX,
    load_vocab,
)
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.seq2seq import Seq2Seq
from src.utils.paths import get_data_paths


DATA_ROOT = os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))
paths = get_data_paths(DATA_ROOT)


def build_demo_vocab():
    english_tokens = [
        "i", "am", "going", "to", "the", "university", "today",
        "hello", "good", "morning", "this", "is", "a", "test",
        "we", "are", "students", "school", "home", "work",
    ]
    amharic_tokens = [
        "ሰላም", "ጤና", "ወደ", "ዩኒቨርሲቲ", "እሄዳለሁ", "ነው", "ዛሬ",
        "እንደ", "እኔ", "እንዴት", "ይሄ", "ነው", "ለምለም", "አስተማማኝ",
        "እኛ", "ተማሪዎች", "ቤት", "ስራ", "እዚህ", "ገንዘብ",
    ]

    eng_vocab = {token: idx + 4 for idx, token in enumerate(english_tokens)}
    amh_vocab = {token: idx + 4 for idx, token in enumerate(amharic_tokens)}
    eng_vocab["<PAD>"] = PAD_IDX
    eng_vocab["<UNK>"] = UNK_IDX
    eng_vocab["<SOS>"] = SOS_IDX
    eng_vocab["<EOS>"] = EOS_IDX
    amh_vocab["<PAD>"] = PAD_IDX
    amh_vocab["<UNK>"] = UNK_IDX
    amh_vocab["<SOS>"] = SOS_IDX
    amh_vocab["<EOS>"] = EOS_IDX
    return eng_vocab, amh_vocab


if paths.eng_vocab_path.exists() and paths.amh_vocab_path.exists():
    eng_vocab = load_vocab(paths.eng_vocab_path)
    amh_vocab = load_vocab(paths.amh_vocab_path)
else:
    eng_vocab, amh_vocab = build_demo_vocab()

inv_amh_vocab = {idx: token for token, idx in amh_vocab.items()}


def source_token_id(token: str) -> int:
    """Match common casing variants before falling back to <UNK>."""
    candidates = (token, token.lower(), token.capitalize(), token.upper())
    for candidate in candidates:
        if candidate in eng_vocab:
            return eng_vocab[candidate]
    return UNK_IDX

encoder = Encoder(input_dim=len(eng_vocab), embedding_dim=256, hidden_dim=512, num_layers=1, dropout=0.2, pad_idx=PAD_IDX)
decoder = Decoder(output_dim=len(amh_vocab), embedding_dim=256, hidden_dim=512, num_layers=1, dropout=0.2, pad_idx=PAD_IDX)
model = Seq2Seq(encoder, decoder)

model_path = PROJECT_ROOT / "models" / "baseline_seq2seq.pt"
if model_path.exists():
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
else:
    model.eval()


def translate_text(text: str) -> str:
    if not text or not text.strip():
        return ""

    tokens = str(text).strip().split()
    src_ids = [SOS_IDX] + [source_token_id(tok) for tok in tokens] + [EOS_IDX]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(1)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
        trg_indexes = [SOS_IDX]

        for _ in range(70):
            current_input = torch.tensor([trg_indexes[-1]], dtype=torch.long)
            output, hidden, cell = model.decoder(current_input, hidden, cell)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == EOS_IDX:
                break

    translated = [
        inv_amh_vocab.get(i, "<UNK>")
        for i in trg_indexes
        if i not in (SOS_IDX, EOS_IDX, PAD_IDX)
    ]
    if not translated:
        return "<UNK>"
    return " ".join(translated)


def app_main(input_text: str):
    translated = translate_text(input_text)
    return translated


iface = gr.Interface(
    fn=app_main,
    inputs=gr.Textbox(lines=3, placeholder="Enter English sentence..."),
    outputs=gr.Textbox(lines=3, label="Amharic translation"),
    title="English → Amharic NMT",
    description="Translate English text into Amharic using the trained Seq2Seq model.",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
