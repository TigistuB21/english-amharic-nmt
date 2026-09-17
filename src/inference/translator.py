"""Inference and translation generation utilities."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.data.vocabulary import (
    DEFAULT_MAX_LEN,
    EOS_IDX,
    PAD_IDX,
    SOS_IDX,
    UNK_IDX,
    sentence_to_ids,
)


def translate_sentence(
    sentence: str,
    model: nn.Module,
    src_vocab: Dict[str, int],
    trg_vocab_inv: Dict[int, str],
    max_len: int = DEFAULT_MAX_LEN,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[List[str], List[int]]:
    """
    Translate an English input sentence to Amharic using greedy decoding.

    Args:
        sentence: English source sentence.
        model: Trained Seq2Seq model.
        src_vocab: English token-to-ID vocabulary dictionary.
        trg_vocab_inv: Amharic ID-to-token dictionary.
        max_len: Maximum length for generated translation.
        device: Device to execute inference on.

    Returns:
        Tuple of (translated_tokens, translated_token_ids).
    """
    model.eval()
    dev = device if device is not None else next(model.parameters()).device

    # Tokenize and numericalize
    tokens = str(sentence).strip().split()
    src_ids = [SOS_IDX] + [src_vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=dev).unsqueeze(1)  # [src_len, 1]

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

        trg_indexes = [SOS_IDX]

        for _ in range(max_len):
            current_input = torch.tensor([trg_indexes[-1]], dtype=torch.long, device=dev)
            output, hidden, cell = model.decoder(current_input, hidden, cell)

            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            if pred_token == EOS_IDX:
                break

    # Convert IDs to words (excluding <SOS> and <EOS>)
    trg_tokens = [
        trg_vocab_inv.get(i, "<UNK>")
        for i in trg_indexes
        if i not in (SOS_IDX, EOS_IDX, PAD_IDX)
    ]

    return trg_tokens, trg_indexes
