import torch

from src.data.vocabulary import build_vocabulary_dict
from src.inference.translator import translate_sentence
from src.models.attention import AttentionDecoder, AttentionSeq2Seq
from src.models.encoder import Encoder


def test_attention_model_can_translate_short_sentence():
    eng_vocab = build_vocabulary_dict(["hello world", "good morning"], min_freq=1)
    amh_vocab = build_vocabulary_dict(["ሰላም ዓለም", "ጤና አለም"], min_freq=1)
    amh_inv = {idx: token for token, idx in amh_vocab.items()}

    encoder = Encoder(len(eng_vocab), embedding_dim=8, hidden_dim=12, num_layers=1, dropout=0.0)
    decoder = AttentionDecoder(len(amh_vocab), embedding_dim=8, hidden_dim=12, num_layers=1, dropout=0.0)
    model = AttentionSeq2Seq(encoder, decoder, device="cpu")

    tokens, token_ids = translate_sentence("hello world", model, eng_vocab, amh_inv, max_len=8, device="cpu")

    assert isinstance(tokens, list)
    assert token_ids[0] == 2
    assert token_ids[-1] in {3, 0}
