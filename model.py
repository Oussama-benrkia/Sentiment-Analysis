"""
Shared inference code: text cleaning, vocabulary, and the BiLSTM model.
Used by both predict.py and app.py.
"""

import re
import pickle
from collections import Counter

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Vocabulary ─────────────────────────────────────────────────────────────────

class Vocabulary:
    """Word-to-index mapping. Index 0 = <PAD>, index 1 = <UNK>."""

    def __init__(self, max_size: int = 25_000):
        self.max_size = max_size
        self.word2idx: dict = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: dict = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts: list) -> None:
        counter = Counter(w for t in texts for w in t.split())
        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str, max_len: int) -> tuple:
        tokens = text.split()[:max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]
        length = max(len(ids), 1)
        ids   += [0] * (max_len - len(ids))
        return ids, length

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Vocabulary":
        # Custom unpickler: maps Vocabulary from any module (e.g. __main__,
        # __mp_main__ used by uvicorn's reloader) to this class.
        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "Vocabulary":
                    return Vocabulary
                return super().find_class(module, name)

        with open(path, "rb") as f:
            return _Unpickler(f).load()

    def __len__(self) -> int:
        return len(self.word2idx)


# ── Model ──────────────────────────────────────────────────────────────────────

class SentimentBiLSTM(nn.Module):
    """
    Embedding → Dropout → BiLSTM (2 layers) → concat hidden states → Linear
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 n_layers, dropout, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc      = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(text))
        packed   = pack_padded_sequence(embedded, lengths.cpu(),
                                        batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden))
