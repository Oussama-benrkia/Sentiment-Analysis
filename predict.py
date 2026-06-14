"""
CLI prediction — uses the model trained in Colab.

Usage:
    python predict.py "This film was absolutely brilliant!"
    python predict.py          # interactive mode
"""

import os
import sys

import torch
import torch.nn.functional as F

import config
from model import Vocabulary, SentimentBiLSTM, clean_text

LABELS = {
    2: {0: "Negative", 1: "Positive"},
    3: {0: "Negative", 1: "Neutral",  2: "Positive"},
}
EMOJIS = {"Positive": "😊", "Negative": "😡", "Neutral": "😐"}


def load_model():
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No model at '{config.MODEL_PATH}'.\n"
            "Download best_model.pth from Colab and rename it to "
            f"{config.MODEL_PATH}"
        )
    if not os.path.exists(config.VOCAB_PATH):
        raise FileNotFoundError(
            f"No vocabulary at '{config.VOCAB_PATH}'.\n"
            "Download vocab.pkl from Colab and place it at "
            f"{config.VOCAB_PATH}"
        )
    vocab = Vocabulary.load(config.VOCAB_PATH)
    model = SentimentBiLSTM(
        len(vocab), config.EMBED_DIM, config.HIDDEN_DIM,
        config.NUM_CLASSES, config.N_LAYERS, config.DROPOUT,
    )
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(torch.device(config.DEVICE))
    model.eval()
    return model, vocab


def predict(text: str, model: SentimentBiLSTM, vocab: Vocabulary) -> tuple:
    ids, length = vocab.encode(clean_text(text), config.MAX_SEQ_LEN)
    ids_t = torch.tensor([ids], dtype=torch.long).to(config.DEVICE)
    len_t = torch.tensor([length])
    with torch.no_grad():
        probs = F.softmax(model(ids_t, len_t), dim=1).squeeze()
    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()
    label      = LABELS[config.NUM_CLASSES][pred_idx]
    return label, EMOJIS[label], confidence


def _print(text, label, emoji, conf):
    bar = "█" * int(conf * 30) + "░" * (30 - int(conf * 30))
    print(f"\n  Text       : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Sentiment  : {emoji}  {label}")
    print(f"  Confidence : [{bar}]  {conf:.1%}\n")


if __name__ == "__main__":
    model, vocab = load_model()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        _print(text, *predict(text, model, vocab))
    else:
        print("Sentiment Predictor — type 'quit' to exit\n")
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in ("quit", "exit", "q"):
                break
            _print(text, *predict(text, model, vocab))
