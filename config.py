"""
Loads all hyperparameters from the .env file in the project root.
To change a value: edit .env, no code changes needed.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

def _int(key, default):   return int(os.getenv(key, default))
def _float(key, default): return float(os.getenv(key, default))
def _str(key, default):   return os.getenv(key, default)

MAX_VOCAB_SIZE = _int  ("MAX_VOCAB_SIZE", 25_000)
MAX_SEQ_LEN    = _int  ("MAX_SEQ_LEN",    256)
EMBED_DIM      = _int  ("EMBED_DIM",      128)
HIDDEN_DIM     = _int  ("HIDDEN_DIM",     256)
N_LAYERS       = _int  ("N_LAYERS",       2)
DROPOUT        = _float("DROPOUT",        0.3)
NUM_CLASSES    = _int  ("NUM_CLASSES",    2)
BATCH_SIZE     = _int  ("BATCH_SIZE",     64)
EPOCHS         = _int  ("EPOCHS",         10)
LEARNING_RATE  = _float("LEARNING_RATE",  1e-3)
MODEL_PATH     = _str  ("MODEL_PATH",     "model/sentiment_lstm.pth")
VOCAB_PATH     = _str  ("VOCAB_PATH",     "model/vocab.pkl")

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
