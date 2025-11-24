# ================================================================
#  UTILS.PY — Shared Utilities for V10-TR CCT-90 System
#  Clean, Render-safe, consistent across all modules
# ================================================================

import numpy as np
import pandas as pd
import torch
import random
import logging


# ================================================================
#  LOGGING SETUP
# ================================================================

def get_logger(name="V10TR", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


logger = get_logger()


# ================================================================
#  SEED CONTROL FOR REPRODUCIBILITY
# ================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
#  TORCH DEVICE SELECTOR
# ================================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
#  ATR FUNCTION (True Range + Rolling Mean)
# ================================================================

def compute_atr(df, window=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = tr.rolling(window).mean()
    return atr


# ================================================================
#  PIPS CONVERSION
# ================================================================

def pips_to_price(symbol, pips):
    if symbol.endswith("JPY=X"):
        return pips * 0.01
    return pips * 0.0001


# ================================================================
#  NORMALIZATION HELPERS
# ================================================================

def minmax_scale(x):
    minv = np.min(x)
    maxv = np.max(x)
    if maxv - minv == 0:
        return np.zeros_like(x)
    return (x - minv) / (maxv - minv)


def zscore_scale(x):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return np.zeros_like(x)
    return (x - mean) / std


# ================================================================
#  SAFE ROLLING WINDOW EXTRACTOR
# ================================================================

def extract_window(arr, window):
    """
    Returns the last `window` rows. If insufficient, pads with zeros.
    """
    if len(arr) >= window:
        return arr[-window:]
    pad = np.zeros((window - len(arr), arr.shape[1]))
    return np.vstack([pad, arr])


# ================================================================
#  CCT-90 SEQUENCE CHUNKER
# ================================================================

def chunk_sequence(arr, chunk_size):
    """
    Splits a 2D array [seq_len, feature_dim] into chunks of size chunk_size.

    Example:
    90 days × 20 features → 6 chunks of 15 days each
    """
    chunks = []
    for i in range(0, len(arr), chunk_size):
        chunk = arr[i:i + chunk_size]
        if len(chunk) < chunk_size:
            pad = np.zeros((chunk_size - len(chunk), arr.shape[1]))
            chunk = np.vstack([pad, chunk])
        chunks.append(chunk)
    return np.array(chunks)  # shape: [num_chunks, chunk_size, n_features]


# ================================================================
#  BATCH PADDING (FOR TRANSFORMER)
# ================================================================

def pad_batch(sequences, max_len=None):
    """
    Pads a list of tensors to a uniform length.
    """
    if max_len is None:
        max_len = max(seq.shape[0] for seq in sequences)

    padded = []
    for seq in sequences:
        pad_len = max_len - seq.shape[0]
        if pad_len > 0:
            pad = torch.zeros((pad_len, seq.shape[1]))
            seq = torch.cat([pad, seq], dim=0)
        padded.append(seq)

    return torch.stack(padded, dim=0)


# ================================================================
#  TIME WINDOW HELPER
# ================================================================

def sliding_indices(total_len, window_size, step_size=1):
    """
    Generates start/end indices for a sliding window.
    Useful for rolling training and walk-forward backtesting.
    """
    indices = []
    for start in range(0, total_len - window_size + 1, step_size):
        end = start + window_size
        indices.append((start, end))
    return indices

