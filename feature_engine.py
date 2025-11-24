# ================================================================
#  FEATURE ENGINE — V10-TR CCT-90
#  Builds classical + reasoning features from daily OHLCV
# ================================================================

from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import V10TRConfig
from utils import compute_atr


# List of numeric feature columns that will feed into CCT-90
FEATURE_COLS = [
    # Returns
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",

    # Volatility
    "vol_5",
    "vol_10",
    "vol_20",

    # Range / amplitude
    "range_pct",

    # EMAs (trend structure)
    "ema_10",
    "ema_20",
    "ema_50",
    "ema_100",
    "ema_200",

    # Trend & strength
    "trend_raw_50_200",
    "trend_strength",

    # ATR-based
    "ATR",
    "atr_pct",

    # Reasoning-style context features
    "trend_compression",
    "vol_ratio_10_20",
    "price_vs_ema20_atr",
    "price_vs_ema50_atr",
]


@dataclass
class FeatureEngine:
    """
    Turns OHLCV into a rich feature set for the Transformer + experts.

    Input DataFrame columns expected:
        Open, High, Low, Close, Volume

    Output DataFrame:
        Original price columns + all FEATURE_COLS
    """
    cfg: V10TRConfig

    # ------------------------------------------------------------
    #  BUILD FEATURES ON A PRICE DATAFRAME
    # ------------------------------------------------------------
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Basic sanity
        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                raise ValueError(f"[FEATURE] Missing required column: {col}")

        # =====================
        # RETURNS
        # =====================
        df["ret_1"] = df["Close"].pct_change(1)
        df["ret_3"] = df["Close"].pct_change(3)
        df["ret_5"] = df["Close"].pct_change(5)
        df["ret_10"] = df["Close"].pct_change(10)

        # =====================
        # VOLATILITY (rolling std of daily returns)
        # =====================
        df["vol_5"] = df["ret_1"].rolling(5).std()
        df["vol_10"] = df["ret_1"].rolling(10).std()
        df["vol_20"] = df["ret_1"].rolling(20).std()

        # =====================
        # RANGE PERCENT
        # =====================
        df["range_pct"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)

        # =====================
        # EXPONENTIAL MOVING AVERAGES
        # =====================
        df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["ema_100"] = df["Close"].ewm(span=100, adjust=False).mean()
        df["ema_200"] = df["Close"].ewm(span=200, adjust=False).mean()

        # =====================
        # TREND STRUCTURE
        # =====================
        df["trend_raw_50_200"] = df["ema_50"] - df["ema_200"]
        df["trend_strength"] = df["trend_raw_50_200"] / (df["Close"] + 1e-9)

        # =====================
        # ATR & ATR-BASED FEATURES
        # =====================
        df["ATR"] = compute_atr(df, window=self.cfg.atr_window)
        df["atr_pct"] = df["ATR"] / (df["Close"] + 1e-9)

        # =====================
        # REASONING-STYLE FEATURES (GPT-like context)
        # =====================

        # Trend compression: is the 50–200 EMA spread small relative to volatility?
        # High compression often precedes big moves (breakouts).
        roll_std_close_20 = df["Close"].rolling(20).std()
        df["trend_compression"] = (
            df["trend_raw_50_200"].abs() / (roll_std_close_20 + 1e-9)
        )

        # Volatility ratio: short-term vol vs longer-term vol.
        # Tells if volatility is expanding or contracting.
        df["vol_ratio_10_20"] = df["vol_10"] / (df["vol_20"] + 1e-9)

        # Price stretched away from mean, scaled by ATR.
        # Captures mean-reversion tension.
        df["price_vs_ema20_atr"] = (df["Close"] - df["ema_20"]) / (df["ATR"] + 1e-9)
        df["price_vs_ema50_atr"] = (df["Close"] - df["ema_50"]) / (df["ATR"] + 1e-9)

        # Drop warm-up rows where indicators are NaN
        df = df.dropna().copy()

        return df

    # ------------------------------------------------------------
    #  EXTRACT FEATURE MATRIX AS NUMPY ARRAY
    # ------------------------------------------------------------
    def get_feature_matrix(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Returns a NumPy matrix [N_days, n_features] in the order of FEATURE_COLS.
        This is what the CCT-90 embedder will consume.
        """
        missing = [c for c in FEATURE_COLS if c not in df_features.columns]
        if missing:
            raise ValueError(f"[FEATURE] Missing feature columns: {missing}")

        mat = df_features[FEATURE_COLS].values.astype(np.float32)
        return mat
