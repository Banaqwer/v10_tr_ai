# regime_engine.py
#
# V10-Daily Regime Engine
# Computes EMAs internally and classifies market regime:
#   +1 = bull, -1 = bear, 0 = neutral

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class RegimeEngine:
    cfg: any

    def __post_init__(self):
        # default params; can be overridden by cfg
        self.fast_span = getattr(self.cfg, "regime_fast_span", 50)
        self.slow_span = getattr(self.cfg, "regime_slow_span", 200)
        # threshold for normalized EMA spread
        self.trend_threshold = getattr(self.cfg, "regime_trend_threshold", 0.001)

    # ---------------------------------------------------------
    # Internal: compute EMA-based trend metrics
    # ---------------------------------------------------------

    def add_trend_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ensure we have close prices
        if "close" not in df.columns:
            raise ValueError("RegimeEngine requires 'close' column in DataFrame.")

        # EMAs computed here (no dependency on FeatureEngine)
        df["ema_50"] = df["close"].ewm(span=self.fast_span, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=self.slow_span, adjust=False).mean()

        # Raw trend and normalized trend
        df["trend_raw"] = df["ema_50"] - df["ema_200"]
        df["trend_strength"] = df["trend_raw"] / (df["close"] + 1e-12)

        return df

    # ---------------------------------------------------------
    # Public: classify regime
    # ---------------------------------------------------------

    def classify_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_trend_metrics(df)

        thr = self.trend_threshold

        # Bull / bear / neutral regimes
        regime_vals = np.where(
            df["trend_strength"] > thr,
            1,
            np.where(df["trend_strength"] < -thr, -1, 0),
        )

        df["regime"] = regime_vals

        # Drop initial NA rows due to EMA warmup
        df = df.dropna(subset=["ema_50", "ema_200"])

        return df
