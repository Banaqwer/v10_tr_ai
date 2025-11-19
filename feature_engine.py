import pandas as pd
import numpy as np

class FeatureEngine:
    def __init__(self):
        pass

    def build_features(self, df):
        """
        Unified feature builder for DAILY data.
        Automatically fixes Yahoo Finance multiple-column issues.
        """

        # --- FIX 1: Ensure df["close"] is ALWAYS a single column ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]  # flatten MultiIndex

        # If columns contain both "Close" and "Adj Close"
        if "Close" in df.columns:
            df["close"] = df["Close"]
        elif "close" not in df.columns:
            raise ValueError("No usable close price in DF")

        # --- Range features (daily) ---
        df["daily_range"] = (df["high"] - df["low"])
        df["range_pct"] = df["daily_range"] / (df["close"] + 1e-12)

        # --- Returns ---
        df["ret_1"] = df["close"].pct_change()
        df["ret_3"] = df["close"].pct_change(3)
        df["ret_5"] = df["close"].pct_change(5)

        # --- Volatility ---
        df["vol_5"]  = df["ret_1"].rolling(5).std()
        df["vol_10"] = df["ret_1"].rolling(10).std()

        # --- Momentum ---
        df["mom_5"]  = df["close"] - df["close"].shift(5)
        df["mom_10"] = df["close"] - df["close"].shift(10)

        # --- Clean NA ---
        df = df.dropna().copy()

        return df

