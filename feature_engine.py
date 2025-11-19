# feature_engine.py

import numpy as np
import pandas as pd
from config import BotConfig


class FeatureEngineer:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    def _ema(self, s, n):
        return s.ewm(span=n, adjust=False).mean()

    def add_mas(self, df):
        df = df.copy()
        for w in self.cfg.ma_windows:
            df[f"ma_{w}"] = df["close"].rolling(w).mean()
            df[f"ema_{w}"] = self._ema(df["close"], w)
        return df

    def add_atr(self, df):
        df = df.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev).abs(),
            (low - prev).abs()
        ], axis=1).max(axis=1)

        df["atr"] = tr.rolling(self.cfg.atr_window).mean()
        return df

    def add_rsi(self, df):
        df = df.copy()
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain_ema = pd.Series(gain, index=df.index).ewm(alpha=1/self.cfg.rsi_window).mean()
        loss_ema = pd.Series(loss, index=df.index).ewm(alpha=1/self.cfg.rsi_window).mean()
        rs = gain_ema / (loss_ema + 1e-12)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def add_bb(self, df):
        df = df.copy()
        mid = df["close"].rolling(self.cfg.bb_window).mean()
        std = df["close"].rolling(self.cfg.bb_window).std()
        df["bb_mid"] = mid
        df["bb_upper"] = mid + self.cfg.bb_std * std
        df["bb_lower"] = mid - self.cfg.bb_std * std
        return df

    def add_breakouts(self, df):
        df = df.copy()
        for lb in self.cfg.breakout_lookbacks:
            df[f"high_{lb}"] = df["high"].rolling(lb).max()
            df[f"low_{lb}"] = df["low"].rolling(lb).min()
        return df

    def add_vol_features(self, df):
        df["vol_short"] = df["log_ret"].rolling(24).std()
        df["vol_long"] = df["log_ret"].rolling(120).std()
        df["vol_ratio"] = df["vol_short"] / (df["vol_long"] + 1e-12)
        return df

    def build_features(self, df):
        df = df.copy()
        df = self.add_mas(df)
        df = self.add_atr(df)
        df = self.add_rsi(df)
        df = self.add_bb(df)
        df = self.add_breakouts(df)
        df = self.add_vol_features(df)
        df = df.dropna()
        return df

