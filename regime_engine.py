# regime_engine.py

import numpy as np
import pandas as pd
from config import BotConfig


class RegimeEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def add_trend_metrics(self, df):
        df["trend_raw"] = df["ema_50"] - df["ema_200"]
        df["trend_z"] = df["trend_raw"] / (df["trend_raw"].rolling(200).std() + 1e-12)
        return df

    def add_vol_z(self, df):
        vol = df["vol_short"]
        mu = vol.rolling(500).mean()
        sigma = vol.rolling(500).std()
        df["vol_z"] = (vol - mu) / (sigma + 1e-12)
        return df

    def classify_regime(self, df):
        df = df.copy()
        df = self.add_trend_metrics(df)
        df = self.add_vol_z(df)

        regimes = []
        for _, r in df.iterrows():
            if pd.isna(r["trend_z"]) or pd.isna(r["vol_z"]):
                regimes.append("unknown")
                continue

            if abs(r["trend_z"]) > self.cfg.trend_z_threshold:
                if r["vol_z"] > self.cfg.vol_z_high:
                    regimes.append("trend_high_vol")
                elif r["vol_z"] < self.cfg.vol_z_low:
                    regimes.append("trend_low_vol")
                else:
                    regimes.append("trend_normal")
            else:
                if r["vol_z"] > self.cfg.vol_z_high:
                    regimes.append("range_high_vol")
                elif r["vol_z"] < self.cfg.vol_z_low:
                    regimes.append("range_low_vol")
                else:
                    regimes.append("range_normal")

        df["regime"] = regimes
        return df
