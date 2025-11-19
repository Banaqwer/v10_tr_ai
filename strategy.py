# strategy.py

import numpy as np
import pandas as pd
from config import BotConfig
from ml_models import MLSignalModel


class StrategyEngine:
    def __init__(self, cfg, ml_model):
        self.cfg = cfg
        self.ml_model = ml_model

    def trend_sig(self, r):
        if r["ema_50"] > r["ema_200"]:
            return 1
        elif r["ema_50"] < r["ema_200"]:
            return -1
        return 0

    def breakout_sig(self, r):
        sig = 0
        c = r["close"]
        for lb in self.cfg.breakout_lookbacks:
            if c > r[f"high_{lb}"]:
                sig += 0.5
            elif c < r[f"low_{lb}"]:
                sig -= 0.5
        return sig

    def mean_rev_sig(self, r):
        sig = 0
        if r["rsi"] > 70:
            sig -= 0.5
        elif r["rsi"] < 30:
            sig += 0.5

        if r["close"] > r["bb_upper"]:
            sig -= 0.5
        elif r["close"] < r["bb_lower"]:
            sig += 0.5

        return sig

    def ml_sig(self, r):
        try:
            return self.ml_model.predict_proba_row(r)
        except:
            return 0

    def weight(self, regime):
        if regime.startswith("trend"):
            return {"trend": 0.6, "break": 0.3, "mr": 0.1, "ml": 0.8}
        elif regime.startswith("range"):
            return {"trend": 0.1, "break": 0.2, "mr": 0.7, "ml": 0.8}
        return {"trend": 0.3, "break": 0.3, "mr": 0.4, "ml": 0.5}

    def generate_signal(self, r):
        t = self.trend_sig(r)
        b = self.breakout_sig(r)
        m = self.mean_rev_sig(r)
        ml = self.ml_sig(r)

        w = self.weight(r["regime"])

        sig = (
            w["trend"] * t +
            w["break"] * b +
            w["mr"] * m +
            w["ml"] * ml
        )
        return max(-1, min(1, sig))
