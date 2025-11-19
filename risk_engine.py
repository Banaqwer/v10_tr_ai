# risk_engine.py

import numpy as np
import pandas as pd
from config import BotConfig


class RiskEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def compute_units(self, equity, r, sig):
        atr = r["atr"]
        if atr <= 0:
            return 0

        risk_amt = equity * self.cfg.risk_per_trade
        sl_dist = 2 * atr

        units = risk_amt / sl_dist
        units *= abs(sig)
        return units if sig > 0 else -units

    def compute_sl_tp(self, r, sig):
        price = r["close"]
        atr = r["atr"]

        if sig > 0:
            sl = price - 2 * atr
            tp = price + 3 * atr
        else:
            sl = price + 2 * atr
            tp = price - 3 * atr
        return sl, tp
