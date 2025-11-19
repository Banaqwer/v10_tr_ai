# risk_engine.py (V10 DAILY – conservative ATR risk)

import numpy as np
import pandas as pd
from config import BotConfig


class RiskEngine:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    def compute_position_size(
        self,
        equity: float,
        row: pd.Series,
        signal: float
    ) -> float:
        """
        Position sizing based on DAILY ATR.

        risk_per_trade is fraction of current equity (e.g. 0.5%).
        Stop distance is ~1.5 * ATR, so units = risk_amount / stop_distance.
        """
        atr = row.get("atr", np.nan)
        if np.isnan(atr) or atr <= 0:
            return 0.0

        risk_amount = equity * self.cfg.risk_per_trade
        sl_distance = 1.5 * atr  # DAILY: 1.5 * ATR stop
        if sl_distance <= 0:
            return 0.0

        units = risk_amount / sl_distance
        # scale by signal strength (0..1)
        units *= abs(signal)

        if signal < 0:
            units = -units
        return units

    def compute_sl_tp(self, row: pd.Series, signal: float) -> (float, float):
        """
        Stop and target for DAILY timeframe.

        - Stop ~ 1.5 ATR away
        - Target ~ 2.5 ATR away

        This gives a reward:risk of ~1.67:1, with high win-rate bias.
        """
        price = row["close"]
        atr = row.get("atr", np.nan)
        if np.isnan(atr) or atr <= 0:
            return np.nan, np.nan

        sl_mult = 1.5
        tp_mult = 2.5

        if signal > 0:  # long
            sl = price - sl_mult * atr
            tp = price + tp_mult * atr
        elif signal < 0:  # short
            sl = price + sl_mult * atr
            tp = price - tp_mult * atr
        else:
            sl, tp = np.nan, np.nan

        return sl, tp

