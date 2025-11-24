# ================================================================
#  RISK ENGINE — V10-TR CCT-90
#  Handles position sizing, TP/SL conversion, PnL logic
# ================================================================

from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import V10TRConfig
from utils import logger


@dataclass
class RiskEngine:
    """
    Handles daily risk for the strategy.

    Responsibilities:
        - position sizing based on ATR & capital
        - convert ATR multiples → price SL/TP levels
        - evaluate intraday hit of TP or SL
        - update equity curve

    Assumes:
        - 1 trade max per day
        - no pyramiding
        - all trade exits via TP or SL within day
    """

    cfg: V10TRConfig
    initial_equity: float = 10000.0   # starting capital

    # ------------------------------------------------------------
    #  ATR-BASED TP / SL CALCULATIONS
    # ------------------------------------------------------------
    def compute_sl_tp(self, entry_price: float, atr: float) -> tuple:
        """
        Given entry price and ATR, compute stop loss & take profit.

        SL = entry - ATR * sl_mult
        TP = entry + ATR * tp_mult
        """
        tp = entry_price + self.cfg.tp_atr_mult * atr
        sl = entry_price - self.cfg.sl_atr_mult * atr
        return sl, tp

    # ------------------------------------------------------------
    #  POSITION SIZING — SIMPLE FIXED FRACTIONAL (ATR-SCALED)
    # ------------------------------------------------------------
    def compute_position_size(self, equity: float, atr: float) -> float:
        """
        Position size is determined by risk per trade (fractional)
        divided by ATR in price terms.

        risk_amount = equity * position_risk
        size = risk_amount / (ATR * sl_mult)

        Ensures risk is consistent across volatility regimes.
        """
        risk_amt = equity * self.cfg.position_risk

        if atr <= 0:
            return 0.0

        # Dollar distance from entry to SL
        sl_distance = atr * self.cfg.sl_atr_mult

        if sl_distance == 0:
            return 0.0

        size = risk_amt / sl_distance
        return max(size, 0.0)

    # ------------------------------------------------------------
    #  CHECK INTRADAY HIT OF TP/SL
    # ------------------------------------------------------------
    def check_exit(self, sl: float, tp: float,
                   day_open: float, day_high: float, day_low: float) -> tuple:
        """
        Returns:
            exit_price, outcome_flag
            outcome_flag: +1 = TP hit, -1 = SL hit, 0 = neither

        Priority: whichever is hit first in the day.
        """
        # TP hit
        if day_high >= tp:
            return tp, +1

        # SL hit
        if day_low <= sl:
            return sl, -1

        # Neither touched
        return day_open, 0

    # ------------------------------------------------------------
    #  APPLY TRADE PNL TO EQUITY
    # ------------------------------------------------------------
    def update_equity(self, equity: float,
                      position: float,
                      entry_price: float,
                      exit_price: float,
                      direction: int,
                      outcome: int) -> float:
        """
        equity: previous equity
        position: number of units
        entry_price, exit_price: floats
        direction: +1 or -1
        outcome:  +1, -1, or 0

        pnl = direction * position * (exit - entry)
        """
        pnl = direction * position * (exit_price - entry_price)
        new_eq = equity + pnl
        return max(new_eq, 0.0)  # no negative equity allowed
