# strategy.py
#
# V10-Daily Strategy Engine
# Uses a trained model (EnsembleSignalModel) to produce trading signals
# and turn them into discrete long/short/flat positions.

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class StrategyEngine:
    cfg: any
    model: any  # any object with predict_signal(df_row) -> float in [-1, 1]

    def compute_signal(self, row: pd.Series) -> float:
        """
        Use the ML model to compute a continuous signal in [-1, +1].
        """
        signal = self.model.predict_signal(row)
        return float(signal)

    def to_position(self, signal: float) -> int:
        """
        Convert continuous signal to discrete position:
          +1 = long
          -1 = short
           0 = flat
        using cfg.min_signal_abs as threshold.
        """
        thr = getattr(self.cfg, "min_signal_abs", 0.25)

        if signal > thr:
            return 1
        elif signal < -thr:
            return -1
        else:
            return 0

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply the strategy over a DataFrame of features, returning
        a Series of discrete positions indexed like df.
        """
        signals = []
        for idx, row in df.iterrows():
            s = self.compute_signal(row)
            pos = self.to_position(s)
            signals.append(pos)

        positions = pd.Series(signals, index=df.index, name="position")
        return positions
