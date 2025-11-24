# ================================================================
#  LABEL ENGINE — V10-TR CCT-90
#  Builds classification labels using ATR TP/SL forward simulation
# ================================================================

from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import V10TRConfig


@dataclass
class LabelEngine:
    cfg: V10TRConfig

    # ------------------------------------------------------------
    #  BUILD LABELS ON FEATURE-ENRICHED DATAFRAME
    # ------------------------------------------------------------
    def build_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a column 'label' to df:
            +1 → Take Profit hit first
            -1 → Stop Loss hit first
             0 → Neither hit within horizon

        df MUST contain:
            Close, High, Low, ATR
        """
        df = df.copy()

        horizon = self.cfg.forward_horizon_days
        tp_mult = self.cfg.tp_atr_mult
        sl_mult = self.cfg.sl_atr_mult

        labels = []

        closes = df["Close"].values
        highs = df["High"].values
        lows = df["Low"].values
        atrs = df["ATR"].values

        N = len(df)

        for i in range(N):
            if i + 1 >= N:
                labels.append(0)
                continue

            entry = closes[i]
            atr = atrs[i]

            tp = entry + tp_mult * atr
            sl = entry - sl_mult * atr

            # Look forward up to horizon days
            hit_tp = False
            hit_sl = False

            end = min(i + horizon, N - 1)

            for j in range(i + 1, end + 1):
                if highs[j] >= tp:
                    hit_tp = True
                    break
                if lows[j] <= sl:
                    hit_sl = True
                    break

            if hit_tp and not hit_sl:
                labels.append(+1)
            elif hit_sl and not hit_tp:
                labels.append(-1)
            else:
                labels.append(0)

        df["label"] = labels
        return df

    # ------------------------------------------------------------
    #  EXTRACT LABEL VECTOR AS NUMPY ARRAY
    # ------------------------------------------------------------
    def get_label_vector(self, df_labeled: pd.DataFrame) -> np.ndarray:
        """
        Returns a numpy vector of labels.
        Values: +1, 0, -1
        """
        if "label" not in df_labeled.columns:
            raise ValueError("[LABEL] Missing 'label' column.")

        return df_labeled["label"].values.astype(np.int64)
