# ================================================================
#  STRATEGY ENGINE — V10-TR CCT-90
#  Converts expert probabilities into trading signals
# ================================================================

from dataclasses import dataclass
import numpy as np

from config import V10TRConfig
from experts_engine import ExpertsEngine
from utils import logger


@dataclass
class Strategy:
    """
    Strategy wrapper around the ExpertsEngine.

    Responsibilities:
      - For a given input row x_row (features + context) and regime,
        query the experts for P(y = -1, 0, +1).
      - Convert probabilities into a discrete signal:
            +1  → go long
             0  → stay flat
            -1  → go short

    Uses an "edge" metric:
        edge = P(+1) - P(-1)

    If edge > threshold_long → go long
    If edge < -threshold_short → go short
    Else → flat
    """

    cfg: V10TRConfig
    experts: ExpertsEngine
    threshold: float = 0.20   # conviction threshold for entering a trade

    # ------------------------------------------------------------
    #  CONVERT PROBABILITIES INTO A SIGNAL
    # ------------------------------------------------------------
    def signal_from_proba(self, proba: np.ndarray) -> int:
        """
        proba: numpy array [P(-1), P(0), P(+1)]
        """
        if proba.shape[0] != 3:
            raise ValueError(f"[STRATEGY] Expected proba of length 3, got {len(proba)}")

        p_neg, p_zero, p_pos = float(proba[0]), float(proba[1]), float(proba[2])

        # Edge in favor of long vs short
        edge = p_pos - p_neg

        if edge > self.threshold:
            return +1
        elif edge < -self.threshold:
            return -1
        else:
            return 0

    # ------------------------------------------------------------
    #  PUBLIC: GENERATE SIGNAL FOR A SINGLE ROW
    # ------------------------------------------------------------
    def generate_signal(self, x_row: np.ndarray, regime: int) -> int:
        """
        x_row: 1D numpy array representing combined features + context
        regime: integer {0, 1, 2} from RegimeEngine

        Returns: +1, 0, or -1
        """
        proba = self.experts.predict_proba(x_row, regime)
        sig = self.signal_from_proba(proba)

        logger.debug(
            f"[STRATEGY] regime={regime}, proba={proba}, signal={sig}"
        )

        return sig
