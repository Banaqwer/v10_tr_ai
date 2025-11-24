# ================================================================
#  EXPERTS ENGINE — V10-TR CCT-90
#  5 Expert models driven by transformer context + features
# ================================================================

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from config import V10TRConfig
from utils import logger


# ------------------------------------------------------------
#  HELPER: unified probability output for classes [-1, 0, 1]
# ------------------------------------------------------------

def _align_proba_to_classes(clf, proba: np.ndarray) -> np.ndarray:
    """
    Ensure probability array is ordered as [-1, 0, 1] even if
    classifier classes_ are a subset or in different order.
    """
    target_classes = [-1, 0, 1]
    out = np.zeros(3, dtype=float)

    for i, cls in enumerate(clf.classes_):
        if cls in target_classes:
            idx = target_classes.index(cls)
            out[idx] = proba[i]

    # If some classes missing, remaining probs stay 0; renormalize to sum 1 if possible
    s = out.sum()
    if s > 0:
        out /= s
    return out


# ------------------------------------------------------------
#  EXPERTS ENGINE
# ------------------------------------------------------------

@dataclass
class ExpertsEngine:
    """
    Maintains 5 expert models:

      - trend_continuation
      - trend_reversal
      - mean_reversion
      - volatility_breakout
      - shock

    Each expert is a classifier predicting P(y = -1, 0, +1)
    given a combined feature + context vector.

    Training is done on subsets of data based on regimes:
      - Regime 1 (trend): feeds trend_continuation & trend_reversal
      - Regime 0 (range): feeds mean_reversion
      - Regime 2 (shock): feeds volatility_breakout & shock
    """

    cfg: V10TRConfig
    experts: Dict[str, Optional[object]] = None

    def __post_init__(self):
        self.experts = {name: None for name in self.cfg.expert_types}

    # --------------------------------------------------------
    #  INTERNAL: create a classifier for a given expert type
    # --------------------------------------------------------
    def _make_expert_model(self, expert_name: str):
        """
        Different experts can use different base models.
        """
        if expert_name in ("trend_continuation", "trend_reversal"):
            # Trend experts: Gradient Boosting (handles complex boundaries well)
            return GradientBoostingClassifier(
                n_estimators=self.cfg.gb_estimators,
                learning_rate=self.cfg.gb_learning_rate,
                max_depth=self.cfg.gb_max_depth,
                subsample=0.9
            )
        elif expert_name in ("mean_reversion", "volatility_breakout", "shock"):
            # Range / breakout / shock: Random Forest (robust to noise)
            return RandomForestClassifier(
                n_estimators=self.cfg.rf_estimators,
                max_depth=self.cfg.rf_max_depth,
                n_jobs=-1
            )
        else:
            # Fallback
            return RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                n_jobs=-1
            )

    # --------------------------------------------------------
    #  PUBLIC: build combined feature + context training matrix
    # --------------------------------------------------------
    def build_training_matrix(
        self,
        feature_matrix: np.ndarray,
        context_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Concatenate last-day features with transformer context vectors.

        feature_matrix: [N, n_features]
        context_matrix: [N, d_model]

        Returns: [N, n_features + d_model]
        """
        if feature_matrix.shape[0] != context_matrix.shape[0]:
            raise ValueError(
                f"[EXPERTS] Feature rows ({feature_matrix.shape[0]}) "
                f"!= Context rows ({context_matrix.shape[0]})"
            )

        X_full = np.concatenate([feature_matrix, context_matrix], axis=1)
        return X_full.astype(np.float32)

    # --------------------------------------------------------
    #  PUBLIC: fit all experts using X, y, regimes
    # --------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, regimes: np.ndarray):
        """
        Fit each expert on the subset of data relevant to its regime.

        X: [N, d_input]     combined features + context
        y: [N]              labels in {-1, 0, +1}
        regimes: [N]        regime ids {0, 1, 2}
        """
        N = len(y)
        if len(regimes) != N or X.shape[0] != N:
            raise ValueError("[EXPERTS] X, y, regimes must have same length.")

        # --- Trend experts: regime == 1
        mask_trend = regimes == 1
        X_trend = X[mask_trend]
        y_trend = y[mask_trend]

        # --- Range expert: regime == 0
        mask_range = regimes == 0
        X_range = X[mask_range]
        y_range = y[mask_range]

        # --- Shock experts: regime == 2
        mask_shock = regimes == 2
        X_shock = X[mask_shock]
        y_shock = y[mask_shock]

        # Fit trend_continuation
        if X_trend.shape[0] > 50:
            model = self._make_expert_model("trend_continuation")
            model.fit(X_trend, y_trend)
            self.experts["trend_continuation"] = model
            logger.info(f"[EXPERTS] Trained trend_continuation on {X_trend.shape[0]} samples.")
        else:
            logger.warning("[EXPERTS] Not enough samples for trend_continuation; leaving None.")

        # Fit trend_reversal
        if X_trend.shape[0] > 50:
            model = self._make_expert_model("trend_reversal")
            model.fit(X_trend, y_trend)
            self.experts["trend_reversal"] = model
            logger.info(f"[EXPERTS] Trained trend_reversal on {X_trend.shape[0]} samples.")
        else:
            logger.warning("[EXPERTS] Not enough samples for trend_reversal; leaving None.")

        # Fit mean_reversion
        if X_range.shape[0] > 50:
            model = self._make_expert_model("mean_reversion")
            model.fit(X_range, y_range)
            self.experts["mean_reversion"] = model
            logger.info(f"[EXPERTS] Trained mean_reversion on {X_range.shape[0]} samples.")
        else:
            logger.warning("[EXPERTS] Not enough samples for mean_reversion; leaving None.")

        # Fit volatility_breakout
        if X_shock.shape[0] > 30:
            model = self._make_expert_model("volatility_breakout")
            model.fit(X_shock, y_shock)
            self.experts["volatility_breakout"] = model
            logger.info(f"[EXPERTS] Trained volatility_breakout on {X_shock.shape[0]} samples.")
        else:
            logger.warning("[EXPERTS] Not enough samples for volatility_breakout; leaving None.")

        # Fit shock expert
        if X_shock.shape[0] > 30:
            model = self._make_expert_model("shock")
            model.fit(X_shock, y_shock)
            self.experts["shock"] = model
            logger.info(f"[EXPERTS] Trained shock expert on {X_shock.shape[0]} samples.")
        else:
            logger.warning("[EXPERTS] Not enough samples for shock expert; leaving None.")

    # --------------------------------------------------------
    #  INTERNAL: predict proba from a single expert
    # --------------------------------------------------------
    def _expert_proba(self, name: str, x_row: np.ndarray) -> np.ndarray:
        model = self.experts.get(name)
        if model is None:
            # no model trained → neutral distribution
            return np.array([1/3, 1/3, 1/3], dtype=float)

        proba = model.predict_proba(x_row.reshape(1, -1))[0]
        return _align_proba_to_classes(model, proba)

    # --------------------------------------------------------
    #  PUBLIC: combined probability given regime + x_row
    # --------------------------------------------------------
    def predict_proba(self, x_row: np.ndarray, regime: int) -> np.ndarray:
        """
        x_row: 1D numpy array [d_input]
        regime: {0, 1, 2}

        Returns: probability vector [P(-1), P(0), P(+1)]
        combining relevant experts.
        """
        # RANGE regime → mean reversion expert
        if regime == 0:
            return self._expert_proba("mean_reversion", x_row)

        # TREND regime → blend continuation and reversal experts
        if regime == 1:
            p_cont = self._expert_proba("trend_continuation", x_row)
            p_rev = self._expert_proba("trend_reversal", x_row)
            # Equal weighting; could be refined later using validation
            p = 0.6 * p_cont + 0.4 * p_rev
            # normalize
            s = p.sum()
            if s > 0:
                p /= s
            return p

        # SHOCK regime → blend breakout + shock experts
        if regime == 2:
            p_brk = self._expert_proba("volatility_breakout", x_row)
            p_shk = self._expert_proba("shock", x_row)
            p = 0.5 * p_brk + 0.5 * p_shk
            s = p.sum()
            if s > 0:
                p /= s
            return p

        # Fallback: neutral if regime is unexpected
        return np.array([1/3, 1/3, 1/3], dtype=float)
