# ml_models.py
#
# V10-Daily ML module
# Provides:
#   - LabelBuilder
#   - EnsembleSignalModel
#
# The model returns a continuous signal in [-1, +1].

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------
# Label Builder (for daily timeframe)
# ---------------------------------------------------------------------

@dataclass
class LabelBuilder:
    cfg: any

    def build_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a future return label (binary: up/down).
        Prediction horizon: cfg.label_horizon_days.
        """

        horizon = getattr(self.cfg, "label_horizon_days", 3)

        df = df.copy()
        df["future_close"] = df["close"].shift(-horizon)
        df["future_ret"] = (df["future_close"] - df["close"]) / df["close"]

        df["label"] = np.where(df["future_ret"] > 0, 1, 0)

        df = df.dropna(subset=["future_close", "future_ret", "label"])
        return df


# ---------------------------------------------------------------------
# Ensemble Signal Model
# Produces a *continuous* signal from 2 ML models
# ---------------------------------------------------------------------

class EnsembleSignalModel:

    def __init__(self, cfg):
        self.cfg = cfg

        # Two complementary learners
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=20,
            random_state=42
        )

        self.gb = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=3,
            random_state=42
        )

        self.features = None
        self.fitted = False

    # -------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------

    def train(self, df: pd.DataFrame):

        # Identify usable columns
        feature_cols = [c for c in df.columns
                        if c not in ["open", "high", "low", "close",
                                     "future_close", "future_ret", "label",
                                     "regime"]]

        self.features = feature_cols

        X = df[feature_cols]
        y = df["label"]

        # Fit each model
        self.rf.fit(X, y)
        self.gb.fit(X, y)

        self.fitted = True

        # Score
        pred = self.predict_proba_raw(X)
        pred_class = (pred > 0.5).astype(int)
        acc = accuracy_score(y, pred_class)

        return {
            "features_used": feature_cols,
            "training_accuracy": acc,
            "rows": len(df)
        }

    # -------------------------------------------------------------
    # Internal: raw prob. prediction
    # -------------------------------------------------------------

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Returns probability of rising market from both models."""
        prf = self.rf.predict_proba(X)[:, 1]
        pgb = self.gb.predict_proba(X)[:, 1]
        return (prf + pgb) / 2

    # -------------------------------------------------------------
    # Public signal output
    # -------------------------------------------------------------

    def predict_signal(self, df_row: pd.Series) -> float:
        """
        Returns continuous signal in range [-1, +1].

        +1 → strong buy
        -1 → strong sell
         0 → neutral
        """

        if not self.fitted:
            raise RuntimeError("Model not trained yet.")

        X = df_row[self.features].values.reshape(1, -1)
        proba = self.predict_proba_raw(pd.DataFrame([df_row[self.features]]))[0]

        # Convert prob into signal
        signal = (proba - 0.5) * 2
        return float(signal)
