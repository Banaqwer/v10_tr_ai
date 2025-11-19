# ml_models.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from config import BotConfig


class LabelBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_labels(self, df):
        df = df.copy()
        future = df["close"].shift(-self.cfg.horizon_bars)
        future_ret = (future - df["close"]) / df["close"]
        df["future_ret"] = future_ret

        thr = self.cfg.label_threshold
        label = np.zeros(len(df), dtype=int)
        label[future_ret > thr] = 1
        label[future_ret < -thr] = -1
        df["label"] = label
        return df


class MLSignalModel:
    def __init__(self, cfg, feature_cols=None):
        self.cfg = cfg
        self.model = RandomForestClassifier(**cfg.ml_model_params)
        self.feature_cols = feature_cols

    def _select_cols(self, df):
        if self.feature_cols:
            return self.feature_cols

        exclude = {
            "open", "high", "low", "close", "volume",
            "future_ret", "label"
        }
        return [c for c in df.columns if c not in exclude]

    def train(self, df, train_frac=0.7):
        df = df.dropna()
        df = df[df["label"] != 0]

        feats = self._select_cols(df)
        self.feature_cols = feats

        X = df[feats].values
        y = df["label"].values

        split = int(len(df) * train_frac)
        self.model.fit(X[:split], y[:split])
        pred = self.model.predict(X[split:])

        rep = classification_report(y[split:], pred, output_dict=True)
        return feats, rep

    def predict_proba_row(self, row):
        x = row[self.feature_cols].values.reshape(1, -1)
        p = self.model.predict_proba(x)[0]
        classes = list(self.model.classes_)
        p_up = p[classes.index(1)] if 1 in classes else 0
        p_down = p[classes.index(-1)] if -1 in classes else 0
        return p_up - p_down
