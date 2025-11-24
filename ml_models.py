# ml_models.py
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

@dataclass
class V10Ensemble:
    cfg: any

    def __post_init__(self):
        # Experts
        self.exp_trend = GradientBoostingClassifier(
            n_estimators=self.cfg.gb_n_estimators,
            learning_rate=self.cfg.gb_learning_rate,
            max_depth=self.cfg.gb_max_depth
        )
        self.exp_range = RandomForestClassifier(
            n_estimators=self.cfg.rf_n_estimators,
            max_depth=self.cfg.rf_max_depth
        )
        self.exp_shock = RandomForestClassifier(
            n_estimators=self.cfg.rf_n_estimators,
            max_depth=self.cfg.rf_max_depth
        )

    def fit(self, X, y, regimes):
        # split by regime
        X_t, y_t = X[regimes==0], y[regimes==0]
        X_r, y_r = X[regimes==1], y[regimes==1]
        X_s, y_s = X[regimes==2], y[regimes==2]

        if len(y_t)>10: self.exp_trend.fit(X_t, y_t)
        if len(y_r)>10: self.exp_range.fit(X_r, y_r)
        if len(y_s)>10: self.exp_shock.fit(X_s, y_s)

    def predict_proba(self, X, regime):
        if regime==0: return self.exp_trend.predict_proba(X)[0]
        if regime==1: return self.exp_range.predict_proba(X)[0]
        return self.exp_shock.predict_proba(X)[0]

    def signal(self, X, regime):
        proba = self.predict_proba(X, regime)
        # class order: [-1,0,1]
        cls = [-1,0,1]
        p_neg = proba[cls.index(-1)]
        p_pos = proba[cls.index(1)]
        return p_pos - p_neg

