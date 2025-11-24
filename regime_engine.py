# ================================================================
#  REGIME ENGINE — V10-TR CCT-90
#  Classifies market into: Trend / Range / Shock
#  Uses data-driven thresholds from features (no heavy ML here)
# ================================================================

from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import V10TRConfig


@dataclass
class RegimeEngine:
    """
    Regime classification based on feature structure.

    Regimes:
        0 = RANGE / NEUTRAL
        1 = TREND (up or down)
        2 = SHOCK / HIGH VOLATILITY

    Uses:
        - trend_strength
        - vol_ratio_10_20
        - atr_pct

    Thresholds are learned from each symbol's own history via quantiles.
    """
    cfg: V10TRConfig

    # thresholds learned from fit()
    trend_strength_thr: float = None
    vol_ratio_thr: float = None
    shock_atr_thr: float = None

    # ------------------------------------------------------------
    #  FIT THRESHOLDS FROM FEATURE DATAFRAME
    # ------------------------------------------------------------
    def fit(self, df_features: pd.DataFrame):
        """
        Learns thresholds from the symbol's feature history.

        df_features must contain:
            trend_strength, vol_ratio_10_20, atr_pct
        """
        if not {"trend_strength", "vol_ratio_10_20", "atr_pct"}.issubset(df_features.columns):
            missing = {"trend_strength", "vol_ratio_10_20", "atr_pct"} - set(df_features.columns)
            raise ValueError(f"[REGIME] Missing columns: {missing}")

        abs_trend = df_features["trend_strength"].abs().values
        vol_ratio = df_features["vol_ratio_10_20"].values
        atr_pct = df_features["atr_pct"].values

        # Trend regime: strong trend when |trend_strength| is above ~60th percentile
        self.trend_strength_thr = np.nanpercentile(abs_trend, 60)

        # Volatility expansion: vol_10 > vol_20 often > 1.0
        # We'll use 1.0 as baseline, but refine using ~55th percentile.
        self.vol_ratio_thr = max(1.0, np.nanpercentile(vol_ratio, 55))

        # Shock regime: when ATR% is extremely high compared to history
        self.shock_atr_thr = np.nanpercentile(atr_pct, 80)

    # ------------------------------------------------------------
    #  ASSIGN A SINGLE ROW TO A REGIME
    # ------------------------------------------------------------
    def assign_regime_row(self, row: pd.Series) -> int:
        """
        Given a feature row, return regime id: 0, 1, or 2.
        """
        if self.trend_strength_thr is None:
            raise RuntimeError("[REGIME] RegimeEngine.fit() must be called before usage.")

        ts = float(row["trend_strength"])
        vol_ratio = float(row["vol_ratio_10_20"])
        atr_pct = float(row["atr_pct"])

        abs_ts = abs(ts)

        # Shock regime: very high ATR% (exceptional volatility)
        if atr_pct >= self.shock_atr_thr:
            return 2  # SHOCK

        # Trend regime: strong trend & vol not collapsing
        if abs_ts >= self.trend_strength_thr and vol_ratio >= self.vol_ratio_thr:
            return 1  # TREND

        # Otherwise: range / neutral
        return 0

    # ------------------------------------------------------------
    #  VECTORIZED REGIME ASSIGNMENT FOR FULL DATAFRAME
    # ------------------------------------------------------------
    def get_regime_vector(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Returns a numpy array of regime ids for all rows in df_features.
        """
        if self.trend_strength_thr is None:
            # auto-fit if not done
            self.fit(df_features)

        regimes = []
        for _, row in df_features.iterrows():
            regimes.append(self.assign_regime_row(row))

        return np.array(regimes, dtype=np.int64)
