# feature_engine.py (V10 DAILY – richer trend/vol/momentum features)

import numpy as np
import pandas as pd
from config import BotConfig


class FeatureEngineer:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def add_basic_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ret"] = df["close"].pct_change()
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def add_daily_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DAILY-specific microstructure proxies:
          - absolute range (high-low)
          - normalized range (range / close)
          - overnight gap (open - prev close)
        """
        df = df.copy()
        df["daily_range"] = df["high"] - df["low"]
        df["range_pct"] = df["daily_range"] / (df["close"] + 1e-12)
        df["gap"] = df["open"] - df["close"].shift(1)
        df["gap_pct"] = df["gap"] / (df["close"].shift(1) + 1e-12)
        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for w in self.cfg.ma_windows:
            df[f"ma_{w}"] = df["close"].rolling(w).mean()
            df[f"ema_{w}"] = self._ema(df["close"], w)
        # multi-horizon price momentum (in days)
        for h in [3, 5, 10, 20]:
            df[f"mom_{h}"] = df["close"] / df["close"].shift(h) - 1.0
        return df

    def add_atr(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        df = df.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df["atr"] = tr.rolling(window).mean()
        # short vs long ATR ratios (daily)
        df["atr_short"] = tr.rolling(window).mean()
        df["atr_long"] = tr.rolling(window * 4).mean()
        df["atr_ratio"] = df["atr_short"] / (df["atr_long"] + 1e-12)
        return df

    def add_rsi(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        df = df.copy()
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        gain_ema = pd.Series(gain, index=df.index).ewm(alpha=1/window, adjust=False).mean()
        loss_ema = pd.Series(loss, index=df.index).ewm(alpha=1/window, adjust=False).mean()
        rs = gain_ema / (loss_ema + 1e-12)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        return df

    def add_bollinger(self, df: pd.DataFrame, window: int, num_std: float) -> pd.DataFrame:
        df = df.copy()
        ma = df["close"].rolling(window).mean()
        std = df["close"].rolling(window).std()
        df["bb_mid"] = ma
        df["bb_upper"] = ma + num_std * std
        df["bb_lower"] = ma - num_std * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (ma + 1e-12)
        return df

    def add_breakout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for lb in self.cfg.breakout_lookbacks:
            df[f"high_{lb}"] = df["high"].rolling(lb).max()
            df[f"low_{lb}"] = df["low"].rolling(lb).min()
            df[f"range_{lb}"] = df[f"high_{lb}"] - df[f"low_{lb}"]
            df[f"close_pos_in_range_{lb}"] = (
                (df["close"] - df[f"low_{lb}"]) /
                (df[f"range_{lb}"] + 1e-12)
            )
        return df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # realized vol at different DAILY horizons
        df["vol_short"] = df["log_ret"].rolling(24).std()   # ~1 month
        df["vol_med"] = df["log_ret"].rolling(72).std()     # ~3 months
        df["vol_long"] = df["log_ret"].rolling(120).std()   # ~6 months
        df["vol_ratio_short_long"] = df["vol_short"] / (df["vol_long"] + 1e-12)
        df["vol_ratio_med_long"] = df["vol_med"] / (df["vol_long"] + 1e-12)
        # vol acceleration
        df["vol_grad"] = df["vol_short"] - df["vol_med"]
        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.add_basic_returns(df)
        df = self.add_daily_range_features(df)
        df = self.add_moving_averages(df)
        df = self.add_atr(df, self.cfg.atr_window)
        df = self.add_rsi(df, self.cfg.rsi_window)
        df = self.add_bollinger(df, self.cfg.bb_window, self.cfg.bb_std)
        df = self.add_breakout_features(df)
        df = self.add_volatility_features(df)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        return df
