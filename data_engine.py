# data_engine.py

import os
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


@dataclass
class MarketDataConfig:
    symbols: List[str]
    timeframe: str
    data_dir: str
    raw_subdir: str = "raw"
    processed_subdir: str = "processed"
    default_start: str = "2018-01-01"
    default_end: Optional[str] = None
    tz: str = "UTC"
    max_missing_ratio: float = 0.05

    @property
    def raw_path(self):
        return os.path.join(self.data_dir, self.raw_subdir)

    @property
    def processed_path(self):
        return os.path.join(self.data_dir, self.processed_subdir)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def parse_date(date_str):
    if date_str is None:
        return dt.datetime.utcnow()
    return dt.datetime.fromisoformat(date_str)


class DataDownloader:
    def __init__(self, config):
        self.config = config
        ensure_dir(config.raw_path)

    def _local_raw_filename(self, symbol):
        safe = symbol.replace("=", "").replace("/", "_")
        return os.path.join(self.config.raw_path, f"{safe}_{self.config.timeframe}_raw.csv")

    def _resample(self, df, rule):
        ohlc = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        out = df.resample(rule).agg(ohlc)
        out.dropna(inplace=True)
        return out

    def _download_from_yfinance(self, symbol, start, end):
        if not HAS_YFINANCE:
            raise ImportError("yfinance not installed")

        interval_map = {
            "1T": "1m",
            "5T": "5m",
            "15T": "15m",
            "1H": "60m",
            "4H": "60m",
            "1D": "1d"
        }
        interval = interval_map.get(self.config.timeframe, "60m")

        df = yf.download(symbol, start=start, end=end, interval=interval)
        if df.empty:
            raise ValueError(f"No data for {symbol}")

        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        df.index = df.index.tz_localize("UTC")

        df = self._resample(df, self.config.timeframe)
        return df

    def fetch_raw(self, symbol, start=None, end=None, force_download=False):
        start_dt = parse_date(start or self.config.default_start)
        end_dt = parse_date(end or self.config.default_end)

        fname = self._local_raw_filename(symbol)

        if os.path.exists(fname) and not force_download:
            df = pd.read_csv(fname, parse_dates=["datetime"], index_col="datetime")
            df.index = df.index.tz_localize(self.config.tz)
            return df

        df = self._download_from_yfinance(symbol, start_dt, end_dt)
        df.index.name = "datetime"
        df.to_csv(fname)
        return df


class DataCleaner:
    def __init__(self, config):
        self.config = config

    def add_returns(self, df):
        df["ret"] = df["close"].pct_change()
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def add_vol(self, df):
        df["vol"] = df["log_ret"].rolling(48).std()
        return df

    def forward_fill(self, df):
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].ffill()
        return df

    def clean(self, df, symbol):
        df = self.forward_fill(df)
        df = self.add_returns(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = self.add_vol(df)
        return df


class MarketDataEngine:
    def __init__(self, config):
        self.config = config
        ensure_dir(config.data_dir)
        ensure_dir(config.raw_path)
        ensure_dir(config.processed_path)

        self.downloader = DataDownloader(config)
        self.cleaner = DataCleaner(config)

    def _processed_filename(self, symbol):
        safe = symbol.replace("=", "").replace("/", "_")
        return os.path.join(self.config.processed_path, f"{safe}_{self.config.timeframe}_proc.csv")

    def get_history(self, symbol, start=None, end=None, processed=True, force_refresh=False):
        fname = self._processed_filename(symbol)

        if processed and os.path.exists(fname) and not force_refresh:
            df = pd.read_csv(fname, parse_dates=["datetime"], index_col="datetime")
            df.index = df.index.tz_localize(self.config.tz)
            return df

        raw = self.downloader.fetch_raw(symbol, start, end, force_download=force_refresh)
        clean = self.cleaner.clean(raw, symbol)
        clean.index.name = "datetime"
        clean.to_csv(fname)
        return clean
