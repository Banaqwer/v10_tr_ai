# data_engine.py
#
# V10 Market Data Engine – simplified and made timeframe-aware.
# Works with DAILY ("1d") or INTRADAY ("1h" → "60m") depending on cfg.timeframe.
#
# It:
#   - downloads data from Yahoo Finance using yfinance
#   - uses cfg.timeframe to choose interval
#   - caches raw data to disk (data/raw/)
#   - returns a clean DataFrame with columns:
#       ["open", "high", "low", "close", "volume"]
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime

import pandas as pd
import yfinance as yf

from config import BotConfig


# ---------------------------------------------------------------------------
# Config wrapper specific to market data
# ---------------------------------------------------------------------------

@dataclass
class MarketDataConfig:
    symbols: List[str]
    timeframe: str          # "1d" or "1h"
    data_dir: str
    default_start: str
    default_end: str


# ---------------------------------------------------------------------------
# MarketDataEngine
# ---------------------------------------------------------------------------

class MarketDataEngine:
    def __init__(self, mdc: MarketDataConfig):
        self.cfg = mdc
        base = Path(mdc.data_dir)
        self.base_dir = base
        self.raw_dir = base / "raw"
        self.processed_dir = base / "processed"

        # ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------- helpers ----------

    @staticmethod
    def _parse_date(dt: Union[str, datetime]) -> datetime:
        if isinstance(dt, datetime):
            return dt
        return datetime.fromisoformat(dt)

    def _symbol_filename(self, symbol: str) -> str:
        """
        Turn 'EURUSD=X' into a safe filename like 'EURUSD=X_1d.csv'
        """
        safe = symbol.replace("/", "_").replace("=", "_")
        return f"{safe}_{self.cfg.timeframe}.csv"

    # ---------- Yahoo download ----------

    def _download_from_yfinance(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> pd.DataFrame:
        """
        Download from Yahoo Finance using the timeframe from our config.

        For V10-Daily:
            cfg.timeframe = "1d"  -> interval="1d"
        If later we revert to intraday:
            cfg.timeframe = "1h"  -> interval="60m"
        """
        interval = self.cfg.timeframe
        if interval == "1h":
            interval = "60m"  # yfinance intraday symbol

        df = yf.download(
            symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if df is None or df.empty:
            raise ValueError(f"No data for {symbol}")

        # normalize column names
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )

        # keep only what we need
        cols = ["open", "high", "low", "close", "volume"]
        df = df[cols]
        df = df.dropna()

        # ensure datetime index and sort
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df

    # ---------- public API ----------

    def get_history(
        self,
        symbol: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Main entry point used by main.py and the rest of V10.

        - symbol: "EURUSD=X" etc.
        - start/end: optional override of default dates
        - force_download: if True, ignore cache and re-pull from Yahoo
        """
        if start is None:
            start = self.cfg.default_start
        if end is None:
            end = self.cfg.default_end

        start_dt = self._parse_date(start)
        end_dt = self._parse_date(end)

        # path for cached raw data
        fname = self._symbol_filename(symbol)
        fpath = self.raw_dir / fname

        if fpath.exists() and not force_download:
            df = pd.read_csv(fpath, parse_dates=["timestamp"])
            df = df.set_index("timestamp")
            df = df.sort_index()
            return df

        # download fresh
        df = self._download_from_yfinance(symbol, start_dt, end_dt)

        # cache to disk
        tmp = df.copy()
        tmp["timestamp"] = tmp.index
        tmp.to_csv(fpath, index=False)

        return df
