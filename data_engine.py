# ================================================================
#  DATA ENGINE — V10-TR CCT-90
#  Fetches daily OHLCV data from Yahoo Finance (Render-safe)
# ================================================================

import os
import time
import pickle
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from config import V10TRConfig
from utils import logger


# ================================================================
#  SIMPLE DISK CACHE (PER SYMBOL)
# ================================================================

def _cache_path(symbol: str) -> str:
    safe = symbol.replace("=", "_").replace("^", "_")
    return f"cache_{safe}.pkl"


def load_from_cache(symbol: str):
    path = _cache_path(symbol)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame) and not df.empty:
                logger.info(f"[DATA] Loaded {symbol} from cache {path}")
                return df
        except Exception as e:
            logger.warning(f"[DATA] Failed to load cache for {symbol}: {e}")
    return None


def save_to_cache(symbol: str, df: pd.DataFrame):
    path = _cache_path(symbol)
    try:
        with open(path, "wb") as f:
            pickle.dump(df, f)
        logger.info(f"[DATA] Saved {symbol} to cache {path}")
    except Exception as e:
        logger.warning(f"[DATA] Failed to save cache for {symbol}: {e}")


# ================================================================
#  DATA ENGINE
# ================================================================

@dataclass
class DataEngine:
    cfg: V10TRConfig

    # ------------------------------------------------------------
    #  PUBLIC METHOD: get_history(symbol) → OHLCV DataFrame
    # ------------------------------------------------------------
    def get_history(self, symbol: str) -> pd.DataFrame:
        """
        Returns a clean daily OHLCV DataFrame for the given symbol.

        Columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex
        """

        # 1) Try cache
        cached = load_from_cache(symbol)
        if cached is not None:
            return cached

        # 2) Download from Yahoo with retries
        df = self._download_from_yahoo(symbol)

        # 3) Clean & standardize
        df = self._clean_dataframe(df)

        # 4) Save to cache
        save_to_cache(symbol, df)

        return df

    # ------------------------------------------------------------
    #  INTERNAL: download via yfinance with basic retry
    # ------------------------------------------------------------
    def _download_from_yahoo(self, symbol: str) -> pd.DataFrame:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"[DATA] Downloading {symbol} (attempt {attempt}/{max_retries})"
                )
                df = yf.download(
                    symbol,
                    start=self.cfg.start_date,
                    end=self.cfg.end_date,
                    interval=self.cfg.timeframe,
                    auto_adjust=False,
                    progress=False,
                )
                if df is not None and not df.empty:
                    return df
                else:
                    logger.warning(f"[DATA] Empty data for {symbol} on attempt {attempt}")
            except Exception as e:
                logger.warning(f"[DATA] Error downloading {symbol}: {e}")
            time.sleep(2.0)

        raise RuntimeError(f"[DATA] Failed to download data for {symbol} after retries")

    # ------------------------------------------------------------
    #  INTERNAL: standardize columns & clean
    # ------------------------------------------------------------
    def _clean_dataframe(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()

        # Standardize column names
        col_map = {
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Close",   # prefer Close but fall back if needed
            "Volume": "Volume",
        }

        # Only keep known columns
        keep_cols = [c for c in df.columns if c in col_map]
        df = df[keep_cols].rename(columns=col_map)

        # If "Close" is missing but "Adj Close" exists
        if "Close" not in df.columns and "Adj Close" in df_raw.columns:
            df["Close"] = df_raw["Adj Close"]

        # Ensure all required columns exist, fill missing with ffill
        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                df[col] = df["Close"]
        if "Volume" not in df.columns:
            df["Volume"] = 0.0

        # Drop rows without close price
        df = df.dropna(subset=["Close"])

        # Sort by date index
        df = df.sort_index()

        return df

