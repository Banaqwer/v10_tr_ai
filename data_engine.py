# data_engine.py
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import pandas as pd
import requests

from utils import logger as _logger


log = _logger.get_logger("DATA") if hasattr(_logger, "get_logger") else _logger


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _oanda_base_url() -> str:
    """
    Choose the correct OANDA API host based on OANDA_ENV / OANDA_DOMAIN.
    practice -> api-fxpractice.oanda.com
    live     -> api-fxtrade.oanda.com
    """
    env = (os.getenv("OANDA_ENV") or os.getenv("OANDA_DOMAIN") or "practice").lower()
    if env == "live":
        return "https://api-fxtrade.oanda.com/v3"
    return "https://api-fxpractice.oanda.com/v3"


def _granularity_from_timeframe(tf: str) -> str:
    """
    Map our config timeframe to OANDA candle granularity.
    Feel free to adjust if you change your config.
    """
    tf = tf.upper()
    mapping = {
        "M1": "M1",
        "M5": "M5",
        "M15": "M15",
        "M30": "M30",
        "H1": "H1",
        "H4": "H4",
        "D": "D",
        "1D": "D",
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe for OANDA: {tf}")
    return mapping[tf]


# ------------------------------------------------------------
# DataEngine – OANDA version
# ------------------------------------------------------------

class DataEngine:
    """
    V10-TR data engine that pulls candles directly from OANDA.

    - No Yahoo Finance.
    - Returns a DataFrame with index = datetime and columns:
      [Open, High, Low, Close, Volume]
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.symbol = cfg.symbol        # OANDA instrument, e.g. "EUR_USD", "XAU_USD"
        self.timeframe = cfg.timeframe  # e.g. "D", "H1", "M5"
        self.max_retries = getattr(cfg, "data_max_retries", 3)
        self.retry_delay = getattr(cfg, "data_retry_delay", 2.0)

        self._session = requests.Session()
        self._base_url = _oanda_base_url()
        self._token = _get_env("OANDA_TOKEN")
        self._account_id = _get_env("OANDA_ACCOUNT_ID")

        log.info(
            f"[DATA] Using OANDA data source | env={os.getenv('OANDA_ENV', 'practice')}, "
            f"symbol={self.symbol}, timeframe={self.timeframe}"
        )

    # ------------------------------------------------------------------    # Public API used by Backtester
    # ------------------------------------------------------------------
    def get_history(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical candles for the given symbol (or cfg.symbol if None),
        clean them, and return OHLCV DataFrame.
        """
        inst = symbol or self.symbol
        log.info(f"[DATA] Requesting OANDA history for {inst}")

        raw_df = self._download_from_oanda(inst)
        df = self._clean_dataframe(raw_df)

        log.info(f"[DATA] Loaded {len(df)} bars for {inst}")
        return df

    # ------------------------------------------------------------------    # OANDA download logic
    # ------------------------------------------------------------------
    def _download_from_oanda(self, instrument: str) -> pd.DataFrame:
        """
        Download all candles between cfg.start_date and cfg.end_date
        (inclusive) from OANDA, chunking if necessary.
        """

        # Expect ISO dates like "2015-01-01" in cfg; fallback if absent.
        start_date = datetime.fromisoformat(getattr(self.cfg, "start_date", "2015-01-01")).replace(
            tzinfo=timezone.utc
        )
        end_date = datetime.fromisoformat(getattr(self.cfg, "end_date", datetime.utcnow().strftime("%Y-%m-%d"))).replace(
            tzinfo=timezone.utc
        ) + timedelta(days=1)  # include last day

        granularity = _granularity_from_timeframe(self.timeframe)

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        url = f"{self._base_url}/instruments/{instrument}/candles"

        all_rows: List[dict] = []
        since = start_date

        log.info(
            f"[DATA] OANDA candles | instrument={instrument}, "
            f"granularity={granularity}, start={start_date.date()}, end={end_date.date()}"
        )

        # OANDA limit is 5000 candles / request. We chunk by time.
        while since < end_date:
            params = {
                "from": since.isoformat().replace("+00:00", "Z"),
                "to": end_date.isoformat().replace("+00:00", "Z"),
                "granularity": granularity,
                "price": "M",  # mid prices
                "count": 5000,
            }

            for attempt in range(1, self.max_retries + 1):
                resp = self._session.get(url, headers=headers, params=params, timeout=30)
                if resp.status_code == 429:
                    log.warning("[DATA] OANDA rate limit. Sleeping before retry...")
                    import time as _time
                    _time.sleep(self.retry_delay * attempt)
                    continue
                if not resp.ok:
                    log.warning(
                        f"[DATA] OANDA request failed (status={resp.status_code}) "
                        f"attempt {attempt}/{self.max_retries}: {resp.text[:200]}"
                    )
                    if attempt == self.max_retries:
                        resp.raise_for_status()
                    import time as _time
                    _time.sleep(self.retry_delay * attempt)
                    continue

                data = resp.json()
                candles = data.get("candles", [])
                if not candles:
                    log.warning("[DATA] OANDA returned 0 candles for this chunk.")
                    return pd.DataFrame()  # let cleaner handle KeyError if empty

                all_rows.extend(candles)
                last_time_str = candles[-1]["time"]
                last_dt = datetime.fromisoformat(last_time_str.replace("Z", "+00:00"))

                # Move cursor forward by one candle to avoid duplicates
                since = last_dt + timedelta(seconds=1)
                break  # success, exit retry loop

        if not all_rows:
            return pd.DataFrame()

        # Build DataFrame
        records = []
        for c in all_rows:
            if not c.get("complete", True):
                continue
            t = datetime.fromisoformat(c["time"].replace("Z", "+00:00"))
            mid = c.get("mid") or c.get("ask") or c.get("bid")
            if mid is None:
                continue
            records.append(
                {
                    "Datetime": t,
                    "Open": float(mid["o"]),
                    "High": float(mid["h"]),
                    "Low": float(mid["l"]),
                    "Close": float(mid["c"]),
                    "Volume": int(c.get("volume", 0)),
                }
            )

        df = pd.DataFrame.from_records(records)
        if df.empty:
            return df

        df.set_index("Datetime", inplace=True)
        df.sort_index(inplace=True)
        return df

    # ------------------------------------------------------------------    # Cleaning
    # ------------------------------------------------------------------
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize columns & drop bad rows so the rest of the AI
        (features, regime engine, etc.) can stay unchanged.
        """

        if df.empty:
            raise KeyError(
                "Received empty dataframe from OANDA. "
                "Check symbol, timeframe, and date range."
            )

        # Ensure required columns exist
        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                raise KeyError(f"Expected column '{col}' in price data, got {df.columns.tolist()}")

        # Drop rows with missing close
        df = df.dropna(subset=["Close"])

        # Optional: keep only OHLCV
        keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep_cols]

        return df
