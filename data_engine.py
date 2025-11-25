from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
import yfinance as yf

from utils import logger


ISO_DATE_FMT = "%Y-%m-%d"


def _parse_cfg_date(value: Optional[str], default: Optional[datetime] = None) -> datetime:
    """
    Parse a YYYY-MM-DD date coming from the config. Assume UTC.
    """
    if value:
        # Allow both "YYYY-MM-DD" and full ISO with time
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            dt = datetime.strptime(value, ISO_DATE_FMT)
    elif default is not None:
        dt = default
    else:
        raise ValueError("No date value supplied and no default provided")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


@dataclass
class EngineDates:
    start: datetime
    end: datetime


class DataEngine:
    """
    Unified data engine for V10-TR.

    - In Yahoo! mode it pulls bars from yfinance.
    - In OANDA mode it uses the REST v3 candles endpoint.

    Expected usage from the rest of the code:

        engine = DataEngine(cfg)
        df = engine.get_history("EUR_USD")
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

        # ------------------------------------------------------------------
        # Basic config
        # ------------------------------------------------------------------
        # default to OANDA now
        self.source = getattr(cfg, "data_source", "oanda").lower()
        self.timeframe = getattr(cfg, "timeframe", "D")

        cfg_start = getattr(cfg, "start_date", "2010-01-01")
        cfg_end = getattr(cfg, "end_date", None)

        start = _parse_cfg_date(cfg_start)

        now = datetime.now(timezone.utc)
        end = _parse_cfg_date(cfg_end, default=now) if cfg_end else now

        # OANDA rejects "to" dates in the future, so always clamp.
        if end > now:
            logger.warning(
                "[DATA] Requested end_date %s is in the future – clamping to now (%s)",
                end.isoformat(), now.isoformat()
            )
            end = now

        self.dates = EngineDates(start=start, end=end)

        # ------------------------------------------------------------------
        # OANDA configuration (from environment variables)
        # ------------------------------------------------------------------
        self.oanda_env = os.environ.get("OANDA_ENV", "practice")
        self.oanda_domain = os.environ.get("OANDA_DOMAIN", "practice")
        self.oanda_account = os.environ.get("OANDA_ACCOUNT_ID")
        self.oanda_token = os.environ.get("OANDA_TOKEN")

        if self.source == "oanda":
            if not self.oanda_token:
                raise RuntimeError(
                    "OANDA_TOKEN environment variable is not set – "
                    "cannot use OANDA as data source."
                )

        self.session = requests.Session()
        if self.oanda_token:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.oanda_token}",
                    "Content-Type": "application/json",
                }
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_history(self, symbol: str) -> pd.DataFrame:
        """
        Fetch OHLCV history for the given symbol between engine.dates.start and
        engine.dates.end (inclusive where possible).
        """
        if self.source == "oanda":
            return self._get_history_oanda(symbol)
        else:
            return self._get_history_yahoo(symbol)

    # ------------------------------------------------------------------ #
    # Yahoo! Finance (kept as fallback)
    # ------------------------------------------------------------------ #
    def _get_history_yahoo(self, symbol: str) -> pd.DataFrame:
        logger.info(
            "[DATA] Using Yahoo! Finance data source | symbol=%s, timeframe=%s",
            symbol, self.timeframe
        )
        start_str = self.dates.start.strftime(ISO_DATE_FMT)
        end_str = self.dates.end.strftime(ISO_DATE_FMT)

        data = yf.download(
            symbol,
            start=start_str,
            end=end_str,
            interval=self._map_timeframe_to_yf(self.timeframe),
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            raise RuntimeError(f"No Yahoo data returned for {symbol}")

        data.index = pd.to_datetime(data.index, utc=True)
        return data[["Open", "High", "Low", "Close", "Volume"]].sort_index()

    # ------------------------------------------------------------------ #
    # OANDA helpers
    # ------------------------------------------------------------------ #
    def _get_history_oanda(self, symbol: str) -> pd.DataFrame:
        env = self.oanda_env
        granularity = self._map_timeframe_to_oanda(self.timeframe)

        # Your config already uses EUR_USD / XAU_USD style, so no transform.
        instrument = symbol

        logger.info(
            "[DATA] Using OANDA data source | env=%s, symbol=%s, timeframe=%s",
            env, instrument, granularity,
        )

        domain = "api-fxpractice.oanda.com" if env == "practice" else "api-fxtrade.oanda.com"
        base_url = f"https://{domain}/v3/instruments/{instrument}/candles"

        start = self.dates.start
        end = self.dates.end

        logger.info(
            "[DATA] Requesting OANDA history for %s | granularity=%s, "
            "start=%s, end=%s",
            instrument, granularity, start.date(), end.date(),
        )

        candles: List[Dict[str, Any]] = []

        # Work *backwards* from end to start using `to` + `count` only.
        # This avoids the forbidden combination `from + to + count`.
        step = self._granularity_to_timedelta(granularity)
        max_count = 5000
        to_time = end

        while True:
            params = {
                "granularity": granularity,
                "price": "M",
                "to": to_time.isoformat().replace("+00:00", "Z"),
                "count": max_count,
            }

            attempt = 1
            while True:
                resp = self.session.get(base_url, params=params, timeout=30)

                if resp.ok:
                    break

                msg = resp.text
                logger.warning(
                    "[DATA] OANDA request failed (status=%s) attempt %d/3: %s",
                    resp.status_code, attempt, msg,
                )
                # If after 3 tries it's still failing, let it crash and show the message.
                if attempt >= 3:
                    resp.raise_for_status()
                attempt += 1

            payload = resp.json()
            batch = payload.get("candles", [])
            if not batch:
                break

            candles.extend(batch)

            # Oldest candle in this batch
            first_time = self._parse_oanda_time(batch[0]["time"])

            # Stop if we have reached or gone before the requested start,
            # or if OANDA returned fewer than max_count candles (no more data).
            if first_time <= start or len(batch) < max_count:
                break

            # Move the window backwards one step.
            to_time = first_time - step

        if not candles:
            raise RuntimeError(f"No OANDA candles returned for {instrument}")

        # Convert candles to a clean OHLCV DataFrame.
        records = []
        for c in candles:
            if not c.get("complete", False):
                continue

            t = self._parse_oanda_time(c["time"])
            if t < start or t > end:
                continue

            mid = c["mid"]
            records.append(
                {
                    "Date": t,
                    "Open": float(mid["o"]),
                    "High": float(mid["h"]),
                    "Low": float(mid["l"]),
                    "Close": float(mid["c"]),
                    "Volume": int(c["volume"]),
                }
            )

        if not records:
            raise RuntimeError(f"OANDA candles empty after filtering for {instrument}")

        df = pd.DataFrame.from_records(records).set_index("Date")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        logger.info(
            "[DATA] OANDA history loaded | rows=%d, first=%s, last=%s",
            len(df), df.index[0].isoformat(), df.index[-1].isoformat(),
        )

        return df[["Open", "High", "Low", "Close", "Volume"]]

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _map_timeframe_to_yf(tf: str) -> str:
        tf = tf.upper()
        if tf == "D":
            return "1d"
        if tf in ("H1", "1H"):
            return "60m"
        if tf in ("M5", "5M"):
            return "5m"
        return "1d"

    @staticmethod
    def _map_timeframe_to_oanda(tf: str) -> str:
        tf = tf.upper()
        if tf in ("D", "1D"):
            return "D"
        if tf in ("H1", "1H"):
            return "H1"
        if tf in ("M5", "5M"):
            return "M5"
        # If it's something else, just pass it through and let OANDA complain.
        return tf

    @staticmethod
    def _granularity_to_timedelta(granularity: str) -> timedelta:
        g = granularity.upper()
        if g == "D":
            return timedelta(days=1)
        if g.startswith("H"):
            hours = int(g[1:]) if len(g) > 1 else 1
            return timedelta(hours=hours)
        if g.startswith("M"):
            minutes = int(g[1:]) if len(g) > 1 else 1
            return timedelta(minutes=minutes)
        return timedelta(days=1)

    @staticmethod
    def _parse_oanda_time(ts: str) -> datetime:
        """
        Parse OANDA's RFC3339 timestamps to aware UTC datetimes.
        Examples:
        - "2017-02-10T22:24:06.000000000Z"
        - "2017-02-10T22:24:06Z"
        """
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        # strip nanoseconds beyond microsecond precision
        if "." in ts:
            date_part, frac_part = ts.split(".")
            if "+" in frac_part:
                frac, tz = frac_part.split("+", 1)
                frac = frac[:6]  # microseconds
                ts = f"{date_part}.{frac}+{tz}"
        return datetime.fromisoformat(ts).astimezone(timezone.utc)

