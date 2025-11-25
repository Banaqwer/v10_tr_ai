# ================================================================
#  BACKTEST ENGINE — V10-TR CCT-90 (ROLLING WINDOW WALK-FORWARD)
#  Option A: Rolling training window, out-of-sample test window
# ================================================================

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch

from config import V10TRConfig
from data_engine import DataEngine
from feature_engine import FeatureEngine, FEATURE_COLS
from label_engine import LabelEngine
from regime_engine import RegimeEngine
from embed_engine import CCT90Embedder
from transformer_engine import TransformerEncoder
from experts_engine import ExpertsEngine
from strategy import Strategy
from risk_engine import RiskEngine

from utils import logger


# ------------------------------------------------------------
# SAFE CONVERSION — tensor -> numpy for JSON / NumPy ops
# ------------------------------------------------------------
def _to_numpy(x):
    """
    Safely convert tensors to NumPy arrays.

    - If x is a torch.Tensor (even with grad), detach + cpu + numpy().
    - Otherwise, fall back to np.asarray.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass
class Backtester:
    """
    Rolling-window walk-forward backtester.

    - Uses cfg.train_window_days as the length of each training window (in bars).
    - Uses cfg.test_window_days as the length of each out-of-sample test window.
    - For each step:
        1) Train experts on [train_start : train_end)
        2) Trade on [train_end : test_end)
        3) Slide forward by test_window_days and repeat.
    """
    cfg: V10TRConfig
    symbol: str = "EUR_USD"

    # ------------------------------------------------------------
    #  MAIN ENTRYPOINT
    # ------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        logger.info(f"[BACKTEST] Rolling-window walk-forward backtest for {self.symbol}")

        # ----------------------------------
        # Load full price history (2010–2025)
        # ----------------------------------
        data_engine = DataEngine(self.cfg)
        df_prices = data_engine.get_history(self.symbol)
        if df_prices.empty:
            raise RuntimeError("[BACKTEST] Price data is empty")

        # ----------------------------------
        # Build features once on full history
        # ----------------------------------
        feat_engine = FeatureEngine(self.cfg)
        df_feat = feat_engine.build_features(df_prices)

        # Labels (for expert training)
        label_engine = LabelEngine(self.cfg)
        df_labeled = label_engine.build_labels(df_feat)

        # Matrices (full period)
        feature_matrix = feat_engine.get_feature_matrix(df_labeled)   # shape [N, F]
        label_vector = label_engine.get_label_vector(df_labeled)      # shape [N]

        # ----------------------------------
        # Regime classification (full period)
        # ----------------------------------
        regime_engine = RegimeEngine(self.cfg)
        regimes = regime_engine.get_regime_vector(df_labeled)         # shape [N]

        n_samples = feature_matrix.shape[0]
        logger.info(f"[BACKTEST] Labeled samples: {n_samples}")

        # ----------------------------------
        # Transformer contexts for full series
        # ----------------------------------
        embedder = CCT90Embedder(self.cfg, n_features=feature_matrix.shape[1])
        transformer = TransformerEncoder(self.cfg)

        context_list: List[np.ndarray] = []
        for i in range(n_samples):
            # past_window includes all data up to i = strictly past+present
            past_window = feature_matrix[: i + 1]
            emb = embedder.transform(past_window)     # torch / numpy, [chunks, embed_dim]
            ctx = transformer(emb)                    # torch tensor [d_model]
            context_list.append(ctx)

        context_matrix = np.vstack([_to_numpy(c) for c in context_list])

        # ----------------------------------
        # Build full training matrix once
        # ----------------------------------
        base_experts_engine = ExpertsEngine(self.cfg)
        X_full = base_experts_engine.build_training_matrix(
            feature_matrix,
            context_matrix
        )  # shape [N, D]

        # ----------------------------------
        # Rolling-window definitions
        # ----------------------------------
        train_window = int(self.cfg.train_window_days)
        test_window = int(self.cfg.test_window_days)

        if train_window + test_window > n_samples:
            raise ValueError(
                f"[BACKTEST] Not enough data for one train+test window: "
                f"train={train_window}, test={test_window}, N={n_samples}"
            )

        logger.info(
            f"[BACKTEST] Rolling windows: train={train_window} bars, "
            f"test={test_window} bars, total={n_samples} bars"
        )

        # Risk engine is stateless with respect to equity; we carry equity ourselves
        risk = RiskEngine(self.cfg)

        equity = risk.initial_equity
        equity_curve: List[float] = []
        signals: List[int] = []
        trades: List[Dict[str, Any]] = []

        close = df_labeled["Close"].values
        high = df_labeled["High"].values
        low = df_labeled["Low"].values
        atr = df_labeled["ATR"].values
        index = df_labeled.index

        # ----------------------------------
        # Walk-forward loop
        # ----------------------------------
        windows = []
        start_train = 0
        while True:
            train_start = start_train
            train_end = train_start + train_window
            test_end = train_end + test_window

            if test_end > n_samples:
                break

            windows.append((train_start, train_end, test_end))
            start_train += test_window  # slide by test window

        logger.info(f"[BACKTEST] Number of walk-forward windows: {len(windows)}")

        # starting from first test index
        for w_idx, (train_start, train_end, test_end) in enumerate(windows):
            logger.info(
                f"[BACKTEST] Window {w_idx+1}/{len(windows)} | "
                f"train[{train_start}:{train_end}) test[{train_end}:{test_end})"
            )

            # ------------------------------
            # Train experts on train window
            # ------------------------------
            X_train = X_full[train_start:train_end]
            y_train = label_vector[train_start:train_end]
            reg_train = regimes[train_start:train_end]

            experts_engine = ExpertsEngine(self.cfg)
            experts_engine.fit(X_train, y_train, reg_train)

            strategy = Strategy(self.cfg, experts_engine)

            # ------------------------------
            # Trade on test window
            # ------------------------------
            for i in range(train_end, test_end):
                x_today = X_full[i]
                reg_today = regimes[i]

                signal = strategy.generate_signal(x_today, reg_today)
                signals.append(int(signal))

                if signal == 0:
                    # No position today
                    equity_curve.append(equity)
                    continue

                # Enter position
                entry_price = float(close[i])
                position_size = risk.compute_position_size(equity, float(atr[i]))

                sl, tp = risk.compute_sl_tp(entry_price, float(atr[i]))

                exit_price, outcome = risk.check_exit(
                    sl=sl,
                    tp=tp,
                    day_open=entry_price,
                    day_high=float(high[i]),
                    day_low=float(low[i]),
                )

                new_equity = risk.update_equity(
                    equity=equity,
                    position=position_size,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction=signal,
                    outcome=outcome,
                )

                trade_pnl = new_equity - equity

                trades.append({
                    "index": int(i),
                    "timestamp": str(index[i]),
                    "signal": int(signal),
                    "entry": float(entry_price),
                    "exit": float(exit_price),
                    "pnl": float(trade_pnl),
                    "equity_before": float(equity),
                    "equity_after": float(new_equity),
                    "regime": int(reg_today),
                    "window_id": int(w_idx),
                    "train_start": int(train_start),
                    "train_end": int(train_end),
                })

                equity = new_equity
                equity_curve.append(equity)

        # ----------------------------------
        # Aggregate results
        # ----------------------------------
        equity_curve_np = np.array(equity_curve, dtype=float)
        signals_np = np.array(signals, dtype=int)

        final_equity = float(equity)
        total_return_pct = (final_equity / risk.initial_equity - 1.0) * 100.0
        win_rate = self._compute_win_rate(trades)

        results = {
            "equity_curve": equity_curve_np,
            "signals": signals_np,
            "trades": trades,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "num_trades": len(trades),
            "win_rate": win_rate,
        }

        logger.info(f"[BACKTEST] Finished walk-forward. Final equity: {final_equity:.2f}")
        logger.info(f"[BACKTEST] Total return: {total_return_pct:.2f}% | "
                    f"Trades: {len(trades)} | Win rate: {win_rate*100:.2f}%")

        return results

    # ------------------------------------------------------------
    #  INTERNAL: Compute win rate
    # ------------------------------------------------------------
    def _compute_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        if len(trades) == 0:
            return 0.0
        wins = [t for t in trades if t["pnl"] > 0]
        return len(wins) / len(trades)
