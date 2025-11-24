# ================================================================
#  BACKTEST ENGINE — V10-TR CCT-90
#  Full daily loop running the entire AI pipeline
# ================================================================

from dataclasses import dataclass
import numpy as np
import pandas as pd

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


@dataclass
class Backtester:
    cfg: V10TRConfig
    symbol: str = "GC=F"

    # ------------------------------------------------------------
    #  RUN FULL BACKTEST
    # ------------------------------------------------------------
    def run(self):
        logger.info(f"[BACKTEST] Starting backtest for {self.symbol}")

        # ----------------------------------
        # Load data
        # ----------------------------------
        data_engine = DataEngine(self.cfg)
        df_prices = data_engine.get_history(self.symbol)
        if df_prices.empty:
            raise RuntimeError("[BACKTEST] Price data is empty")

        # ----------------------------------
        # Build features
        # ----------------------------------
        feat_engine = FeatureEngine(self.cfg)
        df_feat = feat_engine.build_features(df_prices)

        # Labeling (used only for training experts)
        label_engine = LabelEngine(self.cfg)
        df_labeled = label_engine.build_labels(df_feat)

        # Extract matrices
        feature_matrix = feat_engine.get_feature_matrix(df_labeled)
        label_vector = label_engine.get_label_vector(df_labeled)

        # ----------------------------------
        # Regime classification
        # ----------------------------------
        regime_engine = RegimeEngine(self.cfg)
        regimes = regime_engine.get_regime_vector(df_labeled)

        # ----------------------------------
        # Compute Transformer contexts
        # ----------------------------------
        embedder = CCT90Embedder(self.cfg, n_features=feature_matrix.shape[1])
        transformer = TransformerEncoder(self.cfg)

        context_list = []
        for i in range(len(feature_matrix)):
            past_window = feature_matrix[: i + 1]  # all up to day i
            emb = embedder.transform(past_window)   # [6, embed_dim]
            ctx = transformer(emb)                  # [d_model]
            context_list.append(ctx)

        context_matrix = np.vstack(context_list)

        # ----------------------------------
        # Train the experts
        # ----------------------------------
        experts_engine = ExpertsEngine(self.cfg)

        # Combine features + context
        X_full = experts_engine.build_training_matrix(
            feature_matrix,
            context_matrix
        )
        y = label_vector
        expert_regimes = regimes

        experts_engine.fit(X_full, y, expert_regimes)

        # ----------------------------------
        # Strategy + Risk engines
        # ----------------------------------
        strategy = Strategy(self.cfg, experts_engine)
        risk = RiskEngine(self.cfg)

        # ----------------------------------
        # Daily Backtest
        # ----------------------------------
        equity = risk.initial_equity
        equity_curve = []
        signals = []
        trades = []

        close = df_labeled["Close"].values
        high = df_labeled["High"].values
        low = df_labeled["Low"].values
        atr = df_labeled["ATR"].values

        d_model = self.cfg.transformer_model_dim

        for i in range(len(df_labeled)):

            # Build today's input vector
            x_today = X_full[i]
            reg_today = regimes[i]

            signal = strategy.generate_signal(x_today, reg_today)
            signals.append(signal)

            if signal == 0:
                # flat day
                equity_curve.append(equity)
                continue

            # Enter position
            entry_price = close[i]
            position_size = risk.compute_position_size(equity, atr[i])

            sl, tp = risk.compute_sl_tp(entry_price, atr[i])

            exit_price, outcome = risk.check_exit(
                sl=sl,
                tp=tp,
                day_open=entry_price,
                day_high=high[i],
                day_low=low[i],
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
                "day": i,
                "signal": signal,
                "entry": entry_price,
                "exit": exit_price,
                "pnl": trade_pnl,
                "equity_before": equity,
                "equity_after": new_equity,
                "regime": reg_today,
            })

            equity = new_equity
            equity_curve.append(equity)

        # ----------------------------------
        # Build results
        # ----------------------------------
        results = {
            "equity_curve": np.array(equity_curve),
            "signals": np.array(signals),
            "trades": trades,
            "final_equity": equity,
            "total_return_pct": (equity / risk.initial_equity - 1) * 100,
            "num_trades": len(trades),
            "win_rate": self._compute_win_rate(trades),
        }

        logger.info(f"[BACKTEST] Finished. Final equity: {equity:.2f}")

        return results

    # ------------------------------------------------------------
    #  INTERNAL: Compute win rate
    # ------------------------------------------------------------
    def _compute_win_rate(self, trades):
        if len(trades) == 0:
            return 0.0
        wins = [t for t in trades if t["pnl"] > 0]
        return len(wins) / len(trades)
