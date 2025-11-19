# backtester.py

import numpy as np
import pandas as pd
from execution import ExecutionSimulator
from risk_engine import RiskEngine
from strategy import StrategyEngine


class Backtester:
    def __init__(self, cfg, strat, risk):
        self.cfg = cfg
        self.strat = strat
        self.risk = risk
        self.exec = ExecutionSimulator(cfg.spread_cost, cfg.slippage)

    def run(self, df, symbol):
        equity = self.cfg.initial_equity
        positions = []
        curve = []

        for t, r in df.iterrows():
            price = r["close"]

            # update positions
            new_positions = []
            for p in positions:
                pnl = (price - p.entry_price) * p.units
                exit_flag = False

                if p.units > 0:
                    if price <= p.sl or price >= p.tp:
                        exit_flag = True
                else:
                    if price >= p.sl or price <= p.tp:
                        exit_flag = True

                if exit_flag:
                    equity += pnl
                else:
                    new_positions.append(p)

            positions = new_positions

            # compute current risk
            risk_taken = 0
            for p in positions:
                if p.units > 0:
                    risk_taken += max(0, (p.entry_price - p.sl) * abs(p.units))
                else:
                    risk_taken += max(0, (p.sl - p.entry_price) * abs(p.units))
            risk_taken /= equity

            # generate signal
            sig = self.strat.generate_signal(r)

            # open new trade?
            if abs(sig) > 0.1 and risk_taken < self.cfg.max_total_risk:
                units = self.risk.compute_units(equity, r, sig)
                sl, tp = self.risk.compute_sl_tp(r, sig)
                pos = self.exec.execute(symbol, units, price, sl, tp, str(t))
                if pos:
                    positions.append(pos)

            # mark-to-market equity
            mtm = sum([(price - p.entry_price) * p.units for p in positions])
            curve.append(equity + mtm)

        eq = pd.Series(curve, index=df.index, name="equity")
        ret = eq.pct_change().fillna(0)

        ann_factor = 252
        ann_return = (1 + ret.mean())**ann_factor - 1
        ann_vol = ret.std() * np.sqrt(ann_factor)
        sharpe = ann_return / (ann_vol + 1e-12)

        return {
            "final_equity": eq.iloc[-1],
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "equity_curve": eq
        }
