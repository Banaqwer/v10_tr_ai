# optimizer.py
"""
V10 Optimizer – semi-automatic parameter sweeper.

Usage (locally or on Render):
    python optimizer.py

It will:
  - loop over a small grid of high-confidence parameters
  - run backtests for each combo on each symbol
  - log results to optimizer_results.csv
You then inspect that CSV and choose the best config manually.
"""

import itertools
import csv
from copy import deepcopy
from datetime import datetime

from config import BotConfig
from main import run_single_symbol  # we reuse the V10 controller


def run_with_params(base_cfg: BotConfig,
                    symbol: str,
                    ml_edge_threshold: float,
                    min_confluence_score: float,
                    risk_per_trade: float):
    # clone config and tweak
    cfg = deepcopy(base_cfg)
    cfg.ml_edge_threshold = ml_edge_threshold
    cfg.min_confluence_score = min_confluence_score
    cfg.risk_per_trade = risk_per_trade

    # run_single_symbol prints stats and saves equity curve.
    # We slightly modify it here to also return the stats object
    # so we need to adjust main.run_single_symbol to return stats.
    from backtester import Backtester
    from data_engine import MarketDataConfig, MarketDataEngine
    from feature_engine import FeatureEngineer
    from regime_engine import RegimeEngine
    from ml_models import LabelBuilder, EnsembleSignalModel
    from strategy import StrategyEngine
    from risk_engine import RiskEngine

    print(f"\n=== OPT RUN {symbol} | ml_edge={ml_edge_threshold} "
          f"| conf={min_confluence_score} | risk={risk_per_trade} ===")

    mdc = MarketDataConfig(
        symbols=[symbol],
        timeframe=cfg.timeframe,
        data_dir=cfg.data_dir,
        default_start=cfg.start_date,
        default_end=cfg.end_date
    )
    data_engine = MarketDataEngine(mdc)
    df = data_engine.get_history(symbol, start=cfg.start_date, end=cfg.end_date)

    fe = FeatureEngineer(cfg)
    df = fe.build_features(df)

    re = RegimeEngine(cfg)
    df = re.classify_regime(df)

    lb = LabelBuilder(cfg)
    df = lb.build_labels(df)

    ml_model = EnsembleSignalModel(cfg)
    report = ml_model.train(df)

    strat_engine = StrategyEngine(cfg, ml_model)
    risk_engine = RiskEngine(cfg)
    backtester = Backtester(cfg, strat_engine, risk_engine)
    stats = backtester.run(df, symbol)

    return stats, report


def main():
    base_cfg = BotConfig()

    # Small, sane grid – adjust if you want wider search
    ml_edge_vals = [0.18, 0.20, 0.22]
    conf_vals = [1.0, 1.5, 2.0]
    risk_vals = [0.003, 0.005, 0.007]

    # Output CSV
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = f"optimizer_results_{ts}.csv"

    fieldnames = [
        "symbol",
        "ml_edge_threshold",
        "min_confluence_score",
        "risk_per_trade",
        "final_equity",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "test_accuracy",
        "test_precision_up",
        "test_precision_down"
    ]

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for symbol in base_cfg.symbols:
            for ml_edge, conf, risk in itertools.product(
                ml_edge_vals, conf_vals, risk_vals
            ):
                stats, report = run_with_params(
                    base_cfg, symbol, ml_edge, conf, risk
                )

                # Extract some ML test metrics
                acc = report.get("accuracy", 0.0)
                prec_up = report.get("1", {}).get("precision", 0.0)
                prec_down = report.get("-1", {}).get("precision", 0.0)

                row = {
                    "symbol": symbol,
                    "ml_edge_threshold": ml_edge,
                    "min_confluence_score": conf,
                    "risk_per_trade": risk,
                    "final_equity": stats["final_equity"],
                    "ann_return": stats["ann_return"],
                    "ann_vol": stats["ann_vol"],
                    "sharpe": stats["sharpe"],
                    "max_drawdown": stats["max_drawdown"],
                    "test_accuracy": acc,
                    "test_precision_up": prec_up,
                    "test_precision_down": prec_down,
                }
                writer.writerow(row)
                print(f"Logged result: {row}")

    print(f"\nOptimizer finished. Results saved to {out_file}")


if __name__ == "__main__":
    main()
