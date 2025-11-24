# ================================================================
#  MAIN — V10-TR CCT-90
#  Full system launcher for training + backtesting
# ================================================================

from config import V10TRConfig
from backtester import Backtester
from utils import logger
import json
import numpy as np
import os


def save_results(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "final_equity": float(results["final_equity"]),
            "total_return_pct": float(results["total_return_pct"]),
            "win_rate": float(results["win_rate"]),
            "num_trades": int(results["num_trades"]),
        }, f, indent=4)

    # Save equity curve
    np.save(os.path.join(output_dir, "equity_curve.npy"),
            results["equity_curve"])

    # Save trades log
    trades_path = os.path.join(output_dir, "trades.json")
    with open(trades_path, "w") as f:
        json.dump(results["trades"], f, indent=4)

    logger.info(f"[MAIN] Results saved to {output_dir}/")


def main():
    logger.info("[MAIN] Starting V10-TR CCT-90 AI system...")

    # ------------------------------------------------------------
    # LOAD CONFIG
    # ------------------------------------------------------------
    cfg = V10TRConfig()

    # ------------------------------------------------------------
    # RUN BACKTEST
    # ------------------------------------------------------------
    bt = Backtester(cfg, symbol=cfg.symbol)
    results = bt.run()

    # ------------------------------------------------------------
    # PRINT SUMMARY
    # ------------------------------------------------------------
    logger.info("====================================")
    logger.info(f" FINAL EQUITY:     {results['final_equity']:.2f}")
    logger.info(f" TOTAL RETURN:     {results['total_return_pct']:.2f}%")
    logger.info(f" TRADES EXECUTED:  {results['num_trades']}")
    logger.info(f" WIN RATE:         {results['win_rate']*100:.1f}%")
    logger.info("====================================")

    # ------------------------------------------------------------
    # SAVE OUTPUT
    # ------------------------------------------------------------
    save_results(results)

    logger.info("[MAIN] Completed V10-TR CCT-90.")


if __name__ == "__main__":
    main()
