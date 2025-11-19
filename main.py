# main.py
#
# V10-Daily controller script.
# Wires together:
#   - MarketDataEngine  (data_engine.py)
#   - FeatureEngine     (feature_engine.py)
#   - RegimeEngine      (regime_engine.py)
#   - LabelBuilder      (ml_models.py)
#   - EnsembleSignalModel (ml_models.py)
#   - StrategyEngine    (strategy.py)
#   - RiskEngine        (risk_engine.py)
#   - Backtester        (backtester.py)
#
# Runs a full backtest for each symbol listed in BotConfig.symbols.

from __future__ import annotations

import logging
from typing import Dict

from config import BotConfig
from data_engine import MarketDataConfig, MarketDataEngine
from feature_engine import FeatureEngine
from regime_engine import RegimeEngine
from ml_models import LabelBuilder, EnsembleSignalModel
from strategy import StrategyEngine
from risk_engine import RiskEngine
from backtester import Backtester


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-symbol pipeline
# ---------------------------------------------------------------------------

def run_symbol(cfg: BotConfig, symbol: str) -> Dict:
    logger.info(f"=== BACKTESTING {symbol} ===")

    # ----- DATA -----
    mdc = MarketDataConfig(
        symbols=[symbol],
        timeframe=cfg.timeframe,
        data_dir=cfg.data_dir,
        default_start=cfg.start_date,
        default_end=cfg.end_date,
    )
    data_engine = MarketDataEngine(mdc)
    df = data_engine.get_history(symbol, start=cfg.start_date, end=cfg.end_date)

    logger.info(f"Loaded {len(df)} bars for {symbol}")

    # ----- FEATURES -----
    fe = FeatureEngine()        # NOTE: new class name & no cfg arg
    df = fe.build_features(df)

    logger.info(f"After feature engineering: {len(df)} rows, {len(df.columns)} columns")

    # ----- REGIME CLASSIFICATION -----
    reg_engine = RegimeEngine(cfg)
    df = reg_engine.classify_regime(df)

    # ----- LABELS -----
    label_builder = LabelBuilder(cfg)
    df = label_builder.build_labels(df)

    # Filter out rows without labels if your LabelBuilder uses 0 / NaN
    if "label" not in df.columns:
        raise ValueError("Label column 'label' not found after label building.")
    df = df.dropna(subset=["label"])

    logger.info(f"After labeling: {len(df)} rows remain")

    # ----- MODEL TRAINING -----
    model = EnsembleSignalModel(cfg)
    report = model.train(df)
    logger.info(f"Model training report for {symbol}: {report}")

    # ----- STRATEGY & RISK ENGINES -----
    strat_engine = StrategyEngine(cfg, model)
    risk_engine = RiskEngine(cfg)

    # ----- BACKTEST -----
    backtester = Backtester(cfg, strat_engine, risk_engine)
    stats = backtester.run(df, symbol)

    logger.info(f"Backtest stats for {symbol}: {stats}")

    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    cfg = BotConfig()
    all_stats: Dict[str, Dict] = {}

    logger.info("Starting V10-Daily backtests...")
    logger.info(
        f"Symbols: {cfg.symbols}, timeframe={cfg.timeframe}, "
        f"start={cfg.start_date}, end={cfg.end_date}"
    )

    for s in cfg.symbols:
        try:
            stats = run_symbol(cfg, s)
            all_stats[s] = stats
        except Exception as e:
            logger.exception(f"Error while processing {s}: {e}")

    logger.info("=== SUMMARY ===")
    for sym, st in all_stats.items():
        logger.info(
            f"{sym}: final_equity={st.get('final_equity')}, "
            f"ann_return={st.get('ann_return')}, "
            f"sharpe={st.get('sharpe')}, "
            f"max_drawdown={st.get('max_drawdown')}"
        )


if __name__ == "__main__":
    main()
