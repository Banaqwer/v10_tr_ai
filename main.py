# main.py

from config import BotConfig
from data_engine import MarketDataConfig, MarketDataEngine
from feature_engine import FeatureEngine
from regime_engine import RegimeEngine
from ml_models import LabelBuilder, MLSignalModel
from strategy import StrategyEngine
from risk_engine import RiskEngine
from backtester import Backtester


def run_symbol(cfg, symbol):
    print(f"\n=== BACKTESTING {symbol} ===")

    mdc = MarketDataConfig(
        symbols=[symbol],
        timeframe=cfg.timeframe,
        data_dir=cfg.data_dir,
        default_start=cfg.start_date,
        default_end=cfg.end_date
    )

    data_engine = MarketDataEngine(mdc)
    df = data_engine.get_history(symbol)

    print(f"Loaded {len(df)} bars")

    fe = FeatureEngineer(cfg)
    df = fe.build_features(df)

    re = RegimeEngine(cfg)
    df = re.classify_regime(df)

    lb = LabelBuilder(cfg)
    df = lb.build_labels(df)

    ml = MLSignalModel(cfg)
    feats, rep = ml.train(df)

    print("ML model trained on features:", feats)
    print("Accuracy:", rep["accuracy"])

    strat = StrategyEngine(cfg, ml)
    risk = RiskEngine(cfg)
    bt = Backtester(cfg, strat, risk)

    stats = bt.run(df, symbol)

    print("Final equity:", round(stats["final_equity"], 2))
    print("Sharpe:", round(stats["sharpe"], 3))

    fname = f"equity_curve_{symbol.replace('=','').replace('/','_')}.csv"
    stats["equity_curve"].to_csv(fname)
    print("Saved:", fname)


def main():
    cfg = BotConfig()
    for s in cfg.symbols:
        run_symbol(cfg, s)


if __name__ == "__main__":
    main()
