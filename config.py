# config.py
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BotConfig:
    # Instruments and timeframes
    symbols: List[str] = field(
        default_factory=lambda: ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
    )
    timeframe: str = "1H"

    # 4–5 year backtest window
    start_date: str = "2018-01-01"
    end_date: str = "2022-12-31"

    # Data paths
    data_dir: str = "data"
    raw_subdir: str = "raw"
    processed_subdir: str = "processed"

    # Risk and capital
    initial_equity: float = 100000.0
    risk_per_trade: float = 0.01      # 1%
    max_total_risk: float = 0.05      # 5%

    # Feature params
    ma_windows: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    atr_window: int = 14
    rsi_window: int = 14
    bb_window: int = 20
    bb_std: float = 2.0

    # Breakout params
    breakout_lookbacks: List[int] = field(default_factory=lambda: [20, 55])

    # Regime thresholds
    trend_z_threshold: float = 0.5
    vol_z_high: float = 0.7
    vol_z_low: float = -0.7

    # Labeling / ML
    horizon_bars: int = 4
    label_threshold: float = 0.0005    # 5 pips

    # Backtest transaction costs
    spread_cost: float = 0.0001        # 1 pip
    slippage: float = 0.00005

    # ML model config
    ml_model_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 50,
        "random_state": 42
    })
