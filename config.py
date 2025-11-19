# config.py (V10 – DAILY timeframe, high-win-rate focused)
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BotConfig:
    # Instruments and timeframes
    # Yahoo Finance ticker format for FX
    symbols: List[str] = field(
        default_factory=lambda: ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
    )

    # DAILY timeframe for V10-1D
    timeframe: str = "1d"   # used as yfinance "interval"

    # Backtest window – multi-year but reasonable
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"

    # Data paths
    data_dir: str = "data"
    raw_subdir: str = "raw"
    processed_subdir: str = "processed"

    # Capital & risk – keep conservative
    initial_equity: float = 100000.0
    risk_per_trade: float = 0.005        # 0.5% of equity per trade
    max_total_risk: float = 0.03         # 3% total open risk
    max_drawdown_stop: float = 0.20      # Stop opening NEW trades if DD > 20%

    # Feature params (these are in DAYS now)
    ma_windows: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    atr_window: int = 14
    rsi_window: int = 14
    bb_window: int = 20
    bb_std: float = 2.0

    # Breakout params (daily Donchian channels)
    breakout_lookbacks: List[int] = field(default_factory=lambda: [20, 55])

    # Regime thresholds (trend & vol)
    trend_z_threshold: float = 0.5
    vol_z_high: float = 0.7
    vol_z_low: float = -0.7

    # Labeling / ML – DAILY horizon
    horizon_bars: int = 3          # predict ~3 days ahead
    label_threshold: float = 0.003 # 0.3% (~30 pips on EURUSD)

    # ML model base params
    ml_model_params_rf: Dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 6,
        "min_samples_leaf": 50,
        "random_state": 42,
        "n_jobs": -1
    })
    ml_model_params_gb: Dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "random_state": 42
    })

    # High-confidence filters (V10)
    ml_edge_threshold: float = 0.20   # |p_up - p_down| must exceed this
    min_confluence_score: float = 1.0 # sum of trend/breakout/mean-rev signs
    min_signal_abs: float = 0.25      # combined signal must be at least this

    # Transaction costs (still realistic for FX majors)
    spread_cost: float = 0.0001
    slippage: float = 0.00005
