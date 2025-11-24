# ================================================================
#  V10-TR CCT-90 CONFIGURATION FILE
#  Full Transformer + Compressed Context + 5 Experts
#  DAILY timeframe FX trading AI (Render-ready)
# ================================================================

from dataclasses import dataclass, field
from datetime import datetime


# ================================================================
#  MARKET UNIVERSE
# ================================================================

UNIVERSE = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
]


# ================================================================
#  TIMEFRAME & DATA RANGE
# ================================================================

TIMEFRAME = "1d"

START_DATE = "2010-01-01"
END_DATE = datetime.utcnow().strftime("%Y-%m-%d")


# ================================================================
#  TRAIN / TEST SPLITS (rolling window backtest)
# ================================================================

TRAIN_YEARS = 4
TEST_YEARS = 1

TRAIN_WINDOW_DAYS = TRAIN_YEARS * 252
TEST_WINDOW_DAYS = TEST_YEARS * 252


# ================================================================
#  LABEL SETTINGS (TP/SL intraday simulation)
# ================================================================

FORWARD_HORIZON_DAYS = 10

ATR_WINDOW = 14
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0


# ================================================================
#  CCT-90 EMBEDDING SETTINGS
# ================================================================

CCT_WINDOW_DAYS = 90
CCT_CHUNK_SIZE = 15                    # 90 days → 6 chunks
CCT_EMBED_DIM = 32                    # embedding dimension per chunk


# ================================================================
#  TRANSFORMER SETTINGS (Render-safe CPU version)
# ================================================================

TRANSFORMER_LAYERS = 4
TRANSFORMER_HEADS = 4
TRANSFORMER_MODEL_DIM = 64           # dimension of transformer tokens
TRANSFORMER_FEEDFORWARD_DIM = 128    # inner feed-forward layer
TRANSFORMER_DROPOUT = 0.1


# ================================================================
#  REGIME SETTINGS
# ================================================================

N_REGIMES = 3   # Trend / Range / Shock


# ================================================================
#  EXPERT MODELS SETTINGS (5 experts)
# ================================================================

EXPERT_TYPES = [
    "trend_continuation",
    "trend_reversal",
    "mean_reversion",
    "volatility_breakout",
    "shock",
]

RF_ESTIMATORS = 250
RF_MAX_DEPTH = 6

GB_ESTIMATORS = 200
GB_LEARNING_RATE = 0.05
GB_MAX_DEPTH = 3


# ================================================================
#  RISK SETTINGS
# ================================================================

INITIAL_EQUITY = 100_000.0

RISK_PER_TRADE = 0.0075          # 0.75%
MAX_RISK_PER_SYMBOL = 0.02       # 2%
MAX_PORTFOLIO_RISK = 0.05        # 5%

DAILY_MAX_DRAWDOWN = 0.02        # stop trading symbol for the day

SLIPPAGE_PIPS = 0.2
COMMISSION_PER_TRADE = 0.0


# ================================================================
#  LOGGING SETTINGS
# ================================================================

VERBOSE = True


# ================================================================
#  MASTER CONFIG OBJECT
# ================================================================

@dataclass
class V10TRConfig:
    universe: list = field(default_factory=lambda: UNIVERSE)

    timeframe: str = TIMEFRAME
    start_date: str = START_DATE
    end_date: str = END_DATE

    train_window_days: int = TRAIN_WINDOW_DAYS
    test_window_days: int = TEST_WINDOW_DAYS

    forward_horizon_days: int = FORWARD_HORIZON_DAYS
    atr_window: int = ATR_WINDOW
    tp_atr_mult: float = TP_ATR_MULT
    sl_atr_mult: float = SL_ATR_MULT

    cct_window_days: int = CCT_WINDOW_DAYS
    cct_chunk_size: int = CCT_CHUNK_SIZE
    cct_embed_dim: int = CCT_EMBED_DIM

    transformer_layers: int = TRANSFORMER_LAYERS
    transformer_heads: int = TRANSFORMER_HEADS
    transformer_model_dim: int = TRANSFORMER_MODEL_DIM
    transformer_feedforward_dim: int = TRANSFORMER_FEEDFORWARD_DIM
    transformer_dropout: float = TRANSFORMER_DROPOUT

    n_regimes: int = N_REGIMES
    expert_types: list = field(default_factory=lambda: EXPERT_TYPES)

    rf_estimators: int = RF_ESTIMATORS
    rf_max_depth: int = RF_MAX_DEPTH

    gb_estimators: int = GB_ESTIMATORS
    gb_learning_rate: float = GB_LEARNING_RATE
    gb_max_depth: int = GB_MAX_DEPTH

    initial_equity: float = INITIAL_EQUITY
    risk_per_trade: float = RISK_PER_TRADE
    max_risk_per_symbol: float = MAX_RISK_PER_SYMBOL
    max_portfolio_risk: float = MAX_PORTFOLIO_RISK
    daily_max_drawdown: float = DAILY_MAX_DRAWDOWN

    slippage_pips: float = SLIPPAGE_PIPS
    commission_per_trade: float = COMMISSION_PER_TRADE

    verbose: bool = VERBOSE
