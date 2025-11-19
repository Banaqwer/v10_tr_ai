# execution.py

from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    units: float
    entry_price: float
    sl: float
    tp: float
    entry_time: str


class ExecutionSimulator:
    def __init__(self, spread, slip):
        self.spread = spread
        self.slip = slip

    def execute(self, symbol, units, price, sl, tp, t):
        if units == 0:
            return None

        if units > 0:
            fill = price + self.spread/2 + self.slip
        else:
            fill = price - self.spread/2 - self.slip

        return Position(symbol, units, fill, sl, tp, t)
