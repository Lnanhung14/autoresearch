"""
Trading strategies for backtesting optimization.
Each strategy takes OHLCV data and parameters, returns buy/sell signals.

To add a custom strategy:
1. Define a function with signature: (df: pd.DataFrame, **params) -> pd.DataFrame
2. Register it in STRATEGY_REGISTRY with its parameter search space
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable


@dataclass
class StrategyParam:
    """Defines a single parameter's search space."""
    name: str
    min_val: float
    max_val: float
    step: float
    default: float

    def range(self) -> List[float]:
        vals = []
        v = self.min_val
        while v <= self.max_val + 1e-9:
            vals.append(round(v, 6))
            v += self.step
        return vals


@dataclass
class StrategyDef:
    """Defines a strategy and its parameter search space."""
    name: str
    func: Callable
    params: List[StrategyParam]
    description: str = ""

    def default_params(self) -> Dict[str, float]:
        return {p.name: p.default for p in self.params}


# ---------------------------------------------------------------------------
# Strategy implementations
# Each returns a DataFrame with added 'signal' column:
#   1 = buy, -1 = sell, 0 = hold
# ---------------------------------------------------------------------------

def sma_crossover(df: pd.DataFrame, fast_period: int = 5, slow_period: int = 20) -> pd.DataFrame:
    """Simple Moving Average crossover strategy."""
    df = df.copy()
    fast_period, slow_period = int(fast_period), int(slow_period)
    df['sma_fast'] = df['close'].rolling(window=fast_period).mean()
    df['sma_slow'] = df['close'].rolling(window=slow_period).mean()
    df['signal'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
    df.loc[df['sma_fast'] <= df['sma_slow'], 'signal'] = -1
    # Generate trade signals on crossover points
    df['signal'] = df['signal'].diff().fillna(0)
    df.loc[df['signal'] > 0, 'signal'] = 1   # buy
    df.loc[df['signal'] < 0, 'signal'] = -1  # sell
    df.loc[df['signal'] == 0, 'signal'] = 0  # hold
    return df


def ema_crossover(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26) -> pd.DataFrame:
    """Exponential Moving Average crossover strategy."""
    df = df.copy()
    fast_period, slow_period = int(fast_period), int(slow_period)
    df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['signal'] = 0
    df.loc[df['ema_fast'] > df['ema_slow'], 'signal'] = 1
    df.loc[df['ema_fast'] <= df['ema_slow'], 'signal'] = -1
    df['signal'] = df['signal'].diff().fillna(0)
    df.loc[df['signal'] > 0, 'signal'] = 1
    df.loc[df['signal'] < 0, 'signal'] = -1
    df.loc[df['signal'] == 0, 'signal'] = 0
    return df


def rsi_strategy(df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70) -> pd.DataFrame:
    """RSI overbought/oversold strategy."""
    df = df.copy()
    period = int(period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['signal'] = 0
    df.loc[df['rsi'] < oversold, 'signal'] = 1       # oversold → buy
    df.loc[df['rsi'] > overbought, 'signal'] = -1    # overbought → sell
    return df


def macd_strategy(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                   signal_period: int = 9) -> pd.DataFrame:
    """MACD crossover strategy."""
    df = df.copy()
    fast_period, slow_period, signal_period = int(fast_period), int(slow_period), int(signal_period)
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    df['signal'] = 0
    prev_hist = df['macd_hist'].shift(1)
    df.loc[(df['macd_hist'] > 0) & (prev_hist <= 0), 'signal'] = 1   # MACD crosses above signal
    df.loc[(df['macd_hist'] < 0) & (prev_hist >= 0), 'signal'] = -1  # MACD crosses below signal
    return df


def bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands mean reversion strategy."""
    df = df.copy()
    period = int(period)
    df['bb_mid'] = df['close'].rolling(window=period).mean()
    bb_std = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_mid'] + num_std * bb_std
    df['bb_lower'] = df['bb_mid'] - num_std * bb_std

    df['signal'] = 0
    df.loc[df['close'] < df['bb_lower'], 'signal'] = 1    # below lower → buy
    df.loc[df['close'] > df['bb_upper'], 'signal'] = -1   # above upper → sell
    return df


def kdj_strategy(df: pd.DataFrame, k_period: int = 9, d_period: int = 3,
                  oversold: float = 20, overbought: float = 80) -> pd.DataFrame:
    """KDJ (Stochastic) oscillator strategy."""
    df = df.copy()
    k_period, d_period = int(k_period), int(d_period)
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    rsv = (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan) * 100

    df['k'] = rsv.ewm(com=d_period - 1, adjust=False).mean()
    df['d'] = df['k'].ewm(com=d_period - 1, adjust=False).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']

    df['signal'] = 0
    # Golden cross in oversold zone
    df.loc[(df['k'] > df['d']) & (df['k'].shift(1) <= df['d'].shift(1)) & (df['k'] < oversold), 'signal'] = 1
    # Death cross in overbought zone
    df.loc[(df['k'] < df['d']) & (df['k'].shift(1) >= df['d'].shift(1)) & (df['k'] > overbought), 'signal'] = -1
    return df


def volume_price_strategy(df: pd.DataFrame, vol_ma_period: int = 20,
                           vol_threshold: float = 1.5, price_ma_period: int = 10) -> pd.DataFrame:
    """Volume-price breakout strategy: high volume + price above MA → buy."""
    df = df.copy()
    vol_ma_period, price_ma_period = int(vol_ma_period), int(price_ma_period)
    df['vol_ma'] = df['volume'].rolling(window=vol_ma_period).mean()
    df['price_ma'] = df['close'].rolling(window=price_ma_period).mean()

    df['signal'] = 0
    high_vol = df['volume'] > vol_threshold * df['vol_ma']
    df.loc[high_vol & (df['close'] > df['price_ma']), 'signal'] = 1
    df.loc[high_vol & (df['close'] < df['price_ma']), 'signal'] = -1
    return df


def dual_ma_rsi_filter(df: pd.DataFrame, fast_period: int = 5, slow_period: int = 20,
                        rsi_period: int = 14, rsi_threshold: float = 50) -> pd.DataFrame:
    """Dual MA crossover with RSI filter: buy only if RSI confirms trend."""
    df = df.copy()
    fast_period, slow_period, rsi_period = int(fast_period), int(slow_period), int(rsi_period)

    df['sma_fast'] = df['close'].rolling(window=fast_period).mean()
    df['sma_slow'] = df['close'].rolling(window=slow_period).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    ma_cross_up = (df['sma_fast'] > df['sma_slow']) & (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1))
    ma_cross_down = (df['sma_fast'] < df['sma_slow']) & (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1))

    df['signal'] = 0
    df.loc[ma_cross_up & (df['rsi'] > rsi_threshold), 'signal'] = 1
    df.loc[ma_cross_down & (df['rsi'] < rsi_threshold), 'signal'] = -1
    return df


# ---------------------------------------------------------------------------
# Strategy Registry - maps strategy name to StrategyDef
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, StrategyDef] = {
    "sma_crossover": StrategyDef(
        name="sma_crossover",
        func=sma_crossover,
        description="Simple Moving Average crossover",
        params=[
            StrategyParam("fast_period", 3, 30, 1, 5),
            StrategyParam("slow_period", 10, 120, 5, 20),
        ],
    ),
    "ema_crossover": StrategyDef(
        name="ema_crossover",
        func=ema_crossover,
        description="Exponential Moving Average crossover",
        params=[
            StrategyParam("fast_period", 3, 30, 1, 12),
            StrategyParam("slow_period", 10, 120, 5, 26),
        ],
    ),
    "rsi": StrategyDef(
        name="rsi",
        func=rsi_strategy,
        description="RSI overbought/oversold",
        params=[
            StrategyParam("period", 5, 30, 1, 14),
            StrategyParam("oversold", 15, 40, 5, 30),
            StrategyParam("overbought", 60, 85, 5, 70),
        ],
    ),
    "macd": StrategyDef(
        name="macd",
        func=macd_strategy,
        description="MACD crossover",
        params=[
            StrategyParam("fast_period", 6, 20, 2, 12),
            StrategyParam("slow_period", 18, 40, 2, 26),
            StrategyParam("signal_period", 5, 15, 1, 9),
        ],
    ),
    "bollinger_bands": StrategyDef(
        name="bollinger_bands",
        func=bollinger_bands,
        description="Bollinger Bands mean reversion",
        params=[
            StrategyParam("period", 10, 40, 5, 20),
            StrategyParam("num_std", 1.0, 3.0, 0.25, 2.0),
        ],
    ),
    "kdj": StrategyDef(
        name="kdj",
        func=kdj_strategy,
        description="KDJ (Stochastic) oscillator",
        params=[
            StrategyParam("k_period", 5, 21, 2, 9),
            StrategyParam("d_period", 2, 7, 1, 3),
            StrategyParam("oversold", 10, 30, 5, 20),
            StrategyParam("overbought", 70, 90, 5, 80),
        ],
    ),
    "volume_price": StrategyDef(
        name="volume_price",
        func=volume_price_strategy,
        description="Volume-price breakout",
        params=[
            StrategyParam("vol_ma_period", 10, 40, 5, 20),
            StrategyParam("vol_threshold", 1.0, 3.0, 0.25, 1.5),
            StrategyParam("price_ma_period", 5, 30, 5, 10),
        ],
    ),
    "dual_ma_rsi": StrategyDef(
        name="dual_ma_rsi",
        func=dual_ma_rsi_filter,
        description="Dual MA crossover + RSI filter",
        params=[
            StrategyParam("fast_period", 3, 20, 1, 5),
            StrategyParam("slow_period", 10, 60, 5, 20),
            StrategyParam("rsi_period", 5, 25, 1, 14),
            StrategyParam("rsi_threshold", 30, 70, 5, 50),
        ],
    ),
}


def get_strategy(name: str) -> StrategyDef:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> List[str]:
    return list(STRATEGY_REGISTRY.keys())
