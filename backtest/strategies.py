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
                  oversold: float = 15, overbought: float = 90) -> pd.DataFrame:
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
                           vol_threshold: float = 2.5, price_ma_period: int = 10) -> pd.DataFrame:
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
                        rsi_period: int = 14, rsi_threshold: float = 70) -> pd.DataFrame:
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
# Gemini APP2 Stock strategies (12 strategies adapted)
# ---------------------------------------------------------------------------

def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Helper: calculate RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calc_kd(df: pd.DataFrame, k_period: int = 9) -> tuple:
    """Helper: calculate KD values."""
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    rsv = (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d


def multi_factor_strategy(df: pd.DataFrame, momentum_days: int = 60,
                           rsi_period: int = 14, rsi_buy: float = 40,
                           rsi_sell: float = 70) -> pd.DataFrame:
    """Multi-factor ranking strategy: momentum + RSI scoring."""
    df = df.copy()
    momentum_days = int(momentum_days)
    rsi_period = int(rsi_period)

    # Momentum: current close / close N days ago - 1
    df['momentum'] = df['close'] / df['close'].shift(momentum_days) - 1
    df['rsi'] = _calc_rsi(df['close'], rsi_period)

    df['signal'] = 0
    # Buy: positive momentum + RSI not overbought
    df.loc[(df['momentum'] > 0) & (df['rsi'] < rsi_sell) & (df['rsi'] > rsi_buy), 'signal'] = 1
    # Sell: negative momentum or RSI overbought
    df.loc[(df['momentum'] < 0) | (df['rsi'] > rsi_sell), 'signal'] = -1
    # Only trigger on state change
    df['signal'] = df['signal'].diff().fillna(0)
    df.loc[df['signal'] > 0, 'signal'] = 1
    df.loc[df['signal'] < 0, 'signal'] = -1
    df.loc[df['signal'] == 0, 'signal'] = 0
    return df


def kd_macd_signal_strategy(df: pd.DataFrame, kd_oversold: float = 20,
                              kd_overbought: float = 80, macd_fast: int = 12,
                              macd_slow: int = 26, macd_signal: int = 9) -> pd.DataFrame:
    """KD golden cross + MACD histogram crossover strategy."""
    df = df.copy()
    kd_oversold, kd_overbought = float(kd_oversold), float(kd_overbought)
    macd_fast, macd_slow, macd_signal = int(macd_fast), int(macd_slow), int(macd_signal)

    # KD
    k, d = _calc_kd(df)
    df['k'], df['d'] = k, d

    # MACD
    ema_f = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_s = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd_hist'] = (ema_f - ema_s) - (ema_f - ema_s).ewm(span=macd_signal, adjust=False).mean()

    df['signal'] = 0
    k_prev, d_prev = df['k'].shift(1), df['d'].shift(1)
    macd_prev = df['macd_hist'].shift(1)

    # Buy: KD golden cross in oversold zone OR MACD histogram turns positive
    buy = ((k_prev < d_prev) & (df['k'] > df['d']) & (df['k'] < kd_oversold)) | \
          ((macd_prev < 0) & (df['macd_hist'] > 0))
    # Sell: KD death cross in overbought zone OR MACD histogram turns negative
    sell = ((k_prev > d_prev) & (df['k'] < df['d']) & (df['k'] > kd_overbought)) | \
           ((macd_prev > 0) & (df['macd_hist'] < 0))

    df.loc[buy, 'signal'] = 1
    df.loc[sell, 'signal'] = -1
    return df


def ma_breakout_strategy(df: pd.DataFrame, ma_short: int = 5, ma_long: int = 20,
                          ma_trend: int = 60) -> pd.DataFrame:
    """MA breakout trend strategy: price breaks above MA with trend confirmation."""
    df = df.copy()
    ma_short, ma_long, ma_trend = int(ma_short), int(ma_long), int(ma_trend)

    df['ma_short'] = df['close'].rolling(ma_short).mean()
    df['ma_long'] = df['close'].rolling(ma_long).mean()
    df['ma_trend'] = df['close'].rolling(ma_trend).mean()

    df['signal'] = 0
    # Buy: price > ma_long, short MA > long MA > trend MA (uptrend)
    buy = (df['close'] > df['ma_long']) & (df['ma_short'] > df['ma_long']) & (df['ma_long'] > df['ma_trend'])
    # Sell: price < ma_long or short MA < long MA
    sell = (df['close'] < df['ma_long']) | (df['ma_short'] < df['ma_long'])

    df.loc[buy, 'signal'] = 1
    df.loc[sell, 'signal'] = -1
    df['signal'] = df['signal'].diff().fillna(0)
    df.loc[df['signal'] > 0, 'signal'] = 1
    df.loc[df['signal'] < 0, 'signal'] = -1
    df.loc[df['signal'] == 0, 'signal'] = 0
    return df


def foreign_follow_strategy(df: pd.DataFrame, vol_surge: float = 2.0,
                              rsi_min: float = 40, rsi_sell: float = 75) -> pd.DataFrame:
    """Foreign institutional follow strategy (volume surge proxy)."""
    df = df.copy()
    vol_ma = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / vol_ma.replace(0, np.nan)
    df['rsi'] = _calc_rsi(df['close'], 14)

    df['signal'] = 0
    # Buy: volume surge + RSI in favorable range
    df.loc[(df['vol_ratio'] > vol_surge) & (df['rsi'] > rsi_min) & (df['rsi'] < rsi_sell), 'signal'] = 1
    # Sell: RSI overbought
    df.loc[df['rsi'] > rsi_sell, 'signal'] = -1
    return df


def etf_momentum_strategy(df: pd.DataFrame, momentum_days: int = 60,
                            rebal_period: int = 20) -> pd.DataFrame:
    """ETF momentum strategy: buy when momentum is positive, rebalance periodically."""
    df = df.copy()
    momentum_days, rebal_period = int(momentum_days), int(rebal_period)

    df['momentum'] = df['close'] / df['close'].shift(momentum_days) - 1
    df['ma20'] = df['close'].rolling(20).mean()

    df['signal'] = 0
    # Create rebalance points
    rebal_mask = pd.Series(False, index=df.index)
    for i in range(0, len(df), max(rebal_period, 1)):
        rebal_mask.iloc[i] = True

    # On rebalance dates: buy if momentum positive and above MA20, sell otherwise
    df.loc[rebal_mask & (df['momentum'] > 0) & (df['close'] > df['ma20']), 'signal'] = 1
    df.loc[rebal_mask & ((df['momentum'] <= 0) | (df['close'] <= df['ma20'])), 'signal'] = -1
    return df


def vol_price_sync_strategy(df: pd.DataFrame, vol_surge_multiplier: float = 1.5,
                              ma_period: int = 20) -> pd.DataFrame:
    """Volume-price sync strategy: price up + volume surge = bullish confirmation."""
    df = df.copy()
    ma_period = int(ma_period)
    vol_ma = df['volume'].rolling(ma_period).mean()
    df['vol_ratio'] = df['volume'] / vol_ma.replace(0, np.nan)
    df['ma'] = df['close'].rolling(ma_period).mean()
    price_up = df['close'] > df['close'].shift(1)

    df['signal'] = 0
    # Buy: price up + volume surge + above MA
    df.loc[price_up & (df['vol_ratio'] > vol_surge_multiplier) & (df['close'] > df['ma']), 'signal'] = 1
    # Sell: price up but volume shrink (bearish divergence) or below MA
    price_up_no_vol = price_up & (df['vol_ratio'] < 0.7)
    df.loc[price_up_no_vol | (df['close'] < df['ma']), 'signal'] = -1
    return df


def vol_breakout_vcp_strategy(df: pd.DataFrame, vol_surge_multiplier: float = 2.0,
                                range_contract: float = 0.7, lookback: int = 20) -> pd.DataFrame:
    """VCP (Volatility Contraction Pattern) breakout strategy."""
    df = df.copy()
    lookback = int(lookback)
    df['high_n'] = df['high'].rolling(lookback).max()
    df['low_n'] = df['low'].rolling(lookback).min()
    df['range_pct'] = (df['high_n'] - df['low_n']) / df['close'].replace(0, np.nan) * 100
    df['range_prev'] = df['range_pct'].shift(lookback)
    vol_ma = df['volume'].rolling(lookback).mean()
    df['vol_ratio'] = df['volume'] / vol_ma.replace(0, np.nan)
    vol_ma5 = df['volume'].rolling(5).mean()

    df['signal'] = 0
    # Contracting range
    contracting = (df['range_pct'] < df['range_prev'] * range_contract) & (df['range_pct'] < 15)
    # Volume dry then surge
    vol_dry = vol_ma5 < vol_ma * 0.8
    # Breakout above recent high with volume
    breakout = df['close'] >= df['high_n']
    vol_expand = df['vol_ratio'] > vol_surge_multiplier

    df.loc[(contracting | vol_dry) & breakout & vol_expand, 'signal'] = 1
    # Sell: price drops below MA20
    ma20 = df['close'].rolling(20).mean()
    df.loc[df['close'] < ma20, 'signal'] = -1
    return df


def vol_divergence_strategy(df: pd.DataFrame, lookback_days: int = 20) -> pd.DataFrame:
    """Volume-price divergence reversal strategy using OBV."""
    df = df.copy()
    lookback_days = int(lookback_days)

    # OBV
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv
    df['obv_ma'] = obv.rolling(lookback_days).mean()
    df['high_n'] = df['high'].rolling(lookback_days).max()
    df['low_n'] = df['low'].rolling(lookback_days).min()
    df['obv_high_n'] = obv.rolling(lookback_days).max()
    df['obv_low_n'] = obv.rolling(lookback_days).min()

    df['signal'] = 0
    # Bullish divergence: price near low but OBV not at low + OBV above MA
    price_near_low = df['close'] <= df['low_n'] * 1.02
    obv_not_low = df['obv'] > df['obv_low_n'] * 1.05
    df.loc[price_near_low & obv_not_low & (df['obv'] > df['obv_ma']), 'signal'] = 1

    # Bearish divergence: price near high but OBV not at high
    price_near_high = df['close'] >= df['high_n'] * 0.98
    obv_not_high = df['obv'] < df['obv_high_n'] * 0.95
    df.loc[price_near_high & obv_not_high, 'signal'] = -1
    return df


def chip_sedimentation_strategy(df: pd.DataFrame, vol_lookback: int = 20,
                                  vol_shrink_threshold: float = 0.8,
                                  vol_shrink_min_days: int = 2,
                                  price_range_threshold: float = 15.0,
                                  sedimentation_window: int = 10,
                                  vol_surge_multiplier: float = 1.2) -> pd.DataFrame:
    """Chip sedimentation momentum breakout: volume contraction + price consolidation → breakout."""
    df = df.copy()
    n = int(vol_lookback)

    # Dimension 1: Volume shrinkage
    vol_ma = df['volume'].rolling(n).mean()
    is_low_vol = df['volume'] < vol_ma * vol_shrink_threshold
    low_count = is_low_vol.rolling(int(vol_shrink_min_days)).sum()
    dim1 = low_count >= max(vol_shrink_min_days * 0.5, 2)

    # Dimension 2: Price consolidation
    high_n = df['high'].rolling(n).max()
    low_n = df['low'].rolling(n).min()
    ma_n = df['close'].rolling(n).mean()
    price_range = (high_n - low_n) / ma_n.replace(0, np.nan) * 100
    dim2 = price_range < price_range_threshold

    # Sedimentation: volume shrink + price consolidation
    sedimentation = dim1 & dim2
    win = int(sedimentation_window)
    sedimentation_context = sedimentation.rolling(win).max().shift(1).fillna(0).astype(bool)

    # Breakout: price breaks above recent high with volume
    range_high = df['high'].rolling(n).max()
    price_break = df['close'] > range_high.shift(1)
    vol_confirm = df['volume'] > vol_ma * vol_surge_multiplier

    df['signal'] = 0
    df.loc[sedimentation_context & price_break & vol_confirm, 'signal'] = 1
    # Sell: price drops below consolidation low or volume dries up again
    df.loc[df['close'] < low_n.shift(1), 'signal'] = -1
    return df


def chip_sedimentation_inst_strategy(df: pd.DataFrame, vol_lookback: int = 20,
                                       vol_shrink_threshold: float = 0.8,
                                       price_range_threshold: float = 15.0,
                                       sedimentation_window: int = 10,
                                       vol_surge_multiplier: float = 2.0) -> pd.DataFrame:
    """Chip sedimentation + institutional resonance (volume surge > 2x as proxy)."""
    df = df.copy()
    n = int(vol_lookback)

    vol_ma = df['volume'].rolling(n).mean()
    is_low_vol = df['volume'] < vol_ma * vol_shrink_threshold
    low_count = is_low_vol.rolling(2).sum()
    dim1 = low_count >= 1

    high_n = df['high'].rolling(n).max()
    low_n = df['low'].rolling(n).min()
    ma_n = df['close'].rolling(n).mean()
    price_range = (high_n - low_n) / ma_n.replace(0, np.nan) * 100
    dim2 = price_range < price_range_threshold

    sedimentation = dim1 & dim2
    win = int(sedimentation_window)
    sedimentation_context = sedimentation.rolling(win).max().shift(1).fillna(0).astype(bool)

    range_high = df['high'].rolling(n).max()
    price_break = df['close'] > range_high.shift(1)
    # Institutional resonance: require stronger volume surge
    vol_confirm = df['volume'] > vol_ma * vol_surge_multiplier

    df['signal'] = 0
    df.loc[sedimentation_context & price_break & vol_confirm, 'signal'] = 1
    df.loc[df['close'] < low_n.shift(1), 'signal'] = -1
    return df


def chip_sedimentation_rising_strategy(df: pd.DataFrame, vol_lookback: int = 20,
                                         vol_shrink_threshold: float = 0.8,
                                         price_range_threshold: float = 15.0,
                                         sedimentation_window: int = 10,
                                         vol_surge_multiplier: float = 1.2,
                                         rsi_min: float = 40) -> pd.DataFrame:
    """Chip sedimentation + rising point (RSI + MACD momentum confirmation)."""
    df = df.copy()
    n = int(vol_lookback)

    vol_ma = df['volume'].rolling(n).mean()
    is_low_vol = df['volume'] < vol_ma * vol_shrink_threshold
    low_count = is_low_vol.rolling(2).sum()
    dim1 = low_count >= 1

    high_n = df['high'].rolling(n).max()
    low_n = df['low'].rolling(n).min()
    ma_n = df['close'].rolling(n).mean()
    price_range = (high_n - low_n) / ma_n.replace(0, np.nan) * 100
    dim2 = price_range < price_range_threshold

    sedimentation = dim1 & dim2
    win = int(sedimentation_window)
    sedimentation_context = sedimentation.rolling(win).max().shift(1).fillna(0).astype(bool)

    range_high = df['high'].rolling(n).max()
    price_break = df['close'] > range_high.shift(1)
    vol_confirm = df['volume'] > vol_ma * vol_surge_multiplier

    # Momentum confirmation: RSI > rsi_min and MACD histogram > 0
    rsi = _calc_rsi(df['close'], 14)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    momentum_ok = (rsi > rsi_min) & (macd_hist > 0)

    df['signal'] = 0
    df.loc[sedimentation_context & price_break & vol_confirm & momentum_ok, 'signal'] = 1
    df.loc[df['close'] < low_n.shift(1), 'signal'] = -1
    return df


def vol_dry_bottom_strategy(df: pd.DataFrame, vol_dry_ratio: float = 0.7,
                              vol_dry_days: int = 2, kd_oversold: float = 30,
                              rsi_max: float = 55) -> pd.DataFrame:
    """Volume dry bottom reversal: volume contraction + KD oversold + red candle entry."""
    df = df.copy()
    vol_dry_days = int(vol_dry_days)
    vol_ma = df['volume'].rolling(20).mean()
    k, d_val = _calc_kd(df)
    rsi = _calc_rsi(df['close'], 14)

    df['signal'] = 0

    for i in range(max(30, vol_dry_days + 1), len(df)):
        # Condition 1: Recent volume contraction
        recent_vols = df['volume'].iloc[max(0, i - vol_dry_days):i]
        vm = vol_ma.iloc[i]
        if pd.isna(vm) or vm <= 0:
            continue
        dry_count = (recent_vols < vm * vol_dry_ratio).sum()
        if dry_count < vol_dry_days * 0.6:
            continue

        # Condition 2: KD oversold
        k_val = k.iloc[i]
        if pd.isna(k_val) or k_val > kd_oversold:
            continue

        # Condition 3: RSI not overbought
        rsi_val = rsi.iloc[i]
        if pd.isna(rsi_val) or rsi_val > rsi_max:
            continue

        # Entry signal: red candle (close > prev close) + volume increase
        close_now = df['close'].iloc[i]
        close_prev = df['close'].iloc[i - 1]
        vol_now = df['volume'].iloc[i]
        vol_prev = df['volume'].iloc[i - 1]
        if close_now > close_prev and vol_now > vol_prev * 1.1:
            df.iloc[i, df.columns.get_loc('signal')] = 1

    # Sell: KD overbought
    df.loc[(k > 80) & (d_val > 75), 'signal'] = -1
    return df


# ---------------------------------------------------------------------------
# Strategy Registry - maps strategy name to StrategyDef
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, StrategyDef] = {
    "sma_crossover": StrategyDef(
        name="sma_crossover",
        func=sma_crossover,
        description="簡單移動平均線交叉",
        params=[
            StrategyParam("fast_period", 3, 30, 1, 5),
            StrategyParam("slow_period", 10, 120, 5, 20),
        ],
    ),
    "ema_crossover": StrategyDef(
        name="ema_crossover",
        func=ema_crossover,
        description="指數移動平均線交叉",
        params=[
            StrategyParam("fast_period", 3, 30, 1, 12),
            StrategyParam("slow_period", 10, 120, 5, 26),
        ],
    ),
    "rsi": StrategyDef(
        name="rsi",
        func=rsi_strategy,
        description="RSI 超買超賣",
        params=[
            StrategyParam("period", 5, 30, 1, 14),
            StrategyParam("oversold", 15, 40, 5, 30),
            StrategyParam("overbought", 60, 85, 5, 70),
        ],
    ),
    "macd": StrategyDef(
        name="macd",
        func=macd_strategy,
        description="MACD 交叉",
        params=[
            StrategyParam("fast_period", 6, 20, 2, 12),
            StrategyParam("slow_period", 18, 40, 2, 26),
            StrategyParam("signal_period", 5, 15, 1, 9),
        ],
    ),
    "bollinger_bands": StrategyDef(
        name="bollinger_bands",
        func=bollinger_bands,
        description="布林通道均值回歸",
        params=[
            StrategyParam("period", 10, 40, 5, 20),
            StrategyParam("num_std", 1.0, 3.0, 0.25, 2.0),
        ],
    ),
    "kdj": StrategyDef(
        name="kdj",
        func=kdj_strategy,
        description="KDJ 隨機指標",
        params=[
            StrategyParam("k_period", 5, 21, 2, 9),
            StrategyParam("d_period", 2, 7, 1, 3),
            StrategyParam("oversold", 10, 30, 5, 15),
            StrategyParam("overbought", 70, 90, 5, 90),
        ],
    ),
    "volume_price": StrategyDef(
        name="volume_price",
        func=volume_price_strategy,
        description="量價突破",
        params=[
            StrategyParam("vol_ma_period", 10, 40, 5, 20),
            StrategyParam("vol_threshold", 1.0, 3.0, 0.25, 2.5),
            StrategyParam("price_ma_period", 5, 30, 5, 10),
        ],
    ),
    "dual_ma_rsi": StrategyDef(
        name="dual_ma_rsi",
        func=dual_ma_rsi_filter,
        description="雙均線交叉 + RSI 過濾",
        params=[
            StrategyParam("fast_period", 3, 20, 1, 5),
            StrategyParam("slow_period", 10, 60, 5, 20),
            StrategyParam("rsi_period", 5, 25, 1, 14),
            StrategyParam("rsi_threshold", 30, 70, 5, 70),
        ],
    ),
    # --- Gemini APP2 Stock 策略 (12種) ---
    "multi_factor": StrategyDef(
        name="multi_factor",
        func=multi_factor_strategy,
        description="多因子動量 + RSI 排名選股",
        params=[
            StrategyParam("momentum_days", 20, 120, 10, 60),
            StrategyParam("rsi_period", 5, 30, 1, 14),
            StrategyParam("rsi_buy", 20, 50, 5, 40),
            StrategyParam("rsi_sell", 60, 85, 5, 70),
        ],
    ),
    "kd_macd_signal": StrategyDef(
        name="kd_macd_signal",
        func=kd_macd_signal_strategy,
        description="KD 黃金交叉 + MACD 柱狀圖",
        params=[
            StrategyParam("kd_oversold", 10, 30, 5, 20),
            StrategyParam("kd_overbought", 70, 90, 5, 80),
            StrategyParam("macd_fast", 6, 20, 2, 12),
            StrategyParam("macd_slow", 18, 40, 2, 26),
            StrategyParam("macd_signal", 5, 15, 1, 9),
        ],
    ),
    "ma_breakout": StrategyDef(
        name="ma_breakout",
        func=ma_breakout_strategy,
        description="均線突破趨勢跟蹤",
        params=[
            StrategyParam("ma_short", 3, 10, 1, 5),
            StrategyParam("ma_long", 20, 60, 5, 20),
            StrategyParam("ma_trend", 40, 120, 10, 60),
        ],
    ),
    "foreign_follow": StrategyDef(
        name="foreign_follow",
        func=foreign_follow_strategy,
        description="外資跟單策略 (量能放大)",
        params=[
            StrategyParam("vol_surge", 1.2, 3.0, 0.1, 2.0),
            StrategyParam("rsi_min", 30, 50, 5, 40),
            StrategyParam("rsi_sell", 65, 85, 5, 75),
        ],
    ),
    "etf_momentum": StrategyDef(
        name="etf_momentum",
        func=etf_momentum_strategy,
        description="ETF 動量定期再平衡",
        params=[
            StrategyParam("momentum_days", 20, 120, 10, 60),
            StrategyParam("rebal_period", 5, 60, 5, 20),
        ],
    ),
    "vol_price_sync": StrategyDef(
        name="vol_price_sync",
        func=vol_price_sync_strategy,
        description="量價齊揚確認多頭",
        params=[
            StrategyParam("vol_surge_multiplier", 1.0, 3.0, 0.1, 1.5),
            StrategyParam("ma_period", 10, 40, 5, 20),
        ],
    ),
    "vol_breakout_vcp": StrategyDef(
        name="vol_breakout_vcp",
        func=vol_breakout_vcp_strategy,
        description="VCP 量縮突破",
        params=[
            StrategyParam("vol_surge_multiplier", 1.5, 3.0, 0.1, 2.0),
            StrategyParam("range_contract", 0.5, 0.9, 0.1, 0.7),
            StrategyParam("lookback", 10, 40, 5, 20),
        ],
    ),
    "vol_divergence": StrategyDef(
        name="vol_divergence",
        func=vol_divergence_strategy,
        description="量價背離反轉 (OBV)",
        params=[
            StrategyParam("lookback_days", 10, 30, 5, 20),
        ],
    ),
    "chip_sedimentation": StrategyDef(
        name="chip_sedimentation",
        func=chip_sedimentation_strategy,
        description="籌碼沉澱動能突破",
        params=[
            StrategyParam("vol_lookback", 10, 40, 5, 20),
            StrategyParam("vol_shrink_threshold", 0.4, 0.9, 0.1, 0.8),
            StrategyParam("vol_shrink_min_days", 1, 5, 1, 2),
            StrategyParam("price_range_threshold", 5, 25, 1, 15),
            StrategyParam("sedimentation_window", 3, 15, 1, 10),
            StrategyParam("vol_surge_multiplier", 1.0, 2.5, 0.1, 1.2),
        ],
    ),
    "chip_sedimentation_inst": StrategyDef(
        name="chip_sedimentation_inst",
        func=chip_sedimentation_inst_strategy,
        description="籌碼沉澱 + 法人共振",
        params=[
            StrategyParam("vol_lookback", 10, 40, 5, 20),
            StrategyParam("vol_shrink_threshold", 0.4, 0.9, 0.1, 0.8),
            StrategyParam("price_range_threshold", 5, 25, 1, 15),
            StrategyParam("sedimentation_window", 3, 15, 1, 10),
            StrategyParam("vol_surge_multiplier", 1.5, 3.0, 0.1, 2.0),
        ],
    ),
    "chip_sedimentation_rising": StrategyDef(
        name="chip_sedimentation_rising",
        func=chip_sedimentation_rising_strategy,
        description="籌碼沉澱 + 起漲點 (RSI+MACD)",
        params=[
            StrategyParam("vol_lookback", 10, 40, 5, 20),
            StrategyParam("vol_shrink_threshold", 0.4, 0.9, 0.1, 0.8),
            StrategyParam("price_range_threshold", 5, 25, 1, 15),
            StrategyParam("sedimentation_window", 3, 15, 1, 10),
            StrategyParam("vol_surge_multiplier", 1.0, 2.5, 0.1, 1.2),
            StrategyParam("rsi_min", 30, 60, 5, 40),
        ],
    ),
    "vol_dry_bottom": StrategyDef(
        name="vol_dry_bottom",
        func=vol_dry_bottom_strategy,
        description="量縮 KD 低檔反彈",
        params=[
            StrategyParam("vol_dry_ratio", 0.3, 0.8, 0.1, 0.7),
            StrategyParam("vol_dry_days", 2, 5, 1, 2),
            StrategyParam("kd_oversold", 20, 45, 5, 30),
            StrategyParam("rsi_max", 40, 60, 5, 55),
        ],
    ),
}


def get_strategy(name: str) -> StrategyDef:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> List[str]:
    return list(STRATEGY_REGISTRY.keys())
