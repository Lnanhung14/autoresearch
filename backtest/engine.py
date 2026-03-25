"""
Backtesting engine - evaluates a strategy with given parameters on stock data.
Returns performance metrics including win rate, total return, Sharpe ratio, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    strategy_name: str
    params: Dict[str, float]
    symbol: str
    win_rate: float          # primary metric (higher is better)
    total_return: float      # cumulative return %
    num_trades: int
    num_wins: int
    num_losses: int
    avg_win: float           # average winning trade return %
    avg_loss: float          # average losing trade return %
    max_drawdown: float      # maximum drawdown %
    sharpe_ratio: float
    profit_factor: float     # gross profit / gross loss
    expectancy: float        # expected return per trade

    def summary(self) -> str:
        return (
            f"[{self.strategy_name}] {self.symbol} | "
            f"WinRate={self.win_rate:.2%} Trades={self.num_trades} "
            f"Return={self.total_return:.2%} MaxDD={self.max_drawdown:.2%} "
            f"Sharpe={self.sharpe_ratio:.3f} PF={self.profit_factor:.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'strategy': self.strategy_name,
            'symbol': self.symbol,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'num_trades': self.num_trades,
            'num_wins': self.num_wins,
            'num_losses': self.num_losses,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
        }
        for k, v in self.params.items():
            d[f'param_{k}'] = v
        return d


def run_backtest(
    df: pd.DataFrame,
    strategy_func,
    strategy_name: str,
    params: Dict[str, float],
    symbol: str = "UNKNOWN",
    initial_capital: float = 1_000_000,
    commission_rate: float = 0.001425,  # Taiwan stock commission rate
    tax_rate: float = 0.003,            # Taiwan stock transaction tax
    slippage: float = 0.001,            # 0.1% slippage
) -> BacktestResult:
    """
    Run backtest for a strategy with given parameters on OHLCV data.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume
        strategy_func: Strategy function that adds 'signal' column
        strategy_name: Name of the strategy
        params: Strategy parameters
        symbol: Stock symbol
        initial_capital: Starting capital
        commission_rate: Commission rate per trade (buy+sell)
        tax_rate: Transaction tax rate (sell only, Taiwan-specific)
        slippage: Slippage per trade as fraction
    """
    # Run strategy to get signals
    sig_df = strategy_func(df, **params)

    if 'signal' not in sig_df.columns:
        raise ValueError(f"Strategy {strategy_name} did not produce 'signal' column")

    # Extract trades from signals
    trades = _extract_trades(sig_df, commission_rate, tax_rate, slippage)

    if len(trades) == 0:
        return BacktestResult(
            strategy_name=strategy_name, params=params, symbol=symbol,
            win_rate=0.0, total_return=0.0, num_trades=0,
            num_wins=0, num_losses=0, avg_win=0.0, avg_loss=0.0,
            max_drawdown=0.0, sharpe_ratio=0.0, profit_factor=0.0,
            expectancy=0.0,
        )

    # Calculate metrics
    returns = [t['return_pct'] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    num_trades = len(trades)
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = num_wins / num_trades if num_trades > 0 else 0.0

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    # Total return (compounded)
    total_return = 1.0
    for r in returns:
        total_return *= (1 + r)
    total_return -= 1.0

    # Max drawdown
    equity_curve = [initial_capital]
    for r in returns:
        equity_curve.append(equity_curve[-1] * (1 + r))
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Sharpe ratio (annualized, assuming ~252 trading days)
    if len(returns) > 1:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    # Profit factor
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    # Expectancy (average return per trade)
    expectancy = np.mean(returns) if returns else 0.0

    return BacktestResult(
        strategy_name=strategy_name,
        params=params,
        symbol=symbol,
        win_rate=win_rate,
        total_return=total_return,
        num_trades=num_trades,
        num_wins=num_wins,
        num_losses=num_losses,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        profit_factor=profit_factor,
        expectancy=expectancy,
    )


def _extract_trades(
    df: pd.DataFrame,
    commission_rate: float,
    tax_rate: float,
    slippage: float,
) -> List[Dict[str, Any]]:
    """Extract completed trades from signal DataFrame."""
    trades = []
    position = None  # None = no position, 'long' = holding

    buy_price = 0.0
    buy_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        signal = row['signal']
        date = row.get('date', row.name)
        price = row['close']

        if signal == 1 and position is None:
            # Buy signal
            buy_price = price * (1 + slippage)  # slippage on buy
            buy_date = date
            position = 'long'

        elif signal == -1 and position == 'long':
            # Sell signal
            sell_price = price * (1 - slippage)  # slippage on sell
            # Calculate return including costs
            cost = buy_price * commission_rate + sell_price * (commission_rate + tax_rate)
            raw_return = (sell_price - buy_price) / buy_price
            net_return = raw_return - (cost / buy_price)
            trades.append({
                'buy_date': buy_date,
                'sell_date': date,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'return_pct': net_return,
                'holding_days': None,  # can be calculated if dates are datetime
            })
            position = None

    return trades


def run_backtest_multi_stock(
    stock_data: Dict[str, pd.DataFrame],
    strategy_func,
    strategy_name: str,
    params: Dict[str, float],
    **kwargs,
) -> Tuple[float, List[BacktestResult]]:
    """
    Run backtest across multiple stocks. Returns (average_win_rate, results_list).
    """
    results = []
    for symbol, df in stock_data.items():
        try:
            result = run_backtest(df, strategy_func, strategy_name, params, symbol=symbol, **kwargs)
            if result.num_trades > 0:
                results.append(result)
        except Exception as e:
            print(f"  [WARN] {symbol} failed: {e}")

    if not results:
        return 0.0, []

    avg_win_rate = np.mean([r.win_rate for r in results])
    return avg_win_rate, results
