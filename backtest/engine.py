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
    trades: List = None      # detailed trade records
    signal_df: Any = None    # DataFrame with signals for charting

    def summary(self) -> str:
        return (
            f"[{self.strategy_name}] {self.symbol} | "
            f"勝率={self.win_rate:.2%} 交易數={self.num_trades} "
            f"報酬={self.total_return:.2%} 最大回撤={self.max_drawdown:.2%} "
            f"夏普={self.sharpe_ratio:.3f} 獲利因子={self.profit_factor:.2f}"
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
    stop_loss: float = 0.0,            # 停損比例 (0.10 = 10%), 0=不啟用
    take_profit: float = 0.0,          # 停利比例 (0.20 = 20%), 0=不啟用
    max_hold_days: int = 0,            # 最大持有天數, 0=不限制
    trailing_stop: float = 0.0,        # 移動停利回撤% (0.08 = 從高點回落8%), 0=不啟用
    partial_exit: bool = False,        # 分批出場 (漲10%出1/3, 漲20%出1/3, 剩餘移動停利)
    partial_exit_last_mode: str = "trailing",  # 最後1/3: "trailing"=移動停利, "ma5"=破5日均線
) -> BacktestResult:
    """
    Run backtest for a strategy with given parameters on OHLCV data.

    買賣分離架構:
        - 買入: 由 strategy_func 產生的 signal=1 觸發
        - 賣出: 由多層賣出策略決定 (優先順序):
          1. 停損 (stop_loss)
          2. 移動停利 (trailing_stop) — 從最高點回落 N% 賣出
          3. 固定停利 (take_profit)
          4. 最大持有天數 (max_hold_days)
          5. 策略訊號 (signal=-1) — 原始策略的賣出訊號
        - 分批出場 (partial_exit):
            漲10%出1/3 → 漲20%出1/3 → 剩餘1/3依 partial_exit_last_mode:
              "trailing" = 移動停利
              "ma5"      = 收盤跌破5日均線
    """
    # Run strategy to get signals
    sig_df = strategy_func(df, **params)

    if 'signal' not in sig_df.columns:
        raise ValueError(f"Strategy {strategy_name} did not produce 'signal' column")

    # Extract trades from signals (含風險管理 + 賣出策略)
    trades = _extract_trades(
        sig_df, commission_rate, tax_rate, slippage,
        stop_loss=stop_loss, take_profit=take_profit, max_hold_days=max_hold_days,
        trailing_stop=trailing_stop, partial_exit=partial_exit,
        partial_exit_last_mode=partial_exit_last_mode,
    )

    if len(trades) == 0:
        return BacktestResult(
            strategy_name=strategy_name, params=params, symbol=symbol,
            win_rate=0.0, total_return=0.0, num_trades=0,
            num_wins=0, num_losses=0, avg_win=0.0, avg_loss=0.0,
            max_drawdown=0.0, sharpe_ratio=0.0, profit_factor=0.0,
            expectancy=0.0, trades=[], signal_df=sig_df,
        )

    # ====================================================================
    # 分批出場合併: 同一次買入的多筆賣出合併為一筆交易計算 metrics
    # trades 保留原始明細 (給 UI 顯示)，merged_trades 用於計算勝率等指標
    # ====================================================================
    def _merge_partial_trades(raw_trades):
        """將同一個 buy_date + buy_price 的分批出場交易合併為一筆"""
        merged = []
        group = []
        for t in raw_trades:
            ratio = t.get('sell_ratio', 1.0)
            if ratio < 1.0 - 1e-6:
                # 分批出場的一部分
                group.append(t)
            else:
                # 完整出場 (ratio=1 or 剩餘比例)
                if group and group[0]['buy_date'] == t['buy_date']:
                    # 最後一批，和之前的合併
                    group.append(t)
                    # 合併: 加權平均報酬率
                    total_ratio = sum(g.get('sell_ratio', 1.0) for g in group)
                    weighted_return = sum(
                        g['return_pct'] * g.get('sell_ratio', 1.0)
                        for g in group
                    ) / total_ratio if total_ratio > 0 else 0
                    total_profit = sum(g.get('profit', 0) for g in group)
                    merged.append({
                        'return_pct': weighted_return,
                        'profit': total_profit,
                        'buy_date': group[0]['buy_date'],
                        'sell_date': t['sell_date'],
                        'holding_days': t.get('holding_days', 0),
                        '_is_merged': True,
                        '_sub_trades': len(group),
                    })
                    group = []
                else:
                    # 之前有未完成的 group (不應發生，但保險起見)
                    if group:
                        total_ratio = sum(g.get('sell_ratio', 1.0) for g in group)
                        weighted_return = sum(
                            g['return_pct'] * g.get('sell_ratio', 1.0)
                            for g in group
                        ) / total_ratio if total_ratio > 0 else 0
                        total_profit = sum(g.get('profit', 0) for g in group)
                        merged.append({
                            'return_pct': weighted_return,
                            'profit': total_profit,
                            'buy_date': group[0]['buy_date'],
                            'sell_date': group[-1]['sell_date'],
                            'holding_days': group[-1].get('holding_days', 0),
                            '_is_merged': True,
                            '_sub_trades': len(group),
                        })
                        group = []
                    # 這筆是獨立的完整交易
                    merged.append(t)

        # 處理尚未結束的 group (只出了 1/3 或 2/3，還沒完全出場)
        if group:
            total_ratio = sum(g.get('sell_ratio', 1.0) for g in group)
            weighted_return = sum(
                g['return_pct'] * g.get('sell_ratio', 1.0)
                for g in group
            ) / total_ratio if total_ratio > 0 else 0
            total_profit = sum(g.get('profit', 0) for g in group)
            merged.append({
                'return_pct': weighted_return,
                'profit': total_profit,
                'buy_date': group[0]['buy_date'],
                'sell_date': group[-1]['sell_date'],
                'holding_days': group[-1].get('holding_days', 0),
                '_is_merged': True,
                '_sub_trades': len(group),
            })

        return merged

    merged_trades = _merge_partial_trades(trades)

    # Calculate metrics (用合併後的交易計算)
    returns = [t['return_pct'] for t in merged_trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    num_trades = len(merged_trades)
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = num_wins / num_trades if num_trades > 0 else 0.0

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    # Total return (simple sum of each trade's return %)
    total_return = sum(returns)

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
        trades=trades,
        signal_df=sig_df,
    )


def _extract_trades(
    df: pd.DataFrame,
    commission_rate: float,
    tax_rate: float,
    slippage: float,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    max_hold_days: int = 0,
    exit_mode: str = "signal",
    trailing_stop: float = 0.0,
    partial_exit: bool = False,
    partial_exit_last_mode: str = "trailing",  # "trailing"=移動停利, "ma5"=破5日均線
) -> List[Dict[str, Any]]:
    """
    Extract completed trades from signal DataFrame.

    Risk management:
        stop_loss:     停損比例 (如 0.10 = 跌 10% 停損), 0=不啟用
        take_profit:   停利比例 (如 0.20 = 漲 20% 停利), 0=不啟用
        max_hold_days: 最大持有天數, 0=不限制

    Exit strategies (賣出策略):
        exit_mode:     "signal"=策略訊號賣出, "trailing"=移動停利, "partial"=分批出場
        trailing_stop: 移動停利回撤比例 (如 0.08 = 從最高點回落 8% 賣出), 0=不啟用
        partial_exit:  True=啟用分批出場 (漲10%出1/3, 漲20%出1/3, 剩餘移動停利)
        partial_exit_last_mode: 最後 1/3 出場方式
            "trailing" = 移動停利 (預設)
            "ma5"      = 收盤跌破 5 日均線賣出
    """
    trades = []
    position = None  # None = no position, 'long' = holding

    buy_price = 0.0
    buy_raw_price = 0.0
    buy_date = None
    buy_idx = 0
    cumulative_profit = 0.0
    max_price_since_buy = 0.0  # 持倉期間最高價 (移動停利用)

    # 預計算 5 日均線 (分批出場最後 1/3 破均線用)
    if 'close' in df.columns:
        ma5_series = df['close'].rolling(window=5).mean()
    else:
        ma5_series = pd.Series([np.nan] * len(df))

    # 分批出場狀態
    partial_sold_1 = False  # 已出 1/3
    partial_sold_2 = False  # 已出 2/3
    remaining_ratio = 1.0   # 剩餘持倉比例

    buy_reason = ""  # 初始化，在買入時更新

    def _make_trade(sell_date, sell_raw_price, sell_reason_text, ratio=1.0):
        """建立交易記錄的輔助函式"""
        nonlocal cumulative_profit
        sell_price_adj = sell_raw_price * (1 - slippage)
        buy_cost = buy_price * commission_rate * ratio
        sell_cost = sell_price_adj * (commission_rate + tax_rate) * ratio
        total_cost = buy_cost + sell_cost
        raw_return = (sell_price_adj - buy_price) / buy_price
        net_return = raw_return - (total_cost / (buy_price * ratio))
        profit = (sell_price_adj - buy_price) * ratio - total_cost

        try:
            hold_days = (pd.Timestamp(sell_date) - pd.Timestamp(buy_date)).days
        except Exception:
            hold_days = 0

        cumulative_profit += profit

        return {
            'buy_date': buy_date,
            'sell_date': sell_date,
            'buy_price': round(float(buy_raw_price), 2),
            'sell_price': round(float(sell_raw_price), 2),
            'buy_cost': round(float(buy_cost), 2),
            'sell_cost': round(float(sell_cost), 2),
            'return_pct': net_return,
            'profit': round(float(profit), 2),
            'cumulative_profit': round(float(cumulative_profit), 2),
            'holding_days': hold_days,
            'buy_reason': buy_reason,
            'sell_reason': sell_reason_text,
            'result': 'WIN' if net_return > 0 else 'LOSS',
            'sell_ratio': ratio,
        }

    for i in range(len(df)):
        row = df.iloc[i]
        signal = row['signal']
        date = row.get('date', row.name)
        price = row['close']

        if signal == 1 and position is None:
            # Buy signal
            buy_raw_price = price
            buy_price = price * (1 + slippage)
            buy_date = date
            buy_idx = i
            position = 'long'
            max_price_since_buy = price
            partial_sold_1 = False
            partial_sold_2 = False
            remaining_ratio = 1.0
            buy_reason = _detect_reason(row, signal=1)  # noqa: F841 used in _make_trade

        elif position == 'long':
            # 更新持倉最高價
            if price > max_price_since_buy:
                max_price_since_buy = price

            unrealized_return = (price - buy_raw_price) / buy_raw_price if buy_raw_price > 0 else 0
            try:
                hold_days = (pd.Timestamp(date) - pd.Timestamp(buy_date)).days
            except Exception:
                hold_days = i - buy_idx

            # ============================================================
            # 分批出場模式
            # ============================================================
            if partial_exit and remaining_ratio > 0:
                # 第一批: 漲 10% 出 1/3
                if not partial_sold_1 and unrealized_return >= 0.10:
                    trades.append(_make_trade(
                        date, price,
                        f"📊 分批出場 1/3: 漲幅{unrealized_return:.1%} 達 10%",
                        ratio=1/3,
                    ))
                    partial_sold_1 = True
                    remaining_ratio -= 1/3

                # 第二批: 漲 20% 出 1/3
                elif partial_sold_1 and not partial_sold_2 and unrealized_return >= 0.20:
                    trades.append(_make_trade(
                        date, price,
                        f"📊 分批出場 2/3: 漲幅{unrealized_return:.1%} 達 20%",
                        ratio=1/3,
                    ))
                    partial_sold_2 = True
                    remaining_ratio -= 1/3

            # ============================================================
            # 分批出場最後 1/3: 破 5 日均線賣出
            # ============================================================
            if (partial_exit and partial_sold_2 and remaining_ratio > 0
                    and partial_exit_last_mode == "ma5"):
                ma5_val = ma5_series.iloc[i] if i < len(ma5_series) else np.nan
                if not pd.isna(ma5_val) and price < ma5_val:
                    trades.append(_make_trade(
                        date, price,
                        f"📊 分批出場 3/3: 收盤{price:.1f} 跌破 5日均線{ma5_val:.1f}",
                        ratio=remaining_ratio,
                    ))
                    remaining_ratio = 0
                    position = None
                    continue

            # ============================================================
            # 停損 (最高優先)
            # ============================================================
            force_sell = False
            sell_reason_override = None

            if stop_loss > 0 and unrealized_return <= -stop_loss:
                force_sell = True
                sell_reason_override = f"⛔ 停損觸發: 跌幅{unrealized_return:.1%} 超過 -{stop_loss:.0%}"

            # ============================================================
            # 移動停利 (trailing stop)
            # ============================================================
            elif trailing_stop > 0 and max_price_since_buy > buy_raw_price:
                drawdown_from_peak = (max_price_since_buy - price) / max_price_since_buy
                if drawdown_from_peak >= trailing_stop:
                    force_sell = True
                    peak_gain = (max_price_since_buy - buy_raw_price) / buy_raw_price
                    sell_reason_override = (
                        f"📉 移動停利: 最高{max_price_since_buy:.1f}(+{peak_gain:.1%})"
                        f" 回落{drawdown_from_peak:.1%} 達 {trailing_stop:.0%}"
                    )

            # ============================================================
            # 固定停利
            # ============================================================
            elif take_profit > 0 and unrealized_return >= take_profit:
                force_sell = True
                sell_reason_override = f"🎯 停利觸發: 漲幅{unrealized_return:.1%} 達 +{take_profit:.0%}"

            # ============================================================
            # 最大持有天數
            # ============================================================
            elif max_hold_days > 0 and hold_days >= max_hold_days:
                force_sell = True
                sell_reason_override = f"⏰ 持有{hold_days}天達上限{max_hold_days}天，強制出場"

            # ============================================================
            # 策略訊號賣出 或 風控強制賣出
            # ============================================================
            if signal == -1 or force_sell:
                sell_price = price * (1 - slippage)
                buy_cost = buy_price * commission_rate * remaining_ratio
                sell_cost = sell_price * (commission_rate + tax_rate) * remaining_ratio
                total_cost = buy_cost + sell_cost
                raw_return = (sell_price - buy_price) / buy_price
                net_return = raw_return - (total_cost / (buy_price * remaining_ratio)) if remaining_ratio > 0 else raw_return
                profit = (sell_price - buy_price) * remaining_ratio - total_cost

                cumulative_profit += profit

                if sell_reason_override:
                    sell_reason = sell_reason_override
                else:
                    sell_reason = _detect_reason(row, signal=-1)

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': date,
                    'buy_price': round(float(buy_raw_price), 2),
                    'sell_price': round(float(price), 2),
                    'buy_cost': round(float(buy_cost), 2),
                    'sell_cost': round(float(sell_cost), 2),
                    'return_pct': net_return,
                    'profit': round(float(profit), 2),
                    'cumulative_profit': round(float(cumulative_profit), 2),
                    'holding_days': hold_days,
                    'buy_reason': buy_reason,
                    'sell_reason': sell_reason,
                    'result': 'WIN' if net_return > 0 else 'LOSS',
                    'sell_ratio': remaining_ratio,
                })
                position = None

    return trades


def _detect_reason(row, signal: int) -> str:
    """偵測買賣理由 (含實際數值)。"""
    reasons = []
    price = row.get('close', 0)
    if signal == 1:  # 買入
        if 'sma_fast' in row.index and 'sma_slow' in row.index:
            reasons.append(f"短均線({row['sma_fast']:.1f}) 上穿 長均線({row['sma_slow']:.1f})")
        if 'ema_fast' in row.index and 'ema_slow' in row.index:
            reasons.append(f"EMA短({row['ema_fast']:.1f}) 上穿 EMA長({row['ema_slow']:.1f})")
        if 'rsi' in row.index and not pd.isna(row['rsi']):
            reasons.append(f"RSI={row['rsi']:.1f} 進入超賣區")
        if 'macd_hist' in row.index and not pd.isna(row['macd_hist']):
            reasons.append(f"MACD柱={row['macd_hist']:.3f} 翻正/上穿")
        if 'bb_lower' in row.index and not pd.isna(row['bb_lower']):
            reasons.append(f"收盤{price:.1f} < 布林下軌{row['bb_lower']:.1f}")
        if 'k' in row.index and 'd' in row.index:
            reasons.append(f"KD K={row['k']:.1f} 黃金交叉 D={row['d']:.1f}")
        if 'vol_ma' in row.index and not pd.isna(row['vol_ma']):
            reasons.append("量能放大突破 + 股價站上均線")
    else:  # 賣出
        if 'sma_fast' in row.index and 'sma_slow' in row.index:
            reasons.append(f"短均線({row['sma_fast']:.1f}) 下穿 長均線({row['sma_slow']:.1f})")
        if 'ema_fast' in row.index and 'ema_slow' in row.index:
            reasons.append(f"EMA短({row['ema_fast']:.1f}) 下穿 EMA長({row['ema_slow']:.1f})")
        if 'rsi' in row.index and not pd.isna(row['rsi']):
            reasons.append(f"RSI={row['rsi']:.1f} 進入超買區")
        if 'macd_hist' in row.index and not pd.isna(row['macd_hist']):
            reasons.append(f"MACD柱={row['macd_hist']:.3f} 翻負/下穿")
        if 'bb_upper' in row.index and not pd.isna(row['bb_upper']):
            reasons.append(f"收盤{price:.1f} > 布林上軌{row['bb_upper']:.1f}")
        if 'k' in row.index and 'd' in row.index:
            reasons.append(f"KD K={row['k']:.1f} 死亡交叉 D={row['d']:.1f}")
        if 'vol_ma' in row.index and not pd.isna(row['vol_ma']):
            reasons.append("量能萎縮 + 股價跌破均線")

    return reasons[0] if reasons else "策略訊號觸發"


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
