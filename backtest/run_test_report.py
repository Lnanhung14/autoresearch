#!/usr/bin/env python3
"""
Run all 20 strategies on 2454, 1326, 6176 and generate a markdown report.
"""
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_loader import download_yahoo
from backtest.strategies import STRATEGY_REGISTRY, list_strategies
from backtest.engine import run_backtest


def main():
    # Download data
    symbols = ["2454.TW", "1326.TW", "6176.TW"]
    print(f"Downloading data for {symbols}...")
    stock_data = download_yahoo(symbols, start="2020-01-01")
    print(f"Downloaded {len(stock_data)} stocks")

    if not stock_data:
        print("ERROR: No data downloaded")
        sys.exit(1)

    strategies = list_strategies()
    print(f"\nTesting {len(strategies)} strategies on {len(stock_data)} stocks...\n")

    # Run all strategies on all stocks
    all_results = []  # list of dicts
    for strat_name in strategies:
        sdef = STRATEGY_REGISTRY[strat_name]
        params = sdef.default_params()

        for symbol, df in stock_data.items():
            try:
                result = run_backtest(
                    df=df,
                    strategy_func=sdef.func,
                    strategy_name=strat_name,
                    params=params,
                    symbol=symbol,
                )
                all_results.append({
                    "strategy": strat_name,
                    "description": sdef.description,
                    "symbol": symbol,
                    "win_rate": result.win_rate,
                    "total_return": result.total_return,
                    "num_trades": result.num_trades,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "profit_factor": result.profit_factor,
                    "avg_win": result.avg_win,
                    "avg_loss": result.avg_loss,
                })
                status = "OK" if result.num_trades > 0 else "NO TRADES"
                print(f"  {strat_name:30s} | {symbol:10s} | trades={result.num_trades:3d} | "
                      f"WR={result.win_rate:.1%} | ret={result.total_return:.2%} | "
                      f"sharpe={result.sharpe_ratio:.2f} | {status}")
            except Exception as e:
                print(f"  {strat_name:30s} | {symbol:10s} | ERROR: {e}")
                all_results.append({
                    "strategy": strat_name,
                    "description": sdef.description,
                    "symbol": symbol,
                    "win_rate": 0, "total_return": 0, "num_trades": 0,
                    "sharpe_ratio": 0, "max_drawdown": 0, "profit_factor": 0,
                    "avg_win": 0, "avg_loss": 0, "error": str(e),
                })

    # Generate report
    generate_report(all_results, stock_data)
    print(f"\nReport saved to: backtest_test_report.md")


def generate_report(results, stock_data):
    symbol_names = {
        "2454.TW": "2454 聯發科",
        "1326.TW": "1326 台化",
        "6176.TW": "6176 瑞儀",
    }

    lines = []
    lines.append("# 回測策略測試報告")
    lines.append("")
    lines.append(f"**產生時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**資料區間**: 2020-01-01 ~ 至今")
    lines.append(f"**測試標的**: 2454 聯發科、1326 台化、6176 瑞儀")
    lines.append(f"**策略總數**: {len(set(r['strategy'] for r in results))} 種")
    lines.append(f"**交易成本**: 手續費 0.1425% + 證交稅 0.3% + 滑價 0.1%")
    lines.append("")

    # Summary table
    lines.append("## 全部回測結果")
    lines.append("")
    lines.append("| # | 策略代碼 | 策略名稱 | 股票 | 交易次數 | 勝率 | 總報酬 | 夏普比率 | 最大回撤 | 獲利因子 |")
    lines.append("|---|----------|----------|------|----------|------|--------|----------|----------|----------|")

    for i, r in enumerate(sorted(results, key=lambda x: (x['strategy'], x['symbol']))):
        sym_display = symbol_names.get(r['symbol'], r['symbol'])
        error = r.get('error', '')
        if error:
            lines.append(f"| {i+1} | {r['strategy']} | {r['description']} | {sym_display} | 錯誤 | - | - | - | - | - |")
        else:
            lines.append(
                f"| {i+1} | {r['strategy']} | {r['description']} | {sym_display} | "
                f"{r['num_trades']} | {r['win_rate']:.1%} | {r['total_return']:.2%} | "
                f"{r['sharpe_ratio']:.2f} | {r['max_drawdown']:.2%} | {r['profit_factor']:.2f} |"
            )

    # Best strategy per stock
    lines.append("")
    lines.append("## 各股票最佳策略 (依夏普比率)")
    lines.append("")
    symbols_in_results = sorted(set(r['symbol'] for r in results))
    for symbol in symbols_in_results:
        sym_display = symbol_names.get(symbol, symbol)
        stock_results = [r for r in results if r['symbol'] == symbol and r['num_trades'] >= 3]
        if not stock_results:
            lines.append(f"### {sym_display}")
            lines.append("無有效結果 (所有策略交易次數 < 3)")
            lines.append("")
            continue

        best = max(stock_results, key=lambda x: x['sharpe_ratio'])
        lines.append(f"### {sym_display}")
        lines.append(f"- **最佳策略**: `{best['strategy']}` ({best['description']})")
        lines.append(f"- **夏普比率**: {best['sharpe_ratio']:.3f}")
        lines.append(f"- **勝率**: {best['win_rate']:.1%}")
        lines.append(f"- **總報酬**: {best['total_return']:.2%}")
        lines.append(f"- **交易次數**: {best['num_trades']}")
        lines.append(f"- **最大回撤**: {best['max_drawdown']:.2%}")
        lines.append(f"- **獲利因子**: {best['profit_factor']:.2f}")
        lines.append("")

    # Best overall strategy
    lines.append("## 綜合最佳策略 (全部股票平均夏普比率)")
    lines.append("")
    strat_sharpes = {}
    for r in results:
        if r['num_trades'] >= 3:
            strat_sharpes.setdefault(r['strategy'], []).append(r['sharpe_ratio'])

    if strat_sharpes:
        avg_sharpes = {k: sum(v) / len(v) for k, v in strat_sharpes.items()}
        sorted_strats = sorted(avg_sharpes.items(), key=lambda x: x[1], reverse=True)

        lines.append("| 排名 | 策略代碼 | 策略名稱 | 平均夏普 | 測試股數 |")
        lines.append("|------|----------|----------|----------|----------|")
        for rank, (strat, avg_s) in enumerate(sorted_strats[:10], 1):
            desc = STRATEGY_REGISTRY[strat].description
            n = len(strat_sharpes[strat])
            lines.append(f"| {rank} | {strat} | {desc} | {avg_s:.3f} | {n} |")
    else:
        lines.append("無策略達到 >= 3 次交易門檻。")

    lines.append("")

    # Strategy categories
    lines.append("## 策略分類")
    lines.append("")
    lines.append("### 原有策略 (1-8)")
    lines.append("| # | 策略代碼 | 策略名稱 |")
    lines.append("|---|----------|----------|")
    original = ["sma_crossover", "ema_crossover", "rsi", "macd", "bollinger_bands", "kdj", "volume_price", "dual_ma_rsi"]
    for i, s in enumerate(original, 1):
        lines.append(f"| {i} | {s} | {STRATEGY_REGISTRY[s].description} |")

    lines.append("")
    lines.append("### Gemini APP2 Stock 策略 (9-20)")
    lines.append("| # | 策略代碼 | 策略名稱 |")
    lines.append("|---|----------|----------|")
    gemini = [s for s in list_strategies() if s not in original]
    for i, s in enumerate(gemini, 9):
        lines.append(f"| {i} | {s} | {STRATEGY_REGISTRY[s].description} |")

    lines.append("")

    # Write report
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "backtest", "backtest_test_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    main()
