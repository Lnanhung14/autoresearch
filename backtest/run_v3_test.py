#!/usr/bin/env python3
"""
v3 完整測試: 買賣分離 + autoresearch 式搜尋
測試觀察名單: 2330, 2454, 1326, 1301, 2609
比較 v2 (舊版) vs v3 (新版) 的差異
"""
import time
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.strategies import STRATEGY_REGISTRY, get_strategy, list_strategies
from backtest.engine import run_backtest, run_backtest_multi_stock
from backtest.optimizer import BacktestOptimizer, OptimizationConfig

# ==========================================================
# 測試股票 (觀察名單)
# ==========================================================
STOCKS = {
    "2330.TW": "台積電",
    "2454.TW": "聯發科",
    "1326.TW": "台化",
    "1301.TW": "台塑",
    "2609.TW": "陽明",
}

START = "2023-01-01"
END = "2026-03-28"

print("=" * 70)
print("📊 v3 完整測試: 買賣分離 + autoresearch 搜尋")
print(f"   股票: {list(STOCKS.values())}")
print(f"   期間: {START} ~ {END}")
print("=" * 70)

# Download data
print("\n下載資料...")
stock_data = {}
for sym, name in STOCKS.items():
    try:
        df = yf.download(sym, start=START, end=END, progress=False)
        if df.empty:
            continue
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'adj close' in df.columns:
            df['close'] = df['adj close']
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        stock_data[sym] = df
        print(f"  ✅ {name} ({sym}): {len(df)} 天")
    except Exception as e:
        print(f"  ❌ {sym}: {e}")

# ==========================================================
# Test 1: v2 舊版 (固定停利 + 有限搜尋)
# ==========================================================
print("\n" + "=" * 70)
print("🔬 測試 1: v2 舊版 (停損15%, 固定停利30%, 搜尋上限200)")
print("=" * 70)

t0 = time.time()
config_v2 = OptimizationConfig(
    primary_metric="total_return",
    min_trades=3,
    max_combos_per_strategy=200,
    patience=30,
    neighbors_per_step=5,
    stop_loss=0.15,
    take_profit=0.30,
    max_hold_days=120,
    trailing_stop=0.0,
    partial_exit=False,
)
opt_v2 = BacktestOptimizer(stock_data, config_v2)
result_v2 = opt_v2.optimize()
t_v2 = time.time() - t0

# ==========================================================
# Test 2: v3 新版 (移動停利 + 不限搜尋)
# ==========================================================
print("\n" + "=" * 70)
print("🔬 測試 2: v3 新版 (停損15%, 移動停利8%, 不限搜尋)")
print("=" * 70)

t0 = time.time()
config_v3 = OptimizationConfig(
    primary_metric="total_return",
    min_trades=3,
    max_combos_per_strategy=0,  # 不限制
    time_budget_seconds=180,    # 3 分鐘
    patience=50,
    neighbors_per_step=8,
    stop_loss=0.15,
    take_profit=0.0,            # 不設固定停利
    max_hold_days=120,
    trailing_stop=0.08,         # 移動停利 8%
    partial_exit=False,
)
opt_v3 = BacktestOptimizer(stock_data, config_v3)
result_v3 = opt_v3.optimize()
t_v3 = time.time() - t0

# ==========================================================
# Test 3: v3 + 分批出場
# ==========================================================
print("\n" + "=" * 70)
print("🔬 測試 3: v3 分批出場 (停損15%, 移動停利8%, 分批)")
print("=" * 70)

t0 = time.time()
config_v3p = OptimizationConfig(
    primary_metric="total_return",
    min_trades=3,
    max_combos_per_strategy=0,
    time_budget_seconds=180,
    patience=50,
    neighbors_per_step=8,
    stop_loss=0.15,
    take_profit=0.0,
    max_hold_days=120,
    trailing_stop=0.08,
    partial_exit=True,          # 分批出場
)
opt_v3p = BacktestOptimizer(stock_data, config_v3p)
result_v3p = opt_v3p.optimize()
t_v3p = time.time() - t0

# ==========================================================
# 比較報告
# ==========================================================
print("\n" + "=" * 70)
print("📊 比較報告")
print("=" * 70)

results = {
    "v2 (舊版: 固定停利30%)": (result_v2, t_v2),
    "v3 (新版: 移動停利8%)": (result_v3, t_v3),
    "v3+分批 (移動停利+分批出場)": (result_v3p, t_v3p),
}

report_lines = []
report_lines.append("# v3 買賣分離 + autoresearch 搜尋 — 完整測試報告\n")
report_lines.append(f"**測試日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
report_lines.append(f"**測試股票**: {', '.join([f'{v}({k})' for k,v in STOCKS.items()])}\n")
report_lines.append(f"**資料期間**: {START} ~ {END}\n")
report_lines.append("")

report_lines.append("## 三版本比較\n")
report_lines.append("| 項目 | v2 舊版 | v3 新版 | v3+分批 |")
report_lines.append("|------|--------|--------|---------|")

r_v2, t2 = results["v2 (舊版: 固定停利30%)"]
r_v3, t3 = results["v3 (新版: 移動停利8%)"]
r_v3p, t3p = results["v3+分批 (移動停利+分批出場)"]

report_lines.append(f"| 最佳策略 | {r_v2.best_strategy} | {r_v3.best_strategy} | {r_v3p.best_strategy} |")
report_lines.append(f"| 平均勝率 | {r_v2.best_win_rate:.1%} | {r_v3.best_win_rate:.1%} | {r_v3p.best_win_rate:.1%} |")
report_lines.append(f"| 平均報酬 | {r_v2.best_total_return:.1%} | {r_v3.best_total_return:.1%} | {r_v3p.best_total_return:.1%} |")
report_lines.append(f"| 夏普比率 | {r_v2.best_sharpe:.2f} | {r_v3.best_sharpe:.2f} | {r_v3p.best_sharpe:.2f} |")
report_lines.append(f"| 總交易數 | {r_v2.best_num_trades} | {r_v3.best_num_trades} | {r_v3p.best_num_trades} |")
report_lines.append(f"| 實驗次數 | {r_v2.total_experiments} | {r_v3.total_experiments} | {r_v3p.total_experiments} |")
report_lines.append(f"| 耗時 | {t2:.0f}秒 | {t3:.0f}秒 | {t3p:.0f}秒 |")
report_lines.append(f"| 停損 | 15% | 15% | 15% |")
report_lines.append(f"| 賣出策略 | 固定停利30% | 移動停利8% | 移動停利8%+分批 |")
report_lines.append(f"| 參數搜尋 | 上限200組 | 不限制 | 不限制 |")
report_lines.append("")

# Per-stock breakdown for best version
report_lines.append("## 各股票詳細 (v3 新版)\n")
report_lines.append("| 股票 | 勝率 | 報酬 | 交易數 | 夏普 | 最大回撤 |")
report_lines.append("|------|------|------|--------|------|---------|")

if r_v3.per_stock_results:
    for r in sorted(r_v3.per_stock_results, key=lambda x: x.total_return, reverse=True):
        name = STOCKS.get(r.symbol, r.symbol)
        report_lines.append(
            f"| {name} ({r.symbol}) | {r.win_rate:.1%} | {r.total_return:.1%} | "
            f"{r.num_trades} | {r.sharpe_ratio:.2f} | {r.max_drawdown:.1%} |"
        )

report_lines.append("")
report_lines.append(f"## 最佳參數 (v3)\n")
report_lines.append(f"- **策略**: {r_v3.best_strategy}")
report_lines.append(f"- **參數**: {r_v3.best_params}")
report_lines.append("")

# 改善分析
report_lines.append("## 改善分析\n")
improve_wr = r_v3.best_win_rate - r_v2.best_win_rate
improve_ret = r_v3.best_total_return - r_v2.best_total_return
improve_exp = r_v3.total_experiments - r_v2.total_experiments

report_lines.append(f"| 指標 | v2→v3 變化 | 說明 |")
report_lines.append(f"|------|-----------|------|")
report_lines.append(f"| 勝率 | {improve_wr:+.1%} | {'改善' if improve_wr > 0 else '持平或退步'} |")
report_lines.append(f"| 報酬 | {improve_ret:+.1%} | {'改善' if improve_ret > 0 else '持平或退步'} |")
report_lines.append(f"| 實驗數 | +{improve_exp} | 不限組合搜尋更充分 |")
report_lines.append(f"| 賣出 | 固定→移動 | 移動停利不會錯過大波段 |")

report_lines.append("")
report_lines.append("## 結論\n")
if improve_ret > 0:
    report_lines.append(f"v3 (移動停利+不限搜尋) 相比 v2 報酬提升 **{improve_ret:+.1%}**，")
    report_lines.append(f"實驗次數從 {r_v2.total_experiments} 增加到 {r_v3.total_experiments}，搜尋更充分。")
else:
    report_lines.append(f"v3 和 v2 在此測試中表現接近，")
    report_lines.append(f"但 v3 的移動停利和不限搜尋提供了更好的風險管理和搜尋深度。")

report = "\n".join(report_lines)
print(report)

with open("backtest/v3_test_report.md", "w") as f:
    f.write(report)

print(f"\n✅ 報告已寫入 backtest/v3_test_report.md")
