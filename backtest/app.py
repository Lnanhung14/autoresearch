#!/usr/bin/env python3
"""
Stock Backtest Optimizer Dashboard
====================================
輸入股票代碼，一鍵找出最佳策略與參數。

Run:
    streamlit run backtest/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.fmp_data import fetch_multiple_stocks, search_symbol
from backtest.strategies import STRATEGY_REGISTRY, list_strategies, get_strategy
from backtest.engine import run_backtest, BacktestResult
from backtest.optimizer import BacktestOptimizer, OptimizationConfig
from backtest.data_loader import generate_sample_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FMP_API_KEY = "w56DzLDLRiPWJtKCgEaPLuW8fbgyJWn4"

STRATEGY_CN = {
    "sma_crossover": "SMA 均線交叉",
    "ema_crossover": "EMA 均線交叉",
    "rsi": "RSI 超買超賣",
    "macd": "MACD 交叉",
    "bollinger_bands": "布林通道",
    "kdj": "KDJ 隨機指標",
    "volume_price": "量價突破",
    "dual_ma_rsi": "雙均線 + RSI",
}

METRIC_CN = {
    "win_rate": "勝率",
    "total_return": "總報酬率",
    "sharpe_ratio": "Sharpe Ratio",
    "profit_factor": "獲利因子",
}

PARAM_CN = {
    "fast_period": "快線週期",
    "slow_period": "慢線週期",
    "period": "週期",
    "oversold": "超賣線",
    "overbought": "超買線",
    "signal_period": "訊號線週期",
    "num_std": "標準差倍數",
    "k_period": "K 線週期",
    "d_period": "D 線週期",
    "vol_ma_period": "成交量均線",
    "vol_threshold": "量能倍數門檻",
    "price_ma_period": "價格均線",
    "rsi_period": "RSI 週期",
    "rsi_threshold": "RSI 門檻",
}

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="策略最佳化分析", page_icon="📈", layout="wide")

st.markdown("""
<style>
    /* Main container */
    .block-container { padding-top: 1.5rem; }

    /* Hero card */
    .hero-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 16px; padding: 2rem 2.5rem; color: white;
        margin-bottom: 1.5rem;
    }
    .hero-card h1 { color: white; margin: 0 0 0.3rem 0; font-size: 2rem; }
    .hero-card p  { color: #b0cfea; margin: 0; font-size: 1rem; }

    /* Result cards */
    .result-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1.2rem; text-align: center;
        border: 1px solid #e9ecef;
    }
    .result-card .value {
        font-size: 1.8rem; font-weight: 700; color: #1e3a5f;
        margin: 0.3rem 0;
    }
    .result-card .label {
        font-size: 0.8rem; color: #6c757d; text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .win { color: #28a745 !important; }
    .lose { color: #dc3545 !important; }

    /* Strategy badge */
    .strat-badge {
        display: inline-block; background: #1e3a5f; color: white;
        padding: 0.4rem 1rem; border-radius: 20px; font-size: 1.1rem;
        font-weight: 600; margin-right: 0.5rem;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Input area styling */
    .stTextArea textarea { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("stock_data", {}), ("result", None), ("logs", []),
    ("running", False), ("phase_text", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <h1>📈 自選股策略最佳化分析</h1>
    <p>輸入股票代碼 → 自動抓取資料 → AI 爬山法收斂最佳策略與參數</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main input area — simple and prominent
# ---------------------------------------------------------------------------
col_input, col_go = st.columns([4, 1])

with col_input:
    symbols_input = st.text_input(
        "輸入股票代碼（逗號或空格分隔）",
        value="AAPL, MSFT, GOOGL, NVDA, AMZN",
        placeholder="例：AAPL, TSLA, 2330.TW",
        label_visibility="collapsed",
    )
    st.caption("輸入股票代碼，以逗號或空格分隔。台股請加 .TW（如 2330.TW）")

with col_go:
    go_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)

# Advanced settings in expander — not in the way
with st.expander("⚙️ 進階設定（選填）", expanded=False):
    adv_col1, adv_col2, adv_col3 = st.columns(3)

    with adv_col1:
        st.markdown("**資料範圍**")
        start_date = st.date_input("開始日期", value=datetime(2021, 1, 1))
        end_date = st.date_input("結束日期", value=datetime.now())

    with adv_col2:
        st.markdown("**最佳化設定**")
        primary_metric = st.selectbox(
            "優化目標",
            options=list(METRIC_CN.keys()),
            format_func=lambda x: METRIC_CN[x],
        )
        min_trades = st.number_input("最少交易次數", value=5, min_value=1, max_value=30)

    with adv_col3:
        st.markdown("**交易成本**")
        commission = st.number_input("手續費率", value=0.001425, format="%.6f",
                                     help="台股券商手續費 0.1425%")
        tax = st.number_input("交易稅率", value=0.003, format="%.4f",
                              help="台股交易稅 0.3%")

    st.markdown("**策略選擇**（不選 = 全部測試）")
    selected_strategies = st.multiselect(
        "策略",
        options=list_strategies(),
        default=[],
        format_func=lambda x: f"{STRATEGY_CN.get(x, x)}",
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Run optimization
# ---------------------------------------------------------------------------
if go_btn:
    # Parse symbols
    raw = symbols_input.replace(",", " ").replace(";", " ").replace("\n", " ")
    symbols = [s.strip().upper() for s in raw.split() if s.strip()]

    if not symbols:
        st.error("請輸入至少一個股票代碼")
        st.stop()

    # --- Phase: Download ---
    status_container = st.container()
    progress_bar = st.progress(0, text="正在下載股票資料...")

    with st.spinner(f"正在從 FMP 下載 {len(symbols)} 檔股票資料..."):
        stock_data = fetch_multiple_stocks(
            symbols, FMP_API_KEY,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

    if not stock_data:
        st.error("❌ 無法下載任何股票資料，請確認代碼是否正確")
        st.stop()

    st.session_state.stock_data = stock_data
    progress_bar.progress(0.1, text=f"✅ 已載入 {len(stock_data)}/{len(symbols)} 檔 → 開始最佳化...")

    # --- Phase: Optimize ---
    config = OptimizationConfig(
        primary_metric=primary_metric,
        min_trades=min_trades,
        results_file="backtest_results.tsv",
        strategies=selected_strategies if selected_strategies else None,
        max_combos_per_strategy=150,
        patience=20,
        commission_rate=commission,
        tax_rate=tax,
        slippage=0.001,
    )

    optimizer = BacktestOptimizer(stock_data, config)

    # Patch to track progress
    original_try = optimizer._try_experiment
    logs = []

    def _patched_try(strategy_name, params, description=""):
        res = original_try(strategy_name, params, description)
        entry = optimizer.results_log[-1] if optimizer.results_log else {}
        logs.append(entry)
        n = len(logs)
        pct = min(0.1 + 0.85 * (n / max(n + 40, 80)), 0.95)
        wr = entry.get("win_rate", 0)
        progress_bar.progress(pct, text=f"實驗 #{n} ── {STRATEGY_CN.get(strategy_name, strategy_name)} ── 勝率 {wr:.1%}")
        return res

    optimizer._try_experiment = _patched_try

    t0 = time.time()
    result = optimizer.optimize()
    elapsed = time.time() - t0

    progress_bar.progress(1.0, text=f"✅ 完成！共 {result.total_experiments} 次實驗，耗時 {elapsed:.0f} 秒")

    st.session_state.result = result
    st.session_state.logs = logs
    st.rerun()

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
if st.session_state.result:
    result = st.session_state.result
    logs = st.session_state.logs

    st.markdown("---")

    # ============ Best strategy hero ============
    st.markdown("### 🏆 最佳策略")

    c1, c2, c3, c4, c5 = st.columns(5)
    _strat_label = STRATEGY_CN.get(result.best_strategy, result.best_strategy)
    c1.markdown(f"""<div class="result-card">
        <div class="label">策略</div>
        <div class="value" style="font-size:1.3rem;">{_strat_label}</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="result-card">
        <div class="label">平均勝率</div>
        <div class="value win">{result.best_win_rate:.1%}</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="result-card">
        <div class="label">平均報酬</div>
        <div class="value {'win' if result.best_total_return >= 0 else 'lose'}">{result.best_total_return:.1%}</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="result-card">
        <div class="label">Sharpe Ratio</div>
        <div class="value">{result.best_sharpe:.2f}</div>
    </div>""", unsafe_allow_html=True)
    c5.markdown(f"""<div class="result-card">
        <div class="label">總交易次數</div>
        <div class="value">{result.best_num_trades}</div>
    </div>""", unsafe_allow_html=True)

    # ============ Best params table ============
    st.markdown("### 🔑 最佳參數")
    param_rows = []
    strategy_def = get_strategy(result.best_strategy)
    param_lookup = {p.name: p for p in strategy_def.params}
    for k, v in result.best_params.items():
        p = param_lookup.get(k)
        param_rows.append({
            "參數": PARAM_CN.get(k, k),
            "最佳值": int(v) if float(v) == int(v) else round(float(v), 2),
            "搜索範圍": f"{p.min_val} ~ {p.max_val}" if p else "",
        })
    st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

    # ============ Per-stock breakdown ============
    if result.per_stock_results:
        st.markdown("### 📋 各股票回測明細")
        rows = []
        for r in sorted(result.per_stock_results, key=lambda x: x.win_rate, reverse=True):
            if r.num_trades > 0:
                rows.append({
                    "股票": r.symbol,
                    "勝率": f"{r.win_rate:.1%}",
                    "總報酬": f"{r.total_return:.1%}",
                    "交易數": r.num_trades,
                    "勝": r.num_wins,
                    "負": r.num_losses,
                    "均獲利": f"{r.avg_win:.2%}" if r.avg_win else "-",
                    "均虧損": f"{r.avg_loss:.2%}" if r.avg_loss else "-",
                    "最大回撤": f"{r.max_drawdown:.1%}",
                    "Sharpe": f"{r.sharpe_ratio:.2f}",
                    "獲利因子": f"{r.profit_factor:.1f}" if r.profit_factor < 100 else "∞",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ============ Convergence chart ============
    if logs:
        st.markdown("### 📉 勝率收斂曲線")
        chart_df = pd.DataFrame(logs)
        if "win_rate" in chart_df.columns:
            chart_df["最佳勝率"] = chart_df["win_rate"].cummax()
            chart_df = chart_df.rename(columns={"win_rate": "當次勝率"})
            st.line_chart(chart_df[["當次勝率", "最佳勝率"]], use_container_width=True, height=300)

    # ============ Strategy comparison ============
    if logs:
        st.markdown("### 📊 各策略最佳勝率比較")
        log_df = pd.DataFrame(logs)
        if "strategy" in log_df.columns and "win_rate" in log_df.columns:
            strat_best = log_df.groupby("strategy")["win_rate"].max().sort_values(ascending=True)
            strat_best.index = [STRATEGY_CN.get(s, s) for s in strat_best.index]
            st.bar_chart(strat_best, horizontal=True, height=max(200, len(strat_best) * 50))

    # ============ Experiment log ============
    if logs:
        with st.expander(f"📝 完整實驗記錄（{len(logs)} 筆）", expanded=False):
            log_df = pd.DataFrame(logs)
            cols = ["experiment", "strategy", "win_rate", "sharpe",
                    "total_trades", "valid_stocks", "status", "description"]
            available = [c for c in cols if c in log_df.columns]
            display = log_df[available].copy()
            if "strategy" in display.columns:
                display["strategy"] = display["strategy"].map(
                    lambda x: STRATEGY_CN.get(x, x))
            if "win_rate" in display.columns:
                display["win_rate"] = display["win_rate"].apply(lambda x: f"{x:.1%}")
            if "sharpe" in display.columns:
                display["sharpe"] = display["sharpe"].apply(lambda x: f"{x:.2f}")
            display.columns = ["#", "策略", "勝率", "Sharpe", "交易數", "有效股數", "結果", "說明"][:len(available)]
            st.dataframe(display, use_container_width=True, height=400, hide_index=True)

    # ============ Export ============
    st.markdown("### 💾 匯出")
    ecol1, ecol2 = st.columns(2)
    best_json = {
        "strategy": result.best_strategy,
        "strategy_cn": STRATEGY_CN.get(result.best_strategy, ""),
        "params": {k: float(v) for k, v in result.best_params.items()},
        "metrics": {
            "win_rate": float(result.best_win_rate),
            "sharpe_ratio": float(result.best_sharpe),
            "total_return": float(result.best_total_return),
            "total_trades": int(result.best_num_trades),
        },
    }
    ecol1.download_button(
        "📥 下載最佳策略 JSON", use_container_width=True,
        data=json.dumps(best_json, indent=2, ensure_ascii=False),
        file_name="best_strategy.json", mime="application/json",
    )
    if logs:
        ecol2.download_button(
            "📥 下載實驗記錄 CSV", use_container_width=True,
            data=pd.DataFrame(logs).to_csv(index=False),
            file_name="experiment_log.csv", mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Empty state — when no results yet
# ---------------------------------------------------------------------------
if not st.session_state.result and not go_btn:
    st.markdown("---")
    st.markdown("#### 💡 使用方式")
    st.markdown("""
    1. 在上方輸入框填入股票代碼（如 `AAPL, TSLA, NVDA`）
    2. 點擊 **🚀 開始分析**
    3. 系統自動下載資料 → 測試 8 種策略 × 多種參數 → 收斂出最佳組合
    """)

    st.markdown("#### 📌 支援的策略")
    strat_cols = st.columns(4)
    for i, (key, cn) in enumerate(STRATEGY_CN.items()):
        sdef = STRATEGY_REGISTRY[key]
        params_str = "、".join([PARAM_CN.get(p.name, p.name) for p in sdef.params])
        strat_cols[i % 4].markdown(f"**{cn}**  \n<span style='color:gray;font-size:0.85em;'>參數：{params_str}</span>", unsafe_allow_html=True)

    st.markdown("#### 🌍 股票代碼範例")
    ex1, ex2, ex3 = st.columns(3)
    ex1.markdown("**美股**  \n`AAPL, MSFT, GOOGL, NVDA, TSLA`")
    ex2.markdown("**台股**  \n`2330.TW, 2317.TW, 2454.TW`")
    ex3.markdown("**ETF**  \n`SPY, QQQ, VTI, 0050.TW`")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; color:#aaa; font-size:0.8em; padding:2rem 0 1rem;">
    Stock Backtest Optimizer &middot; Autoresearch Hill-Climbing Engine &middot; FMP API
</div>
""", unsafe_allow_html=True)
