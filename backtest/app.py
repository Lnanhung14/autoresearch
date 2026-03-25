#!/usr/bin/env python3
"""
Stock Backtest Optimizer - Streamlit Web App
=============================================

使用 FMP API 抓取歷史資料，透過 autoresearch 風格的爬山法
收斂出自選股最高勝率的策略及參數。

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
from io import StringIO

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.fmp_data import fetch_multiple_stocks, search_symbol, validate_api_key
from backtest.strategies import STRATEGY_REGISTRY, list_strategies, get_strategy
from backtest.engine import run_backtest, run_backtest_multi_stock, BacktestResult
from backtest.optimizer import BacktestOptimizer, OptimizationConfig
from backtest.data_loader import generate_sample_data

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Backtest Optimizer",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "stock_data" not in st.session_state:
    st.session_state.stock_data = {}
if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None
if "log_output" not in st.session_state:
    st.session_state.log_output = ""

# ---------------------------------------------------------------------------
# Sidebar - API Key & Stock Input
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ 設定")

# API Key
api_key = st.sidebar.text_input(
    "FMP API Key",
    value=os.environ.get("FMP_API_KEY", "w56DzLDLRiPWJtKCgEaPLuW8fbgyJWn4"),
    type="password",
    help="Financial Modeling Prep API key",
)

st.sidebar.divider()

# Stock symbols input
st.sidebar.subheader("📋 自選股清單")
default_symbols = "AAPL\nMSFT\nGOOGL\nAMZN\nNVDA"
symbols_text = st.sidebar.text_area(
    "輸入股票代碼（每行一個）",
    value=default_symbols,
    height=150,
    help="支援美股、台股(如 2330.TW)等 FMP 支援的股票代碼",
)

# Date range
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input(
    "開始日期",
    value=datetime(2021, 1, 1),
)
end_date = col2.date_input(
    "結束日期",
    value=datetime.now(),
)

# Demo mode
use_demo = st.sidebar.checkbox("使用模擬資料（Demo 模式）", value=False)

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Sidebar - Optimization Settings
# ---------------------------------------------------------------------------
st.sidebar.subheader("🔧 最佳化參數")

selected_strategies = st.sidebar.multiselect(
    "選擇策略",
    options=list_strategies(),
    default=list_strategies(),
    help="選擇要測試的交易策略",
)

primary_metric = st.sidebar.selectbox(
    "主要指標",
    options=["win_rate", "total_return", "sharpe_ratio", "profit_factor"],
    index=0,
    help="優化目標：勝率 / 總報酬 / Sharpe / 獲利因子",
)

min_trades = st.sidebar.slider("最少交易次數", 1, 20, 5)
patience = st.sidebar.slider("收斂耐心值", 5, 50, 20, help="連續N次無改善則停止")
max_combos = st.sidebar.slider("每策略最大參數組合", 50, 500, 150)

st.sidebar.divider()

# Trading costs
st.sidebar.subheader("💰 交易成本")
commission = st.sidebar.number_input("手續費率", value=0.001425, format="%.6f")
tax = st.sidebar.number_input("交易稅率", value=0.003, format="%.6f")
slippage = st.sidebar.number_input("滑價", value=0.001, format="%.4f")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("📈 Stock Backtest Strategy Optimizer")
st.markdown("""
使用 **autoresearch 風格的貪婪爬山法**，自動搜尋最佳交易策略與參數組合。

**流程：** 粗掃描所有策略 → 網格搜索最佳策略參數 → 爬山法微調收斂
""")

# ---------------------------------------------------------------------------
# Step 1: Load Data
# ---------------------------------------------------------------------------
st.header("Step 1️⃣ 載入股票資料")

col_load, col_status = st.columns([1, 2])

with col_load:
    load_btn = st.button("🔄 載入資料", type="primary", use_container_width=True)

if load_btn:
    if use_demo:
        with st.spinner("生成模擬資料..."):
            stock_data = {}
            for i, name in enumerate(["DEMO_A", "DEMO_B", "DEMO_C", "DEMO_D", "DEMO_E"]):
                stock_data[name] = generate_sample_data(name, days=500, start_price=100 + i * 50, seed=42 + i)
            st.session_state.stock_data = stock_data
            st.success(f"✅ 已生成 {len(stock_data)} 檔模擬資料")
    else:
        symbols = [s.strip() for s in symbols_text.strip().split("\n") if s.strip()]
        if not symbols:
            st.error("請輸入至少一個股票代碼")
        elif not api_key:
            st.error("請輸入 FMP API Key")
        else:
            with st.spinner(f"正在從 FMP 下載 {len(symbols)} 檔股票資料..."):
                stock_data = fetch_multiple_stocks(
                    symbols, api_key,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                )
                st.session_state.stock_data = stock_data
                if stock_data:
                    st.success(f"✅ 成功載入 {len(stock_data)} 檔股票資料")
                else:
                    st.error("❌ 無法載入任何股票資料，請確認代碼及 API Key")

# Show loaded data summary
if st.session_state.stock_data:
    with st.expander(f"📊 已載入 {len(st.session_state.stock_data)} 檔股票資料", expanded=False):
        summary_data = []
        for symbol, df in st.session_state.stock_data.items():
            summary_data.append({
                "股票": symbol,
                "資料筆數": len(df),
                "起始日": df["date"].min().strftime("%Y-%m-%d") if "date" in df.columns else "N/A",
                "結束日": df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns else "N/A",
                "最新收盤": f"{df['close'].iloc[-1]:.2f}",
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ---------------------------------------------------------------------------
# Step 2: Run Optimization
# ---------------------------------------------------------------------------
st.header("Step 2️⃣ 執行策略最佳化")

optimize_btn = st.button(
    "🚀 開始最佳化",
    type="primary",
    use_container_width=True,
    disabled=not st.session_state.stock_data,
)

if optimize_btn and st.session_state.stock_data:
    config = OptimizationConfig(
        primary_metric=primary_metric,
        min_trades=min_trades,
        results_file="backtest_results.tsv",
        strategies=selected_strategies if selected_strategies else None,
        max_combos_per_strategy=max_combos,
        patience=patience,
        commission_rate=commission,
        tax_rate=tax,
        slippage=slippage,
    )

    # Capture print output
    log_capture = StringIO()

    progress_bar = st.progress(0, text="正在初始化...")
    log_area = st.empty()

    # Run optimization with progress tracking
    optimizer = BacktestOptimizer(st.session_state.stock_data, config)

    # Monkey-patch _try_experiment to update progress
    original_try = optimizer._try_experiment
    experiment_logs = []

    def patched_try(strategy_name, params, description=""):
        result = original_try(strategy_name, params, description)
        entry = optimizer.results_log[-1] if optimizer.results_log else {}
        experiment_logs.append(entry)

        # Update progress (estimate total experiments)
        n = len(experiment_logs)
        estimated_total = max(n + 50, 100)  # rough estimate
        progress = min(n / estimated_total, 0.95)
        progress_bar.progress(progress, text=f"實驗 #{n} | {strategy_name} | WR={entry.get('win_rate', 0):.2%}")

        # Update log
        status_icon = "✅" if entry.get("status") == "keep" else "  "
        log_line = (f"{status_icon} #{n:3d} [{strategy_name:15s}] "
                    f"WR={entry.get('win_rate', 0):.2%} "
                    f"Sharpe={entry.get('sharpe', 0):.3f} "
                    f"Trades={entry.get('total_trades', 0):3d}")
        experiment_logs[-1]["_log_line"] = log_line
        return result

    optimizer._try_experiment = patched_try

    start_time = time.time()
    result = optimizer.optimize()
    elapsed = time.time() - start_time

    progress_bar.progress(1.0, text=f"✅ 完成! {result.total_experiments} 次實驗，耗時 {elapsed:.1f}s")

    st.session_state.optimization_result = result
    st.session_state.experiment_logs = experiment_logs

# ---------------------------------------------------------------------------
# Step 3: Show Results
# ---------------------------------------------------------------------------
if st.session_state.optimization_result:
    result = st.session_state.optimization_result

    st.header("Step 3️⃣ 最佳化結果")

    # Best strategy highlight
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏆 最佳策略", result.best_strategy)
    col2.metric("🎯 平均勝率", f"{result.best_win_rate:.2%}")
    col3.metric("📈 平均報酬", f"{result.best_total_return:.2%}")
    col4.metric("📊 Sharpe", f"{result.best_sharpe:.3f}")

    st.markdown("---")

    # Best parameters
    st.subheader("🔑 最佳參數")
    params_df = pd.DataFrame([
        {"參數": k, "最佳值": v} for k, v in result.best_params.items()
    ])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    # Per-stock breakdown
    if result.per_stock_results:
        st.subheader("📋 各股票回測結果")
        stock_results = []
        for r in sorted(result.per_stock_results, key=lambda x: x.win_rate, reverse=True):
            if r.num_trades > 0:
                stock_results.append({
                    "股票": r.symbol,
                    "勝率": f"{r.win_rate:.2%}",
                    "總報酬": f"{r.total_return:.2%}",
                    "交易次數": r.num_trades,
                    "贏": r.num_wins,
                    "輸": r.num_losses,
                    "平均獲利": f"{r.avg_win:.2%}" if r.avg_win else "N/A",
                    "平均虧損": f"{r.avg_loss:.2%}" if r.avg_loss else "N/A",
                    "最大回撤": f"{r.max_drawdown:.2%}",
                    "Sharpe": f"{r.sharpe_ratio:.3f}",
                    "獲利因子": f"{r.profit_factor:.2f}",
                })
        st.dataframe(pd.DataFrame(stock_results), use_container_width=True, hide_index=True)

    # Experiment log
    if hasattr(st.session_state, "experiment_logs") and st.session_state.experiment_logs:
        st.subheader("📝 實驗記錄")

        log_df = pd.DataFrame(st.session_state.experiment_logs)
        display_cols = ["experiment", "strategy", "win_rate", "sharpe",
                        "total_trades", "valid_stocks", "status", "description"]
        available_cols = [c for c in display_cols if c in log_df.columns]
        log_display = log_df[available_cols].copy()

        if "win_rate" in log_display.columns:
            log_display["win_rate"] = log_display["win_rate"].apply(lambda x: f"{x:.2%}")
        if "sharpe" in log_display.columns:
            log_display["sharpe"] = log_display["sharpe"].apply(lambda x: f"{x:.3f}")

        # Color kept experiments
        st.dataframe(log_display, use_container_width=True, height=400, hide_index=True)

        # Win rate convergence chart
        st.subheader("📉 勝率收斂曲線")
        chart_data = pd.DataFrame(st.session_state.experiment_logs)
        if "win_rate" in chart_data.columns:
            chart_data["best_so_far"] = chart_data["win_rate"].cummax()
            st.line_chart(
                chart_data[["win_rate", "best_so_far"]].rename(
                    columns={"win_rate": "當次勝率", "best_so_far": "最佳勝率"}
                ),
                use_container_width=True,
            )

    # Export
    st.subheader("💾 匯出結果")
    col_json, col_tsv = st.columns(2)

    best_config = {
        "strategy": result.best_strategy,
        "params": {k: float(v) for k, v in result.best_params.items()},
        "metrics": {
            "win_rate": float(result.best_win_rate),
            "sharpe_ratio": float(result.best_sharpe),
            "total_return": float(result.best_total_return),
            "total_trades": int(result.best_num_trades),
        },
    }
    col_json.download_button(
        "📥 下載最佳策略 (JSON)",
        data=json.dumps(best_config, indent=2, ensure_ascii=False),
        file_name="best_strategy.json",
        mime="application/json",
    )

    if os.path.exists("backtest_results.tsv"):
        with open("backtest_results.tsv", "r") as f:
            tsv_data = f.read()
        col_tsv.download_button(
            "📥 下載完整記錄 (TSV)",
            data=tsv_data,
            file_name="backtest_results.tsv",
            mime="text/tab-separated-values",
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "Stock Backtest Optimizer | Powered by autoresearch hill-climbing logic | FMP API"
    "</div>",
    unsafe_allow_html=True,
)
