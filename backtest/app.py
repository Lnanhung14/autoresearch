#!/usr/bin/env python3
"""
Stock Backtest Optimizer Dashboard
====================================
Run:  streamlit run backtest/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import sys
import io
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest.strategies import STRATEGY_REGISTRY, list_strategies, get_strategy
from backtest.engine import run_backtest, BacktestResult
from backtest.optimizer import BacktestOptimizer, OptimizationConfig
from backtest.data_loader import generate_sample_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRATEGY_CN = {
    # 原有 8 種策略
    "sma_crossover": "簡單移動平均線交叉",
    "ema_crossover": "指數移動平均線交叉",
    "rsi": "RSI 超買超賣",
    "macd": "MACD 交叉",
    "bollinger_bands": "布林通道均值回歸",
    "kdj": "KDJ 隨機指標",
    "volume_price": "量價突破",
    "dual_ma_rsi": "雙均線交叉 + RSI 過濾",
    # Gemini APP2 Stock 12 種策略
    "multi_factor": "多因子動量 + RSI 排名選股",
    "kd_macd_signal": "KD 黃金交叉 + MACD 柱狀圖",
    "ma_breakout": "均線突破趨勢跟蹤",
    "foreign_follow": "外資跟單策略 (量能放大)",
    "etf_momentum": "ETF 動量定期再平衡",
    "vol_price_sync": "量價齊揚確認多頭",
    "vol_breakout_vcp": "VCP 量縮突破",
    "vol_divergence": "量價背離反轉 (OBV)",
    "chip_sedimentation": "籌碼沉澱動能突破",
    "chip_sedimentation_inst": "籌碼沉澱 + 法人共振",
    "chip_sedimentation_rising": "籌碼沉澱 + 起漲點 (RSI+MACD)",
    "vol_dry_bottom": "量縮 KD 低檔反彈",
}

METRIC_CN = {
    "win_rate": "勝率",
    "total_return": "總報酬",
    "sharpe_ratio": "夏普比率",
    "profit_factor": "獲利因子",
}

PARAM_CN = {
    # 原有參數
    "fast_period": "快線週期",
    "slow_period": "慢線週期",
    "period": "計算週期",
    "oversold": "超賣線",
    "overbought": "超買線",
    "signal_period": "訊號線週期",
    "num_std": "標準差倍數",
    "k_period": "K 線週期",
    "d_period": "D 線週期",
    "vol_ma_period": "成交量均線週期",
    "vol_threshold": "成交量門檻倍數",
    "price_ma_period": "價格均線週期",
    "rsi_period": "RSI 計算週期",
    "rsi_threshold": "RSI 門檻",
    # 新增策略參數
    "momentum_days": "動量計算天數",
    "rsi_buy": "RSI 買入門檻",
    "rsi_sell": "RSI 賣出門檻",
    "kd_oversold": "KD 超賣線",
    "kd_overbought": "KD 超買線",
    "macd_fast": "MACD 快線",
    "macd_slow": "MACD 慢線",
    "macd_signal": "MACD 訊號線",
    "ma_short": "短均線",
    "ma_long": "長均線",
    "ma_trend": "趨勢均線",
    "vol_surge": "量能放大倍數",
    "rsi_min": "RSI 下限",
    "rebal_period": "再平衡週期(天)",
    "vol_surge_multiplier": "放量倍數門檻",
    "ma_period": "均線週期",
    "range_contract": "收斂比例",
    "lookback": "回看天數",
    "lookback_days": "回看天數",
    "vol_lookback": "量能回看週期",
    "vol_shrink_threshold": "量縮門檻(均量倍)",
    "vol_shrink_min_days": "量縮最少天數",
    "price_range_threshold": "價格收斂門檻(%)",
    "sedimentation_window": "沉澱窗口(天)",
    "vol_dry_ratio": "量縮門檻(均量倍)",
    "vol_dry_days": "量縮天數",
    "rsi_max": "RSI 上限",
}

# ---------------------------------------------------------------------------
# 台股代碼 ↔ 名稱 對照表
# ---------------------------------------------------------------------------
TW_STOCK_NAMES = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2308": "台達電",
    "2382": "廣達", "2881": "富邦金", "2882": "國泰金", "2891": "中信金",
    "2303": "聯電", "2412": "中華電", "3711": "日月光投控", "2886": "兆豐金",
    "2884": "玉山金", "1301": "台塑", "1326": "台化", "2357": "華碩",
    "3037": "欣興", "5871": "中租-KY", "2327": "國巨", "2395": "研華",
    "3008": "大立光", "2345": "智邦", "6505": "台塑化", "1303": "南亞",
    "2002": "中鋼", "1216": "統一", "2207": "和泰車", "5880": "合庫金",
    "2892": "第一金", "3045": "台灣大", "2880": "華南金", "2887": "台新金",
    "4904": "遠傳", "2379": "瑞昱", "2301": "光寶科", "9910": "豐泰",
    "2885": "元大金", "4938": "和碩", "3034": "聯詠", "2883": "凱基金",
    "6669": "緯穎", "2912": "統一超", "5876": "上海商銀", "1101": "台泥",
    "2888": "新光金", "3231": "緯創", "2603": "長榮", "6446": "藥華藥",
    "2105": "正新", "8046": "南電", "6176": "瑞儀", "3443": "創意",
    "8150": "南茂", "2344": "華邦電", "3023": "信邦", "3044": "健鼎",
    "2383": "台光電", "6239": "力成", "3665": "貿聯-KY", "2049": "上銀",
    "8464": "億豐", "6285": "啟碁", "3661": "世芯-KY", "3035": "智原",
    "2474": "可成", "2458": "義隆", "6409": "旭隼", "1477": "聚陽",
    "2542": "興富發", "5269": "祥碩", "6592": "和潤企業", "6770": "力積電",
    "2404": "漢唐", "3017": "奇鋐", "2801": "彰銀", "6531": "愛普",
    "1476": "儒鴻", "2548": "華固", "9921": "巨大", "2615": "萬海",
    "2634": "漢翔", "1802": "台玻", "1590": "亞德客-KY", "2059": "川湖",
    "9945": "潤泰新", "1795": "美時", "6415": "矽力-KY", "2845": "遠東銀",
    "8069": "元太", "2618": "長榮航", "2006": "東和鋼鐵", "1504": "東元",
    "3653": "健策", "2227": "裕日車", "2408": "南亞科", "6456": "GIS-KY",
    "3036": "文曄", "8454": "富邦媒", "6278": "台表科", "2353": "宏碁",
    "2356": "英業達", "2376": "技嘉", "2377": "微星", "2385": "群光",
    "2324": "仁寶", "2360": "致茂", "2347": "聯強", "3706": "神達",
    "4958": "臻鼎-KY", "2889": "國票金", "2023": "燁輝", "2201": "裕隆",
    "2101": "南港", "3515": "華擎", "6550": "北極星藥業-KY",
    "3576": "聯合再生", "1304": "台聚", "5388": "中磊", "2439": "美律",
    "6789": "采鈺", "3016": "嘉晶", "4919": "新唐", "2610": "華航",
    "6116": "彩晶", "1560": "中砂", "2354": "鴻準", "2609": "陽明",
    "1605": "華新", "3702": "大聯大", "2923": "鼎固-KY",
    "3189": "景碩", "3533": "嘉澤", "6472": "保瑞",
    "2337": "旺宏", "5534": "長虹", "6257": "矽格",
    "2349": "錸德", "2603": "長榮",
}

# 名稱 → 代碼 反向對照 (含簡稱)
_NAME_TO_CODE = {}
for _code, _name in TW_STOCK_NAMES.items():
    _NAME_TO_CODE[_name] = _code
    # 去掉 -KY 的簡稱
    if "-KY" in _name:
        _NAME_TO_CODE[_name.replace("-KY", "")] = _code


def resolve_symbol(raw_input: str) -> tuple:
    """
    解析使用者輸入的股票代碼或名稱。
    回傳 (yahoo_symbol, display_name)
    支援: "2330", "2330.TW", "台積電"
    """
    s = raw_input.strip()
    if not s:
        return None, None

    # 已經是完整 Yahoo 格式 (含 . 如 AAPL, 2330.TW)
    if '.' in s:
        code = s.split('.')[0]
        name = TW_STOCK_NAMES.get(code, "")
        display = f"{name} ({s})" if name else s
        return s.upper(), display

    # 純數字 → 台股代碼，自動加 .TW
    if s.isdigit():
        yahoo_sym = f"{s}.TW"
        name = TW_STOCK_NAMES.get(s, "")
        display = f"{name} ({yahoo_sym})" if name else yahoo_sym
        return yahoo_sym, display

    # 英文字母 → 美股代碼
    if s.isascii() and s.isalpha():
        return s.upper(), s.upper()

    # 中文名稱查找
    if s in _NAME_TO_CODE:
        code = _NAME_TO_CODE[s]
        yahoo_sym = f"{code}.TW"
        return yahoo_sym, f"{s} ({yahoo_sym})"

    # 模糊搜尋 (名稱包含)
    for name, code in _NAME_TO_CODE.items():
        if s in name:
            yahoo_sym = f"{code}.TW"
            return yahoo_sym, f"{name} ({yahoo_sym})"

    # 無法識別，原樣返回加 .TW
    if not s.isascii():
        return None, None
    return s.upper(), s.upper()


def resolve_symbols_input(raw_text: str) -> list:
    """
    解析多個股票輸入 (逗號/空格分隔)。
    回傳 list of (yahoo_symbol, display_name)。
    """
    raw = raw_text.replace(",", " ").replace(";", " ").replace("\n", " ")
    tokens = [t.strip() for t in raw.split() if t.strip()]
    results = []
    seen = set()
    for t in tokens:
        sym, display = resolve_symbol(t)
        if sym and sym not in seen:
            seen.add(sym)
            results.append((sym, display))
    return results


def get_stock_display(symbol: str) -> str:
    """取得股票顯示名稱 (名稱 + 代碼)"""
    code = symbol.replace(".TW", "").replace(".TWO", "")
    name = TW_STOCK_NAMES.get(code, "")
    if name:
        return f"{name} ({symbol})"
    return symbol


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="回測策略最佳化", page_icon="chart_with_upwards_trend", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .hero-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 16px; padding: 2rem 2.5rem; color: white;
        margin-bottom: 1.5rem;
    }
    .hero-card h1 { color: white; margin: 0 0 0.3rem 0; font-size: 2rem; }
    .hero-card p  { color: #b0cfea; margin: 0; font-size: 1rem; }
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
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data persistence helpers
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data")
os.makedirs(_DATA_DIR, exist_ok=True)

_PORTFOLIO_FILE = os.path.join(_DATA_DIR, "portfolio.json")
_STRATEGIES_FILE = os.path.join(_DATA_DIR, "saved_strategies.json")
_WATCHLIST_FILE = os.path.join(_DATA_DIR, "watchlist.json")


def _load_json(path, default=None):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return default if default is not None else {}


def _save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _fetch_current_price(symbol):
    """取得即時價格"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        price = info.get('regularMarketPrice') or info.get('currentPrice')
        if price:
            return float(price)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("stock_data", {}), ("result", None), ("logs", []),
    ("running", False), ("per_stock_bt", {}), ("best_strategy_name", None),
    ("best_params", None),
    ("portfolio", _load_json(_PORTFOLIO_FILE, [])),
    ("saved_strategies", _load_json(_STRATEGIES_FILE, {})),
    ("watchlist", _load_json(_WATCHLIST_FILE, [])),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <h1>自選股策略最佳化分析</h1>
    <p>輸入股票代碼 -> 自動抓取資料 -> AI 爬山法收斂最佳策略與參數</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: 持股追蹤 + 觀察名單
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 持股追蹤")

    # 新增持股表單
    with st.expander("新增持股", expanded=False):
        _add_symbol = st.text_input("股票代碼或名稱", placeholder="如 2330 或 台積電", key="add_port_symbol")
        # 即時顯示解析結果
        if _add_symbol.strip():
            _resolved_sym, _resolved_disp = resolve_symbol(_add_symbol)
            if _resolved_sym:
                st.caption(f"→ {_resolved_disp}")
            else:
                st.caption("⚠️ 無法識別")
        _add_cols = st.columns(2)
        _add_shares = _add_cols[0].number_input("股數 (股)", min_value=1, value=1000, step=1000, key="add_port_shares")
        _add_price = _add_cols[1].number_input("買入價", min_value=0.01, value=100.0, step=0.5, key="add_port_price")
        _add_date = st.date_input("買入日期", value=datetime.now(), key="add_port_date")
        if st.button("加入持股", type="primary", use_container_width=True, key="btn_add_port"):
            sym, _ = resolve_symbol(_add_symbol)
            if sym:
                st.session_state.portfolio.append({
                    'symbol': sym,
                    'shares': int(_add_shares),
                    'buy_price': float(_add_price),
                    'buy_date': _add_date.strftime('%Y-%m-%d'),
                })
                _save_json(_PORTFOLIO_FILE, st.session_state.portfolio)
                st.rerun()
            else:
                st.warning("請輸入股票代碼")

    # 顯示持股
    if st.session_state.portfolio:
        total_cost = 0
        total_market = 0
        total_pnl = 0
        rows_to_show = []

        for i, pos in enumerate(st.session_state.portfolio):
            current_price = _fetch_current_price(pos['symbol'])
            cost = pos['shares'] * pos['buy_price']
            total_cost += cost

            if current_price:
                market_val = pos['shares'] * current_price
                pnl = market_val - cost
                ret = (current_price - pos['buy_price']) / pos['buy_price']
                total_market += market_val
                total_pnl += pnl
                rows_to_show.append({
                    'idx': i,
                    'symbol': pos['symbol'],
                    'shares': pos['shares'],
                    'buy_price': pos['buy_price'],
                    'current': current_price,
                    'return': ret,
                    'pnl': pnl,
                })
            else:
                total_market += cost
                rows_to_show.append({
                    'idx': i,
                    'symbol': pos['symbol'],
                    'shares': pos['shares'],
                    'buy_price': pos['buy_price'],
                    'current': None,
                    'return': 0,
                    'pnl': 0,
                })

        # 總損益
        total_ret = (total_pnl / total_cost) if total_cost > 0 else 0
        pnl_color = "#28a745" if total_pnl >= 0 else "#dc3545"
        st.markdown(
            f"<div style='text-align:center; padding:8px; "
            f"background:{'#e8f5e9' if total_pnl >= 0 else '#ffebee'}; "
            f"border-radius:8px; margin-bottom:8px;'>"
            f"<div style='font-size:0.8rem; color:#666;'>總損益</div>"
            f"<div style='font-size:1.5rem; font-weight:700; color:{pnl_color};'>"
            f"{total_pnl:+,.0f}</div>"
            f"<div style='font-size:0.9rem; color:{pnl_color};'>{total_ret:+.2%}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # 個股明細
        for r in rows_to_show:
            col_info, col_del = st.columns([5, 1])
            with col_info:
                if r['current']:
                    c = "🟢" if r['pnl'] >= 0 else "🔴"
                    disp = get_stock_display(r['symbol'])
                    st.markdown(
                        f"{c} **{disp}** {r['shares']}股 "
                        f"| 買{r['buy_price']:.2f} → {r['current']:.2f} "
                        f"| **{r['return']:+.1%}** ({r['pnl']:+,.0f})"
                    )
                else:
                    disp = get_stock_display(r['symbol'])
                    st.markdown(f"⚪ **{disp}** {r['shares']}股 | 買{r['buy_price']:.2f} | 無法取得現價")
            with col_del:
                if st.button("✕", key=f"del_port_{r['idx']}", help="刪除"):
                    st.session_state.portfolio.pop(r['idx'])
                    _save_json(_PORTFOLIO_FILE, st.session_state.portfolio)
                    st.rerun()

        # 賣點提示
        st.markdown("---")
        st.markdown("#### 賣點提示")
        has_alert = False
        for pos in st.session_state.portfolio:
            try:
                ticker = yf.Ticker(pos['symbol'])
                hist = ticker.history(period="3mo")
                if hist.empty:
                    continue
                hist.columns = [c.lower() for c in hist.columns]
                alerts = detect_sell_signals(pos['symbol'], hist)
                if alerts:
                    has_alert = True
                    disp = get_stock_display(pos['symbol'])
                    for a in alerts:
                        icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(a['urgency'], "⚪")
                        st.markdown(f"{icon} **{disp}** [{a['strategy']}]  \n{a['reason']}")
            except Exception:
                pass
        if not has_alert:
            st.info("目前持股無賣出警示")

        if st.button("更新現價", use_container_width=True, key="btn_refresh_port"):
            st.rerun()
    else:
        st.info("尚無持股，請新增")

    # 觀察名單
    st.markdown("---")
    st.markdown("## 觀察名單")

    if st.session_state.watchlist:
        for i, item in enumerate(st.session_state.watchlist):
            wl_col1, wl_col2 = st.columns([5, 1])
            with wl_col1:
                label = get_stock_display(item.get('symbol', ''))
                strat = item.get('strategy_cn', '')
                wr = item.get('win_rate', 0)
                st.markdown(f"**{label}** | {strat} | 勝率 {wr:.0%}")
            with wl_col2:
                if st.button("✕", key=f"del_wl_{i}", help="移除"):
                    st.session_state.watchlist.pop(i)
                    _save_json(_WATCHLIST_FILE, st.session_state.watchlist)
                    st.rerun()
    else:
        st.caption("觀察名單為空")

    _wl_input = st.text_input("快速加入觀察", placeholder="代碼或名稱", key="wl_quick_add")
    if _wl_input.strip():
        _wl_resolved, _wl_disp = resolve_symbol(_wl_input)
        if _wl_resolved:
            st.caption(f"→ {_wl_disp}")
    if st.button("加入觀察", use_container_width=True, key="btn_add_wl"):
        sym, _ = resolve_symbol(_wl_input)
        if sym:
            st.session_state.watchlist.append({
                'symbol': sym,
                'strategy_cn': '待分析',
                'win_rate': 0,
                'added_date': datetime.now().strftime('%Y-%m-%d'),
            })
            _save_json(_WATCHLIST_FILE, st.session_state.watchlist)
            st.rerun()

    # 已儲存策略列表
    if st.session_state.saved_strategies:
        st.markdown("---")
        st.markdown("## 已儲存最佳策略")
        for key_label, data in st.session_state.saved_strategies.items():
            strat_cn = STRATEGY_CN.get(data.get('strategy', ''), data.get('strategy', ''))
            symbols_list = data.get('symbols', [])
            symbols_display = ", ".join([get_stock_display(s) for s in symbols_list]) if symbols_list else key_label

            with st.container():
                st.markdown(
                    f"**{symbols_display}**  \n"
                    f"📋 {strat_cn}  \n"
                    f"勝率 {data.get('win_rate', 0):.0%} | "
                    f"夏普 {data.get('sharpe', 0):.2f} | "
                    f"報酬 {data.get('total_return', 0):.1%}  \n"
                    f"🕐 {data.get('saved_date', '')}"
                )
                sb_col1, sb_col2, sb_col3 = st.columns(3)
                # 用 hash 確保 key 唯一且無特殊字元
                safe_key = str(abs(hash(key_label)))[:12]
                with sb_col1:
                    if st.button("▶️ 重跑", key=f"sb_r_{safe_key}", use_container_width=True):
                        st.session_state['_reload_strategy'] = data.copy()
                        st.session_state['_reload_symbols'] = list(symbols_list)
                        st.session_state['_input_override'] = ", ".join(symbols_list)
                        st.rerun()
                with sb_col2:
                    if st.button("🔬 優化", key=f"sb_o_{safe_key}", use_container_width=True):
                        st.session_state['_reload_optimize'] = data.copy()
                        st.session_state['_reload_symbols'] = list(symbols_list)
                        st.session_state['_input_override'] = ", ".join(symbols_list)
                        st.rerun()
                with sb_col3:
                    if st.button("🗑️", key=f"sb_d_{safe_key}", use_container_width=True):
                        del st.session_state.saved_strategies[key_label]
                        _save_json(_STRATEGIES_FILE, st.session_state.saved_strategies)
                        st.rerun()
                st.markdown("---")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

# 如果有載入的股票，更新輸入框預設值
_default_symbols = "2330, 2317, 2454"
if '_reload_symbols' in st.session_state and st.session_state['_reload_symbols']:
    _default_symbols = ", ".join(st.session_state['_reload_symbols'])
if '_input_override' in st.session_state:
    _default_symbols = st.session_state.pop('_input_override')

col_input, col_go = st.columns([4, 1])

with col_input:
    symbols_input = st.text_input(
        "輸入股票代碼或名稱",
        value=_default_symbols,
        placeholder="例: 台積電, 2330, AAPL (台股可省略 .TW)",
        label_visibility="collapsed",
    )
    # 即時預覽解析結果
    if symbols_input.strip():
        resolved = resolve_symbols_input(symbols_input)
        if resolved:
            preview = " | ".join([disp for _, disp in resolved])
            st.caption(f"📋 {preview}")
        else:
            st.caption("⚠️ 無法識別輸入的股票")
    else:
        st.caption("支援代碼 (2330)、名稱 (台積電)、美股 (AAPL)，逗號或空格分隔，台股可省略 .TW")

with col_go:
    go_btn = st.button("開始分析", type="primary", use_container_width=True)

with st.expander("進階設定 (選填)", expanded=False):
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

    st.markdown("---")
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    with risk_col1:
        st.markdown("**🛡️ 風險管理**")
        stop_loss_pct = st.number_input(
            "停損 %", value=15, min_value=0, max_value=50, step=1,
            help="股價跌幅超過此比例強制賣出。0=不啟用",
        )
    with risk_col2:
        st.markdown("&nbsp;")  # spacer
        take_profit_pct = st.number_input(
            "停利 %", value=0, min_value=0, max_value=500, step=10,
            help="股價漲幅超過此比例強制賣出。0=不啟用。強勢股建議不設或設高 (如 100~200%)",
        )
    with risk_col3:
        st.markdown("&nbsp;")  # spacer
        max_hold = st.number_input(
            "最大持有天數", value=120, min_value=0, max_value=365, step=10,
            help="持有超過此天數強制賣出。0=不限制",
        )

    exit_col1, exit_col2, exit_col3 = st.columns(3)
    with exit_col1:
        st.markdown("**📊 賣出策略 (買賣分離)**")
        trailing_stop_pct = st.number_input(
            "移動停利 %", value=8, min_value=0, max_value=30, step=1,
            help="從持倉最高點回落此比例賣出。例: 8% = 股價從高點回落8%時賣出。0=不啟用",
        )
    with exit_col2:
        st.markdown("&nbsp;")
        partial_exit = st.checkbox(
            "分批出場",
            value=False,
            help="漲10%出1/3 → 漲20%出1/3 → 剩餘依選擇方式出場。適合大波段行情",
        )
        if partial_exit:
            partial_last_mode = st.radio(
                "最後 1/3 出場方式",
                options=["trailing", "ma5"],
                format_func=lambda x: {
                    "trailing": "📉 移動停利 (從高點回落)",
                    "ma5": "📊 破 5 日均線賣出",
                }[x],
                index=0,
                help="移動停利: 從最高點回落N%賣出。\n破5日均線: 收盤價跌破5日移動平均線時賣出，適合追蹤短線趨勢。",
                horizontal=True,
            )
        else:
            partial_last_mode = "trailing"
    with exit_col3:
        st.markdown("&nbsp;")
        time_budget = st.number_input(
            "搜尋時間 (秒)", value=0, min_value=0, max_value=3600, step=30,
            help="最佳化時間預算。0=搜尋到收斂為止 (不限時間)。建議 120~600 秒",
        )

    st.markdown("**策略選擇** (不選 = 全部測試)")
    selected_strategies = st.multiselect(
        "策略",
        options=list_strategies(),
        default=[],
        format_func=lambda x: f"{STRATEGY_CN.get(x, x)}",
        label_visibility="collapsed",
    )


# ---------------------------------------------------------------------------
# K-line chart (Interactive Plotly)
# ---------------------------------------------------------------------------
def draw_kline_chart(symbol, df, signal_df, trades, params, strategy_name, period="daily"):
    """Draw interactive K-line chart with technical indicators, trade markers, and hover info."""

    chart_df = df.copy()
    if not isinstance(chart_df.index, pd.DatetimeIndex):
        if 'date' in chart_df.columns:
            chart_df.index = pd.to_datetime(chart_df['date'])
        else:
            chart_df.index = pd.to_datetime(chart_df.index)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in chart_df.columns:
            return None

    if period == "weekly":
        chart_df = chart_df.resample('W').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

    # --- 計算技術指標 ---
    chart_df['MA5'] = chart_df['close'].rolling(5).mean()
    chart_df['MA10'] = chart_df['close'].rolling(10).mean()
    chart_df['MA20'] = chart_df['close'].rolling(20).mean()
    chart_df['MA60'] = chart_df['close'].rolling(60).mean()

    # EMA
    chart_df['EMA12'] = chart_df['close'].ewm(span=12, adjust=False).mean()
    chart_df['EMA26'] = chart_df['close'].ewm(span=26, adjust=False).mean()

    # RSI
    delta = chart_df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss_s = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss_s.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    chart_df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    chart_df['MACD'] = chart_df['EMA12'] - chart_df['EMA26']
    chart_df['MACD_signal'] = chart_df['MACD'].ewm(span=9, adjust=False).mean()
    chart_df['MACD_hist'] = chart_df['MACD'] - chart_df['MACD_signal']

    # Bollinger Bands
    chart_df['BB_mid'] = chart_df['close'].rolling(20).mean()
    bb_std = chart_df['close'].rolling(20).std()
    chart_df['BB_upper'] = chart_df['BB_mid'] + 2 * bb_std
    chart_df['BB_lower'] = chart_df['BB_mid'] - 2 * bb_std

    # KD
    low_min = chart_df['low'].rolling(9).min()
    high_max = chart_df['high'].rolling(9).max()
    rsv = (chart_df['close'] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    chart_df['K'] = rsv.ewm(com=2, adjust=False).mean()
    chart_df['D'] = chart_df['K'].ewm(com=2, adjust=False).mean()

    # 漲跌幅
    chart_df['change_pct'] = chart_df['close'].pct_change() * 100

    # --- 建立 Plotly 子圖 (5 行) ---
    period_label = "日K" if period == "daily" else "週K"
    wins = sum(1 for t in trades if t.get("return_pct", 0) > 0)
    display_name = get_stock_display(symbol)
    title = (f'{display_name} — {STRATEGY_CN.get(strategy_name, strategy_name)} '
             f'({period_label}) | 勝/總: {wins}/{len(trades)}')

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.40, 0.12, 0.16, 0.16, 0.16],
        subplot_titles=["", "成交量", "RSI", "KD", "MACD"],
    )

    # ====== Row 1: K 線 + 均線 + 布林帶 ======

    # 自訂 hover 文字 (日期 + 開高低收 + 漲跌幅 + 量 + 指標)
    hover_texts = []
    for i, row in chart_df.iterrows():
        chg = row.get('change_pct', 0)
        chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
        chg_icon = "🟢" if chg >= 0 else "🔴"
        vol_str = f"{row['volume']/1000:.0f}張" if row['volume'] < 1e8 else f"{row['volume']/1e8:.1f}億"
        lines = [
            f"<b>{i.strftime('%Y-%m-%d')} ({['一','二','三','四','五','六','日'][i.weekday()]})</b>",
            f"{chg_icon} 漲跌: {chg_str}",
            f"開: {row['open']:.2f}  高: {row['high']:.2f}",
            f"低: {row['low']:.2f}  收: <b>{row['close']:.2f}</b>",
            f"量: {vol_str}",
            f"─────────",
            f"MA5: {row['MA5']:.2f}" if pd.notna(row.get('MA5')) else "",
            f"MA20: {row['MA20']:.2f}" if pd.notna(row.get('MA20')) else "",
            f"MA60: {row['MA60']:.2f}" if pd.notna(row.get('MA60')) else "",
            f"RSI: {row['RSI']:.1f}" if pd.notna(row.get('RSI')) else "",
            f"K: {row['K']:.1f} D: {row['D']:.1f}" if pd.notna(row.get('K')) else "",
            f"MACD: {row['MACD']:.3f}" if pd.notna(row.get('MACD')) else "",
            f"BB: {row['BB_lower']:.1f} ~ {row['BB_upper']:.1f}" if pd.notna(row.get('BB_lower')) else "",
        ]
        hover_texts.append("<br>".join([l for l in lines if l]))

    # K 線 (Candlestick)
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['open'], high=chart_df['high'],
        low=chart_df['low'], close=chart_df['close'],
        increasing_line_color='#ef5350', increasing_fillcolor='#ef5350',  # 台股: 紅漲
        decreasing_line_color='#26a69a', decreasing_fillcolor='#26a69a',  # 綠跌
        name='K線',
        text=hover_texts,
        hoverinfo='text',
    ), row=1, col=1)

    # 均線
    ma_configs = [
        ('MA5', '#ff9800', 1.0),
        ('MA10', '#4caf50', 1.0),
        ('MA20', '#2196f3', 1.2),
        ('MA60', '#9c27b0', 1.5),
    ]
    for col_name, color, width in ma_configs:
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df[col_name],
            mode='lines', name=col_name,
            line=dict(color=color, width=width),
            hovertemplate=f'{col_name}: %{{y:.2f}}<extra></extra>',
        ), row=1, col=1)

    # 布林帶 (填充區域)
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['BB_upper'],
        mode='lines', name='BB上軌',
        line=dict(color='rgba(33,150,243,0.3)', width=1, dash='dash'),
        hovertemplate='BB上軌: %{y:.2f}<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['BB_lower'],
        mode='lines', name='BB下軌', fill='tonexty',
        line=dict(color='rgba(33,150,243,0.3)', width=1, dash='dash'),
        fillcolor='rgba(33,150,243,0.06)',
        hovertemplate='BB下軌: %{y:.2f}<extra></extra>',
    ), row=1, col=1)

    # 買賣標記
    for t in trades:
        buy_dt = pd.Timestamp(t['buy_date'])
        sell_dt = pd.Timestamp(t['sell_date'])
        ret = t.get('return_pct', 0)

        if buy_dt >= chart_df.index[0]:
            fig.add_trace(go.Scatter(
                x=[buy_dt], y=[t['buy_price']],
                mode='markers+text', text=['買'],
                textposition='bottom center',
                marker=dict(symbol='triangle-up', size=14, color='#ff1744',
                            line=dict(width=1, color='white')),
                name='',
                hovertemplate=(
                    f"<b>🔺 買入</b><br>"
                    f"日期: {buy_dt.strftime('%Y-%m-%d')}<br>"
                    f"價格: {t['buy_price']:.2f}<br>"
                    f"理由: {t.get('buy_reason', '')}<extra></extra>"
                ),
                showlegend=False,
            ), row=1, col=1)

        if sell_dt >= chart_df.index[0]:
            sell_color = '#26a69a' if ret > 0 else '#ef5350'
            sell_icon = "🟢" if ret > 0 else "🔴"
            fig.add_trace(go.Scatter(
                x=[sell_dt], y=[t['sell_price']],
                mode='markers+text', text=['賣'],
                textposition='top center',
                marker=dict(symbol='triangle-down', size=14, color=sell_color,
                            line=dict(width=1, color='white')),
                name='',
                hovertemplate=(
                    f"<b>🔻 賣出</b> {sell_icon}<br>"
                    f"日期: {sell_dt.strftime('%Y-%m-%d')}<br>"
                    f"價格: {t['sell_price']:.2f}<br>"
                    f"報酬: {ret:+.2%}<br>"
                    f"持有: {t.get('holding_days', 0)}天<br>"
                    f"理由: {t.get('sell_reason', '')}<extra></extra>"
                ),
                showlegend=False,
            ), row=1, col=1)

    # ====== Row 2: 成交量 ======
    vol_colors = ['#ef5350' if c >= o else '#26a69a'
                  for c, o in zip(chart_df['close'], chart_df['open'])]
    fig.add_trace(go.Bar(
        x=chart_df.index, y=chart_df['volume'],
        marker_color=vol_colors, opacity=0.6, name='成交量',
        hovertemplate='量: %{y:,.0f}<extra></extra>',
    ), row=2, col=1)

    # ====== Row 3: RSI ======
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['RSI'],
        mode='lines', name='RSI(14)',
        line=dict(color='#ff9800', width=1.5),
        hovertemplate='RSI: %{y:.1f}<extra></extra>',
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", line_width=0.8,
                  annotation_text="超買 70", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", line_width=0.8,
                  annotation_text="超賣 30", row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.04, row=3, col=1)

    # ====== Row 4: KD ======
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['K'],
        mode='lines', name='K(9)',
        line=dict(color='#2196f3', width=1.5),
        hovertemplate='K: %{y:.1f}<extra></extra>',
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['D'],
        mode='lines', name='D(9)',
        line=dict(color='#ff9800', width=1.5),
        hovertemplate='D: %{y:.1f}<extra></extra>',
    ), row=4, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="#ef5350", line_width=0.8,
                  annotation_text="超買 80", row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="#26a69a", line_width=0.8,
                  annotation_text="超賣 20", row=4, col=1)

    # ====== Row 5: MACD ======
    macd_colors = ['#ef5350' if v >= 0 else '#26a69a' for v in chart_df['MACD_hist'].fillna(0)]
    fig.add_trace(go.Bar(
        x=chart_df.index, y=chart_df['MACD_hist'],
        marker_color=macd_colors, opacity=0.6, name='MACD柱',
        hovertemplate='柱: %{y:.3f}<extra></extra>',
    ), row=5, col=1)
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['MACD'],
        mode='lines', name='DIF',
        line=dict(color='#2196f3', width=1.2),
        hovertemplate='DIF: %{y:.3f}<extra></extra>',
    ), row=5, col=1)
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['MACD_signal'],
        mode='lines', name='DEA',
        line=dict(color='#ff9800', width=1.2),
        hovertemplate='DEA: %{y:.3f}<extra></extra>',
    ), row=5, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5, row=5, col=1)

    # ====== 全局版面配置 ======
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=900,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01,
            xanchor='left', x=0, font=dict(size=10),
        ),
        margin=dict(l=60, r=20, t=80, b=40),
    )

    # Y 軸標籤
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="量", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="KD", range=[0, 100], row=4, col=1)
    fig.update_yaxes(title_text="MACD", row=5, col=1)

    # X 軸格式
    fig.update_xaxes(
        type='date',
        tickformat='%Y-%m',
        dtick='M3',
        row=5, col=1,
    )

    # 十字游標
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                     spikethickness=0.5, spikecolor='gray', spikedash='dot')
    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor',
                     spikethickness=0.5, spikecolor='gray', spikedash='dot')

    return fig


# ---------------------------------------------------------------------------
# Detect recent buy signals
# ---------------------------------------------------------------------------
def detect_recent_signals(stock_data, strategy_name, params, lookback_days=10):
    signals = []
    strategy_def = get_strategy(strategy_name)
    for symbol, df in stock_data.items():
        try:
            sig_df = strategy_def.func(df, **params)
            recent = sig_df.tail(lookback_days)
            for i in range(len(recent)):
                row = recent.iloc[i]
                if row.get('signal', 0) == 1:
                    date = row.get('date', recent.index[i])
                    signals.append({
                        'symbol': symbol,
                        'date': date,
                        'price': round(float(row['close']), 2),
                        'reason': _get_signal_reason(row),
                    })
        except Exception:
            pass
    return signals


def _get_signal_reason(row):
    if 'rsi' in row.index and not pd.isna(row.get('rsi', np.nan)):
        return f"RSI = {row['rsi']:.1f}"
    if 'macd_hist' in row.index and not pd.isna(row.get('macd_hist', np.nan)):
        return f"MACD Histogram = {row['macd_hist']:.3f}"
    if 'sma_fast' in row.index and 'sma_slow' in row.index:
        return f"SMA Fast({row['sma_fast']:.1f}) > Slow({row['sma_slow']:.1f})"
    if 'ema_fast' in row.index and 'ema_slow' in row.index:
        return f"EMA Fast({row['ema_fast']:.1f}) > Slow({row['ema_slow']:.1f})"
    if 'bb_lower' in row.index and not pd.isna(row.get('bb_lower', np.nan)):
        return f"Price < BB Lower({row['bb_lower']:.1f})"
    if 'k' in row.index and 'd' in row.index:
        return f"KDJ K={row['k']:.1f} D={row['d']:.1f}"
    if 'vol_ma' in row.index:
        return "Volume breakout + Price > MA"
    return "Buy signal triggered"


# ---------------------------------------------------------------------------
# Sell signal detection
# ---------------------------------------------------------------------------
def detect_sell_signals(symbol, df, strategies_to_check=None):
    """
    偵測接近賣點的訊號，回傳 list of {strategy, reason, urgency}
    urgency: 'high'=已觸發賣出, 'medium'=接近賣出, 'low'=注意
    """
    alerts = []
    if strategies_to_check is None:
        strategies_to_check = list(STRATEGY_REGISTRY.keys())

    for strat_name in strategies_to_check:
        try:
            sdef = STRATEGY_REGISTRY[strat_name]
            params = sdef.default_params()
            sig_df = sdef.func(df, **params)
            if sig_df.empty:
                continue
            last = sig_df.iloc[-1]
            prev = sig_df.iloc[-2] if len(sig_df) > 1 else last

            # 已觸發賣出訊號
            if last.get('signal', 0) == -1:
                alerts.append({
                    'strategy': STRATEGY_CN.get(strat_name, strat_name),
                    'reason': _get_sell_reason(strat_name, last),
                    'urgency': 'high',
                })
                continue

            # 接近賣出條件 (預警)
            warn = _check_near_sell(strat_name, last, prev)
            if warn:
                alerts.append({
                    'strategy': STRATEGY_CN.get(strat_name, strat_name),
                    'reason': warn,
                    'urgency': 'medium',
                })
        except Exception:
            pass

    # 通用技術面警告
    try:
        close = float(df['close'].iloc[-1])
        # RSI 超買
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss_s = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss_s.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        if rsi > 75:
            alerts.append({
                'strategy': '技術面總覽',
                'reason': f'RSI={rsi:.1f} 進入超買區 (>75)，注意回檔風險',
                'urgency': 'medium',
            })
        # 跌破 20 日均線
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        if close < ma20:
            alerts.append({
                'strategy': '技術面總覽',
                'reason': f'股價 {close:.2f} 跌破 20 日均線 {ma20:.2f}',
                'urgency': 'low',
            })
        # 量能萎縮
        vol = df['volume'].iloc[-1]
        vol_ma = df['volume'].rolling(20).mean().iloc[-1]
        if vol_ma > 0 and vol / vol_ma < 0.5:
            alerts.append({
                'strategy': '技術面總覽',
                'reason': f'成交量僅均量 {vol/vol_ma:.0%}，量能萎縮',
                'urgency': 'low',
            })
    except Exception:
        pass

    return alerts


def _get_sell_reason(strat_name, row):
    if 'rsi' in row.index and pd.notna(row.get('rsi', np.nan)):
        return f"RSI={row['rsi']:.1f} 超買，觸發賣出"
    if 'k' in row.index and pd.notna(row.get('k', np.nan)):
        return f"KD K={row['k']:.1f} 死亡交叉，觸發賣出"
    if 'macd_hist' in row.index and pd.notna(row.get('macd_hist', np.nan)):
        return f"MACD 柱狀圖轉負 ({row['macd_hist']:.3f})，觸發賣出"
    return "策略賣出訊號觸發"


def _check_near_sell(strat_name, last, prev):
    # RSI 類策略: RSI > 65 且仍在上升
    if 'rsi' in last.index and pd.notna(last.get('rsi', np.nan)):
        rsi = last['rsi']
        if 65 < rsi < 75:
            return f"RSI={rsi:.1f} 接近超買區 (70)，留意賣出時機"
    # KD 類策略: K 值 > 75
    if 'k' in last.index and pd.notna(last.get('k', np.nan)):
        k = last['k']
        d = last.get('d', 0)
        if k > 75 and k < d:
            return f"KD K={k:.1f} D={d:.1f}，K 值在高檔向下彎頭"
    # MACD: 柱狀圖縮小
    if 'macd_hist' in last.index and 'macd_hist' in prev.index:
        cur = last.get('macd_hist', 0)
        pre = prev.get('macd_hist', 0)
        if pd.notna(cur) and pd.notna(pre) and cur > 0 and cur < pre:
            return f"MACD 柱狀圖縮小 ({pre:.3f}→{cur:.3f})，多方動能減弱"
    return None


# ---------------------------------------------------------------------------
# 載入已儲存策略 → 直接重跑 or 繼續優化
# ---------------------------------------------------------------------------
_reload_strat = st.session_state.pop('_reload_strategy', None)
_reload_opt = st.session_state.pop('_reload_optimize', None)
_reload_syms = st.session_state.pop('_reload_symbols', None)

if _reload_strat and _reload_syms:
    st.markdown("---")
    st.markdown("## 📂 載入已儲存策略 — 重新回測")
    _r_symbols = _reload_syms
    _r_symbols_display = ", ".join([get_stock_display(s) for s in _r_symbols])
    st.success(f"**{_reload_strat['strategy_cn']}** → {_r_symbols_display}")
    _r_stock_data = {}
    for sym in _r_symbols:
        try:
            _r_df = yf.download(sym, start=start_date.strftime("%Y-%m-%d"),
                                end=end_date.strftime("%Y-%m-%d"), progress=False)
            if len(_r_df) > 0:
                _r_df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in _r_df.columns]
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in _r_df.columns:
                        _r_df[col] = pd.to_numeric(_r_df[col], errors='coerce')
                _r_df = _r_df.dropna(subset=['close'])
                _r_stock_data[sym] = _r_df
        except Exception:
            pass

    if _r_stock_data:
        _r_strategy_name = _reload_strat['strategy']
        _r_params = _reload_strat['params']
        _r_strategy_def = get_strategy(_r_strategy_name)

        st.markdown("### 📊 載入回測結果")
        st.markdown(f"**策略**: {_reload_strat['strategy_cn']} | **參數**: {_r_params}")

        for sym, _r_df in _r_stock_data.items():
            try:
                bt = run_backtest(
                    _r_df, _r_strategy_def.func, _r_strategy_name, _r_params,
                    symbol=sym,
                    commission_rate=commission, tax_rate=tax, slippage=0.001,
                    stop_loss=stop_loss_pct / 100.0 if stop_loss_pct > 0 else 0.0,
                    take_profit=take_profit_pct / 100.0 if take_profit_pct > 0 else 0.0,
                    max_hold_days=max_hold if max_hold > 0 else 0,
                    trailing_stop=trailing_stop_pct / 100.0 if trailing_stop_pct > 0 else 0.0,
                    partial_exit=partial_exit,
                    partial_exit_last_mode=partial_last_mode,
                )
                disp = get_stock_display(sym)
                total_pnl = sum(t.get('profit', 0) for t in bt.trades)
                pnl_icon = "🟢" if total_pnl >= 0 else "🔴"
                st.markdown(
                    f"**{disp}** | 交易數: {bt.num_trades} | "
                    f"勝率: {bt.win_rate:.1%} | 報酬: {bt.total_return:.1%} | "
                    f"夏普: {bt.sharpe_ratio:.2f} | {pnl_icon} 損益: {total_pnl:+,.0f}"
                )

                # 交易明細
                if bt.trades:
                    with st.expander(f"📋 {disp} 交易明細 ({bt.num_trades} 筆)", expanded=False):
                        trade_rows = []
                        # 計算總張數 (百萬本金)
                        total_lots = max(1, int(1_000_000 / (bt.trades[0]['buy_price'] * 1000))) if bt.trades[0]['buy_price'] > 0 else 1
                        for idx, t in enumerate(bt.trades, 1):
                            ratio = t.get('sell_ratio', 1.0)
                            lots = max(1, round(total_lots * ratio))
                            total_shares = lots * 1000
                            buy_amount = round(t['buy_price'] * total_shares)
                            sell_amount = round(t['sell_price'] * total_shares)
                            pnl = sell_amount - buy_amount
                            ratio_label = f" ({ratio:.0%})" if ratio < 1.0 else ""
                            trade_rows.append({
                                "#": idx,
                                "買入日期": t['buy_date'],
                                "買入價": t['buy_price'],
                                "買入理由": t.get('buy_reason', ''),
                                "張數": f"{lots}{ratio_label}",
                                "買入金額": f"{buy_amount:,}",
                                "賣出日期": t['sell_date'],
                                "賣出價": t['sell_price'],
                                "賣出理由": t.get('sell_reason', ''),
                                "賣出金額": f"{sell_amount:,}",
                                "報酬%": f"{t['return_pct']:.2%}",
                                "損益": f"{pnl:+,}",
                                "持有天數": t.get('holding_days', 0),
                                "結果": t['result'],
                            })
                        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

                    # K 線圖
                    kl1, kl2 = st.columns(2)
                    with kl1:
                        if st.button(f"📊 日K線圖", key=f"reload_daily_{sym}"):
                            fig = draw_kline_chart(sym, _r_df, bt.signal_df, bt.trades, _r_params, _r_strategy_name, "daily")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    with kl2:
                        if st.button(f"📊 週K線圖", key=f"reload_weekly_{sym}"):
                            fig = draw_kline_chart(sym, _r_df, bt.signal_df, bt.trades, _r_params, _r_strategy_name, "weekly")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"{sym}: {e}")

        # 提供操作按鈕
        st.markdown("---")
        _rc1, _rc2, _rc3 = st.columns(3)
        with _rc1:
            if st.button("🔬 以此策略繼續優化", key="reload_to_opt", use_container_width=True):
                st.session_state['_reload_optimize'] = _reload_strat.copy()
                st.session_state['_reload_symbols'] = list(_reload_syms)
                st.session_state['_input_override'] = ", ".join(_reload_syms)
                st.rerun()
        with _rc2:
            if st.button("💾 重新儲存", key="reload_resave", use_container_width=True):
                key_label = ", ".join(_reload_syms)
                st.session_state.saved_strategies[key_label] = _reload_strat.copy()
                st.session_state.saved_strategies[key_label]['saved_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                _save_json(_STRATEGIES_FILE, st.session_state.saved_strategies)
                st.success("已重新儲存!")
        with _rc3:
            if st.button("🏠 回到主頁", key="reload_home", use_container_width=True):
                st.rerun()

        st.stop()  # 停止渲染，不顯示下方舊內容

    else:
        st.error("無法下載股票資料")
        st.stop()

elif _reload_opt and _reload_syms:
    st.info(
        f"🔬 以 **{_reload_opt['strategy_cn']}** 為起點繼續優化 — "
        f"股票: {', '.join([get_stock_display(s) for s in _reload_syms])}"
    )
    # 設定 session state，讓下面的 go_btn 邏輯自動啟動
    st.session_state['_auto_optimize'] = True
    st.session_state['_auto_opt_data'] = _reload_opt
    st.session_state['_auto_opt_symbols'] = _reload_syms

# ---------------------------------------------------------------------------
# Run optimization (手動觸發 or 載入繼續優化)
# ---------------------------------------------------------------------------
_auto_optimize = st.session_state.pop('_auto_optimize', False)
_auto_opt_data = st.session_state.pop('_auto_opt_data', None)
_auto_opt_symbols = st.session_state.pop('_auto_opt_symbols', None)

if go_btn or _auto_optimize:
    if _auto_optimize and _auto_opt_symbols:
        symbols = _auto_opt_symbols
        symbol_display = {sym: get_stock_display(sym) for sym in symbols}
    else:
        resolved = resolve_symbols_input(symbols_input)
        symbols = [sym for sym, _ in resolved]
        symbol_display = {sym: disp for sym, disp in resolved}

    if not symbols:
        st.error("請輸入至少一個股票代碼或名稱")
        st.stop()

    # 初始化 session 的顯示名稱對照
    if "symbol_display" not in st.session_state:
        st.session_state.symbol_display = {}
    st.session_state.symbol_display.update(symbol_display)

    progress_bar = st.progress(0, text="正在下載股票資料...")

    stock_data = {}
    for i, symbol in enumerate(symbols):
        try:
            df = yf.download(
                symbol,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            if len(df) >= 30:
                df["date"] = df.index
                stock_data[symbol] = df
            progress_bar.progress(
                (i + 1) / len(symbols) * 0.1,
                text=f"已下載 {get_stock_display(symbol)} ({len(df)} 天)",
            )
        except Exception as e:
            st.warning(f"{symbol} 下載失敗: {e}")

    if not stock_data:
        st.error("無法下載任何股票資料，請確認代碼是否正確")
        st.stop()

    st.session_state.stock_data = stock_data
    progress_bar.progress(0.1, text=f"已載入 {len(stock_data)}/{len(symbols)} 檔 -> 開始最佳化...")

    # 如果是從載入的策略繼續優化，優先使用該策略作為起點
    _opt_strategies = selected_strategies if selected_strategies else None
    if _auto_optimize and _auto_opt_data:
        _loaded_strat = _auto_opt_data.get('strategy')
        if _loaded_strat:
            st.info(f"🔬 以 **{_auto_opt_data.get('strategy_cn', _loaded_strat)}** 為起點，搜尋全部 20 策略")

    config = OptimizationConfig(
        primary_metric=primary_metric,
        min_trades=min_trades,
        results_file="backtest_results.tsv",
        strategies=_opt_strategies,
        max_combos_per_strategy=0,   # 不限制參數組合數
        time_budget_seconds=time_budget if time_budget > 0 else 0,
        patience=50,
        neighbors_per_step=8,
        commission_rate=commission,
        tax_rate=tax,
        slippage=0.001,
        stop_loss=stop_loss_pct / 100.0 if stop_loss_pct > 0 else 0.0,
        take_profit=take_profit_pct / 100.0 if take_profit_pct > 0 else 0.0,
        max_hold_days=max_hold if max_hold > 0 else 0,
        trailing_stop=trailing_stop_pct / 100.0 if trailing_stop_pct > 0 else 0.0,
        partial_exit=partial_exit,
        partial_exit_last_mode=partial_last_mode,
    )

    optimizer = BacktestOptimizer(stock_data, config)
    original_try = optimizer._try_experiment
    logs = []

    def _patched_try(strategy_name, params, description=""):
        try:
            res = original_try(strategy_name, params, description)
        except BrokenPipeError:
            return False
        entry = optimizer.results_log[-1] if optimizer.results_log else {}
        logs.append(entry)
        n = len(logs)
        pct = min(0.1 + 0.85 * (n / max(n + 40, 80)), 0.95)
        wr = entry.get("win_rate", 0)
        try:
            progress_bar.progress(pct, text=f"實驗 #{n} -- {STRATEGY_CN.get(strategy_name, strategy_name)} -- 勝率 {wr:.1%}")
        except Exception:
            pass
        return res

    optimizer._try_experiment = _patched_try

    t0 = time.time()
    result = optimizer.optimize()
    elapsed = time.time() - t0

    progress_bar.progress(1.0, text=f"完成! 共 {result.total_experiments} 次實驗，耗時 {elapsed:.0f} 秒")

    # Re-run best strategy on each stock to get detailed trades
    # 重要: 必須傳入與最佳化相同的風險管理參數，否則交易明細不一致
    best_strat_def = get_strategy(result.best_strategy)
    per_stock_bt = {}
    _sl = stop_loss_pct / 100.0 if stop_loss_pct > 0 else 0.0
    _tp = take_profit_pct / 100.0 if take_profit_pct > 0 else 0.0
    _mh = max_hold if max_hold > 0 else 0
    _ts = trailing_stop_pct / 100.0 if trailing_stop_pct > 0 else 0.0
    for symbol, df in stock_data.items():
        try:
            bt = run_backtest(
                df, best_strat_def.func, result.best_strategy,
                result.best_params, symbol=symbol,
                commission_rate=commission, tax_rate=tax, slippage=0.001,
                stop_loss=_sl, take_profit=_tp, max_hold_days=_mh,
                trailing_stop=_ts, partial_exit=partial_exit,
                partial_exit_last_mode=partial_last_mode,
            )
            per_stock_bt[symbol] = bt
        except Exception:
            pass

    st.session_state.result = result
    st.session_state.logs = logs
    st.session_state.per_stock_bt = per_stock_bt
    st.session_state.best_strategy_name = result.best_strategy
    st.session_state.best_params = result.best_params
    st.rerun()


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
if st.session_state.result:
    result = st.session_state.result
    logs = st.session_state.logs
    per_stock_bt = st.session_state.per_stock_bt
    stock_data = st.session_state.stock_data

    st.markdown("---")

    # ====== 最佳策略 ======
    # 使用 per_stock_bt (含正確風控參數) 重新計算統計數據，確保一致性
    st.markdown("### 最佳策略")

    _valid_bt = [bt for bt in per_stock_bt.values() if bt.num_trades > 0]
    _display_win_rate = np.mean([bt.win_rate for bt in _valid_bt]) if _valid_bt else 0.0
    _display_return = np.mean([bt.total_return for bt in _valid_bt]) if _valid_bt else 0.0
    _display_sharpe = np.mean([bt.sharpe_ratio for bt in _valid_bt]) if _valid_bt else 0.0
    _display_trades = sum(bt.num_trades for bt in _valid_bt)

    c1, c2, c3, c4, c5 = st.columns(5)
    _strat_label = STRATEGY_CN.get(result.best_strategy, result.best_strategy)
    c1.markdown(f"""<div class="result-card">
        <div class="label">策略</div>
        <div class="value" style="font-size:1.3rem;">{_strat_label}</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="result-card">
        <div class="label">平均勝率</div>
        <div class="value win">{_display_win_rate:.1%}</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="result-card">
        <div class="label">總報酬 (加總)</div>
        <div class="value {'win' if _display_return >= 0 else 'lose'}">{_display_return:.1%}</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="result-card">
        <div class="label">Sharpe Ratio</div>
        <div class="value">{_display_sharpe:.2f}</div>
    </div>""", unsafe_allow_html=True)
    c5.markdown(f"""<div class="result-card">
        <div class="label">總交易次數</div>
        <div class="value">{_display_trades}</div>
    </div>""", unsafe_allow_html=True)

    # ====== 儲存 / 載入 / 觀察名單 ======
    save_col1, save_col2, save_col3 = st.columns(3)

    with save_col1:
        if st.button("💾 儲存最佳策略", use_container_width=True, key="btn_save_strat"):
            symbols_tested = list(stock_data.keys()) if stock_data else []
            key_label = ", ".join(symbols_tested) if symbols_tested else "未知"
            st.session_state.saved_strategies[key_label] = {
                'strategy': result.best_strategy,
                'strategy_cn': STRATEGY_CN.get(result.best_strategy, result.best_strategy),
                'params': {k: float(v) for k, v in result.best_params.items()},
                'win_rate': float(result.best_win_rate),
                'sharpe': float(result.best_sharpe),
                'total_return': float(result.best_total_return),
                'num_trades': int(result.best_num_trades),
                'symbols': symbols_tested,
                'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            }
            _save_json(_STRATEGIES_FILE, st.session_state.saved_strategies)
            st.success(f"已儲存 [{key_label}] 的最佳策略!")

    with save_col2:
        saved_keys = list(st.session_state.saved_strategies.keys())
        if saved_keys:
            load_key = st.selectbox("載入已儲存策略", options=[""] + saved_keys,
                                     format_func=lambda x: "選擇..." if x == "" else x,
                                     key="sel_load_strat", label_visibility="collapsed")
            if load_key and load_key in st.session_state.saved_strategies:
                loaded = st.session_state.saved_strategies[load_key]
                st.info(
                    f"**{loaded['strategy_cn']}** | "
                    f"勝率 {loaded['win_rate']:.0%} | "
                    f"夏普 {loaded['sharpe']:.2f} | "
                    f"參數: {loaded['params']}"
                )
                # 載入操作按鈕
                load_action_col1, load_action_col2 = st.columns(2)
                with load_action_col1:
                    if st.button("🔄 以此策略重新回測", key="btn_reload_bt", use_container_width=True):
                        _syms = loaded.get('symbols', [])
                        st.session_state['_reload_strategy'] = loaded
                        st.session_state['_reload_symbols'] = _syms
                        st.session_state['_input_override'] = ", ".join(_syms)
                        st.rerun()
                with load_action_col2:
                    if st.button("🔬 以此為基礎繼續優化", key="btn_reload_opt", use_container_width=True):
                        _syms = loaded.get('symbols', [])
                        st.session_state['_reload_optimize'] = loaded
                        st.session_state['_reload_symbols'] = _syms
                        st.session_state['_input_override'] = ", ".join(_syms)
                        st.rerun()
        else:
            st.caption("尚無儲存的策略")

    with save_col3:
        if st.button("⭐ 加入觀察名單", use_container_width=True, key="btn_add_to_wl"):
            symbols_tested = list(stock_data.keys()) if stock_data else []
            for sym in symbols_tested:
                # 避免重複
                existing = [w['symbol'] for w in st.session_state.watchlist]
                if sym not in existing:
                    st.session_state.watchlist.append({
                        'symbol': sym,
                        'strategy_cn': STRATEGY_CN.get(result.best_strategy, result.best_strategy),
                        'win_rate': float(result.best_win_rate),
                        'added_date': datetime.now().strftime('%Y-%m-%d'),
                    })
            _save_json(_WATCHLIST_FILE, st.session_state.watchlist)
            st.success(f"已將 {', '.join(symbols_tested)} 加入觀察名單!")

    # ====== 策略參數 ======
    st.markdown("### 策略與操作參數")
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

    # ====== 策略量化說明 ======
    _bp = result.best_params
    _bs = result.best_strategy
    strategy_logic_lines = []

    if _bs == 'sma_crossover':
        strategy_logic_lines = [
            f"📈 **買入**: 短均線 SMA({int(_bp.get('fast_period',10))}) 上穿 長均線 SMA({int(_bp.get('slow_period',30))})",
            f"📉 **賣出**: 短均線 SMA({int(_bp.get('fast_period',10))}) 下穿 長均線 SMA({int(_bp.get('slow_period',30))})",
        ]
    elif _bs == 'ema_crossover':
        strategy_logic_lines = [
            f"📈 **買入**: EMA({int(_bp.get('fast_period',12))}) 上穿 EMA({int(_bp.get('slow_period',26))})",
            f"📉 **賣出**: EMA({int(_bp.get('fast_period',12))}) 下穿 EMA({int(_bp.get('slow_period',26))})",
        ]
    elif _bs == 'rsi':
        strategy_logic_lines = [
            f"📈 **買入**: RSI({int(_bp.get('period',14))}) < {int(_bp.get('oversold',30))} (超賣)",
            f"📉 **賣出**: RSI({int(_bp.get('period',14))}) > {int(_bp.get('overbought',70))} (超買)",
        ]
    elif _bs == 'macd':
        strategy_logic_lines = [
            f"📈 **買入**: MACD({int(_bp.get('fast',12))},{int(_bp.get('slow',26))}) 柱狀圖由負轉正 (signal={int(_bp.get('signal_period',9))})",
            f"📉 **賣出**: MACD 柱狀圖由正轉負",
        ]
    elif _bs == 'bollinger_bands':
        strategy_logic_lines = [
            f"📈 **買入**: 收盤價 < 布林下軌 (MA{int(_bp.get('period',20))} - {_bp.get('num_std',2.0):.1f}σ)",
            f"📉 **賣出**: 收盤價 > 布林上軌 (MA{int(_bp.get('period',20))} + {_bp.get('num_std',2.0):.1f}σ)",
            f"💡 布林帶寬 = {_bp.get('num_std',2.0):.2f} 倍標準差，週期 {int(_bp.get('period',20))} 日",
        ]
    elif _bs == 'kdj':
        strategy_logic_lines = [
            f"📈 **買入**: K({int(_bp.get('k_period',9))}) 值 < {int(_bp.get('oversold',20))} 且 K 上穿 D",
            f"📉 **賣出**: K 值 > {int(_bp.get('overbought',80))} 且 K 下穿 D",
        ]
    elif _bs == 'volume_price':
        strategy_logic_lines = [
            f"📈 **買入**: 成交量 > {_bp.get('vol_multiplier',1.5):.1f}× 均量(MA{int(_bp.get('ma_period',20))}) 且 股價站上均線",
            f"📉 **賣出**: 股價跌破均線",
        ]
    elif _bs == 'dual_ma_rsi':
        strategy_logic_lines = [
            f"📈 **買入**: SMA({int(_bp.get('fast_period',10))}) 上穿 SMA({int(_bp.get('slow_period',30))}) 且 RSI({int(_bp.get('rsi_period',14))}) < {int(_bp.get('rsi_threshold',60))}",
            f"📉 **賣出**: SMA 短線下穿長線",
        ]
    elif _bs == 'kd_macd_signal':
        strategy_logic_lines = [
            f"📈 **買入**: KD K < {int(_bp.get('kd_oversold',25))} 黃金交叉 + MACD柱 > 0",
            f"📉 **賣出**: KD K > {int(_bp.get('kd_overbought',75))} 死亡交叉 或 MACD柱翻負",
        ]
    elif _bs == 'ma_breakout':
        strategy_logic_lines = [
            f"📈 **買入**: 收盤價突破 MA{int(_bp.get('ma_period',60))}",
            f"📉 **賣出**: 收盤價跌破 MA{int(_bp.get('ma_period',60))}",
        ]
    elif _bs == 'vol_price_sync':
        strategy_logic_lines = [
            f"📈 **買入**: 量增價漲，成交量 > {_bp.get('vol_ratio',1.5):.1f}× 均量 且 連續上漲",
            f"📉 **賣出**: 量縮價跌",
        ]
    elif _bs == 'vol_breakout_vcp':
        strategy_logic_lines = [
            f"📈 **買入**: 量縮整理後放量突破 (量比 > {_bp.get('breakout_vol_ratio',2.0):.1f}×)",
            f"📉 **賣出**: 跌破突破點均線",
        ]
    elif _bs == 'vol_divergence':
        strategy_logic_lines = [
            f"📈 **買入**: 股價創新低但 OBV 未創新低 (量價背離反轉)",
            f"📉 **賣出**: 股價創新高但 OBV 未跟進",
        ]
    elif _bs == 'vol_dry_bottom':
        strategy_logic_lines = [
            f"📈 **買入**: 量縮至均量 {_bp.get('vol_shrink',0.5):.0%} 以下 + KD K < {int(_bp.get('kd_oversold',25))} 超賣",
            f"📉 **賣出**: KD K > {int(_bp.get('kd_overbought',75))} 超買",
        ]
    elif _bs == 'chip_sedimentation':
        strategy_logic_lines = [
            f"📈 **買入**: 籌碼沉澱整理 {int(_bp.get('consolidation_days',20))} 日後放量突破",
            f"📉 **賣出**: 跌破整理區間低點",
        ]
    elif _bs == 'multi_factor':
        strategy_logic_lines = [
            f"📈 **買入**: 動量排名前列 + RSI({int(_bp.get('rsi_period',14))}) 超賣區",
            f"📉 **賣出**: 動量轉弱 或 RSI 超買",
        ]
    elif _bs == 'foreign_follow':
        strategy_logic_lines = [
            f"📈 **買入**: 量能放大 > {_bp.get('vol_threshold',1.5):.1f}× 均量模擬外資進場",
            f"📉 **賣出**: 量能萎縮跌破均線",
        ]
    elif _bs == 'etf_momentum':
        strategy_logic_lines = [
            f"📈 **買入**: {int(_bp.get('momentum_period',20))} 日動量為正",
            f"📉 **賣出**: 動量轉負",
        ]
    else:
        strategy_logic_lines = [f"ℹ️ 策略 `{_bs}` — 詳見策略原始碼"]

    if strategy_logic_lines:
        st.markdown("#### 📋 策略進出場規則")
        for line in strategy_logic_lines:
            st.markdown(line)

        # 顯示賣出策略設定
        st.markdown("#### 📊 賣出策略 (買賣分離)")
        sell_lines = []
        sell_lines.append(f"**買入**: 由上方策略產生訊號 (signal=1)")
        sell_lines.append(f"**賣出**: 多層賣出策略 (按優先順序):")
        if stop_loss_pct > 0:
            sell_lines.append(f"  1. ⛔ **停損**: 跌幅達 {stop_loss_pct}% 強制賣出")
        if trailing_stop_pct > 0:
            sell_lines.append(f"  2. 📉 **移動停利**: 從最高點回落 {trailing_stop_pct}% 賣出")
        if take_profit_pct > 0:
            sell_lines.append(f"  3. 🎯 **固定停利**: 漲幅達 {take_profit_pct}% 強制賣出")
        if max_hold > 0:
            sell_lines.append(f"  4. ⏰ **最大持有**: {max_hold} 天強制出場")
        sell_lines.append(f"  5. 📋 **策略訊號**: 原始策略的賣出訊號 (signal=-1)")
        if partial_exit:
            if partial_last_mode == "ma5":
                sell_lines.append(f"  📊 **分批出場**: 漲10%出1/3 → 漲20%出1/3 → 最後1/3 **破5日均線**賣出")
            else:
                sell_lines.append(f"  📊 **分批出場**: 漲10%出1/3 → 漲20%出1/3 → 最後1/3 **移動停利**賣出")
        for line in sell_lines:
            st.markdown(line)
        if not any([stop_loss_pct > 0, trailing_stop_pct > 0, take_profit_pct > 0, max_hold > 0]):
            st.warning("⚠️ 未啟用任何風險管理，僅靠策略訊號賣出")

    # ====== 交易明細 ======
    st.markdown("### 交易明細")

    if per_stock_bt:
        stock_tabs = st.tabs([get_stock_display(s) for s in per_stock_bt.keys()])

        for tab, (symbol, bt) in zip(stock_tabs, per_stock_bt.items()):
            with tab:
                if bt.trades:
                    trade_rows = []
                    cumulative_pnl = 0  # 累積損益 (元)
                    # 計算初始總張數 (百萬本金)
                    _first_buy = bt.trades[0]['buy_price'] if bt.trades else 1
                    _base_lots = max(1, int(1_000_000 / (_first_buy * 1000))) if _first_buy > 0 else 1

                    # 分組: 同一 buy_date + buy_price 的交易歸為同一組
                    trade_group = 0
                    prev_buy_key = None
                    group_pnl = 0  # 每組的累計損益

                    for idx, t in enumerate(bt.trades, 1):
                        buy_date_str = pd.Timestamp(t['buy_date']).strftime('%Y-%m-%d')
                        sell_date_str = pd.Timestamp(t['sell_date']).strftime('%Y-%m-%d')
                        cur_buy_key = f"{buy_date_str}_{t['buy_price']}"

                        # 新的交易組
                        if cur_buy_key != prev_buy_key:
                            trade_group += 1
                            group_pnl = 0
                            prev_buy_key = cur_buy_key

                        # 根據 sell_ratio 計算實際賣出張數
                        ratio = t.get('sell_ratio', 1.0)
                        base_lots = max(1, int(1_000_000 / (t['buy_price'] * 1000))) if t['buy_price'] > 0 else 1
                        lots = max(1, round(base_lots * ratio))
                        total_shares = lots * 1000  # 股數
                        buy_amount = t['buy_price'] * total_shares
                        sell_amount = t['sell_price'] * total_shares

                        # 計算實際損益 (含手續費及稅)
                        commission_rate = 0.001425
                        tax_rate = 0.003
                        buy_fee = buy_amount * commission_rate
                        sell_fee = sell_amount * commission_rate
                        sell_tax = sell_amount * tax_rate
                        actual_pnl = sell_amount - buy_amount - buy_fee - sell_fee - sell_tax
                        cumulative_pnl += actual_pnl
                        group_pnl += actual_pnl

                        # 顯示標籤
                        ratio_label = f" ({ratio:.0%})" if ratio < 1.0 else ""
                        group_label = f"#{trade_group}"
                        if ratio < 1.0 - 1e-6:
                            group_label += f" ↳"  # 分批子交易標記

                        trade_rows.append({
                            "交易": group_label,
                            "買入日期": buy_date_str,
                            "買入價": f"{t['buy_price']:.2f}",
                            "買入理由": t.get('buy_reason', ''),
                            "張數": f"{lots}{ratio_label}",
                            "買入金額": f"{buy_amount:,.0f}",
                            "賣出日期": sell_date_str,
                            "賣出價": f"{t['sell_price']:.2f}",
                            "賣出理由": t.get('sell_reason', ''),
                            "賣出金額": f"{sell_amount:,.0f}",
                            "報酬%": f"{t['return_pct']:.2%}",
                            "損益": f"{actual_pnl:+,.0f}",
                            "累積損益": f"{cumulative_pnl:+,.0f}",
                            "持有天數": t.get('holding_days', ''),
                            "結果": t.get('result', ''),
                        })

                    # Summary line with actual P&L
                    simple_return = sum(t['return_pct'] for t in bt.trades)
                    pnl_sign = "🟢" if cumulative_pnl >= 0 else "🔴"
                    st.markdown(
                        f"**{get_stock_display(symbol)}** | 交易數: {bt.num_trades} | "
                        f"勝率: {bt.win_rate:.1%} | "
                        f"總報酬: {simple_return:.1%} | "
                        f"{pnl_sign} 總損益: **{cumulative_pnl:+,.0f} 元**"
                    )

                    trade_df = pd.DataFrame(trade_rows)
                    st.dataframe(
                        trade_df,
                        use_container_width=True,
                        hide_index=True,
                        height=min(400, 35 * len(trade_rows) + 38),
                    )

                    # K線圖 (日K / 週K 切換)
                    kline_col1, kline_col2 = st.columns(2)
                    with kline_col1:
                        show_daily = st.button(f"📊 日K線圖", key=f"daily_{symbol}")
                    with kline_col2:
                        show_weekly = st.button(f"📊 週K線圖", key=f"weekly_{symbol}")

                    if show_daily and symbol in stock_data:
                        fig = draw_kline_chart(
                            symbol, stock_data[symbol], bt.signal_df,
                            bt.trades, result.best_params, result.best_strategy,
                            period="daily"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                    if show_weekly and symbol in stock_data:
                        fig = draw_kline_chart(
                            symbol, stock_data[symbol], bt.signal_df,
                            bt.trades, result.best_params, result.best_strategy,
                            period="weekly"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"{symbol}: 此策略未產生交易")

    # ====== 近日買點追蹤 ======
    st.markdown("### 近日買點追蹤與通知")
    st.caption("以最佳化策略掃描近 10 個交易日是否出現買入訊號")

    if stock_data and result.best_strategy and result.best_params:
        recent_signals = detect_recent_signals(
            stock_data, result.best_strategy, result.best_params, lookback_days=10
        )
        if recent_signals:
            sig_df = pd.DataFrame(recent_signals)
            sig_df['date'] = pd.to_datetime(sig_df['date']).dt.strftime('%Y-%m-%d')
            sig_df.columns = ['股票', '日期', '價格', '買入理由']
            sig_df = sig_df.sort_values('日期', ascending=False)
            st.dataframe(sig_df, use_container_width=True, hide_index=True)

            st.success(
                f"近 10 個交易日發現 {len(recent_signals)} 個買入訊號! "
                f"策略: {STRATEGY_CN.get(result.best_strategy, result.best_strategy)}"
            )
        else:
            st.info("近 10 個交易日未偵測到買入訊號。")

    # ====== 賣點提示 ======
    st.markdown("### 接近賣點提示")
    st.caption("以 20 種策略偵測是否接近賣出訊號")

    if stock_data:
        sell_alert_count = 0
        for symbol, df in stock_data.items():
            alerts = detect_sell_signals(symbol, df)
            if alerts:
                sell_alert_count += len(alerts)
                for a in alerts:
                    icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}.get(a['urgency'], "⚪")
                    level = {"high": "已觸發賣出", "medium": "接近賣出", "low": "留意"}.get(a['urgency'], "")
                    st.markdown(f"{icon} **{get_stock_display(symbol)}** [{level}] — {a['strategy']}  \n{a['reason']}")
        if sell_alert_count == 0:
            st.success("目前測試股票均無賣出警示")

    # ====== 各股票總覽 ======
    # 統一使用 per_stock_bt (含正確風控參數) 而非 optimizer 內部的 per_stock_results
    if per_stock_bt:
        st.markdown("### 各股票回測總覽")
        rows = []
        for symbol, r in sorted(per_stock_bt.items(), key=lambda x: x[1].win_rate, reverse=True):
            if r.num_trades > 0:
                rows.append({
                    "股票": get_stock_display(symbol),
                    "勝率": f"{r.win_rate:.1%}",
                    "總報酬": f"{r.total_return:.1%}",
                    "交易數": r.num_trades,
                    "勝": r.num_wins,
                    "負": r.num_losses,
                    "均獲利": f"{r.avg_win:.2%}" if r.avg_win else "-",
                    "均虧損": f"{r.avg_loss:.2%}" if r.avg_loss else "-",
                    "最大回撤": f"{r.max_drawdown:.1%}",
                    "Sharpe": f"{r.sharpe_ratio:.2f}",
                    "獲利因子": f"{r.profit_factor:.1f}" if r.profit_factor < 100 else "INF",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ====== 收斂曲線 ======
    if logs:
        st.markdown("### 勝率收斂曲線")
        chart_df = pd.DataFrame(logs)
        if "win_rate" in chart_df.columns:
            chart_df["最佳勝率"] = chart_df["win_rate"].cummax()
            chart_df = chart_df.rename(columns={"win_rate": "當次勝率"})
            st.line_chart(chart_df[["當次勝率", "最佳勝率"]], use_container_width=True, height=300)

    # ====== 策略比較 ======
    if logs:
        st.markdown("### 各策略最佳勝率比較")
        log_df = pd.DataFrame(logs)
        if "strategy" in log_df.columns and "win_rate" in log_df.columns:
            strat_best = log_df.groupby("strategy")["win_rate"].max().sort_values(ascending=True)
            strat_best.index = [STRATEGY_CN.get(s, s) for s in strat_best.index]
            st.bar_chart(strat_best, horizontal=True, height=max(200, len(strat_best) * 50))

    # ====== 匯出 ======
    st.markdown("### 匯出")
    ecol1, ecol2, ecol3 = st.columns(3)
    best_json = {
        "strategy": result.best_strategy,
        "strategy_label": STRATEGY_CN.get(result.best_strategy, ""),
        "params": {k: float(v) for k, v in result.best_params.items()},
        "metrics": {
            "win_rate": float(result.best_win_rate),
            "sharpe_ratio": float(result.best_sharpe),
            "total_return": float(result.best_total_return),
            "total_trades": int(result.best_num_trades),
        },
    }
    ecol1.download_button(
        "下載最佳策略 JSON", use_container_width=True,
        data=json.dumps(best_json, indent=2, ensure_ascii=False),
        file_name="best_strategy.json", mime="application/json",
    )
    if logs:
        ecol2.download_button(
            "下載實驗記錄 CSV", use_container_width=True,
            data=pd.DataFrame(logs).to_csv(index=False),
            file_name="experiment_log.csv", mime="text/csv",
        )

    all_trades = []
    for symbol, bt in per_stock_bt.items():
        if bt.trades:
            cum_pnl = 0
            for t in bt.trades:
                ratio = t.get('sell_ratio', 1.0)
                base_lots = max(1, int(1_000_000 / (t['buy_price'] * 1000))) if t['buy_price'] > 0 else 1
                lots = max(1, round(base_lots * ratio))
                total_shares = lots * 1000
                buy_amt = t['buy_price'] * total_shares
                sell_amt = t['sell_price'] * total_shares
                buy_fee = buy_amt * 0.001425
                sell_fee = sell_amt * 0.001425
                sell_tax = sell_amt * 0.003
                pnl = sell_amt - buy_amt - buy_fee - sell_fee - sell_tax
                cum_pnl += pnl
                row = {
                    'symbol': symbol,
                    'stock_name': get_stock_display(symbol),
                    'buy_date': t['buy_date'],
                    'sell_date': t['sell_date'],
                    'buy_price': t['buy_price'],
                    'sell_price': t['sell_price'],
                    'shares_lot': lots,
                    'shares': total_shares,
                    'sell_ratio': ratio,
                    'buy_amount': round(buy_amt),
                    'sell_amount': round(sell_amt),
                    'return_pct': t['return_pct'],
                    'pnl': round(pnl),
                    'cumulative_pnl': round(cum_pnl),
                    'holding_days': t.get('holding_days', ''),
                    'buy_reason': t.get('buy_reason', ''),
                    'sell_reason': t.get('sell_reason', ''),
                    'result': t.get('result', ''),
                }
                all_trades.append(row)
    if all_trades:
        ecol3.download_button(
            "下載全部交易明細 CSV", use_container_width=True,
            data=pd.DataFrame(all_trades).to_csv(index=False),
            file_name="all_trades.csv", mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if not st.session_state.result and not go_btn:
    st.markdown("---")
    st.markdown("#### 使用方式")
    st.markdown("""
    1. 在上方輸入框填入股票代碼或名稱 (如 `台積電, 鴻海` 或 `2330, 2317` 或美股 `AAPL`)
    2. 台股可省略 `.TW`，直接輸入代碼或中文名稱
    3. 點擊 **開始分析**
    4. 系統自動下載資料 → 測試 20 種策略 × 多種參數 → 收斂出最佳組合
    """)

    st.markdown("#### 支援的策略 (共 20 種)")
    strat_cols = st.columns(4)
    for i, (key, cn) in enumerate(STRATEGY_CN.items()):
        if key not in STRATEGY_REGISTRY:
            continue
        sdef = STRATEGY_REGISTRY[key]
        params_str = ", ".join([PARAM_CN.get(p.name, p.name) for p in sdef.params])
        strat_cols[i % 4].markdown(
            f"**{cn}**  \n<span style='color:gray;font-size:0.85em;'>參數: {params_str}</span>",
            unsafe_allow_html=True
        )

    st.markdown("#### 股票代碼範例")
    ex1, ex2, ex3 = st.columns(3)
    ex1.markdown("**美股**  \n`AAPL, MSFT, GOOGL, NVDA, TSLA`")
    ex2.markdown("**台股**  \n`2330.TW, 2317.TW, 2454.TW`")
    ex3.markdown("**ETF**  \n`SPY, QQQ, VTI, 0050.TW`")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; color:#aaa; font-size:0.8em; padding:2rem 0 1rem;">
    股票回測策略最佳化 | 自動爬山法收斂引擎 | Yahoo Finance 資料來源
</div>
""", unsafe_allow_html=True)
