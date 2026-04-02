#!/usr/bin/env python3
"""
掃描 2: 先篩再掃
  篩選條件: 去年獲利>10%, 本益比<30, 股本>30億, 近期跌幅>10%
  再對候選股跑完整 20 策略買入訊號掃描
"""
import sys, os, time
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_loader import download_yahoo
from backtest.strategies import STRATEGY_REGISTRY

# ── 候選池: 台灣50 + 中型100 + 其他知名個股 ──
CANDIDATE_POOL = [
    # 台灣50
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW",
    "2881.TW", "2882.TW", "2891.TW", "2303.TW", "2412.TW",
    "3711.TW", "2886.TW", "2884.TW", "1301.TW", "1326.TW",
    "2357.TW", "3037.TW", "5871.TW", "2327.TW", "2395.TW",
    "3008.TW", "2345.TW", "6505.TW", "1303.TW", "2002.TW",
    "1216.TW", "2207.TW", "5880.TW", "2892.TW", "3045.TW",
    "2880.TW", "2887.TW", "4904.TW", "2379.TW", "2301.TW",
    "9910.TW", "2885.TW", "4938.TW", "3034.TW", "2883.TW",
    "6669.TW", "2912.TW", "5876.TW", "1101.TW", "2888.TW",
    "3231.TW", "2603.TW", "6446.TW", "2105.TW", "8046.TW",
    # 中型100 (部分)
    "6176.TW", "3443.TW", "8150.TW", "2344.TW", "3023.TW",
    "3044.TW", "2383.TW", "6239.TW", "3665.TW", "2049.TW",
    "8464.TW", "6285.TW", "3661.TW", "3035.TW", "2474.TW",
    "2458.TW", "6409.TW", "1477.TW", "2542.TW", "5269.TW",
    "6592.TW", "6770.TW", "2404.TW", "3017.TW", "2801.TW",
    "6531.TW", "1476.TW", "2548.TW", "9921.TW", "2615.TW",
    "2634.TW", "1802.TW", "1590.TW", "2059.TW", "9945.TW",
    "1795.TW", "6415.TW", "2845.TW", "8069.TW", "2618.TW",
    "2006.TW", "1504.TW", "5483.TW", "3653.TW", "2227.TW",
    "2408.TW", "6456.TW", "3036.TW", "8454.TW", "6278.TW",
    "2353.TW", "2356.TW", "2376.TW", "2377.TW", "2385.TW",
    "2324.TW", "2360.TW", "2347.TW", "3706.TW", "4958.TW",
]

CANDIDATE_POOL = list(dict.fromkeys(CANDIDATE_POOL))  # dedupe

# 全部 20 策略
ALL_STRATEGIES = list(STRATEGY_REGISTRY.keys())


def fetch_fundamental_data(symbol):
    """
    用 yfinance 取得基本面資料:
    - 本益比 (trailingPE)
    - 去年獲利成長率 (earningsGrowth / revenueGrowth)
    - 市值 (marketCap) 作為股本代理
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info or info.get('regularMarketPrice') is None:
            return None
        return {
            'symbol': symbol,
            'name': info.get('shortName', info.get('longName', symbol)),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'earnings_growth': info.get('earningsGrowth'),       # 獲利成長率
            'revenue_growth': info.get('revenueGrowth'),         # 營收成長率
            'market_cap': info.get('marketCap'),                 # 市值
            'price': info.get('regularMarketPrice', info.get('currentPrice')),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'current_price': info.get('regularMarketPrice', info.get('currentPrice')),
        }
    except Exception:
        return None


def filter_stocks(candidates):
    """
    篩選條件:
    1. 去年獲利成長 > 10% (earningsGrowth > 0.10)
    2. 本益比 < 30
    3. 市值 > 30億 TWD (約 1億 USD)
    4. 股價距近期高點跌幅 > 10%
    """
    print(f"\n正在取得 {len(candidates)} 檔股票基本面資料...")
    fundamentals = []
    for i, symbol in enumerate(candidates):
        if (i + 1) % 20 == 0:
            print(f"  進度: {i+1}/{len(candidates)}")
        data = fetch_fundamental_data(symbol)
        if data:
            fundamentals.append(data)
        time.sleep(0.1)  # rate limit

    print(f"取得 {len(fundamentals)} 檔基本面資料")

    passed = []
    reasons = []
    for f in fundamentals:
        symbol = f['symbol']
        fails = []

        # 條件1: 獲利成長 > 10%
        eg = f.get('earnings_growth')
        rg = f.get('revenue_growth')
        growth = eg if eg is not None else rg
        if growth is None or growth <= 0.10:
            fails.append(f"獲利成長={growth:.1%}" if growth is not None else "無獲利資料")

        # 條件2: PE < 30
        pe = f.get('pe_ratio') or f.get('forward_pe')
        if pe is None or pe >= 30 or pe <= 0:
            fails.append(f"PE={pe}" if pe is not None else "無PE資料")

        # 條件3: 市值 > 30億 TWD ≈ 1億 USD (yfinance 市值單位是 TWD for .TW)
        mc = f.get('market_cap')
        if mc is None or mc < 3_000_000_000:
            fails.append(f"市值={mc/1e8:.1f}億" if mc else "無市值")

        # 條件4: 距 52 週高點跌幅 > 10%
        high = f.get('fifty_two_week_high')
        price = f.get('current_price')
        if high and price and high > 0:
            drawdown = (high - price) / high
            if drawdown < 0.10:
                fails.append(f"跌幅僅{drawdown:.1%}")
        else:
            fails.append("無價格資料")

        if not fails:
            f['growth'] = growth
            f['pe'] = pe
            f['drawdown'] = (high - price) / high if high and price else 0
            passed.append(f)
            reasons.append({'symbol': symbol, 'status': '通過', 'details': ''})
        else:
            reasons.append({'symbol': symbol, 'status': '未通過', 'details': '; '.join(fails)})

    return passed, reasons


def get_signal_reason(strategy_name, df_row):
    reasons = []
    for col, label, fmt in [
        ('k', 'K', '.1f'), ('d', 'D', '.1f'), ('rsi', 'RSI', '.1f'),
        ('macd_hist', 'MACD柱', '.3f'), ('vol_ratio', '量比', '.2f'),
        ('momentum', '動量', '.2%'),
    ]:
        val = df_row.get(col, float('nan'))
        if col in df_row.index and pd.notna(val):
            reasons.append(f"{label}={val:{fmt}}")
    return ", ".join(reasons) if reasons else "策略訊號觸發"


def scan_signals(stock_data, strategies, lookback=5):
    all_signals = []
    total = len(stock_data) * len(strategies)
    done = 0
    for strat_name in strategies:
        sdef = STRATEGY_REGISTRY[strat_name]
        params = sdef.default_params()
        for symbol, df in stock_data.items():
            done += 1
            if done % 20 == 0:
                print(f"  進度: {done}/{total} ({done/total:.0%})")
            try:
                sig_df = sdef.func(df, **params)
                recent = sig_df.tail(lookback)
                for i in range(len(recent)):
                    row = recent.iloc[i]
                    if row.get('signal', 0) == 1:
                        date = recent.index[i]
                        date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
                        all_signals.append({
                            'symbol': symbol,
                            'date': date_str,
                            'price': round(float(row['close']), 2),
                            'strategy': strat_name,
                            'strategy_name': sdef.description,
                            'reason': get_signal_reason(strat_name, row),
                        })
            except Exception:
                pass
    return all_signals


def generate_report(passed_stocks, filter_reasons, all_signals, stock_data, output_path):
    lines = []
    lines.append("# 先篩再掃 — 基本面篩選 + 策略訊號掃描報告")
    lines.append("")
    lines.append(f"**掃描日期**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**候選池**: {len(filter_reasons)} 檔")
    lines.append(f"**篩選條件**: 去年獲利成長>10%, 本益比<30, 市值>30億, 近期跌幅>10%")
    lines.append(f"**通過篩選**: {len(passed_stocks)} 檔")
    lines.append(f"**使用策略**: 全部 {len(ALL_STRATEGIES)} 種策略")
    lines.append("")

    # 篩選通過清單
    lines.append("## 一、基本面篩選結果")
    lines.append("")
    if passed_stocks:
        lines.append("### 通過篩選的股票")
        lines.append("")
        lines.append("| 代碼 | 名稱 | 獲利成長 | 本益比 | 市值(億) | 距高點跌幅 | 現價 |")
        lines.append("|------|------|---------|--------|---------|-----------|------|")
        for s in passed_stocks:
            mc_str = f"{s.get('market_cap', 0)/1e8:.0f}" if s.get('market_cap') else "N/A"
            lines.append(f"| {s['symbol']} | {s.get('name', '')} | "
                         f"{s.get('growth', 0):.1%} | {s.get('pe', 0):.1f} | "
                         f"{mc_str} | {s.get('drawdown', 0):.1%} | {s.get('price', 0):.2f} |")
        lines.append("")
    else:
        lines.append("**無股票通過全部篩選條件。**")
        lines.append("")

    # 篩選統計
    passed_count = sum(1 for r in filter_reasons if r['status'] == '通過')
    failed_count = sum(1 for r in filter_reasons if r['status'] != '通過')
    lines.append(f"篩選統計: 通過 {passed_count} 檔 / 未通過 {failed_count} 檔")
    lines.append("")

    # 訊號掃描結果
    lines.append("## 二、策略訊號掃描結果")
    lines.append("")

    if not all_signals:
        lines.append("通過篩選的股票近 5 個交易日未發現買入訊號。")
        lines.append("")
    else:
        signal_df = pd.DataFrame(all_signals)
        ranking = signal_df.groupby('symbol').agg(
            策略數=('strategy', 'nunique'),
            最新訊號日=('date', 'max'),
            最新價格=('price', 'last'),
            策略列表=('strategy_name', lambda x: '、'.join(sorted(set(x)))),
        ).reset_index().sort_values('策略數', ascending=False)

        # 合併基本面資訊
        fund_map = {s['symbol']: s for s in passed_stocks}

        lines.append("### 買入訊號排名 (依多策略共振排序)")
        lines.append("")
        lines.append("| 排名 | 代碼 | 名稱 | 策略數 | 現價 | PE | 獲利成長 | 跌幅 | 看好策略 |")
        lines.append("|------|------|------|--------|------|-----|---------|------|---------|")
        for rank, (_, row) in enumerate(ranking.iterrows(), 1):
            f = fund_map.get(row['symbol'], {})
            lines.append(f"| {rank} | {row['symbol']} | {f.get('name', '')} | "
                         f"{row['策略數']} | {row['最新價格']:.2f} | "
                         f"{f.get('pe', 0):.1f} | {f.get('growth', 0):.1%} | "
                         f"{f.get('drawdown', 0):.1%} | {row['策略列表']} |")
        lines.append("")

        # 明細
        hot_symbols = ranking.head(10)['symbol'].tolist()
        if hot_symbols:
            lines.append("### 重點股票訊號明細")
            lines.append("")
            for symbol in hot_symbols:
                f = fund_map.get(symbol, {})
                stock_signals = signal_df[signal_df['symbol'] == symbol].sort_values('date', ascending=False)
                lines.append(f"#### {symbol} {f.get('name', '')}")
                lines.append(f"- 本益比: {f.get('pe', 'N/A')}, 獲利成長: {f.get('growth', 0):.1%}, 跌幅: {f.get('drawdown', 0):.1%}")
                lines.append("")
                lines.append("| 日期 | 策略 | 價格 | 買入理由 |")
                lines.append("|------|------|------|----------|")
                for _, sig in stock_signals.iterrows():
                    lines.append(f"| {sig['date']} | {sig['strategy_name']} | "
                                 f"{sig['price']:.2f} | {sig['reason']} |")
                lines.append("")

    lines.append("---")
    lines.append("> **免責聲明**: 本報告僅供參考，不構成投資建議。投資有風險，請自行評估。")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n報告已儲存至: {output_path}")


def main():
    t0 = time.time()
    print("=" * 70)
    print("掃描 2: 先篩再掃 (基本面篩選 + 全策略訊號)")
    print(f"掃描日期: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"候選池: {len(CANDIDATE_POOL)} 檔")
    print("篩選條件: 獲利>10%, PE<30, 市值>30億, 近期跌>10%")
    print("=" * 70)

    # Step 1: 基本面篩選
    print("\n[1/3] 基本面篩選...")
    passed_stocks, filter_reasons = filter_stocks(CANDIDATE_POOL)
    print(f"\n通過篩選: {len(passed_stocks)} 檔")
    for s in passed_stocks:
        print(f"  {s['symbol']} {s.get('name','')} | PE={s.get('pe',0):.1f} "
              f"獲利={s.get('growth',0):.1%} 跌幅={s.get('drawdown',0):.1%}")

    if not passed_stocks:
        print("\n無股票通過篩選，產生空報告。")
        output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_filtered_signals.md")
        generate_report(passed_stocks, filter_reasons, [], {}, output)
        return

    # Step 2: 下載通過股票的歷史資料
    passed_symbols = [s['symbol'] for s in passed_stocks]
    print(f"\n[2/3] 下載 {len(passed_symbols)} 檔歷史資料...")
    stock_data = download_yahoo(passed_symbols, start="2024-01-01")
    print(f"成功下載 {len(stock_data)} 檔")

    # Step 3: 全策略掃描
    print(f"\n[3/3] 使用全部 {len(ALL_STRATEGIES)} 種策略掃描...")
    signals = scan_signals(stock_data, ALL_STRATEGIES)
    elapsed = time.time() - t0

    print(f"\n掃描完成! 發現 {len(signals)} 個買入訊號 (耗時 {elapsed:.0f}s)")

    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_filtered_signals.md")
    generate_report(passed_stocks, filter_reasons, signals, stock_data, output)


if __name__ == "__main__":
    main()
