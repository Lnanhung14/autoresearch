#!/usr/bin/env python3
"""
掃描 1: 台灣50 + 中型100 成分股 (共 150 檔) 買入訊號掃描
"""
import sys, os, time
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_loader import download_yahoo
from backtest.strategies import STRATEGY_REGISTRY

# ── 台灣50成分股 (0050) ──
TW50 = [
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
]

# ── 中型100成分股 (0051) ──
MID100 = [
    "2353.TW", "2356.TW", "2376.TW", "2377.TW", "2385.TW",
    "2324.TW", "2360.TW", "2347.TW", "3706.TW", "4958.TW",
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
    "6257.TW", "1605.TW", "3702.TW", "2609.TW", "6271.TW",
    "2923.TW", "1513.TW", "2368.TW", "4743.TW", "2889.TW",
    "5534.TW", "3533.TW", "6472.TW", "1589.TW", "2201.TW",
    "2337.TW", "3515.TW", "6550.TW", "8261.TW", "3189.TW",
    "3576.TW", "1304.TW", "6538.TW", "5388.TW", "3529.TW",
    "6547.TW", "5347.TW", "2439.TW", "2023.TW", "6789.TW",
    "3016.TW", "4919.TW", "2610.TW", "6116.TW", "2101.TW",
    "8299.TW", "3037.TW", "1560.TW", "6488.TW", "2354.TW",
]

ALL_SYMBOLS = list(dict.fromkeys(TW50 + MID100))  # dedupe

# 高勝率策略
TOP_STRATEGIES = [
    "kdj", "vol_dry_bottom", "foreign_follow", "bollinger_bands",
    "dual_ma_rsi", "rsi", "chip_sedimentation", "etf_momentum",
    "chip_sedimentation_rising", "volume_price",
]


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


def scan_signals(stock_data, lookback=5):
    all_signals = []
    total = len(stock_data) * len(TOP_STRATEGIES)
    done = 0
    for strat_name in TOP_STRATEGIES:
        sdef = STRATEGY_REGISTRY[strat_name]
        params = sdef.default_params()
        for symbol, df in stock_data.items():
            done += 1
            if done % 50 == 0:
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


def generate_report(all_signals, stock_data, output_path):
    if not all_signals:
        report = "# 台灣50+中型100 買入訊號掃描報告\n\n近 5 個交易日未發現買入訊號。\n"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print("未發現買入訊號。")
        return

    signal_df = pd.DataFrame(all_signals)

    # 多策略共振排名
    ranking = signal_df.groupby('symbol').agg(
        策略數=('strategy', 'nunique'),
        最新訊號日=('date', 'max'),
        最新價格=('price', 'last'),
        策略列表=('strategy_name', lambda x: '、'.join(sorted(set(x)))),
    ).reset_index().sort_values('策略數', ascending=False)

    lines = []
    lines.append("# 台灣50 + 中型100 成分股 — 買入訊號掃描報告")
    lines.append("")
    lines.append(f"**掃描日期**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**掃描範圍**: 台灣50 + 中型100 = {len(stock_data)} 檔")
    lines.append(f"**使用策略**: {len(TOP_STRATEGIES)} 種高勝率策略")
    lines.append(f"**訊號區間**: 近 5 個交易日")
    lines.append(f"**發現訊號**: {len(ranking)} 檔股票有買入訊號")
    lines.append("")

    # 多策略共振 (>=2)
    hot = ranking[ranking['策略數'] >= 2]
    if len(hot) > 0:
        lines.append("## 多策略共振 (2個以上策略同時看好)")
        lines.append("")
        lines.append("| 排名 | 代碼 | 策略看好數 | 現價 | 最新訊號日 | 看好策略 |")
        lines.append("|------|------|-----------|------|-----------|---------|")
        for rank, (_, row) in enumerate(hot.iterrows(), 1):
            lines.append(f"| {rank} | {row['symbol']} | {row['策略數']} | "
                         f"{row['最新價格']:.2f} | {row['最新訊號日']} | {row['策略列表']} |")
        lines.append("")

    # 單策略訊號
    single = ranking[ranking['策略數'] == 1]
    if len(single) > 0:
        lines.append("## 單一策略買入訊號")
        lines.append("")
        lines.append("| 代碼 | 現價 | 最新訊號日 | 策略 |")
        lines.append("|------|------|-----------|------|")
        for _, row in single.iterrows():
            lines.append(f"| {row['symbol']} | {row['最新價格']:.2f} | "
                         f"{row['最新訊號日']} | {row['策略列表']} |")
        lines.append("")

    # 重點股票明細 (多策略共振)
    hot_symbols = hot['symbol'].tolist() if len(hot) > 0 else ranking.head(5)['symbol'].tolist()
    if hot_symbols:
        lines.append("## 重點股票訊號明細")
        lines.append("")
        for symbol in hot_symbols:
            stock_signals = signal_df[signal_df['symbol'] == symbol].sort_values('date', ascending=False)
            lines.append(f"### {symbol}")
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
    print("掃描 1: 台灣50 + 中型100 成分股買入訊號")
    print(f"掃描日期: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"標的數量: {len(ALL_SYMBOLS)} 檔")
    print(f"使用策略: {len(TOP_STRATEGIES)} 種")
    print("=" * 70)

    print("\n[1/2] 下載股票資料 (Yahoo Finance)...")
    stock_data = download_yahoo(ALL_SYMBOLS, start="2024-01-01")
    print(f"成功下載 {len(stock_data)} 檔 (耗時 {time.time()-t0:.0f}s)")

    print(f"\n[2/2] 掃描近 5 日買入訊號...")
    signals = scan_signals(stock_data)
    elapsed = time.time() - t0

    print(f"\n掃描完成! 發現 {len(signals)} 個買入訊號 (耗時 {elapsed:.0f}s)")

    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_tw150_signals.md")
    generate_report(signals, stock_data, output)


if __name__ == "__main__":
    main()
