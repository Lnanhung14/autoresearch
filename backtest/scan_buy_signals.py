#!/usr/bin/env python3
"""
掃描台股近期買入訊號 — 使用 20 種策略找出近日適合買入的股票
"""
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_loader import download_yahoo
from backtest.strategies import STRATEGY_REGISTRY, list_strategies


# 台股熱門標的
TW_STOCKS = [
    "2330.TW",  # 台積電
    "2317.TW",  # 鴻海
    "2454.TW",  # 聯發科
    "2308.TW",  # 台達電
    "2382.TW",  # 廣達
    "2881.TW",  # 富邦金
    "2882.TW",  # 國泰金
    "2891.TW",  # 中信金
    "2412.TW",  # 中華電
    "1301.TW",  # 台塑
    "1326.TW",  # 台化
    "2303.TW",  # 聯電
    "3711.TW",  # 日月光
    "2886.TW",  # 兆豐金
    "2884.TW",  # 玉山金
    "6176.TW",  # 瑞儀
    "3037.TW",  # 欣興
    "2357.TW",  # 華碩
    "5871.TW",  # 中租-KY
    "2327.TW",  # 國巨
]

STOCK_NAMES = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科",
    "2308.TW": "台達電", "2382.TW": "廣達", "2881.TW": "富邦金",
    "2882.TW": "國泰金", "2891.TW": "中信金", "2412.TW": "中華電",
    "1301.TW": "台塑", "1326.TW": "台化", "2303.TW": "聯電",
    "3711.TW": "日月光", "2886.TW": "兆豐金", "2884.TW": "玉山金",
    "6176.TW": "瑞儀", "3037.TW": "欣興", "2357.TW": "華碩",
    "5871.TW": "中租-KY", "2327.TW": "國巨",
}

# 高勝率策略 (根據回測結果篩選)
TOP_STRATEGIES = [
    "kdj", "vol_dry_bottom", "foreign_follow", "bollinger_bands",
    "dual_ma_rsi", "rsi", "chip_sedimentation", "etf_momentum",
    "chip_sedimentation_rising", "volume_price",
]


def get_signal_reason(strategy_name, df_row):
    """取得買入理由描述"""
    reasons = []
    if 'k' in df_row.index and not pd.isna(df_row.get('k', float('nan'))):
        reasons.append(f"KD K={df_row['k']:.1f}")
    if 'd' in df_row.index and not pd.isna(df_row.get('d', float('nan'))):
        reasons.append(f"D={df_row['d']:.1f}")
    if 'rsi' in df_row.index and not pd.isna(df_row.get('rsi', float('nan'))):
        reasons.append(f"RSI={df_row['rsi']:.1f}")
    if 'macd_hist' in df_row.index and not pd.isna(df_row.get('macd_hist', float('nan'))):
        reasons.append(f"MACD柱={df_row['macd_hist']:.3f}")
    if 'vol_ratio' in df_row.index and not pd.isna(df_row.get('vol_ratio', float('nan'))):
        reasons.append(f"量比={df_row['vol_ratio']:.2f}")
    if 'momentum' in df_row.index and not pd.isna(df_row.get('momentum', float('nan'))):
        reasons.append(f"動量={df_row['momentum']:.2%}")
    if 'bb_lower' in df_row.index and not pd.isna(df_row.get('bb_lower', float('nan'))):
        reasons.append(f"布林下軌={df_row['bb_lower']:.1f}")
    if not reasons:
        reasons.append("策略訊號觸發")
    return ", ".join(reasons)


def main():
    print("=" * 70)
    print("台股近期買入訊號掃描")
    print(f"掃描日期: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"標的數量: {len(TW_STOCKS)} 檔")
    print(f"使用策略: {len(TOP_STRATEGIES)} 種高勝率策略")
    print("=" * 70)

    # 下載資料
    print("\n正在下載股票資料...")
    stock_data = download_yahoo(TW_STOCKS, start="2023-01-01")
    print(f"成功下載 {len(stock_data)} 檔\n")

    # 掃描近 5 個交易日的買入訊號
    lookback = 5
    all_signals = []

    for strat_name in TOP_STRATEGIES:
        sdef = STRATEGY_REGISTRY[strat_name]
        params = sdef.default_params()

        for symbol, df in stock_data.items():
            try:
                sig_df = sdef.func(df, **params)
                recent = sig_df.tail(lookback)

                for i in range(len(recent)):
                    row = recent.iloc[i]
                    if row.get('signal', 0) == 1:
                        date = recent.index[i]
                        if isinstance(date, pd.Timestamp):
                            date_str = date.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date)

                        reason = get_signal_reason(strat_name, row)
                        all_signals.append({
                            'symbol': symbol,
                            'name': STOCK_NAMES.get(symbol, symbol),
                            'date': date_str,
                            'price': round(float(row['close']), 2),
                            'strategy': strat_name,
                            'strategy_name': sdef.description,
                            'reason': reason,
                        })
            except Exception:
                pass

    if not all_signals:
        print("近 5 個交易日未發現買入訊號。")
        return

    # 按股票統計被多少策略同時看好
    signal_df = pd.DataFrame(all_signals)
    stock_strategy_count = signal_df.groupby(['symbol', 'name']).agg(
        策略數=('strategy', 'nunique'),
        最新訊號日=('date', 'max'),
        最新價格=('price', 'last'),
    ).reset_index().sort_values('策略數', ascending=False)

    print("=" * 70)
    print("近 5 日買入訊號彙整 (依多策略共振排序)")
    print("=" * 70)
    print(f"\n{'排名':>4} {'代碼':<10} {'名稱':<8} {'策略看好數':>10} {'最新訊號日':>12} {'現價':>8}")
    print("-" * 60)
    for rank, (_, row) in enumerate(stock_strategy_count.iterrows(), 1):
        print(f"{rank:>4} {row['symbol']:<10} {row['name']:<8} "
              f"{row['策略數']:>10} {row['最新訊號日']:>12} {row['最新價格']:>8.2f}")

    # 詳細訊號
    print("\n" + "=" * 70)
    print("各股票買入訊號明細")
    print("=" * 70)

    # 只顯示被 >= 2 個策略看好的
    hot_stocks = stock_strategy_count[stock_strategy_count['策略數'] >= 2]['symbol'].tolist()
    if not hot_stocks:
        hot_stocks = stock_strategy_count.head(5)['symbol'].tolist()

    for symbol in hot_stocks:
        name = STOCK_NAMES.get(symbol, symbol)
        stock_signals = signal_df[signal_df['symbol'] == symbol].sort_values('date', ascending=False)
        print(f"\n### {symbol} {name}")
        print(f"{'日期':<12} {'策略':<20} {'價格':>8} {'買入理由'}")
        print("-" * 70)
        for _, sig in stock_signals.iterrows():
            print(f"{sig['date']:<12} {sig['strategy_name']:<20} "
                  f"{sig['price']:>8.2f} {sig['reason']}")

    # 生成報告
    report_lines = []
    report_lines.append("# 台股近期買入訊號掃描報告")
    report_lines.append("")
    report_lines.append(f"**掃描日期**: {datetime.now().strftime('%Y-%m-%d')}")
    report_lines.append(f"**掃描標的**: {len(stock_data)} 檔台股")
    report_lines.append(f"**使用策略**: {len(TOP_STRATEGIES)} 種高勝率策略")
    report_lines.append(f"**訊號區間**: 近 {lookback} 個交易日")
    report_lines.append("")

    report_lines.append("## 多策略共振排名 (越多策略同時看好，訊號越強)")
    report_lines.append("")
    report_lines.append("| 排名 | 代碼 | 名稱 | 策略看好數 | 最新訊號日 | 現價 |")
    report_lines.append("|------|------|------|-----------|-----------|------|")
    for rank, (_, row) in enumerate(stock_strategy_count.iterrows(), 1):
        report_lines.append(
            f"| {rank} | {row['symbol']} | {row['name']} | "
            f"{row['策略數']} | {row['最新訊號日']} | {row['最新價格']:.2f} |"
        )

    report_lines.append("")
    report_lines.append("## 重點股票買入訊號明細")
    report_lines.append("")

    for symbol in hot_stocks:
        name = STOCK_NAMES.get(symbol, symbol)
        stock_signals = signal_df[signal_df['symbol'] == symbol].sort_values('date', ascending=False)
        report_lines.append(f"### {symbol} {name}")
        report_lines.append("")
        report_lines.append("| 日期 | 策略 | 價格 | 買入理由 |")
        report_lines.append("|------|------|------|----------|")
        for _, sig in stock_signals.iterrows():
            report_lines.append(
                f"| {sig['date']} | {sig['strategy_name']} | "
                f"{sig['price']:.2f} | {sig['reason']} |"
            )
        report_lines.append("")

    report_lines.append("## 使用策略說明")
    report_lines.append("")
    report_lines.append("| 策略代碼 | 策略名稱 | 說明 |")
    report_lines.append("|----------|----------|------|")
    strategy_explanations = {
        "kdj": "KD 值在超賣區形成黃金交叉時買入",
        "vol_dry_bottom": "成交量萎縮後 KD 低檔出現紅K反彈訊號",
        "foreign_follow": "成交量異常放大 (模擬外資大量買超)",
        "bollinger_bands": "股價跌破布林通道下軌後反彈買入",
        "dual_ma_rsi": "短均線上穿長均線且 RSI 確認趨勢向上",
        "rsi": "RSI 進入超賣區 (<30) 後反彈",
        "chip_sedimentation": "量縮整理後放量突破整理區間高點",
        "etf_momentum": "動量為正且股價站上 20 日均線",
        "chip_sedimentation_rising": "籌碼沉澱後 RSI+MACD 確認起漲點",
        "volume_price": "放量且股價站上均線確認突破",
    }
    for s in TOP_STRATEGIES:
        desc = STRATEGY_REGISTRY[s].description
        expl = strategy_explanations.get(s, "")
        report_lines.append(f"| {s} | {desc} | {expl} |")

    report_lines.append("")
    report_lines.append("> **免責聲明**: 本報告僅供參考，不構成投資建議。投資有風險，請自行評估。")

    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "backtest", "tw_stock_buy_signals.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n報告已儲存至: tw_stock_buy_signals.md")


if __name__ == "__main__":
    main()
