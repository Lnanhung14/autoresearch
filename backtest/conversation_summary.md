# 回測策略最佳化系統 — 開發紀錄

**日期**: 2026-03-27

---

## 一、整合 20 種回測策略

### 任務說明
將 `/Gemini APP2 Stock` 專案中的 12 種策略，加入 `/autoresearch/backtest` 專案原有的 8 種策略，合計 20 種回測策略，擴充策略選擇，其他程式邏輯不變。

### 原有 8 種策略 (backtest 專案)

| # | 策略代碼 | 策略名稱 |
|---|----------|----------|
| 1 | sma_crossover | 簡單移動平均線交叉 |
| 2 | ema_crossover | 指數移動平均線交叉 |
| 3 | rsi | RSI 超買超賣 |
| 4 | macd | MACD 交叉 |
| 5 | bollinger_bands | 布林通道均值回歸 |
| 6 | kdj | KDJ 隨機指標 |
| 7 | volume_price | 量價突破 |
| 8 | dual_ma_rsi | 雙均線交叉 + RSI 過濾 |

### 新增 12 種策略 (來自 Gemini APP2 Stock)

| # | 策略代碼 | 策略名稱 |
|---|----------|----------|
| 9 | multi_factor | 多因子動量 + RSI 排名選股 |
| 10 | kd_macd_signal | KD 黃金交叉 + MACD 柱狀圖 |
| 11 | ma_breakout | 均線突破趨勢跟蹤 |
| 12 | foreign_follow | 外資跟單策略 (量能放大) |
| 13 | etf_momentum | ETF 動量定期再平衡 |
| 14 | vol_price_sync | 量價齊揚確認多頭 |
| 15 | vol_breakout_vcp | VCP 量縮突破 |
| 16 | vol_divergence | 量價背離反轉 (OBV) |
| 17 | chip_sedimentation | 籌碼沉澱動能突破 |
| 18 | chip_sedimentation_inst | 籌碼沉澱 + 法人共振 |
| 19 | chip_sedimentation_rising | 籌碼沉澱 + 起漲點 (RSI+MACD) |
| 20 | vol_dry_bottom | 量縮 KD 低檔反彈 |

### 技術適配方式
Gemini APP2 Stock 的策略原本使用大寫欄位 (Close, High, Low, Volume) 與複雜的投資組合管理邏輯 (持倉、停損停利、次日開盤執行)。適配時：
- 統一使用小寫欄位名 (`close`, `high`, `low`, `volume`)
- 每個策略函式回傳 DataFrame 含 `signal` 欄位 (1=買, -1=賣, 0=持有)
- 每個策略註冊 `StrategyParam` 定義可調參數的搜索空間
- 提取核心訊號邏輯，移除投資組合管理層 (由 backtest engine 統一處理)

### 修改檔案
- `backtest/strategies.py` — 新增 12 個策略函式 + 註冊至 `STRATEGY_REGISTRY`

---

## 二、回測測試報告 (2454 / 1326 / 6176)

### 測試條件
- **資料區間**: 2020-01-01 ~ 至今
- **測試標的**: 2454 聯發科、1326 台化、6176 瑞儀
- **交易成本**: 手續費 0.1425% + 證交稅 0.3% + 滑價 0.1%
- **策略數**: 20 種，使用預設參數

### 各股票最佳策略

| 股票 | 最佳策略 | 夏普比率 | 勝率 | 總報酬 | 來源 |
|------|----------|----------|------|--------|------|
| 2454 聯發科 | kdj (KDJ 隨機指標) | 27.01 | 100% | +155.8% | 原有 |
| 1326 台化 | vol_dry_bottom (量縮 KD 低檔反彈) | 2.09 | 85.7% | +11.9% | **新增** |
| 6176 瑞儀 | vol_dry_bottom (量縮 KD 低檔反彈) | 10.79 | 75.0% | +13.9% | **新增** |

### 綜合最佳策略排名 (全部股票平均夏普比率)

| 排名 | 策略 | 平均夏普 | 來源 |
|------|------|----------|------|
| 1 | kdj (KDJ 隨機指標) | 10.26 | 原有 |
| 2 | vol_dry_bottom (量縮 KD 低檔反彈) | 8.66 | **新增** |
| 3 | foreign_follow (外資跟單策略) | 2.76 | **新增** |
| 4 | bollinger_bands (布林通道均值回歸) | 2.57 | 原有 |
| 5 | dual_ma_rsi (雙均線交叉 + RSI 過濾) | 1.81 | 原有 |
| 6 | etf_momentum (ETF 動量定期再平衡) | 1.47 | **新增** |

### 結論
新增的 Gemini 策略中，`vol_dry_bottom`(量縮 KD 低檔反彈) 表現最為突出，在 2 檔股票 (台化、瑞儀) 中都是最佳策略；`foreign_follow`(外資跟單) 在三檔股票中都有正夏普比率，穩定性最佳。

詳細報告: `backtest/backtest_test_report.md`

---

## 三、繁體中文化

### 中文化範圍
將所有使用者介面、策略描述、輸出訊息從英文改為繁體中文。

### 修改檔案

| 檔案 | 修改內容 |
|------|----------|
| `strategies.py` | 20 個策略的 `description` 欄位全部改為繁體中文 |
| `app.py` | `STRATEGY_CN` (20 策略中文名)、`METRIC_CN` (勝率/報酬/夏普/獲利因子)、`PARAM_CN` (所有參數中文名)、頁面標題、頁尾 |
| `optimizer.py` | 最佳化引擎輸出訊息 (階段1/2/3、保留/略過、最終報告) |
| `engine.py` | 回測結果摘要 (勝率/報酬/最大回撤/夏普) |
| `run_optimize.py` | CLI 輸出訊息 |
| `run_test_report.py` | 報告標題、表頭、欄位名全部中文化 |

### 中文化對照範例

| 原文 | 繁體中文 |
|------|----------|
| Win Rate | 勝率 |
| Total Return | 總報酬 |
| Sharpe Ratio | 夏普比率 |
| Max Drawdown | 最大回撤 |
| Profit Factor | 獲利因子 |
| [KEEP] | [保留] |
| [skip] | [略過] |
| Phase 1: Coarse scan | 階段 1: 全策略粗掃描 |
| Hill-climbing refinement | 爬山法精細調整 |

---

## 四、台股近期買入訊號掃描

### 掃描條件
- **掃描日期**: 2026-03-27
- **掃描標的**: 20 檔台股熱門標的
- **使用策略**: 10 種回測中表現最好的高勝率策略
- **訊號區間**: 近 5 個交易日

### 掃描結果 (多策略共振排名)

| 排名 | 代碼 | 名稱 | 策略看好數 | 買入理由摘要 |
|------|------|------|-----------|-------------|
| 1 | 2317 | 鴻海 | 2 策略 | KD 超賣 (K=19.2)、RSI 超賣 (25~29) |
| 2 | 2884 | 玉山金 | 2 策略 | KD 超賣 (K=17.6)、RSI 極低 (12~27) |
| 3 | 6176 | 瑞儀 | 2 策略 | KD 超賣 (K=12.8)、RSI 極低 (17~29) |
| 4 | 5871 | 中租-KY | 2 策略 | 籌碼沉澱突破 + 起漲點確認 |
| 5 | 1301 | 台塑 | 1 策略 | 近日出現買入訊號 |
| 6 | 2891 | 中信金 | 1 策略 | 近日出現買入訊號 |

### 重點個股分析

**鴻海 (2317)** — RSI 跌至 25~29 極度超賣區，KD 值 K=19.2 低檔形成黃金交叉，技術面超跌反彈機會高。

**玉山金 (2884)** — RSI 一度跌至 12.1 極端超賣，KD K=17.6 同步低檔交叉，屬於金融股中超跌最嚴重標的。

**瑞儀 (6176)** — RSI=17.9 為近期最低，KD K=12.8 極度超賣，短線反彈動能蓄積中。

**中租-KY (5871)** — 籌碼沉澱後放量突破，且 RSI+MACD 同時確認起漲點，屬於「量縮整理完畢 → 突破」型態。

詳細報告: `backtest/tw_stock_buy_signals.md`

> **免責聲明**: 本分析僅供參考，不構成投資建議。投資有風險，請自行評估。

---

## 檔案清單

| 檔案 | 說明 |
|------|------|
| `backtest/strategies.py` | 20 種策略定義與註冊 |
| `backtest/engine.py` | 回測引擎 |
| `backtest/optimizer.py` | 爬山法最佳化引擎 |
| `backtest/app.py` | Streamlit 網頁介面 |
| `backtest/run_optimize.py` | CLI 執行入口 |
| `backtest/run_test_report.py` | 批次回測報告產生器 |
| `backtest/scan_buy_signals.py` | 台股買入訊號掃描器 |
| `backtest/backtest_test_report.md` | 回測測試報告 (2454/1326/6176) |
| `backtest/tw_stock_buy_signals.md` | 台股近期買入訊號報告 |
| `backtest/conversation_summary.md` | 本次開發紀錄 (本文件) |
