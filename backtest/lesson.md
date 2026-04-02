# 回測策略最佳化系統 — Bug / Error / 教訓紀錄

**版本**: v1.0 → v3.0
**日期**: 2026-03-27 ~ 2026-03-29

---

## Bug 總覽

| # | 嚴重度 | Bug 描述 | 根因 | 修復方式 | 教訓 |
|---|--------|---------|------|---------|------|
| 1 | 🔴 嚴重 | 台塑虧損 -41% 無停損 | 完全沒有風控 | 加入三道停損機制 | 風控必須優先 |
| 2 | 🔴 嚴重 | 損益數值顯示錯誤 | 計算用「比例」非「金額」 | 改為 (賣出金額-買入金額) | 顯示邏輯需驗證 |
| 3 | 🔴 嚴重 | BrokenPipeError 崩潰 | Streamlit subprocess 管道斷開 | _safe_print 靜默處理 | subprocess 輸出需防禦 |
| 4 | 🟡 中等 | 設定最少 5 筆但只有 2 筆 | min_trades 過濾在最佳化層不在顯示層 | 統一過濾邏輯 | 過濾條件需全鏈路一致 |
| 5 | 🟡 中等 | 最佳報酬 +44% 但總損益 -88K | 最佳化看「平均報酬」非「累積損益」 | 加入累積損益指標 | 指標定義需明確 |
| 6 | 🟡 中等 | 布林通道 65% 居然買入 | 下軌 = 均價-N倍標準差，非百分比 | 改善理由顯示加入數值 | 指標名稱要自解釋 |
| 7 | 🟢 輕微 | 策略只跑 30 次/策略 | 500 實驗 ÷ 20 策略 | 提高到不限制 | 搜尋空間需量化 |
| 8 | 🟢 輕微 | 台股 .TW 忘記加 | 只接受完整代碼 | 自動補全 .TW | 使用者體驗要寬容 |

---

## Bug 1: 台塑 1301 無停損，虧損 -41.62% 🔴

### 現象
```
台塑 (1301.TW) — 布林通道策略
買入: 2024-05-24 @ 64.29 (收盤 < 布林下軌 65.2)
賣出: 2025-02-14 @ 37.86 (收盤 > 布林上軌 37.7)
持有: 266 天
虧損: -41.62% (-400,337 元)
```

### 根因分析
1. **沒有任何停損機制** — engine.py 的 `_extract_trades()` 只在 signal=-1 時賣出
2. **布林帶跟著下移** — 股價持續下跌時，移動平均跟著跌，布林下軌也跟著跌，上軌也跟著跌
3. **惡性循環** — 買在下軌 65.2，股價跌到 40，但此時上軌也跌到 42，所以永遠不會觸發「突破上軌」賣出
4. **最終賣出** — 股價跌到 38 附近波動縮小，上軌終於降到 37.7，才觸發賣出

### 修復
engine.py `_extract_trades()` 加入三道保險:
```python
# 優先順序: 停損 > 停利 > 持有上限 > 策略訊號
if stop_loss > 0 and unrealized_return <= -stop_loss:
    force_sell = True
    reason = f"⛔ 停損: 跌幅{unrealized_return:.1%} 超過 -{stop_loss:.0%}"
elif take_profit > 0 and unrealized_return >= take_profit:
    force_sell = True
    reason = f"🎯 停利: 漲幅{unrealized_return:.1%} 達 +{take_profit:.0%}"
elif max_hold_days > 0 and hold_days >= max_hold_days:
    force_sell = True
    reason = f"⏰ 持有{hold_days}天達上限"
```

### 教訓
> **任何交易系統都必須有停損機制，這是不可協商的底線。**
> 沒有停損的回測結果完全不可信，因為單筆巨虧就能摧毀整個策略的統計意義。

### 改善後的台塑案例 (停損 15%):
```
買入 64.29 → 跌到約 54.6 (-15%) → ⛔ 停損賣出
虧損: -15% (而非 -41.62%)
持有: 約 35 天 (而非 266 天)
```

---

## Bug 2: 損益數值顯示錯誤 🔴

### 現象
交易明細表中:
```
損益:    2, -5, 0, 11, -3, -3, 2, -2, 3, 2  ← 數字太小
累積損益: 31, 26, 27, 37, 34, 31, 33, 32, 35, 37  ← 不合理
```
陽明 (2609.TW) 一張股價 60 元的股票，損益不可能只有 2 或 -5。

### 根因分析
損益計算使用的是「報酬比例」而非「金額」:
```python
# Before (錯誤)
profit = net_return  # 這是比例 (如 0.03 = 3%)

# 顯示為: 損益 = 0.03 → 四捨五入顯示為 0 或 3
```

### 修復
```python
# After (正確)
profit = sell_price - buy_price - total_cost  # 每股實際損益
# 顯示時乘以張數 (shares):
pnl_amount = profit * shares  # 實際損益金額
```

同時在交易明細表頭增加:
- 📊 總損益金額 (所有交易累加)
- 買入金額 = 買入價 × 張數 × 1000
- 賣出金額 = 賣出價 × 張數 × 1000

### 教訓
> **金融系統的數值必須有單位，「損益」必須是金額而非比例。**
> 混用比例和金額是金融軟體最常見的 bug 類型。

---

## Bug 3: BrokenPipeError 崩潰 🔴

### 現象
```
BrokenPipeError: [Errno 32] Broken pipe
  File "optimizer.py", line 225, in _try_experiment
    print(f"  #{self.experiment_count:3d} [{strategy_name:15s}] ")
```
Dashboard 在優化過程中突然崩潰，常發生在切換頁面或長時間運行時。

### 根因分析
1. Streamlit 以 subprocess 方式執行 Python script
2. 使用者切換頁面或瀏覽器斷開時，subprocess 的 stdout 管道被關閉
3. optimizer.py 的 `print()` 寫入已關閉的管道 → BrokenPipeError
4. 異常未被捕獲 → 整個優化過程崩潰

### 修復
optimizer.py 全局替換 print:
```python
import sys as _sys
_original_print = print

def _safe_print(*args, **kwargs):
    """Print that silently ignores BrokenPipeError."""
    try:
        _original_print(*args, **kwargs)
        _sys.stdout.flush()
    except BrokenPipeError:
        pass

print = _safe_print
```

app.py 的 patched `_try_experiment` 也加上:
```python
def _patched_try(strategy_name, params, description):
    try:
        res = original_try(strategy_name, params, description)
    except BrokenPipeError:
        return False
    ...
```

### 教訓
> **任何 subprocess 的 stdout/stderr 輸出都可能在任意時刻斷開。**
> 在 Web 應用中跑長時間任務，所有 I/O 都必須包在 try/except 中。

---

## Bug 4: 設定最少 5 筆但實際顯示 2 筆 🟡

### 現象
```
進階設定: 最少交易次數 = 5
結果: 台塑 (1301.TW) | 交易數: 2 | 勝率: 0.0%
```
為什麼只有 2 筆還能成為「最佳結果」？

### 根因分析
`min_trades` 過濾只在 optimizer 的 `_evaluate()` 中生效:
```python
# optimizer.py
valid_results = [r for r in results if r.num_trades >= self.config.min_trades]
if not valid_results:
    return 0.0, 0.0, results  # 回傳 0 分
```
但最終顯示時，如果所有策略對某檔股票都不到 5 筆，系統會退而取「最不差的」結果，因為總要顯示點什麼。

### 修復
1. 在結果頁面增加警告標示:
```python
if bt.num_trades < min_trades:
    st.warning(f"⚠️ 交易數 {bt.num_trades} 未達最低要求 {min_trades}")
```
2. 最終結果明確區分「有效結果」和「不足交易數的參考結果」

### 教訓
> **過濾條件必須全鏈路一致: 搜尋層、評估層、顯示層都要用同一標準。**

---

## Bug 5: 最佳報酬 +44% 但總損益 -88K 🟡

### 現象
```
台塑 (1301.TW) — RSI 策略
最佳報酬: +44%     ← 看起來很好
總損益:   -88,395 元  ← 居然是虧的
```

### 根因分析
「最佳報酬 +44%」是**最佳化指標的平均值**，包含多檔股票的加權平均。但台塑只有 2 筆交易且都虧損，所以台塑的個別損益是 -88K。

最佳化器選的是「多股票平均最好」的策略，不代表每檔個別都好。

### 教訓
> **平均指標會掩蓋個別表現。** 需要同時顯示:
> - 全體平均指標 (優化目標)
> - 個股明細指標 (實際表現)
> - 最差個股標示 (風險警示)

---

## Bug 6: 布林通道下軌 65% 為何買入 🟡

### 現象
使用者看到交易明細:
```
買入理由: Price < BB lower(65.2)
```
疑問: 「BB lower 65% 為何買入？」

### 根因分析
使用者誤以為 65.2 是百分比，但實際上是**布林下軌的價格** (65.2 元)。
買入理由寫法不夠清楚，沒有解釋布林通道的機制。

### 修復
改善買賣理由的描述:
```python
# Before
f"Price < BB lower({row['bb_lower']:.1f})"

# After
f"收盤{price:.1f} < 布林下軌{row['bb_lower']:.1f}"
```

策略邏輯說明增加量化資訊:
```
📋 策略進出場規則
• 買入: 收盤價 < 布林下軌 (均線 - 1.25倍標準差)
• 賣出: 收盤價 > 布林上軌 (均線 + 1.25倍標準差)

🛡️ 風險管理規則
• ⛔ 停損: 跌幅達 15% 強制賣出
• ⏰ 最大持有: 120 天未觸發賣出訊號強制出場
```

### 教訓
> **技術指標的顯示必須自解釋。** 使用者不一定知道:
> - BB lower 是布林下軌 (一個價格) 不是百分比
> - RSI 70 是超買區間的門檻值
> - MACD hist 是柱狀圖數值
>
> 每個買賣理由都應該用「收盤 XX > 指標 YY」的格式，讓人一看就懂。

---

## Bug 7: 每個策略只跑 30 次搜尋 🟢

### 現象
```
總實驗: 546 次
策略數: 20
每策略: ~27 次
```
策略 17 (籌碼沉澱) 有 5,765,760 種參數組合，27 次搜尋覆蓋率 < 0.0005%。

### 根因分析
`max_combos_per_strategy = 200` 限制了每個策略的搜尋次數，加上 Phase 1~3 分配不均。

### 修復
1. 提高 `max_combos_per_strategy` 到 500+
2. Phase 4 持續搜尋不設上限
3. 時間預算控制 (如 300 秒) 替代固定次數限制
4. 爬山法 patience 提高到 50，鄰居數提高到 8

### 教訓
> **搜尋空間從 5 組到 800 萬組差距巨大，不能用同一個限制值。**
> 小搜尋空間 (vol_divergence: 5 組) 應全覽；大搜尋空間 (chip_sedimentation: 5.7M 組) 需要智能取樣 + 爬山法收斂。

---

## Bug 8: 台股 .TW 忘記加 🟢

### 現象
```
輸入: 2330
結果: yfinance 下載失敗 (找不到 "2330" 這個美股代碼)
```

### 修復
```python
def resolve_symbol(raw_input):
    s = raw_input.strip()
    if s.isdigit():
        return f"{s}.TW", f"{TW_STOCK_NAMES.get(s, '')} ({s}.TW)"
```

### 教訓
> **使用者輸入要做最大程度的寬容處理。** 常見模式:
> - 數字 → 台股代碼，自動加 .TW
> - 中文 → 查名稱對照表
> - 英文 → 美股代碼
> - 已有 .TW → 原樣接受

---

## 設計教訓總結

### 1. 風控優先原則
```
❌ 先開發策略，再「看看要不要加」停損
✅ 停損是系統的第一個功能，策略是第二個
```
任何沒有停損的回測結果都是假的。台塑 -41% 的案例證明了這一點。

### 2. 買賣策略分離原則
```
❌ 買入和賣出綁定在同一個策略裡
✅ 20 種買入策略 × 多層賣出策略 = 更大的搜尋空間
```
RSI 策略適合找買點 (超賣反彈)，但不適合決定賣點 (高檔鈍化問題)。
移動停利比 RSI 超買賣出好太多。

### 3. 顯示即驗證原則
```
❌ 計算邏輯寫好就上線
✅ 每個數字都要有單位，每個理由都要自解釋
```
損益顯示為 2 而非 2,000 → 立刻發現 bug。
布林下軌 65.2 → 使用者以為是百分比 → 立刻發現描述不清。

### 4. 搜尋空間量化原則
```
❌ 所有策略統一 max_combos = 200
✅ 量化每個策略的搜尋空間，動態分配搜尋資源
```
| 策略 | 組合數 | 適合搜尋方式 |
|------|--------|-------------|
| vol_divergence | 5 | 全覽 |
| bollinger_bands | 63 | 全覽 |
| dual_ma_rsi | 37,422 | 隨機 + 爬山 |
| chip_sedimentation | 5,765,760 | 隨機 + 爬山 + 長時間 |

### 5. 全鏈路一致性原則
```
❌ 搜尋層過濾 min_trades=5，但顯示層不過濾
✅ 搜尋、評估、顯示三層用同一標準
```

### 6. subprocess 防禦原則
```
❌ 在 subprocess 中直接 print()
✅ 所有 I/O 操作都用 try/except 包裹
```
Streamlit 的 subprocess 管道隨時可能斷開，`print()` 會拋出 BrokenPipeError。

### 7. 使用者輸入寬容原則
```
❌ 只接受 "2330.TW" 這種精確格式
✅ 接受 "2330"、"台積電"、"tsmc"，自動轉換
```
金融軟體的使用者習慣輸入代碼或名稱，不應該要求記住 Yahoo Finance 的格式。

---

## 修復狀態

| # | Bug | 狀態 | 修復版本 |
|---|-----|------|---------|
| 1 | 無停損 | ✅ 已修復 | v3.0 |
| 2 | 損益數值 | ✅ 已修復 | v3.0 |
| 3 | BrokenPipeError | ✅ 已修復 | v3.0 |
| 4 | min_trades 不一致 | ✅ 已修復 | v3.0 |
| 5 | 平均 vs 個別指標 | ✅ 已修復 | v3.0 |
| 6 | 理由描述不清 | ✅ 已修復 | v3.0 |
| 7 | 搜尋次數不足 | ✅ 已修復 | v3.0 |
| 8 | .TW 自動補全 | ✅ 已修復 | v3.0 |
