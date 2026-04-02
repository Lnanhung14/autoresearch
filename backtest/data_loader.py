"""
Stock data loader - supports multiple data sources.
Users can plug in their own data or use built-in downloaders.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading from CSV files (most common for Taiwan stocks)
# ---------------------------------------------------------------------------

def load_csv(filepath: str, date_col: str = 'date') -> pd.DataFrame:
    """
    Load stock OHLCV data from CSV file.
    Expected columns: date, open, high, low, close, volume
    """
    df = pd.read_csv(filepath)
    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Common column name mappings (支援中文欄位名)
    col_map = {
        '日期': 'date', '開盤價': 'open', '最高價': 'high',
        '最低價': 'low', '收盤價': 'close', '成交量': 'volume',
        '成交股數': 'volume', 'adj close': 'adj_close',
    }
    df.rename(columns=col_map, inplace=True)

    if date_col in df.columns:
        df['date'] = pd.to_datetime(df[date_col])
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True, drop=False)

    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    # Clean data
    df = df.dropna(subset=['close'])
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

    return df


def load_stock_folder(folder: str, pattern: str = "*.csv") -> Dict[str, pd.DataFrame]:
    """Load all CSV files from a folder. Returns {symbol: DataFrame}."""
    folder = Path(folder)
    stock_data = {}
    for f in sorted(folder.glob(pattern)):
        symbol = f.stem  # filename without extension as symbol
        try:
            df = load_csv(str(f))
            if len(df) > 30:  # need minimum data for meaningful backtest
                stock_data[symbol] = df
        except Exception as e:
            print(f"  [WARN] Failed to load {f.name}: {e}")
    return stock_data


# ---------------------------------------------------------------------------
# Yahoo Finance downloader (optional, requires yfinance)
# ---------------------------------------------------------------------------

def download_yahoo(symbols: List[str], start: str = "2020-01-01",
                    end: Optional[str] = None, save_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Download stock data from Yahoo Finance.

    Args:
        symbols: List of stock symbols (e.g., ["2330.TW", "2317.TW"] for Taiwan stocks)
        start: Start date string
        end: End date string (default: today)
        save_dir: If provided, save CSVs to this directory
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    stock_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)
            if df.empty:
                print(f"  [WARN] No data for {symbol}")
                continue

            df.columns = [c.lower() for c in df.columns]
            df['date'] = df.index
            df = df.reset_index(drop=True)

            # Rename columns to standard format
            rename_map = {'stock splits': 'splits', 'capital gains': 'cap_gains'}
            df.rename(columns=rename_map, inplace=True)

            stock_data[symbol] = df

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                safe_name = symbol.replace('.', '_').replace('/', '_')
                df.to_csv(os.path.join(save_dir, f"{safe_name}.csv"), index=False)
                print(f"  Saved {symbol} -> {safe_name}.csv ({len(df)} rows)")

        except Exception as e:
            print(f"  [WARN] Failed to download {symbol}: {e}")

    return stock_data


# ---------------------------------------------------------------------------
# Watchlist (自選股) management
# ---------------------------------------------------------------------------

def load_watchlist(filepath: str) -> List[str]:
    """
    Load watchlist from a text/JSON file.

    Supports:
    - .txt: one symbol per line
    - .json: list of symbols or {"symbols": [...]}
    """
    filepath = Path(filepath)
    if filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'symbols' in data:
            return data['symbols']
        else:
            raise ValueError("JSON watchlist must be a list or {\"symbols\": [...]}")
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def save_watchlist(symbols: List[str], filepath: str):
    """Save watchlist to file."""
    filepath = Path(filepath)
    if filepath.suffix == '.json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"symbols": symbols}, f, indent=2, ensure_ascii=False)
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            for s in symbols:
                f.write(f"{s}\n")


# ---------------------------------------------------------------------------
# Generate sample data for testing
# ---------------------------------------------------------------------------

def generate_sample_data(symbol: str = "SAMPLE", days: int = 500,
                          start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=days)

    price = start_price
    data = []
    for date in dates:
        daily_return = rng.normal(0.0005, 0.02)  # slight upward drift
        open_price = price
        close_price = price * (1 + daily_return)
        high_price = max(open_price, close_price) * (1 + abs(rng.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(rng.normal(0, 0.005)))
        volume = int(rng.lognormal(15, 1))

        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
        })
        price = close_price

    df = pd.DataFrame(data)
    df.set_index('date', inplace=True, drop=False)
    return df
