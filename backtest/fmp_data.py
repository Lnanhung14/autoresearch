"""
FMP (Financial Modeling Prep) API data fetcher.
Fetches historical OHLCV data for stocks.
"""

import requests
import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta


FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


def fetch_historical_data(
    symbol: str,
    api_key: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch historical daily OHLCV data from FMP API.

    Args:
        symbol: Stock symbol (e.g., "AAPL", "2330.TW")
        api_key: FMP API key
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    url = f"{FMP_BASE_URL}/historical-price-full/{symbol}"
    params = {
        "from": start_date,
        "to": end_date,
        "apikey": api_key,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "historical" not in data or not data["historical"]:
        raise ValueError(f"No historical data for {symbol}")

    df = pd.DataFrame(data["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Standardize column names
    df = df.rename(columns={"adjClose": "adj_close"})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.set_index("date", inplace=True, drop=False)
    return df


def fetch_multiple_stocks(
    symbols: List[str],
    api_key: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    delay: float = 0.3,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple stocks.

    Args:
        symbols: List of stock symbols
        api_key: FMP API key
        start_date: Start date
        end_date: End date
        delay: Delay between requests (seconds) to respect rate limits

    Returns:
        Dict of {symbol: DataFrame}
    """
    stock_data = {}
    for i, symbol in enumerate(symbols):
        try:
            df = fetch_historical_data(symbol, api_key, start_date, end_date)
            if len(df) >= 30:
                stock_data[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} days loaded")
            else:
                print(f"  ✗ {symbol}: insufficient data ({len(df)} days)")
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")

        if i < len(symbols) - 1:
            time.sleep(delay)

    return stock_data


def search_symbol(query: str, api_key: str, limit: int = 10) -> List[Dict]:
    """
    Search for stock symbols on FMP.

    Returns list of dicts with: symbol, name, currency, stockExchange
    """
    url = f"{FMP_BASE_URL}/search"
    params = {"query": query, "limit": limit, "apikey": api_key}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def validate_api_key(api_key: str) -> bool:
    """Check if the FMP API key is valid."""
    try:
        url = f"{FMP_BASE_URL}/historical-price-full/AAPL"
        params = {"timeseries": 1, "apikey": api_key}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return "historical" in data
    except Exception:
        return False
