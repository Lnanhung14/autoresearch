#!/usr/bin/env python3
"""
Stock Backtest Optimizer - Main Entry Point
============================================

Usage:
  # 1. Use sample data (quick test)
  python -m backtest.run_optimize --demo

  # 2. Use Yahoo Finance data for Taiwan stocks
  python -m backtest.run_optimize --yahoo "2330.TW,2317.TW,2454.TW"

  # 3. Use CSV files in a folder
  python -m backtest.run_optimize --data-dir /path/to/csv/folder

  # 4. Use watchlist file + Yahoo Finance
  python -m backtest.run_optimize --watchlist watchlist.txt --yahoo

  # 5. Specify strategies to test
  python -m backtest.run_optimize --demo --strategies "sma_crossover,rsi,macd"

This script follows the autoresearch optimization loop:
  1. Try strategy + params → 2. Evaluate win_rate → 3. Keep/Discard → 4. Repeat
"""

import argparse
import sys
import os
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data_loader import (
    load_stock_folder, download_yahoo, load_watchlist,
    generate_sample_data, load_csv,
)
from backtest.optimizer import BacktestOptimizer, OptimizationConfig
from backtest.strategies import list_strategies, STRATEGY_REGISTRY


def main():
    parser = argparse.ArgumentParser(
        description="Stock Backtest Strategy Optimizer (autoresearch-style hill climbing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo                          # Quick test with sample data
  %(prog)s --yahoo "2330.TW,2317.TW"       # Taiwan stocks via Yahoo Finance
  %(prog)s --data-dir ./stock_data          # Load CSVs from folder
  %(prog)s --watchlist stocks.txt --yahoo   # Watchlist + Yahoo Finance download
  %(prog)s --demo --strategies "rsi,macd"   # Only test specific strategies
        """
    )

    # Data source options
    data_group = parser.add_argument_group("Data Source (pick one)")
    data_group.add_argument("--demo", action="store_true",
                            help="Use generated sample data for testing")
    data_group.add_argument("--data-dir", type=str,
                            help="Directory containing CSV files (one per stock)")
    data_group.add_argument("--csv", type=str,
                            help="Single CSV file path")
    data_group.add_argument("--yahoo", type=str, nargs='?', const="",
                            help="Download from Yahoo Finance. Comma-separated symbols or use with --watchlist")
    data_group.add_argument("--watchlist", type=str,
                            help="Watchlist file (.txt or .json) with stock symbols")

    # Optimization options
    opt_group = parser.add_argument_group("Optimization Settings")
    opt_group.add_argument("--strategies", type=str, default=None,
                           help="Comma-separated strategy names to test (default: all)")
    opt_group.add_argument("--metric", type=str, default="win_rate",
                           choices=["win_rate", "total_return", "sharpe_ratio", "profit_factor"],
                           help="Primary metric to optimize (default: win_rate)")
    opt_group.add_argument("--min-trades", type=int, default=5,
                           help="Minimum trades for valid result (default: 5)")
    opt_group.add_argument("--patience", type=int, default=30,
                           help="Hill-climbing patience (default: 30)")
    opt_group.add_argument("--max-combos", type=int, default=200,
                           help="Max parameter combos per strategy in grid search (default: 200)")
    opt_group.add_argument("--output", type=str, default="backtest_results.tsv",
                           help="Results output file (default: backtest_results.tsv)")

    # Trading cost options
    cost_group = parser.add_argument_group("Trading Costs (Taiwan defaults)")
    cost_group.add_argument("--commission", type=float, default=0.001425,
                            help="Commission rate (default: 0.001425 for Taiwan)")
    cost_group.add_argument("--tax", type=float, default=0.003,
                            help="Transaction tax rate (default: 0.003 for Taiwan)")
    cost_group.add_argument("--slippage", type=float, default=0.001,
                            help="Slippage (default: 0.001)")

    # Yahoo options
    yahoo_group = parser.add_argument_group("Yahoo Finance Options")
    yahoo_group.add_argument("--start-date", type=str, default="2020-01-01",
                              help="Data start date (default: 2020-01-01)")
    yahoo_group.add_argument("--save-data", type=str, default=None,
                              help="Save downloaded data to this directory")

    parser.add_argument("--list-strategies", action="store_true",
                        help="List all available strategies and exit")

    args = parser.parse_args()

    # List strategies mode
    if args.list_strategies:
        print("\nAvailable Strategies:")
        print("-" * 60)
        for name, sdef in STRATEGY_REGISTRY.items():
            print(f"\n  {name}: {sdef.description}")
            for p in sdef.params:
                print(f"    {p.name}: [{p.min_val} ~ {p.max_val}] step={p.step} default={p.default}")
        print()
        return

    # Load stock data
    stock_data = {}

    if args.demo:
        print("\n[INFO] Using demo data (5 synthetic stocks)")
        for i, name in enumerate(["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D", "STOCK_E"]):
            stock_data[name] = generate_sample_data(name, days=500, start_price=100 + i * 50, seed=42 + i)
        print(f"   Generated {len(stock_data)} stocks, ~500 days each")

    elif args.data_dir:
        print(f"\n[INFO] Loading CSVs from: {args.data_dir}")
        stock_data = load_stock_folder(args.data_dir)
        print(f"   Loaded {len(stock_data)} stocks")

    elif args.csv:
        print(f"\n[INFO] Loading single CSV: {args.csv}")
        symbol = os.path.splitext(os.path.basename(args.csv))[0]
        stock_data[symbol] = load_csv(args.csv)
        print(f"   Loaded {symbol}: {len(stock_data[symbol])} rows")

    elif args.yahoo is not None:
        symbols = []
        if args.watchlist:
            symbols = load_watchlist(args.watchlist)
            print(f"\n[INFO] Loaded watchlist: {len(symbols)} symbols from {args.watchlist}")
        if args.yahoo:  # explicit symbols provided
            symbols.extend(args.yahoo.split(","))
        if not symbols:
            # Default Taiwan blue chips for demo
            symbols = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "3711.TW",
                       "2882.TW", "2881.TW", "1301.TW", "2891.TW", "2303.TW"]
            print(f"\n[INFO] No symbols specified, using Taiwan top 10: {symbols}")
        else:
            print(f"\n[INFO] Downloading {len(symbols)} stocks from Yahoo Finance")

        stock_data = download_yahoo(symbols, start=args.start_date, save_dir=args.save_data)
        print(f"   Downloaded {len(stock_data)} stocks successfully")

    elif args.watchlist:
        symbols = load_watchlist(args.watchlist)
        print(f"\n[INFO] Watchlist loaded ({len(symbols)} symbols), downloading from Yahoo Finance...")
        stock_data = download_yahoo(symbols, start=args.start_date, save_dir=args.save_data)
        print(f"   Downloaded {len(stock_data)} stocks successfully")

    else:
        parser.print_help()
        print("\n[ERROR] Please specify a data source: --demo, --data-dir, --csv, --yahoo, or --watchlist")
        sys.exit(1)

    if not stock_data:
        print("[ERROR] No stock data loaded. Check your data source.")
        sys.exit(1)

    # Configure optimizer
    strategies = args.strategies.split(",") if args.strategies else None
    config = OptimizationConfig(
        primary_metric=args.metric,
        min_trades=args.min_trades,
        results_file=args.output,
        strategies=strategies,
        max_combos_per_strategy=args.max_combos,
        patience=args.patience,
        commission_rate=args.commission,
        tax_rate=args.tax,
        slippage=args.slippage,
    )

    # Run optimization
    start_time = time.time()
    optimizer = BacktestOptimizer(stock_data, config)
    result = optimizer.optimize()
    elapsed = time.time() - start_time

    print(f"\n[耗時]  總計: {elapsed:.1f} 秒 ({result.total_experiments} 次實驗)")

    # Save final best config as JSON for easy reuse
    best_config = {
        "strategy": result.best_strategy,
        "params": {k: float(v) for k, v in result.best_params.items()},
        "metrics": {
            "win_rate": result.best_win_rate,
            "sharpe_ratio": result.best_sharpe,
            "total_return": result.best_total_return,
            "total_trades": result.best_num_trades,
        },
        "optimization_config": {
            "primary_metric": args.metric,
            "min_trades": args.min_trades,
            "commission_rate": args.commission,
            "tax_rate": args.tax,
        }
    }
    best_config_file = args.output.replace('.tsv', '_best.json')
    with open(best_config_file, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    print(f"   最佳策略設定已儲存至: {best_config_file}")


if __name__ == "__main__":
    main()
