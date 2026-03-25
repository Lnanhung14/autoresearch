"""
Core optimization engine - applies autoresearch's greedy hill-climbing logic
to find the best strategy + parameters for stock backtesting.

Optimization Loop (mirrors autoresearch/program.md):
1. Pick a strategy + parameter combination
2. Run backtest → evaluate win_rate (primary metric)
3. If win_rate improved → KEEP (update best)
4. If win_rate worse/equal → DISCARD
5. Record result in results.tsv
6. Loop → step 1

Search Methods:
- Phase 1: Coarse scan across all strategies with default params
- Phase 2: Fine-tune best strategy's parameters via hill-climbing
- Phase 3: Neighborhood search around best params for convergence
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from .strategies import STRATEGY_REGISTRY, StrategyDef, StrategyParam, get_strategy
from .engine import run_backtest, run_backtest_multi_stock, BacktestResult


@dataclass
class OptimizationConfig:
    """Configuration for the optimization run."""
    # Primary metric to optimize (higher is better)
    primary_metric: str = "win_rate"
    # Secondary metric for tie-breaking
    secondary_metric: str = "sharpe_ratio"
    # Minimum number of trades to consider a result valid
    min_trades: int = 5
    # Results output file
    results_file: str = "backtest_results.tsv"
    # Strategies to include (None = all)
    strategies: Optional[List[str]] = None
    # Max parameter combinations per strategy in grid search
    max_combos_per_strategy: int = 200
    # Hill-climbing: max iterations without improvement before stopping
    patience: int = 30
    # Hill-climbing: number of neighbors to sample per iteration
    neighbors_per_step: int = 5
    # Commission and tax rates
    commission_rate: float = 0.001425
    tax_rate: float = 0.003
    slippage: float = 0.001


@dataclass
class OptimizationResult:
    """Final result of the optimization process."""
    best_strategy: str
    best_params: Dict[str, float]
    best_win_rate: float
    best_sharpe: float
    best_total_return: float
    best_num_trades: int
    all_results: List[Dict[str, Any]]
    total_experiments: int
    per_stock_results: List[BacktestResult] = field(default_factory=list)


class BacktestOptimizer:
    """
    Greedy hill-climbing optimizer for stock backtesting strategies.
    Mirrors the autoresearch optimization loop.
    """

    def __init__(self, stock_data: Dict[str, pd.DataFrame], config: Optional[OptimizationConfig] = None):
        self.stock_data = stock_data
        self.config = config or OptimizationConfig()
        self.results_log: List[Dict[str, Any]] = []
        self.best_metric = -float('inf')
        self.best_strategy = None
        self.best_params = None
        self.best_results: List[BacktestResult] = []
        self.experiment_count = 0

    def optimize(self) -> OptimizationResult:
        """
        Run full optimization: coarse scan → fine-tune → converge.
        Returns the best strategy + parameters found.
        """
        print("=" * 70)
        print("🔬 Stock Backtest Optimizer (autoresearch-style hill climbing)")
        print(f"   Stocks: {len(self.stock_data)} | Metric: {self.config.primary_metric}")
        print(f"   Strategies: {self._get_strategy_names()}")
        print("=" * 70)

        # Phase 1: Coarse scan - test all strategies with default params
        print("\n📊 Phase 1: Coarse scan across all strategies (default params)")
        print("-" * 50)
        self._phase1_coarse_scan()

        if self.best_strategy is None:
            print("\n[ERROR] No valid results from any strategy. Check your data.")
            return self._make_result()

        print(f"\n  ✅ Phase 1 best: {self.best_strategy} | "
              f"win_rate={self.best_metric:.2%}")

        # Phase 2: Grid search on best strategy's parameters
        print(f"\n📊 Phase 2: Parameter grid search for [{self.best_strategy}]")
        print("-" * 50)
        self._phase2_grid_search(self.best_strategy)
        print(f"\n  ✅ Phase 2 best: {self.best_strategy} | "
              f"win_rate={self.best_metric:.2%} | params={self.best_params}")

        # Phase 3: Hill-climbing fine-tune around best params
        print(f"\n📊 Phase 3: Hill-climbing refinement")
        print("-" * 50)
        self._phase3_hill_climb()
        print(f"\n  ✅ Phase 3 best: {self.best_strategy} | "
              f"win_rate={self.best_metric:.2%} | params={self.best_params}")

        # Also try top-3 strategies from Phase 1 for grid search
        top_strategies = self._get_top_n_strategies(3)
        for strat_name in top_strategies:
            if strat_name != self.best_strategy:
                print(f"\n📊 Extra: Grid search for [{strat_name}]")
                print("-" * 50)
                self._phase2_grid_search(strat_name)

        # Final hill-climb on overall best
        print(f"\n📊 Final hill-climbing on best: [{self.best_strategy}]")
        print("-" * 50)
        self._phase3_hill_climb()

        # Save results
        self._save_results()

        result = self._make_result()
        self._print_final_report(result)
        return result

    def _get_strategy_names(self) -> List[str]:
        if self.config.strategies:
            return self.config.strategies
        return list(STRATEGY_REGISTRY.keys())

    def _evaluate(self, strategy_name: str, params: Dict[str, float]) -> Tuple[float, float, List[BacktestResult]]:
        """
        Evaluate a strategy+params combination across all stocks.
        Returns (primary_metric_value, secondary_metric_value, results).
        """
        strategy_def = get_strategy(strategy_name)
        avg_metric, results = run_backtest_multi_stock(
            self.stock_data, strategy_def.func, strategy_name, params,
            commission_rate=self.config.commission_rate,
            tax_rate=self.config.tax_rate,
            slippage=self.config.slippage,
        )

        # Filter results with minimum trades
        valid_results = [r for r in results if r.num_trades >= self.config.min_trades]

        if not valid_results:
            return 0.0, 0.0, results

        primary = np.mean([getattr(r, self.config.primary_metric) for r in valid_results])
        secondary = np.mean([getattr(r, self.config.secondary_metric) for r in valid_results])

        return primary, secondary, results

    def _try_experiment(self, strategy_name: str, params: Dict[str, float],
                         description: str = "") -> bool:
        """
        Try one experiment. Returns True if it's the new best.
        This is the core of the autoresearch loop.
        """
        self.experiment_count += 1
        primary, secondary, results = self._evaluate(strategy_name, params)

        valid_results = [r for r in results if r.num_trades >= self.config.min_trades]
        total_trades = sum(r.num_trades for r in valid_results) if valid_results else 0
        avg_return = np.mean([r.total_return for r in valid_results]) if valid_results else 0.0

        # Record result
        status = "discard"
        is_better = False
        if primary > self.best_metric:
            is_better = True
            status = "keep"
            self.best_metric = primary
            self.best_strategy = strategy_name
            self.best_params = params.copy()
            self.best_results = results

        elif primary == self.best_metric and secondary > 0:
            # Tie-break on secondary metric
            current_secondary = 0.0
            if self.best_strategy and self.best_params:
                _, current_secondary, _ = self._evaluate(self.best_strategy, self.best_params)
            if secondary > current_secondary:
                is_better = True
                status = "keep"
                self.best_metric = primary
                self.best_strategy = strategy_name
                self.best_params = params.copy()
                self.best_results = results

        log_entry = {
            'experiment': self.experiment_count,
            'strategy': strategy_name,
            'params': str(params),
            'win_rate': primary,
            'sharpe': secondary,
            'avg_return': avg_return,
            'total_trades': total_trades,
            'valid_stocks': len(valid_results),
            'status': status,
            'description': description,
        }
        self.results_log.append(log_entry)

        marker = "✅ KEEP" if is_better else "  discard"
        print(f"  #{self.experiment_count:3d} [{strategy_name:15s}] "
              f"WR={primary:.2%} Sharpe={secondary:.3f} "
              f"Trades={total_trades:3d} → {marker}")

        return is_better

    # ------------------------------------------------------------------
    # Phase 1: Coarse scan
    # ------------------------------------------------------------------

    def _phase1_coarse_scan(self):
        for name in self._get_strategy_names():
            strategy_def = get_strategy(name)
            default_params = strategy_def.default_params()
            self._try_experiment(name, default_params, f"default params for {name}")

    # ------------------------------------------------------------------
    # Phase 2: Grid search (smart sampling for large search spaces)
    # ------------------------------------------------------------------

    def _phase2_grid_search(self, strategy_name: str):
        strategy_def = get_strategy(strategy_name)
        param_ranges = {p.name: p.range() for p in strategy_def.params}

        # Calculate total combinations
        total_combos = 1
        for vals in param_ranges.values():
            total_combos *= len(vals)

        if total_combos <= self.config.max_combos_per_strategy:
            # Exhaustive grid search
            combos = list(itertools.product(*param_ranges.values()))
            param_names = list(param_ranges.keys())
            print(f"  Exhaustive grid: {total_combos} combinations")
            for combo in combos:
                params = dict(zip(param_names, combo))
                # Skip invalid combos (e.g., fast > slow period)
                if not self._validate_params(strategy_name, params):
                    continue
                self._try_experiment(strategy_name, params, f"grid search")
        else:
            # Random sampling from grid
            print(f"  Random sampling {self.config.max_combos_per_strategy} "
                  f"from {total_combos} combinations")
            param_names = list(param_ranges.keys())
            rng = np.random.RandomState(42)
            seen = set()
            for _ in range(self.config.max_combos_per_strategy):
                combo = tuple(rng.choice(vals) for vals in param_ranges.values())
                key = tuple(round(v, 4) for v in combo)
                if key in seen:
                    continue
                seen.add(key)
                params = dict(zip(param_names, combo))
                if not self._validate_params(strategy_name, params):
                    continue
                self._try_experiment(strategy_name, params, f"random grid sample")

    def _validate_params(self, strategy_name: str, params: Dict[str, float]) -> bool:
        """Validate parameter constraints (e.g., fast < slow period)."""
        if 'fast_period' in params and 'slow_period' in params:
            if params['fast_period'] >= params['slow_period']:
                return False
        if 'oversold' in params and 'overbought' in params:
            if params['oversold'] >= params['overbought']:
                return False
        return True

    # ------------------------------------------------------------------
    # Phase 3: Hill-climbing refinement
    # ------------------------------------------------------------------

    def _phase3_hill_climb(self):
        """
        Hill-climbing around the current best parameters.
        Mirrors autoresearch's greedy keep/discard loop.
        """
        if not self.best_strategy or not self.best_params:
            return

        strategy_def = get_strategy(self.best_strategy)
        param_defs = {p.name: p for p in strategy_def.params}

        no_improvement_count = 0
        iteration = 0

        while no_improvement_count < self.config.patience:
            iteration += 1
            improved_this_round = False

            # Generate neighbors by perturbing one parameter at a time
            neighbors = self._generate_neighbors(
                self.best_params, param_defs, self.config.neighbors_per_step
            )

            for neighbor_params in neighbors:
                if not self._validate_params(self.best_strategy, neighbor_params):
                    continue
                if self._try_experiment(self.best_strategy, neighbor_params,
                                         f"hill-climb iter {iteration}"):
                    improved_this_round = True

            if improved_this_round:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        print(f"  Converged after {iteration} iterations "
              f"({self.config.patience} without improvement)")

    def _generate_neighbors(self, current_params: Dict[str, float],
                             param_defs: Dict[str, StrategyParam],
                             n: int) -> List[Dict[str, float]]:
        """Generate neighbor parameter sets by small perturbations."""
        neighbors = []
        rng = np.random.RandomState()
        param_names = list(current_params.keys())

        for _ in range(n):
            new_params = current_params.copy()
            # Pick 1-2 parameters to perturb
            n_perturb = rng.choice([1, 2])
            perturb_params = rng.choice(param_names, size=min(n_perturb, len(param_names)), replace=False)

            for pname in perturb_params:
                pdef = param_defs[pname]
                current_val = current_params[pname]
                # Perturb by 1-3 steps in either direction
                step_count = rng.choice([-3, -2, -1, 1, 2, 3])
                new_val = current_val + step_count * pdef.step
                new_val = max(pdef.min_val, min(pdef.max_val, new_val))
                new_val = round(new_val, 6)
                new_params[pname] = new_val

            if new_params != current_params:
                neighbors.append(new_params)

        return neighbors

    # ------------------------------------------------------------------
    # Results management
    # ------------------------------------------------------------------

    def _get_top_n_strategies(self, n: int) -> List[str]:
        """Get top N strategies by win rate from results so far."""
        strat_best = {}
        for entry in self.results_log:
            strat = entry['strategy']
            wr = entry['win_rate']
            if strat not in strat_best or wr > strat_best[strat]:
                strat_best[strat] = wr
        sorted_strats = sorted(strat_best.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_strats[:n]]

    def _save_results(self):
        """Save all results to TSV file."""
        if not self.results_log:
            return
        df = pd.DataFrame(self.results_log)
        df.to_csv(self.config.results_file, sep='\t', index=False)
        print(f"\n  Results saved to: {self.config.results_file}")

    def _make_result(self) -> OptimizationResult:
        valid = [r for r in self.best_results if r.num_trades >= self.config.min_trades]
        return OptimizationResult(
            best_strategy=self.best_strategy or "",
            best_params=self.best_params or {},
            best_win_rate=self.best_metric,
            best_sharpe=np.mean([r.sharpe_ratio for r in valid]) if valid else 0.0,
            best_total_return=np.mean([r.total_return for r in valid]) if valid else 0.0,
            best_num_trades=sum(r.num_trades for r in valid),
            all_results=self.results_log,
            total_experiments=self.experiment_count,
            per_stock_results=self.best_results,
        )

    def _print_final_report(self, result: OptimizationResult):
        """Print the final optimization report."""
        print("\n" + "=" * 70)
        print("🏆 OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"  Total experiments:  {result.total_experiments}")
        print(f"  Best strategy:      {result.best_strategy}")
        print(f"  Best parameters:    {result.best_params}")
        print(f"  Average win rate:   {result.best_win_rate:.2%}")
        print(f"  Average Sharpe:     {result.best_sharpe:.3f}")
        print(f"  Average return:     {result.best_total_return:.2%}")
        print(f"  Total trades:       {result.best_num_trades}")

        if result.per_stock_results:
            print(f"\n  Per-stock breakdown:")
            print(f"  {'Symbol':<12} {'WinRate':>8} {'Return':>10} {'Trades':>7} {'Sharpe':>8}")
            print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*7} {'-'*8}")
            for r in sorted(result.per_stock_results, key=lambda x: x.win_rate, reverse=True):
                if r.num_trades >= self.config.min_trades:
                    print(f"  {r.symbol:<12} {r.win_rate:>7.2%} {r.total_return:>9.2%} "
                          f"{r.num_trades:>7d} {r.sharpe_ratio:>8.3f}")

        print("=" * 70)
