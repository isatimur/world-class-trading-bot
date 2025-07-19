"""
Backtesting framework for trading strategies.

This package provides comprehensive backtesting capabilities for all
trading strategies with detailed performance analysis and visualization.
"""

from .backtest_engine import BacktestEngine, BacktestConfig, BacktestResult

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
] 