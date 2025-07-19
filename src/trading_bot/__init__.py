"""
Trading Bot - World-Class AI-Powered Trading System

A comprehensive trading bot with advanced ML models, real trading strategies,
and professional-grade backtesting capabilities.
"""

__version__ = "2.0.0"
__author__ = "Trading Bot Team"
__description__ = "World-Class AI-Powered Trading System with ML Models and Real Strategies"

# Configuration
from .config.settings import Settings

# Utils
from .utils.logging import get_logger

# Models
from .models.market_data import MarketData, StockData, CryptoData, MarketSentiment, MarketOverview
from .models.portfolio import Position, Trade, Portfolio, PerformanceMetrics, PortfolioSummary

# Strategies
from .strategies.base_strategy import BaseStrategy, Signal, SignalType, StrategyType, StrategyPerformance
from .strategies.strategy_manager import StrategyManager, StrategyAllocation
from .strategies.grid_strategy import GridStrategy
from .strategies.ml_strategy import MLStrategy
from .strategies.mean_reversion_strategy import MeanReversionStrategy
from .strategies.momentum_strategy import MomentumStrategy

# Backtesting
from .backtesting import BacktestEngine, BacktestConfig, BacktestResult

# Telegram Bot
from .telegram.telegram_bot import TelegramTradingBot

__all__ = [
    "Settings",
    "get_logger",
    "MarketData",
    "StockData", 
    "CryptoData",
    "MarketSentiment",
    "MarketOverview",
    "Position",
    "Trade",
    "Portfolio",
    "PerformanceMetrics",
    "PortfolioSummary",
    "BaseStrategy",
    "Signal",
    "SignalType",
    "StrategyType",
    "StrategyPerformance",
    "StrategyManager",
    "StrategyAllocation",
    "GridStrategy",
    "MLStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "TelegramTradingBot",
] 