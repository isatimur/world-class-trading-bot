#!/usr/bin/env python3
"""
Telegram Bot Main Entry Point

This script runs the Telegram trading bot with full integration
to the World-Class Trading Bot system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from trading_bot import TelegramTradingBot, Settings, get_logger
from trading_bot.strategies import StrategyManager, StrategyAllocation
from trading_bot.backtesting import BacktestEngine, BacktestConfig
from trading_bot.strategies import GridStrategy, MLStrategy, MeanReversionStrategy, MomentumStrategy

logger = get_logger(__name__)


async def main():
    """Main function to run the Telegram trading bot."""
    
    logger.info("🚀 Starting World-Class Trading Bot with Telegram Integration")
    
    # Check for Telegram token
    if not Settings.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not configured. Please set it in your environment.")
        logger.info("You can set it in your .env file or as an environment variable.")
        return
    
    # Initialize strategy manager
    logger.info("📊 Initializing strategy manager...")
    strategy_manager = StrategyManager(
        manager_id="telegram_bot_portfolio",
        total_capital=100000.0,
        allocation_method="equal_weight",
        rebalance_frequency=30,
        risk_free_rate=0.02,
        max_portfolio_risk=0.15
    )
    
    # Create and add strategies
    logger.info("🔧 Creating trading strategies...")
    
    # Grid Strategy
    grid_strategy = GridStrategy(
        strategy_id="grid_advanced",
        symbol="AAPL",
        grid_levels=15,
        grid_spacing_pct=0.015,
        ml_enabled=True,
        adaptive_grid=True,
        initial_capital=20000.0,
        risk_per_trade=0.02
    )
    
    # Mean Reversion Strategy
    mean_reversion_strategy = MeanReversionStrategy(
        strategy_id="mean_reversion",
        symbol="MSFT",
        lookback_period=50,
        z_score_threshold=2.0,
        volatility_lookback=20,
        mean_reversion_strength=0.7,
        volatility_adjustment=True,
        ml_enabled=True,
        initial_capital=20000.0,
        risk_per_trade=0.02
    )
    
    # ML Strategy
    ml_strategy = MLStrategy(
        strategy_id="ml_ensemble",
        symbol="GOOGL",
        feature_lookback=50,
        prediction_horizon=5,
        ensemble_method='weighted',
        retrain_frequency=500,
        initial_capital=20000.0,
        risk_per_trade=0.015
    )
    
    # Momentum Strategy
    momentum_strategy = MomentumStrategy(
        strategy_id="momentum",
        symbol="TSLA",
        short_period=10,
        long_period=30,
        momentum_threshold=0.02,
        trend_confirmation_periods=3,
        volatility_lookback=20,
        momentum_strength=0.8,
        dynamic_sizing=True,
        ml_enabled=True,
        initial_capital=20000.0,
        risk_per_trade=0.02
    )
    
    # Create strategy allocations
    grid_allocation = StrategyAllocation(
        strategy_id="grid_advanced",
        weight=0.25,
        max_allocation=0.35,
        min_allocation=0.15,
        risk_budget=0.25,
        is_active=True
    )
    
    mean_reversion_allocation = StrategyAllocation(
        strategy_id="mean_reversion",
        weight=0.30,
        max_allocation=0.40,
        min_allocation=0.20,
        risk_budget=0.30,
        is_active=True
    )
    
    ml_allocation = StrategyAllocation(
        strategy_id="ml_ensemble",
        weight=0.25,
        max_allocation=0.35,
        min_allocation=0.15,
        risk_budget=0.25,
        is_active=True
    )
    
    momentum_allocation = StrategyAllocation(
        strategy_id="momentum",
        weight=0.20,
        max_allocation=0.30,
        min_allocation=0.10,
        risk_budget=0.20,
        is_active=True
    )
    
    # Add strategies to manager
    strategy_manager.add_strategy(grid_strategy, grid_allocation)
    strategy_manager.add_strategy(mean_reversion_strategy, mean_reversion_allocation)
    strategy_manager.add_strategy(ml_strategy, ml_allocation)
    strategy_manager.add_strategy(momentum_strategy, momentum_allocation)
    
    # Initialize Telegram bot
    logger.info("🤖 Initializing Telegram bot...")
    telegram_bot = TelegramTradingBot(
        token=Settings.TELEGRAM_BOT_TOKEN,
        strategy_manager=strategy_manager
    )
    
    # Send startup notification
    startup_message = """
🚀 **World-Class Trading Bot Started!**

**System Status**: Online
**Trading Mode**: Paper Trading
**Active Strategies**: 4
**Total Capital**: $100,000

**Active Strategies:**
• Grid Strategy (AAPL) - 25% allocation
• Mean Reversion (MSFT) - 30% allocation
• ML Ensemble (GOOGL) - 25% allocation
• Momentum (TSLA) - 20% allocation

**Commands Available:**
• `/start` - Initialize bot
• `/status` - System status
• `/portfolio` - Portfolio overview
• `/strategies` - Strategy management
• `/performance` - Performance metrics
• `/signals` - Trading signals
• `/backtest` - Run backtesting
• `/help` - Show all commands

**Ready to trade!** 📈
    """
    
    try:
        # Start the Telegram bot
        logger.info("📱 Starting Telegram bot...")
        await telegram_bot.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal...")
    except Exception as e:
        logger.error(f"❌ Error running Telegram bot: {e}")
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up...")
        await telegram_bot.stop()
        logger.info("✅ Telegram bot stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1) 