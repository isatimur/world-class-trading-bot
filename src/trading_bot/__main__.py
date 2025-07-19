"""
Main entry point for the Trading Bot application.

This module provides the main entry point for running the trading bot
from the command line.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from trading_bot.telegram.telegram_bot import TelegramTradingBot
from trading_bot.config.settings import Settings
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


async def main():
    """Main entry point for the trading bot."""
    try:
        logger.info("Starting Trading Bot...")
        
        # Check if Telegram token is configured
        if not Settings.TELEGRAM_BOT_TOKEN:
            logger.error("TELEGRAM_BOT_TOKEN not configured. Please set it in your environment.")
            sys.exit(1)
        
        # Create and run the telegram bot
        bot = TelegramTradingBot(
            token=Settings.TELEGRAM_BOT_TOKEN
        )
        
        logger.info("Trading Bot started successfully")
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Trading Bot stopped by user")
    except Exception as e:
        logger.error(f"Error starting Trading Bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 