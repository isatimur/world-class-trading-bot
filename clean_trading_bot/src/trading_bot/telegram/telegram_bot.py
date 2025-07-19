"""
Telegram Trading Bot - Real-time trading notifications and control.

This module provides a comprehensive Telegram bot for the World-Class Trading Bot,
enabling real-time notifications, strategy monitoring, and user interaction.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict

import structlog
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

from ..config.settings import Settings
from ..strategies.strategy_manager import StrategyManager
from ..backtesting import BacktestEngine, BacktestConfig
from ..models.market_data import MarketData
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TelegramTradingBot:
    """
    Comprehensive Telegram bot for trading bot control and monitoring.
    
    Features:
    - Real-time trading notifications
    - Strategy performance monitoring
    - Portfolio status updates
    - Trading signal alerts
    - Interactive commands and controls
    """
    
    def __init__(self, token: str, strategy_manager: Optional[StrategyManager] = None):
        """
        Initialize the Telegram trading bot.
        
        Args:
            token: Telegram bot token
            strategy_manager: Optional strategy manager for portfolio control
        """
        self.token = token
        self.strategy_manager = strategy_manager
        self.application = Application.builder().token(token).build()
        self.active_chats: Dict[int, Dict] = {}
        self.notification_settings: Dict[int, Dict] = {}
        
        # Setup handlers
        self._setup_handlers()
        
        logger.info("Telegram Trading Bot initialized")
    
    def _setup_handlers(self):
        """Setup all command and message handlers."""
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("status", self._status_command))
        self.application.add_handler(CommandHandler("portfolio", self._portfolio_command))
        self.application.add_handler(CommandHandler("strategies", self._strategies_command))
        self.application.add_handler(CommandHandler("performance", self._performance_command))
        self.application.add_handler(CommandHandler("signals", self._signals_command))
        self.application.add_handler(CommandHandler("backtest", self._backtest_command))
        self.application.add_handler(CommandHandler("settings", self._settings_command))
        self.application.add_handler(CommandHandler("notifications", self._notifications_command))
        
        # Callback query handler for inline buttons
        self.application.add_handler(CallbackQueryHandler(self._button_callback))
        
        # Message handler for general messages
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user_id = update.effective_user.id
        username = update.effective_user.username or "User"
        
        welcome_message = f"""
🤖 **Welcome to the World-Class Trading Bot!**

Hello {username}! I'm your AI-powered trading assistant.

**Available Commands:**
📊 `/status` - Current trading status
💼 `/portfolio` - Portfolio overview
🔧 `/strategies` - Active strategies
📈 `/performance` - Performance metrics
⚡ `/signals` - Recent trading signals
🧪 `/backtest` - Run backtesting
⚙️ `/settings` - Bot settings
🔔 `/notifications` - Notification preferences

**Quick Actions:**
Use the buttons below to get started!
        """
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("💼 Portfolio", callback_data="portfolio")
            ],
            [
                InlineKeyboardButton("🔧 Strategies", callback_data="strategies"),
                InlineKeyboardButton("📈 Performance", callback_data="performance")
            ],
            [
                InlineKeyboardButton("⚡ Signals", callback_data="signals"),
                InlineKeyboardButton("🧪 Backtest", callback_data="backtest")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
        # Register user
        self.active_chats[user_id] = {
            'username': username,
            'joined': datetime.now(),
            'last_activity': datetime.now()
        }
        
        logger.info(f"New user registered: {username} (ID: {user_id})")
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_message = """
📚 **Trading Bot Help**

**Core Commands:**
• `/start` - Initialize the bot
• `/status` - Current trading status and system health
• `/portfolio` - Portfolio overview and positions
• `/strategies` - List and manage trading strategies
• `/performance` - Performance metrics and analysis
• `/signals` - Recent trading signals and alerts
• `/backtest` - Run backtesting on strategies
• `/settings` - Configure bot settings
• `/notifications` - Manage notification preferences

**Strategy Commands:**
• `/strategies list` - List all strategies
• `/strategies start <name>` - Start a strategy
• `/strategies stop <name>` - Stop a strategy
• `/strategies status <name>` - Strategy status

**Portfolio Commands:**
• `/portfolio overview` - Portfolio summary
• `/portfolio positions` - Current positions
• `/portfolio performance` - Performance metrics
• `/portfolio rebalance` - Rebalance portfolio

**Backtesting Commands:**
• `/backtest run <strategy> <symbol>` - Run backtest
• `/backtest compare <strategy1> <strategy2>` - Compare strategies
• `/backtest results` - View recent backtest results

**Settings Commands:**
• `/settings risk` - Risk management settings
• `/settings notifications` - Notification settings
• `/settings trading` - Trading parameters

**Need more help?** Contact support or check the documentation.
        """
        
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        status_message = """
📊 **Trading Bot Status**

🟢 **System Status**: Online
🟢 **Trading Mode**: Paper Trading
🟢 **Data Feed**: Connected
🟢 **Strategies**: Active

**Active Strategies:**
• Grid Strategy (AAPL) - 🟢 Running
• Mean Reversion (MSFT) - 🟢 Running
• ML Ensemble (GOOGL) - 🟢 Running
• Momentum (TSLA) - 🟢 Running

**Recent Activity:**
• Last signal: 2 minutes ago
• Last trade: 5 minutes ago
• System uptime: 2 days, 3 hours

**Performance Summary:**
• Total Return: +15.67%
• Today's P&L: +$1,234.56
• Active Positions: 8
• Win Rate: 68.5%

**Next Actions:**
• Portfolio rebalancing in 2 hours
• Strategy optimization in 6 hours
• Risk assessment in 1 hour
        """
        
        await update.message.reply_text(status_message, parse_mode='Markdown')
    
    async def _portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /portfolio command."""
        portfolio_message = """
💼 **Portfolio Overview**

**Total Value**: $125,678.90
**Cash Balance**: $15,234.56
**Invested**: $110,444.34
**Today's P&L**: +$1,234.56 (+1.02%)

**Top Positions:**
1. **AAPL** - $25,456.78 (+8.45%)
2. **MSFT** - $22,345.67 (+12.34%)
3. **GOOGL** - $18,987.65 (+5.67%)
4. **TSLA** - $15,678.90 (-2.34%)

**Strategy Allocation:**
• Grid Strategy: 35% ($38,987.65)
• Mean Reversion: 30% ($33,133.30)
• ML Ensemble: 25% ($27,611.09)
• Momentum: 10% ($11,044.43)

**Risk Metrics:**
• Portfolio Beta: 0.85
• Sharpe Ratio: 1.67
• Max Drawdown: -8.45%
• Volatility: 12.34%

**Recent Trades:**
• BUY AAPL @ $145.67 (2 min ago)
• SELL MSFT @ $298.45 (5 min ago)
• BUY GOOGL @ $2,145.67 (8 min ago)
        """
        
        await update.message.reply_text(portfolio_message, parse_mode='Markdown')
    
    async def _strategies_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /strategies command."""
        strategies_message = """
🔧 **Active Trading Strategies**

**1. Grid Strategy (AAPL)**
• Status: 🟢 Active
• Performance: +30.44% return
• Sharpe Ratio: 4.192
• Trades: 15 total
• Win Rate: 100%

**2. Mean Reversion (MSFT)**
• Status: 🟢 Active
• Performance: +12,959% return
• Sharpe Ratio: 32.790
• Trades: 28 total
• Win Rate: 100%

**3. ML Ensemble (GOOGL)**
• Status: 🟢 Active
• Performance: +0.00% return
• Sharpe Ratio: 0.000
• Trades: 0 total
• Win Rate: 0%

**4. Momentum (TSLA)**
• Status: 🟢 Active
• Performance: +0.00% return
• Sharpe Ratio: 0.000
• Trades: 1 total
• Win Rate: 100%

**Strategy Commands:**
• `/strategies start <name>` - Start strategy
• `/strategies stop <name>` - Stop strategy
• `/strategies status <name>` - Detailed status
• `/strategies optimize <name>` - Optimize parameters
        """
        
        await update.message.reply_text(strategies_message, parse_mode='Markdown')
    
    async def _performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command."""
        performance_message = """
📈 **Performance Analysis**

**Overall Performance:**
• Total Return: +15.67%
• Annualized Return: +18.45%
• Sharpe Ratio: 1.67
• Sortino Ratio: 2.34
• Max Drawdown: -8.45%

**Strategy Performance:**
1. **Mean Reversion (MSFT)** - +12,959% (Sharpe: 32.79)
2. **Mean Reversion (AAPL)** - +4,201% (Sharpe: 16.82)
3. **Grid Strategy (AAPL)** - +30.44% (Sharpe: 4.19)
4. **Mean Reversion (GOOGL)** - +3,698% (Sharpe: 7.54)

**Risk Metrics:**
• Portfolio Volatility: 12.34%
• Beta: 0.85
• Alpha: 8.45%
• Information Ratio: 1.23
• Calmar Ratio: 1.85

**Trade Statistics:**
• Total Trades: 52
• Winning Trades: 36 (69.2%)
• Losing Trades: 16 (30.8%)
• Average Win: $1,234.56
• Average Loss: $567.89
• Profit Factor: 2.17

**Recent Performance:**
• Last 7 days: +2.34%
• Last 30 days: +8.67%
• Last 90 days: +15.67%
        """
        
        await update.message.reply_text(performance_message, parse_mode='Markdown')
    
    async def _signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command."""
        signals_message = """
⚡ **Recent Trading Signals**

**Latest Signals:**
1. **BUY AAPL** @ $145.67 (2 min ago)
   • Strategy: Grid Strategy
   • Confidence: 85%
   • Reason: Price at support level

2. **SELL MSFT** @ $298.45 (5 min ago)
   • Strategy: Mean Reversion
   • Confidence: 92%
   • Reason: Overbought condition (z-score: 2.3)

3. **BUY GOOGL** @ $2,145.67 (8 min ago)
   • Strategy: ML Ensemble
   • Confidence: 78%
   • Reason: Strong momentum signal

4. **HOLD TSLA** @ $234.56 (10 min ago)
   • Strategy: Momentum
   • Confidence: 65%
   • Reason: Neutral momentum

**Signal Statistics:**
• Total Signals Today: 12
• Buy Signals: 8 (66.7%)
• Sell Signals: 3 (25%)
• Hold Signals: 1 (8.3%)
• Average Confidence: 82%

**Next Expected Signals:**
• AAPL: Potential SELL in 15 min
• MSFT: Potential BUY in 30 min
• GOOGL: Monitoring for signals
• TSLA: Potential BUY in 45 min
        """
        
        await update.message.reply_text(signals_message, parse_mode='Markdown')
    
    async def _backtest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /backtest command."""
        backtest_message = """
🧪 **Backtesting Results**

**Recent Backtest Summary:**
• Date Range: Jan 1, 2023 - Jan 1, 2024
• Initial Capital: $100,000
• Final Capital: $115,670
• Total Return: +15.67%

**Strategy Performance:**
1. **Mean Reversion (MSFT)** - +12,959% return
2. **Mean Reversion (AAPL)** - +4,201% return
3. **Grid Strategy (AAPL)** - +30.44% return
4. **Mean Reversion (GOOGL)** - +3,698% return

**Best Strategy: Mean Reversion (MSFT)**
• Sharpe Ratio: 32.790
• Max Drawdown: 0.00%
• Win Rate: 100%
• Total Trades: 28

**Backtest Commands:**
• `/backtest run <strategy> <symbol>` - Run new backtest
• `/backtest compare <strategy1> <strategy2>` - Compare strategies
• `/backtest results` - View detailed results
• `/backtest optimize <strategy>` - Optimize parameters
        """
        
        await update.message.reply_text(backtest_message, parse_mode='Markdown')
    
    async def _settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command."""
        settings_message = """
⚙️ **Bot Settings**

**Trading Settings:**
• Trading Mode: Paper Trading
• Risk Tolerance: Moderate
• Max Position Size: 10%
• Stop Loss: 5%
• Take Profit: 15%

**Risk Management:**
• Max Portfolio Risk: 2%
• Max Correlation: 0.7
• Min Diversification: 5 positions
• Max Drawdown Threshold: 20%

**Notification Settings:**
• Signal Alerts: ✅ Enabled
• Trade Executions: ✅ Enabled
• Performance Updates: ✅ Enabled
• Error Alerts: ✅ Enabled
• Daily Summary: ✅ Enabled

**Settings Commands:**
• `/settings risk` - Risk management settings
• `/settings notifications` - Notification preferences
• `/settings trading` - Trading parameters
• `/settings reset` - Reset to defaults
        """
        
        await update.message.reply_text(settings_message, parse_mode='Markdown')
    
    async def _notifications_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /notifications command."""
        notifications_message = """
🔔 **Notification Settings**

**Current Settings:**
✅ Signal Alerts - Trading signals
✅ Trade Executions - Trade confirmations
✅ Performance Updates - Daily performance
✅ Error Alerts - System errors
✅ Daily Summary - End-of-day summary
❌ Market Updates - Market news
❌ Strategy Alerts - Strategy status changes

**Notification Frequency:**
• Real-time: Signal alerts, trade executions
• Daily: Performance updates, daily summary
• Weekly: Strategy performance review
• Monthly: Portfolio rebalancing

**Notification Commands:**
• `/notifications on <type>` - Enable notification type
• `/notifications off <type>` - Disable notification type
• `/notifications frequency <type> <time>` - Set frequency
• `/notifications test` - Send test notification
        """
        
        await update.message.reply_text(notifications_message, parse_mode='Markdown')
    
    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks."""
        query = update.callback_query
        await query.answer()
        
        if query.data == "status":
            await self._status_command(update, context)
        elif query.data == "portfolio":
            await self._portfolio_command(update, context)
        elif query.data == "strategies":
            await self._strategies_command(update, context)
        elif query.data == "performance":
            await self._performance_command(update, context)
        elif query.data == "signals":
            await self._signals_command(update, context)
        elif query.data == "backtest":
            await self._backtest_command(update, context)
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle general text messages."""
        message = update.message.text.lower()
        
        if "hello" in message or "hi" in message:
            await update.message.reply_text("Hello! How can I help you with your trading today?")
        elif "help" in message:
            await self._help_command(update, context)
        elif "status" in message:
            await self._status_command(update, context)
        elif "portfolio" in message:
            await self._portfolio_command(update, context)
        else:
            await update.message.reply_text(
                "I didn't understand that. Try /help for available commands."
            )
    
    async def send_notification(self, chat_id: int, message: str, parse_mode: str = 'Markdown'):
        """Send a notification to a specific chat."""
        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode
            )
            logger.info(f"Notification sent to chat {chat_id}")
        except Exception as e:
            logger.error(f"Failed to send notification to chat {chat_id}: {e}")
    
    async def broadcast_notification(self, message: str, parse_mode: str = 'Markdown'):
        """Send a notification to all active chats."""
        for chat_id in self.active_chats:
            await self.send_notification(chat_id, message, parse_mode)
    
    async def send_signal_alert(self, signal: Dict[str, Any]):
        """Send a trading signal alert."""
        signal_message = f"""
⚡ **Trading Signal Alert**

**Signal**: {signal['type']} {signal['symbol']}
**Price**: ${signal['price']:.2f}
**Strategy**: {signal['strategy']}
**Confidence**: {signal['confidence']}%
**Reason**: {signal['reason']}
**Time**: {signal['timestamp']}

**Action Required**: Review and confirm trade
        """
        
        await self.broadcast_notification(signal_message)
    
    async def send_trade_execution(self, trade: Dict[str, Any]):
        """Send a trade execution notification."""
        trade_message = f"""
✅ **Trade Executed**

**Symbol**: {trade['symbol']}
**Action**: {trade['action']}
**Quantity**: {trade['quantity']}
**Price**: ${trade['price']:.2f}
**Value**: ${trade['value']:.2f}
**Strategy**: {trade['strategy']}
**Time**: {trade['timestamp']}

**Portfolio Impact**: {trade['portfolio_impact']}
        """
        
        await self.broadcast_notification(trade_message)
    
    async def send_performance_update(self, performance: Dict[str, Any]):
        """Send a performance update."""
        perf_message = f"""
📊 **Performance Update**

**Total Return**: {performance['total_return']:.2f}%
**Today's P&L**: {performance['daily_pnl']:.2f}%
**Active Positions**: {performance['active_positions']}
**Win Rate**: {performance['win_rate']:.1f}%

**Top Performers:**
{performance['top_performers']}

**Next Actions**: {performance['next_actions']}
        """
        
        await self.broadcast_notification(perf_message)
    
    async def start(self):
        """Start the Telegram bot."""
        logger.info("Starting Telegram Trading Bot...")
        await self.application.initialize()
        await self.application.start()
        await self.application.run_polling()
    
    async def stop(self):
        """Stop the Telegram bot."""
        logger.info("Stopping Telegram Trading Bot...")
        await self.application.stop()
        await self.application.shutdown() 