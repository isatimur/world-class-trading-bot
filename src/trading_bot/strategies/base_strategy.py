"""
Base strategy class for all trading strategies.

This module provides the foundation for implementing real trading strategies
with proper risk management, position sizing, and performance tracking.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from ..models.market_data import MarketData, StockData, CryptoData
from ..models.portfolio import Position, Trade, PerformanceMetrics
from ..config.settings import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class StrategyType(Enum):
    """Strategy types"""
    GRID = "GRID"
    MEAN_REVERSION = "MEAN_REVERSION"
    MOMENTUM = "MOMENTUM"
    ML = "ML"
    CUSTOM = "CUSTOM"


@dataclass
class Signal:
    """Trading signal with metadata"""
    signal_type: SignalType
    symbol: str
    price: float
    quantity: float
    confidence: float
    timestamp: datetime
    reasoning: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class StrategyState:
    """Strategy state and parameters"""
    strategy_id: str
    symbol: str
    is_active: bool = True
    current_position: Optional[Position] = None
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    parameters: Dict = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)


class StrategyPerformance(BaseModel):
    """Strategy performance metrics"""
    strategy_id: str
    symbol: str
    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    win_rate: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    total_trades: int = Field(..., description="Total number of trades")
    avg_trade_duration: float = Field(..., description="Average trade duration in hours")
    best_trade: float = Field(..., description="Best single trade profit")
    worst_trade: float = Field(..., description="Worst single trade loss")
    timestamp: datetime = Field(default_factory=datetime.now)


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    This class provides the foundation for implementing real trading strategies
    with proper risk management, position sizing, and performance tracking.
    """
    
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        strategy_type: StrategyType,
        initial_capital: float = 10000.0,
        risk_per_trade: float = 0.02,  # 2% risk per trade
        max_position_size: float = 0.1,  # 10% max position size
        stop_loss_pct: float = 0.05,  # 5% stop loss
        take_profit_pct: float = 0.15,  # 15% take profit
        **kwargs
    ):
        """
        Initialize the base strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            symbol: Trading symbol
            strategy_type: Type of strategy
            initial_capital: Initial capital for the strategy
            risk_per_trade: Risk per trade as percentage of capital
            max_position_size: Maximum position size as percentage of capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.strategy_type = strategy_type
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Strategy state
        self.state = StrategyState(
            strategy_id=strategy_id,
            symbol=symbol,
            parameters=kwargs
        )
        
        # Performance tracking
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []
        self.performance_history: List[StrategyPerformance] = []
        
        # Data storage
        self.price_history: List[float] = []
        self.volume_history: List[int] = []
        self.timestamps: List[datetime] = []
        
        logger.info(f"Initialized {strategy_type.value} strategy for {symbol}")
    
    @abstractmethod
    async def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """
        Generate trading signal based on market data.
        
        Args:
            market_data: Current market data
            
        Returns:
            Trading signal or None if no signal
        """
        pass
    
    @abstractmethod
    async def calculate_position_size(self, signal: Signal, capital: float) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Trading signal
            capital: Available capital
            
        Returns:
            Position size in base currency
        """
        pass
    
    async def execute_signal(self, signal: Signal) -> bool:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            True if execution was successful
        """
        try:
            # Calculate position size
            position_size = await self.calculate_position_size(signal, self.current_capital)
            
            if position_size <= 0:
                logger.warning(f"Invalid position size: {position_size}")
                return False
            
            # Create trade
            trade = Trade(
                id=f"{self.strategy_id}_{len(self.trades)}",
                symbol=signal.symbol,
                side=signal.signal_type.value,
                quantity=position_size,
                price=signal.price,
                value=position_size * signal.price,
                commission=0.0,  # Will be calculated by exchange
                net_value=position_size * signal.price,
                timestamp=signal.timestamp,
                order_type="MARKET",
                status="EXECUTED"
            )
            
            # Update strategy state
            if signal.signal_type == SignalType.BUY:
                self.state.current_position = Position(
                    symbol=signal.symbol,
                    quantity=position_size,
                    entry_price=signal.price,
                    current_price=signal.price,
                    market_value=position_size * signal.price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    realized_pnl=0.0,
                    total_pnl=0.0,
                    weight=position_size * signal.price / self.current_capital,
                    entry_date=signal.timestamp,
                    last_updated=signal.timestamp
                )
                self.state.entry_price = signal.price
                self.state.entry_time = signal.timestamp
            
            elif signal.signal_type == SignalType.SELL:
                if self.state.current_position:
                    # Calculate P&L
                    entry_price = self.state.entry_price or signal.price
                    pnl = (signal.price - entry_price) * position_size
                    self.state.total_pnl += pnl
                    self.current_capital += pnl
                    
                    # Update performance
                    self.state.total_trades += 1
                    if pnl > 0:
                        self.state.winning_trades += 1
                    
                    # Clear position
                    self.state.current_position = None
                    self.state.entry_price = None
                    self.state.entry_time = None
            
            # Record trade and signal
            self.trades.append(trade)
            self.signals.append(signal)
            
            logger.info(f"Executed {signal.signal_type.value} signal for {signal.symbol} at {signal.price}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    async def update_position(self, current_price: float, timestamp: datetime):
        """
        Update current position with new price data.
        
        Args:
            current_price: Current market price
            timestamp: Price timestamp
        """
        if not self.state.current_position:
            return
        
        # Update position
        self.state.current_position.current_price = current_price
        self.state.current_position.market_value = (
            self.state.current_position.quantity * current_price
        )
        self.state.current_position.unrealized_pnl = (
            current_price - self.state.current_position.entry_price
        ) * self.state.current_position.quantity
        self.state.current_position.unrealized_pnl_pct = (
            self.state.current_position.unrealized_pnl / 
            (self.state.current_position.entry_price * self.state.current_position.quantity)
        )
        self.state.current_position.last_updated = timestamp
        
        # Check stop loss and take profit
        entry_price = self.state.entry_price or current_price
        pnl_pct = (current_price - entry_price) / entry_price
        
        if pnl_pct <= -self.stop_loss_pct:
            # Stop loss hit
            await self.execute_signal(Signal(
                signal_type=SignalType.SELL,
                symbol=self.symbol,
                price=current_price,
                quantity=self.state.current_position.quantity,
                confidence=1.0,
                timestamp=timestamp,
                reasoning="Stop loss triggered"
            ))
        
        elif pnl_pct >= self.take_profit_pct:
            # Take profit hit
            await self.execute_signal(Signal(
                signal_type=SignalType.SELL,
                symbol=self.symbol,
                price=current_price,
                quantity=self.state.current_position.quantity,
                confidence=1.0,
                timestamp=timestamp,
                reasoning="Take profit triggered"
            ))
    
    def calculate_performance(self) -> StrategyPerformance:
        """
        Calculate strategy performance metrics.
        
        Returns:
            Strategy performance metrics
        """
        if not self.trades:
            return StrategyPerformance(
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_duration=0.0,
                best_trade=0.0,
                worst_trade=0.0
            )
        
        # Calculate basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.net_value > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate returns
        total_return = ((self.current_capital - self.initial_capital) / 
                       self.initial_capital) * 100
        
        # Calculate profit factor
        gross_profit = sum([t.net_value for t in self.trades if t.net_value > 0])
        gross_loss = abs(sum([t.net_value for t in self.trades if t.net_value < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate best and worst trades
        trade_values = [t.net_value for t in self.trades]
        best_trade = max(trade_values) if trade_values else 0.0
        worst_trade = min(trade_values) if trade_values else 0.0
        
        # Calculate average trade duration
        if len(self.trades) >= 2:
            durations = []
            for i in range(1, len(self.trades)):
                duration = (self.trades[i].timestamp - self.trades[i-1].timestamp).total_seconds() / 3600
                durations.append(duration)
            avg_trade_duration = np.mean(durations) if durations else 0.0
        else:
            avg_trade_duration = 0.0
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(self.trades)):
            ret = (self.trades[i].net_value - self.trades[i-1].net_value) / self.initial_capital
            returns.append(ret)
        
        sharpe_ratio = np.mean(returns) / np.std(returns) if returns and np.std(returns) > 0 else 0.0
        
        # Calculate max drawdown
        capital_history = [self.initial_capital]
        for trade in self.trades:
            capital_history.append(capital_history[-1] + trade.net_value)
        
        peak = capital_history[0]
        max_drawdown = 0.0
        for capital in capital_history:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return StrategyPerformance(
            strategy_id=self.strategy_id,
            symbol=self.symbol,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown * 100,
            win_rate=win_rate * 100,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_duration=avg_trade_duration,
            best_trade=best_trade,
            worst_trade=worst_trade
        )
    
    async def run(self, market_data_stream):
        """
        Run the strategy with live market data.
        
        Args:
            market_data_stream: Async generator of market data
        """
        logger.info(f"Starting {self.strategy_type.value} strategy for {self.symbol}")
        
        try:
            async for market_data in market_data_stream:
                # Update position with current price
                await self.update_position(market_data.current_price, market_data.timestamp)
                
                # Generate signal
                signal = await self.generate_signal(market_data)
                
                if signal:
                    # Execute signal
                    await self.execute_signal(signal)
                
                # Update performance periodically
                if len(self.trades) % 10 == 0:
                    performance = self.calculate_performance()
                    self.performance_history.append(performance)
                    logger.info(f"Performance update: {performance.total_return:.2f}% return")
                
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            raise
    
    def get_state(self) -> StrategyState:
        """Get current strategy state."""
        return self.state
    
    def get_performance(self) -> StrategyPerformance:
        """Get current performance metrics."""
        return self.calculate_performance()
    
    def get_trades(self) -> List[Trade]:
        """Get all trades."""
        return self.trades
    
    def get_signals(self) -> List[Signal]:
        """Get all signals."""
        return self.signals 