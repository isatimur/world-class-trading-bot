"""
Grid Trading Strategy implementation.

This module implements a sophisticated grid trading strategy that places
buy and sell orders at predetermined price levels. The strategy is enhanced
with ML models to optimize grid levels and adapt to market conditions.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyType
from ..models.market_data import MarketData
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GridLevel:
    """Grid level with order information"""
    level: float
    order_type: str  # 'BUY' or 'SELL'
    quantity: float
    is_filled: bool = False
    fill_time: Optional[datetime] = None
    fill_price: Optional[float] = None


class GridStrategy(BaseStrategy):
    """
    Advanced Grid Trading Strategy.
    
    This strategy places buy and sell orders at predetermined price levels
    and uses ML models to optimize grid spacing and adapt to market volatility.
    """
    
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        grid_levels: int = 10,
        grid_spacing_pct: float = 0.02,  # 2% spacing
        base_price: Optional[float] = None,
        volatility_lookback: int = 20,
        ml_enabled: bool = True,
        adaptive_grid: bool = True,
        **kwargs
    ):
        """
        Initialize grid strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            symbol: Trading symbol
            grid_levels: Number of grid levels above and below base price
            grid_spacing_pct: Percentage spacing between grid levels
            base_price: Base price for grid (if None, will be set dynamically)
            volatility_lookback: Lookback period for volatility calculation
            ml_enabled: Enable ML-based grid optimization
            adaptive_grid: Enable adaptive grid spacing
        """
        super().__init__(
            strategy_id=strategy_id,
            symbol=symbol,
            strategy_type=StrategyType.GRID,
            **kwargs
        )
        
        self.grid_levels = grid_levels
        self.grid_spacing_pct = grid_spacing_pct
        self.base_price = base_price
        self.volatility_lookback = volatility_lookback
        self.ml_enabled = ml_enabled
        self.adaptive_grid = adaptive_grid
        
        # Grid management
        self.grid_levels_list: List[GridLevel] = []
        self.active_orders: Dict[str, GridLevel] = {}
        self.filled_orders: List[GridLevel] = []
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'price', 'volume', 'rsi', 'macd', 'bollinger_position',
            'volatility', 'trend_strength', 'support_resistance'
        ]
        
        # Performance tracking
        self.grid_performance = {
            'total_buys': 0,
            'total_sells': 0,
            'grid_profit': 0.0,
            'avg_hold_time': 0.0
        }
        
        logger.info(f"Initialized Grid Strategy with {grid_levels} levels")
    
    async def initialize_grid(self, market_data: MarketData):
        """
        Initialize grid levels based on current market data.
        
        Args:
            market_data: Current market data
        """
        if self.base_price is None:
            self.base_price = market_data.current_price
        
        # Calculate grid levels
        self.grid_levels_list = []
        
        for i in range(-self.grid_levels, self.grid_levels + 1):
            level_price = self.base_price * (1 + i * self.grid_spacing_pct)
            
            if i < 0:
                # Buy levels below base price
                grid_level = GridLevel(
                    level=level_price,
                    order_type='BUY',
                    quantity=self.calculate_grid_quantity(level_price)
                )
            elif i > 0:
                # Sell levels above base price
                grid_level = GridLevel(
                    level=level_price,
                    order_type='SELL',
                    quantity=self.calculate_grid_quantity(level_price)
                )
            else:
                # Base level - no order
                continue
            
            self.grid_levels_list.append(grid_level)
        
        logger.info(f"Initialized grid with {len(self.grid_levels_list)} levels")
    
    def calculate_grid_quantity(self, price: float) -> float:
        """
        Calculate quantity for grid level based on risk management.
        
        Args:
            price: Grid level price
            
        Returns:
            Quantity to trade at this level
        """
        # Use Kelly Criterion for position sizing
        capital_at_risk = self.current_capital * self.risk_per_trade
        position_value = min(capital_at_risk, self.current_capital * self.max_position_size)
        
        return position_value / price
    
    async def train_ml_model(self, historical_data: pd.DataFrame):
        """
        Train ML model to predict optimal grid levels.
        
        Args:
            historical_data: Historical price and indicator data
        """
        if not self.ml_enabled:
            return
        
        try:
            # Prepare features
            features = historical_data[self.feature_columns].dropna()
            
            # Create target: optimal grid spacing
            features['price_change'] = features['price'].pct_change()
            features['volatility'] = features['price_change'].rolling(20).std()
            
            # Target: optimal grid spacing based on volatility
            features['optimal_spacing'] = features['volatility'] * 2  # 2x volatility
            
            # Remove NaN values
            features = features.dropna()
            
            if len(features) < 100:
                logger.warning("Insufficient data for ML model training")
                return
            
            # Split features and target
            X = features[self.feature_columns]
            y = features['optimal_spacing']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.ml_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.ml_model.fit(X_scaled, y)
            
            logger.info("ML model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
    
    async def predict_optimal_spacing(self, market_data: MarketData) -> float:
        """
        Predict optimal grid spacing using ML model.
        
        Args:
            market_data: Current market data
            
        Returns:
            Predicted optimal grid spacing
        """
        if not self.ml_enabled or self.ml_model is None:
            return self.grid_spacing_pct
        
        try:
            # Prepare features
            features = pd.DataFrame([{
                'price': market_data.current_price,
                'volume': market_data.volume,
                'rsi': getattr(market_data, 'rsi', 50),
                'macd': getattr(market_data, 'macd', 0),
                'bollinger_position': getattr(market_data, 'bollinger_position', 0.5),
                'volatility': self.calculate_volatility(),
                'trend_strength': self.calculate_trend_strength(),
                'support_resistance': self.calculate_support_resistance()
            }])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict optimal spacing
            predicted_spacing = self.ml_model.predict(features_scaled)[0]
            
            # Clamp to reasonable range
            predicted_spacing = np.clip(predicted_spacing, 0.005, 0.1)
            
            return predicted_spacing
            
        except Exception as e:
            logger.error(f"Error predicting optimal spacing: {e}")
            return self.grid_spacing_pct
    
    def calculate_volatility(self) -> float:
        """Calculate current volatility."""
        if len(self.price_history) < self.volatility_lookback:
            return 0.02  # Default 2% volatility
        
        returns = pd.Series(self.price_history).pct_change().dropna()
        return returns.tail(self.volatility_lookback).std()
    
    def calculate_trend_strength(self) -> float:
        """Calculate trend strength using linear regression."""
        if len(self.price_history) < 20:
            return 0.0
        
        prices = np.array(self.price_history[-20:])
        x = np.arange(len(prices))
        
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        
        # Return R-squared as trend strength
        return r_value ** 2
    
    def calculate_support_resistance(self) -> float:
        """Calculate support/resistance levels."""
        if len(self.price_history) < 50:
            return 0.5
        
        prices = np.array(self.price_history[-50:])
        current_price = prices[-1]
        
        # Find support and resistance levels
        support = np.percentile(prices, 25)
        resistance = np.percentile(prices, 75)
        
        # Calculate position between support and resistance
        if resistance > support:
            return (current_price - support) / (resistance - support)
        else:
            return 0.5
    
    async def adapt_grid(self, market_data: MarketData):
        """
        Adapt grid levels based on market conditions.
        
        Args:
            market_data: Current market data
        """
        if not self.adaptive_grid:
            return
        
        # Update price history
        self.price_history.append(market_data.current_price)
        self.volume_history.append(market_data.volume)
        self.timestamps.append(market_data.timestamp)
        
        # Keep only recent history
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]
            self.volume_history = self.volume_history[-1000:]
            self.timestamps = self.timestamps[-1000:]
        
        # Predict optimal spacing
        optimal_spacing = await self.predict_optimal_spacing(market_data)
        
        # Check if grid needs adjustment
        current_spacing = self.grid_spacing_pct
        spacing_change = abs(optimal_spacing - current_spacing) / current_spacing
        
        if spacing_change > 0.2:  # 20% change threshold
            logger.info(f"Adjusting grid spacing from {current_spacing:.4f} to {optimal_spacing:.4f}")
            self.grid_spacing_pct = optimal_spacing
            
            # Reinitialize grid with new spacing
            await self.initialize_grid(market_data)
    
    async def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """
        Generate trading signal based on grid levels.
        
        Args:
            market_data: Current market data
            
        Returns:
            Trading signal or None
        """
        # Initialize grid if not done
        if not self.grid_levels_list:
            await self.initialize_grid(market_data)
        
        # Adapt grid if enabled
        if self.adaptive_grid:
            await self.adapt_grid(market_data)
        
        current_price = market_data.current_price
        
        # Check for grid level hits
        for grid_level in self.grid_levels_list:
            if grid_level.is_filled:
                continue
            
            # Check if price hit the grid level
            if grid_level.order_type == 'BUY' and current_price <= grid_level.level:
                # Buy signal
                grid_level.is_filled = True
                grid_level.fill_time = market_data.timestamp
                grid_level.fill_price = current_price
                
                self.grid_performance['total_buys'] += 1
                
                return Signal(
                    signal_type=SignalType.BUY,
                    symbol=self.symbol,
                    price=current_price,
                    quantity=grid_level.quantity,
                    confidence=0.8,
                    timestamp=market_data.timestamp,
                    reasoning=f"Grid buy level hit at {grid_level.level:.2f}",
                    stop_loss=current_price * (1 - self.stop_loss_pct),
                    take_profit=current_price * (1 + self.take_profit_pct)
                )
            
            elif grid_level.order_type == 'SELL' and current_price >= grid_level.level:
                # Sell signal
                grid_level.is_filled = True
                grid_level.fill_time = market_data.timestamp
                grid_level.fill_price = current_price
                
                self.grid_performance['total_sells'] += 1
                
                return Signal(
                    signal_type=SignalType.SELL,
                    symbol=self.symbol,
                    price=current_price,
                    quantity=grid_level.quantity,
                    confidence=0.8,
                    timestamp=market_data.timestamp,
                    reasoning=f"Grid sell level hit at {grid_level.level:.2f}",
                    stop_loss=current_price * (1 + self.stop_loss_pct),
                    take_profit=current_price * (1 - self.take_profit_pct)
                )
        
        return None
    
    async def calculate_position_size(self, signal: Signal, capital: float) -> float:
        """
        Calculate position size for grid strategy.
        
        Args:
            signal: Trading signal
            capital: Available capital
            
        Returns:
            Position size
        """
        # For grid strategy, use fixed position size per level
        position_value = capital * self.risk_per_trade
        
        # Adjust based on grid level
        if signal.signal_type == SignalType.BUY:
            # Buy more at lower prices
            price_ratio = signal.price / self.base_price
            position_value *= (1 + (1 - price_ratio))
        else:
            # Sell more at higher prices
            price_ratio = signal.price / self.base_price
            position_value *= (1 + (price_ratio - 1))
        
        return position_value / signal.price
    
    def get_grid_status(self) -> Dict:
        """
        Get current grid status.
        
        Returns:
            Grid status information
        """
        filled_levels = [level for level in self.grid_levels_list if level.is_filled]
        active_levels = [level for level in self.grid_levels_list if not level.is_filled]
        
        return {
            'total_levels': len(self.grid_levels_list),
            'filled_levels': len(filled_levels),
            'active_levels': len(active_levels),
            'base_price': self.base_price,
            'current_spacing': self.grid_spacing_pct,
            'grid_performance': self.grid_performance,
            'buy_levels': [level.level for level in self.grid_levels_list if level.order_type == 'BUY'],
            'sell_levels': [level.level for level in self.grid_levels_list if level.order_type == 'SELL']
        }
    
    async def reset_grid(self, new_base_price: float):
        """
        Reset grid with new base price.
        
        Args:
            new_base_price: New base price for grid
        """
        self.base_price = new_base_price
        self.grid_levels_list = []
        self.filled_orders = []
        
        logger.info(f"Reset grid with new base price: {new_base_price}")
    
    def get_strategy_summary(self) -> Dict:
        """
        Get comprehensive strategy summary.
        
        Returns:
            Strategy summary
        """
        performance = self.calculate_performance()
        grid_status = self.get_grid_status()
        
        return {
            'strategy_id': self.strategy_id,
            'strategy_type': self.strategy_type.value,
            'symbol': self.symbol,
            'performance': performance.dict(),
            'grid_status': grid_status,
            'ml_enabled': self.ml_enabled,
            'adaptive_grid': self.adaptive_grid,
            'total_trades': len(self.trades),
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital
        } 