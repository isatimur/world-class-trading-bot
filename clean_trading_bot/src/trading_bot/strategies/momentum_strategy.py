"""
Momentum Trading Strategy implementation.

This module implements a sophisticated momentum strategy that follows trends
with advanced risk management and dynamic position sizing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyType
from ..models.market_data import MarketData
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MomentumSignal:
    """Momentum signal with trend metrics"""
    signal_type: SignalType
    symbol: str
    price: float
    quantity: float
    confidence: float
    timestamp: datetime
    reasoning: str
    trend_strength: float
    momentum_score: float
    trend_direction: str
    volatility_regime: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None


class MomentumStrategy(BaseStrategy):
    """
    Advanced Momentum Trading Strategy.
    
    This strategy identifies and follows trends using multiple momentum indicators
    with sophisticated risk management and dynamic position sizing.
    """
    
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        short_period: int = 10,
        long_period: int = 30,
        momentum_threshold: float = 0.02,
        trend_confirmation_periods: int = 3,
        volatility_lookback: int = 20,
        momentum_strength: float = 0.8,
        dynamic_sizing: bool = True,
        ml_enabled: bool = True,
        **kwargs
    ):
        """
        Initialize momentum strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            symbol: Trading symbol
            short_period: Short-term momentum period
            long_period: Long-term momentum period
            momentum_threshold: Minimum momentum threshold for signals
            trend_confirmation_periods: Periods required for trend confirmation
            volatility_lookback: Period for volatility calculation
            momentum_strength: Strength of momentum signals (0-1)
            dynamic_sizing: Enable dynamic position sizing
            ml_enabled: Enable ML-based signal optimization
        """
        super().__init__(
            strategy_id=strategy_id,
            symbol=symbol,
            strategy_type=StrategyType.MOMENTUM,
            **kwargs
        )
        
        self.short_period = short_period
        self.long_period = long_period
        self.momentum_threshold = momentum_threshold
        self.trend_confirmation_periods = trend_confirmation_periods
        self.volatility_lookback = volatility_lookback
        self.momentum_strength = momentum_strength
        self.dynamic_sizing = dynamic_sizing
        self.ml_enabled = ml_enabled
        
        # Momentum tracking
        self.price_history: List[float] = []
        self.volume_history: List[int] = []
        self.short_momentum: List[float] = []
        self.long_momentum: List[float] = []
        self.trend_direction: List[str] = []
        self.volatility_history: List[float] = []
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'price', 'volume', 'rsi', 'macd', 'bollinger_position',
            'short_momentum', 'long_momentum', 'momentum_ratio',
            'trend_strength', 'volatility', 'volume_momentum',
            'price_acceleration', 'trend_consistency', 'support_resistance'
        ]
        
        # Performance tracking
        self.momentum_performance = {
            'total_signals': 0,
            'successful_trends': 0,
            'avg_trend_duration': 0.0,
            'avg_profit_per_trend': 0.0,
            'trend_breakdown': {'uptrend': 0, 'downtrend': 0, 'sideways': 0}
        }
        
        logger.info(f"Initialized Momentum Strategy for {symbol}")
    
    def calculate_momentum_indicators(self, current_price: float, current_volume: int):
        """
        Calculate momentum indicators.
        
        Args:
            current_price: Current market price
            current_volume: Current volume
        """
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        
        # Keep only recent history
        max_periods = max(self.long_period * 2, 100)
        if len(self.price_history) > max_periods:
            self.price_history = self.price_history[-max_periods:]
            self.volume_history = self.volume_history[-max_periods:]
        
        if len(self.price_history) >= self.long_period:
            # Calculate short-term momentum
            short_prices = self.price_history[-self.short_period:]
            short_momentum = (short_prices[-1] - short_prices[0]) / short_prices[0]
            self.short_momentum.append(short_momentum)
            
            # Calculate long-term momentum
            long_prices = self.price_history[-self.long_period:]
            long_momentum = (long_prices[-1] - long_prices[0]) / long_prices[0]
            self.long_momentum.append(long_momentum)
            
            # Determine trend direction
            if short_momentum > 0 and long_momentum > 0:
                trend = 'uptrend'
            elif short_momentum < 0 and long_momentum < 0:
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            self.trend_direction.append(trend)
            
            # Calculate volatility
            if len(self.price_history) >= self.volatility_lookback:
                recent_returns = np.diff(self.price_history[-self.volatility_lookback:])
                volatility = np.std(recent_returns) if len(recent_returns) > 0 else 0
                self.volatility_history.append(volatility)
    
    def calculate_trend_strength(self) -> float:
        """
        Calculate trend strength using multiple indicators.
        
        Returns:
            Trend strength (0-1)
        """
        if len(self.short_momentum) < 5:
            return 0.5
        
        # Use recent momentum values
        recent_short = self.short_momentum[-5:]
        recent_long = self.long_momentum[-5:]
        
        # Calculate consistency
        short_consistency = np.std(recent_short)
        long_consistency = np.std(recent_long)
        
        # Calculate momentum ratio
        avg_short = np.mean(recent_short)
        avg_long = np.mean(recent_long)
        
        if abs(avg_long) < 0.001:
            momentum_ratio = 1.0
        else:
            momentum_ratio = abs(avg_short / avg_long)
        
        # Calculate trend strength
        strength = min(momentum_ratio * (1 - short_consistency) * (1 - long_consistency), 1.0)
        
        return max(strength, 0.0)
    
    def calculate_momentum_score(self) -> float:
        """
        Calculate overall momentum score.
        
        Returns:
            Momentum score (-1 to 1)
        """
        if len(self.short_momentum) == 0 or len(self.long_momentum) == 0:
            return 0.0
        
        # Weighted combination of short and long momentum
        short_weight = 0.6
        long_weight = 0.4
        
        recent_short = self.short_momentum[-1]
        recent_long = self.long_momentum[-1]
        
        momentum_score = (short_weight * recent_short + long_weight * recent_long)
        
        # Normalize to [-1, 1] range
        return np.clip(momentum_score, -1.0, 1.0)
    
    def detect_volatility_regime(self) -> str:
        """
        Detect current volatility regime.
        
        Returns:
            Volatility regime ('low', 'medium', 'high')
        """
        if len(self.volatility_history) < 10:
            return 'medium'
        
        recent_volatility = np.mean(self.volatility_history[-10:])
        historical_volatility = np.mean(self.volatility_history)
        
        if recent_volatility < historical_volatility * 0.7:
            return 'low'
        elif recent_volatility > historical_volatility * 1.3:
            return 'high'
        else:
            return 'medium'
    
    def calculate_volume_momentum(self) -> float:
        """
        Calculate volume momentum indicator.
        
        Returns:
            Volume momentum (-1 to 1)
        """
        if len(self.volume_history) < self.short_period:
            return 0.0
        
        recent_volume = self.volume_history[-self.short_period:]
        avg_volume = np.mean(self.volume_history[-self.long_period:])
        
        if avg_volume == 0:
            return 0.0
        
        volume_momentum = (recent_volume[-1] - avg_volume) / avg_volume
        return np.clip(volume_momentum, -1.0, 1.0)
    
    def calculate_price_acceleration(self) -> float:
        """
        Calculate price acceleration (second derivative).
        
        Returns:
            Price acceleration
        """
        if len(self.price_history) < 10:
            return 0.0
        
        recent_prices = self.price_history[-10:]
        x = np.arange(len(recent_prices))
        
        try:
            # Fit quadratic function to get acceleration
            coeffs = np.polyfit(x, recent_prices, 2)
            acceleration = coeffs[0] * 2  # Second derivative
            return acceleration
        except:
            return 0.0
    
    def calculate_trend_consistency(self) -> float:
        """
        Calculate trend consistency over recent periods.
        
        Returns:
            Trend consistency (0-1)
        """
        if len(self.trend_direction) < self.trend_confirmation_periods:
            return 0.0
        
        recent_trends = self.trend_direction[-self.trend_confirmation_periods:]
        
        if len(set(recent_trends)) == 1:
            # All trends are the same
            return 1.0
        else:
            # Mixed trends
            return 0.5
    
    async def train_ml_model(self, historical_data: pd.DataFrame):
        """
        Train ML model to predict optimal momentum parameters.
        
        Args:
            historical_data: Historical price and indicator data
        """
        if not self.ml_enabled:
            return
        
        try:
            # Prepare features
            features = historical_data[self.feature_columns].dropna()
            
            # Create target: optimal momentum threshold based on market conditions
            features['price_change'] = features['price'].pct_change()
            features['volatility'] = features['price_change'].rolling(20).std()
            features['optimal_threshold'] = 0.02 + features['volatility'] * 5
            
            # Remove NaN values
            features = features.dropna()
            
            if len(features) < 100:
                logger.warning("Insufficient data for ML model training")
                return
            
            # Split features and target
            X = features[self.feature_columns]
            y = features['optimal_threshold']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.ml_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            self.ml_model.fit(X_scaled, y)
            
            logger.info("ML model trained successfully for momentum optimization")
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
    
    async def predict_optimal_threshold(self, market_data: MarketData) -> float:
        """
        Predict optimal momentum threshold using ML model.
        
        Args:
            market_data: Current market data
            
        Returns:
            Optimal momentum threshold
        """
        if self.ml_model is None:
            return self.momentum_threshold
        
        try:
            # Prepare features
            features = await self.prepare_features(market_data)
            
            if features is None or features.empty:
                return self.momentum_threshold
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict optimal threshold
            optimal_threshold = self.ml_model.predict(features_scaled)[0]
            
            # Constrain to reasonable range
            return np.clip(optimal_threshold, 0.01, 0.05)
            
        except Exception as e:
            logger.error(f"Error predicting optimal threshold: {e}")
            return self.momentum_threshold
    
    async def prepare_features(self, market_data: MarketData) -> Optional[pd.DataFrame]:
        """
        Prepare features for ML model.
        
        Args:
            market_data: Current market data
            
        Returns:
            Feature DataFrame or None
        """
        try:
            features = {}
            
            # Basic price features
            features['price'] = market_data.current_price
            features['volume'] = market_data.volume
            
            # Technical indicators
            features['rsi'] = market_data.rsi if hasattr(market_data, 'rsi') else 50
            features['macd'] = market_data.macd if hasattr(market_data, 'macd') else 0
            features['bollinger_position'] = market_data.bollinger_position if hasattr(market_data, 'bollinger_position') else 0.5
            
            # Momentum features
            features['short_momentum'] = self.short_momentum[-1] if self.short_momentum else 0
            features['long_momentum'] = self.long_momentum[-1] if self.long_momentum else 0
            features['momentum_ratio'] = (
                abs(features['short_momentum']) / max(abs(features['long_momentum']), 0.001)
            )
            
            # Additional features
            features['trend_strength'] = self.calculate_trend_strength()
            features['volatility'] = np.mean(self.volatility_history[-5:]) if self.volatility_history else 0
            features['volume_momentum'] = self.calculate_volume_momentum()
            features['price_acceleration'] = self.calculate_price_acceleration()
            features['trend_consistency'] = self.calculate_trend_consistency()
            features['support_resistance'] = self.calculate_support_resistance()
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def calculate_support_resistance(self) -> float:
        """
        Calculate support/resistance level proximity.
        
        Returns:
            Position relative to support/resistance (0-1)
        """
        if len(self.price_history) < 20:
            return 0.5
        
        current_price = self.price_history[-1]
        recent_prices = self.price_history[-20:]
        
        support = np.percentile(recent_prices, 25)
        resistance = np.percentile(recent_prices, 75)
        
        if resistance == support:
            return 0.5
        
        return (current_price - support) / (resistance - support)
    
    async def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """
        Generate momentum trading signal.
        
        Args:
            market_data: Current market data
            
        Returns:
            Trading signal or None
        """
        try:
            # Update momentum indicators
            self.calculate_momentum_indicators(market_data.current_price, market_data.volume)
            
            if len(self.short_momentum) == 0 or len(self.long_momentum) == 0:
                return None
            
            # Calculate momentum metrics
            momentum_score = self.calculate_momentum_score()
            trend_strength = self.calculate_trend_strength()
            trend_consistency = self.calculate_trend_consistency()
            
            # Predict optimal threshold
            optimal_threshold = await self.predict_optimal_threshold(market_data)
            
            # Generate signal based on momentum
            signal_type = None
            confidence = 0.0
            reasoning = ""
            
            if momentum_score > optimal_threshold and trend_consistency > 0.7:
                # Strong uptrend - buy signal
                signal_type = SignalType.BUY
                confidence = min(momentum_score / optimal_threshold * trend_strength, 1.0)
                reasoning = f"Strong uptrend (momentum: {momentum_score:.3f}, strength: {trend_strength:.2f})"
                
            elif momentum_score < -optimal_threshold and trend_consistency > 0.7:
                # Strong downtrend - sell signal
                signal_type = SignalType.SELL
                confidence = min(abs(momentum_score) / optimal_threshold * trend_strength, 1.0)
                reasoning = f"Strong downtrend (momentum: {momentum_score:.3f}, strength: {trend_strength:.2f})"
            
            if signal_type is None:
                return None
            
            # Adjust confidence based on volatility regime
            volatility_regime = self.detect_volatility_regime()
            if volatility_regime == 'high':
                confidence *= 0.8  # Reduce confidence in high volatility
            elif volatility_regime == 'low':
                confidence *= 1.1  # Slightly increase confidence in low volatility
            
            # Apply momentum strength
            confidence *= self.momentum_strength
            
            # Minimum confidence threshold
            if confidence < 0.4:
                return None
            
            # Calculate position size
            position_size = await self.calculate_position_size(
                Signal(signal_type, self.symbol, market_data.current_price, 0, confidence, datetime.now(), reasoning),
                self.current_capital
            )
            
            # Create momentum signal
            momentum_signal = MomentumSignal(
                signal_type=signal_type,
                symbol=self.symbol,
                price=market_data.current_price,
                quantity=position_size,
                confidence=confidence,
                timestamp=datetime.now(),
                reasoning=reasoning,
                trend_strength=trend_strength,
                momentum_score=momentum_score,
                trend_direction=self.trend_direction[-1] if self.trend_direction else 'sideways',
                volatility_regime=volatility_regime,
                stop_loss=self.calculate_stop_loss(signal_type, market_data.current_price),
                take_profit=self.calculate_take_profit(signal_type, market_data.current_price, momentum_score)
            )
            
            # Update performance tracking
            self.momentum_performance['total_signals'] += 1
            if self.trend_direction:
                self.momentum_performance['trend_breakdown'][self.trend_direction[-1]] += 1
            
            logger.info(f"Generated {signal_type.value} signal: {reasoning} (confidence: {confidence:.2f})")
            
            return momentum_signal
            
        except Exception as e:
            logger.error(f"Error generating momentum signal: {e}")
            return None
    
    def calculate_stop_loss(self, signal_type: SignalType, current_price: float) -> float:
        """
        Calculate stop loss level.
        
        Args:
            signal_type: Type of signal
            current_price: Current price
            
        Returns:
            Stop loss price
        """
        if signal_type == SignalType.BUY:
            # For buy signals, stop loss below current price
            return current_price * (1 - self.stop_loss_pct)
        else:
            # For sell signals, stop loss above current price
            return current_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, signal_type: SignalType, current_price: float, momentum_score: float) -> float:
        """
        Calculate take profit level based on momentum strength.
        
        Args:
            signal_type: Type of signal
            current_price: Current price
            momentum_score: Momentum score
            
        Returns:
            Take profit price
        """
        # Adjust take profit based on momentum strength
        momentum_multiplier = 1 + abs(momentum_score)
        
        if signal_type == SignalType.BUY:
            return current_price * (1 + self.take_profit_pct * momentum_multiplier)
        else:
            return current_price * (1 - self.take_profit_pct * momentum_multiplier)
    
    async def calculate_position_size(self, signal: Signal, capital: float) -> float:
        """
        Calculate position size with dynamic sizing.
        
        Args:
            signal: Trading signal
            capital: Available capital
            
        Returns:
            Position size in base currency
        """
        # Base position size using Kelly Criterion
        base_size = capital * self.risk_per_trade * signal.confidence
        
        if not self.dynamic_sizing:
            return min(base_size, capital * self.max_position_size)
        
        # Adjust for trend strength and momentum
        trend_strength = self.calculate_trend_strength()
        momentum_score = self.calculate_momentum_score()
        
        # Dynamic multiplier based on trend strength and momentum
        dynamic_multiplier = 1.0 + (trend_strength * 0.5) + (abs(momentum_score) * 0.3)
        
        # Adjust for volatility regime
        volatility_regime = self.detect_volatility_regime()
        if volatility_regime == 'high':
            dynamic_multiplier *= 0.8
        elif volatility_regime == 'low':
            dynamic_multiplier *= 1.2
        
        adjusted_size = base_size * dynamic_multiplier
        
        # Apply maximum position size limit
        return min(adjusted_size, capital * self.max_position_size)
    
    def get_strategy_summary(self) -> Dict:
        """
        Get strategy performance summary.
        
        Returns:
            Strategy summary dictionary
        """
        return {
            'strategy_id': self.strategy_id,
            'strategy_type': self.strategy_type.value,
            'symbol': self.symbol,
            'total_signals': self.momentum_performance['total_signals'],
            'successful_trends': self.momentum_performance['successful_trends'],
            'success_rate': (
                self.momentum_performance['successful_trends'] / 
                max(self.momentum_performance['total_signals'], 1)
            ),
            'avg_trend_duration': self.momentum_performance['avg_trend_duration'],
            'avg_profit_per_trend': self.momentum_performance['avg_profit_per_trend'],
            'trend_breakdown': self.momentum_performance['trend_breakdown'],
            'current_momentum_score': self.calculate_momentum_score(),
            'current_trend_strength': self.calculate_trend_strength(),
            'current_trend_direction': self.trend_direction[-1] if self.trend_direction else 'sideways',
            'volatility_regime': self.detect_volatility_regime()
        } 