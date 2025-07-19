"""
Mean Reversion Trading Strategy implementation.

This module implements a sophisticated mean reversion strategy that identifies
overbought/oversold conditions and trades against the trend with statistical
arbitrage principles and volatility adjustment.
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
class MeanReversionSignal:
    """Mean reversion signal with statistical metrics"""
    signal_type: SignalType
    symbol: str
    price: float
    quantity: float
    confidence: float
    timestamp: datetime
    reasoning: str
    z_score: float
    mean_price: float
    std_dev: float
    volatility_regime: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None


class MeanReversionStrategy(BaseStrategy):
    """
    Advanced Mean Reversion Trading Strategy.
    
    This strategy identifies overbought/oversold conditions using statistical
    methods and trades against the trend with proper risk management.
    """
    
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        lookback_period: int = 50,
        z_score_threshold: float = 2.0,
        volatility_lookback: int = 20,
        mean_reversion_strength: float = 0.7,
        volatility_adjustment: bool = True,
        ml_enabled: bool = True,
        **kwargs
    ):
        """
        Initialize mean reversion strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            symbol: Trading symbol
            lookback_period: Period for calculating mean and standard deviation
            z_score_threshold: Z-score threshold for signal generation
            volatility_lookback: Period for volatility calculation
            mean_reversion_strength: Strength of mean reversion (0-1)
            volatility_adjustment: Enable volatility-based position sizing
            ml_enabled: Enable ML-based parameter optimization
        """
        super().__init__(
            strategy_id=strategy_id,
            symbol=symbol,
            strategy_type=StrategyType.MEAN_REVERSION,
            **kwargs
        )
        
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.volatility_lookback = volatility_lookback
        self.mean_reversion_strength = mean_reversion_strength
        self.volatility_adjustment = volatility_adjustment
        self.ml_enabled = ml_enabled
        
        # Statistical tracking
        self.price_history: List[float] = []
        self.rolling_mean: List[float] = []
        self.rolling_std: List[float] = []
        self.z_scores: List[float] = []
        self.volatility_history: List[float] = []
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'price', 'volume', 'rsi', 'macd', 'bollinger_position',
            'volatility', 'z_score', 'mean_deviation', 'volatility_regime',
            'trend_strength', 'support_resistance', 'volume_profile'
        ]
        
        # Performance tracking
        self.mean_reversion_performance = {
            'total_signals': 0,
            'successful_reversions': 0,
            'avg_reversion_time': 0.0,
            'avg_profit_per_reversion': 0.0
        }
        
        logger.info(f"Initialized Mean Reversion Strategy for {symbol}")
    
    def calculate_rolling_statistics(self, current_price: float):
        """
        Calculate rolling mean, standard deviation, and z-score.
        
        Args:
            current_price: Current market price
        """
        self.price_history.append(current_price)
        
        # Keep only recent history
        if len(self.price_history) > self.lookback_period * 2:
            self.price_history = self.price_history[-self.lookback_period * 2:]
        
        if len(self.price_history) >= self.lookback_period:
            recent_prices = self.price_history[-self.lookback_period:]
            mean_price = np.mean(recent_prices)
            std_price = np.std(recent_prices)
            
            self.rolling_mean.append(mean_price)
            self.rolling_std.append(std_price)
            
            # Calculate z-score
            z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
            self.z_scores.append(z_score)
            
            # Calculate volatility
            if len(self.price_history) >= self.volatility_lookback:
                recent_returns = np.diff(self.price_history[-self.volatility_lookback:])
                volatility = np.std(recent_returns) if len(recent_returns) > 0 else 0
                self.volatility_history.append(volatility)
    
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
    
    def calculate_mean_deviation(self) -> float:
        """
        Calculate current price deviation from mean.
        
        Returns:
            Deviation as percentage
        """
        if len(self.rolling_mean) == 0:
            return 0.0
        
        current_price = self.price_history[-1]
        current_mean = self.rolling_mean[-1]
        
        return (current_price - current_mean) / current_mean
    
    async def train_ml_model(self, historical_data: pd.DataFrame):
        """
        Train ML model to predict optimal mean reversion parameters.
        
        Args:
            historical_data: Historical price and indicator data
        """
        if not self.ml_enabled:
            return
        
        try:
            # Prepare features
            features = historical_data[self.feature_columns].dropna()
            
            # Create target: optimal z-score threshold based on volatility
            features['volatility'] = features['price'].pct_change().rolling(20).std()
            features['optimal_threshold'] = 2.0 + features['volatility'] * 10
            
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
            
            logger.info("ML model trained successfully for mean reversion optimization")
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
    
    async def predict_optimal_threshold(self, market_data: MarketData) -> float:
        """
        Predict optimal z-score threshold using ML model.
        
        Args:
            market_data: Current market data
            
        Returns:
            Optimal z-score threshold
        """
        if self.ml_model is None:
            return self.z_score_threshold
        
        try:
            # Prepare features
            features = await self.prepare_features(market_data)
            
            if features is None or features.empty:
                return self.z_score_threshold
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict optimal threshold
            optimal_threshold = self.ml_model.predict(features_scaled)[0]
            
            # Constrain to reasonable range
            return np.clip(optimal_threshold, 1.5, 3.5)
            
        except Exception as e:
            logger.error(f"Error predicting optimal threshold: {e}")
            return self.z_score_threshold
    
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
            
            # Statistical features
            features['volatility'] = np.mean(self.volatility_history[-5:]) if self.volatility_history else 0
            features['z_score'] = self.z_scores[-1] if self.z_scores else 0
            features['mean_deviation'] = self.calculate_mean_deviation()
            
            # Regime features
            volatility_regime = self.detect_volatility_regime()
            features['volatility_regime'] = {'low': 0, 'medium': 1, 'high': 2}[volatility_regime]
            
            # Additional features
            features['trend_strength'] = self.calculate_trend_strength()
            features['support_resistance'] = self.calculate_support_resistance()
            features['volume_profile'] = self.calculate_volume_profile()
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def calculate_trend_strength(self) -> float:
        """
        Calculate trend strength using linear regression.
        
        Returns:
            Trend strength (0-1)
        """
        if len(self.price_history) < 10:
            return 0.5
        
        recent_prices = self.price_history[-10:]
        x = np.arange(len(recent_prices))
        
        try:
            slope, _, r_value, _, _ = stats.linregress(x, recent_prices)
            return abs(r_value)  # R-squared value
        except:
            return 0.5
    
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
    
    def calculate_volume_profile(self) -> float:
        """
        Calculate volume profile indicator.
        
        Returns:
            Volume profile value (0-1)
        """
        # Simplified volume profile calculation
        # In a real implementation, this would analyze volume at different price levels
        return 0.5
    
    async def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """
        Generate mean reversion trading signal.
        
        Args:
            market_data: Current market data
            
        Returns:
            Trading signal or None
        """
        try:
            # Update rolling statistics
            self.calculate_rolling_statistics(market_data.current_price)
            
            if len(self.z_scores) == 0:
                return None
            
            current_z_score = self.z_scores[-1]
            current_price = market_data.current_price
            current_mean = self.rolling_mean[-1] if self.rolling_mean else current_price
            
            # Predict optimal threshold
            optimal_threshold = await self.predict_optimal_threshold(market_data)
            
            # Generate signal based on z-score
            signal_type = None
            confidence = 0.0
            reasoning = ""
            
            if current_z_score > optimal_threshold:
                # Overbought condition - potential sell signal
                signal_type = SignalType.SELL
                confidence = min(abs(current_z_score) / optimal_threshold, 1.0)
                reasoning = f"Overbought condition (z-score: {current_z_score:.2f})"
                
            elif current_z_score < -optimal_threshold:
                # Oversold condition - potential buy signal
                signal_type = SignalType.BUY
                confidence = min(abs(current_z_score) / optimal_threshold, 1.0)
                reasoning = f"Oversold condition (z-score: {current_z_score:.2f})"
            
            if signal_type is None:
                return None
            
            # Adjust confidence based on volatility regime
            volatility_regime = self.detect_volatility_regime()
            if volatility_regime == 'high':
                confidence *= 0.8  # Reduce confidence in high volatility
            elif volatility_regime == 'low':
                confidence *= 1.2  # Increase confidence in low volatility
            
            # Apply mean reversion strength
            confidence *= self.mean_reversion_strength
            
            # Minimum confidence threshold
            if confidence < 0.3:
                return None
            
            # Calculate position size
            position_size = await self.calculate_position_size(
                Signal(signal_type, self.symbol, current_price, 0, confidence, datetime.now(), reasoning),
                self.current_capital
            )
            
            # Create mean reversion signal
            mean_reversion_signal = MeanReversionSignal(
                signal_type=signal_type,
                symbol=self.symbol,
                price=current_price,
                quantity=position_size,
                confidence=confidence,
                timestamp=datetime.now(),
                reasoning=reasoning,
                z_score=current_z_score,
                mean_price=current_mean,
                std_dev=self.rolling_std[-1] if self.rolling_std else 0,
                volatility_regime=volatility_regime,
                stop_loss=self.calculate_stop_loss(signal_type, current_price),
                take_profit=self.calculate_take_profit(signal_type, current_price, current_mean)
            )
            
            # Update performance tracking
            self.mean_reversion_performance['total_signals'] += 1
            
            logger.info(f"Generated {signal_type.value} signal: {reasoning} (confidence: {confidence:.2f})")
            
            return mean_reversion_signal
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signal: {e}")
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
    
    def calculate_take_profit(self, signal_type: SignalType, current_price: float, mean_price: float) -> float:
        """
        Calculate take profit level.
        
        Args:
            signal_type: Type of signal
            current_price: Current price
            mean_price: Mean price
            
        Returns:
            Take profit price
        """
        if signal_type == SignalType.BUY:
            # For buy signals, take profit at mean price or higher
            return max(mean_price, current_price * (1 + self.take_profit_pct))
        else:
            # For sell signals, take profit at mean price or lower
            return min(mean_price, current_price * (1 - self.take_profit_pct))
    
    async def calculate_position_size(self, signal: Signal, capital: float) -> float:
        """
        Calculate position size with volatility adjustment.
        
        Args:
            signal: Trading signal
            capital: Available capital
            
        Returns:
            Position size in base currency
        """
        # Base position size using Kelly Criterion
        base_size = capital * self.risk_per_trade * signal.confidence
        
        if not self.volatility_adjustment:
            return min(base_size, capital * self.max_position_size)
        
        # Adjust for volatility
        volatility_regime = self.detect_volatility_regime()
        
        if volatility_regime == 'high':
            # Reduce position size in high volatility
            volatility_multiplier = 0.7
        elif volatility_regime == 'low':
            # Increase position size in low volatility
            volatility_multiplier = 1.3
        else:
            volatility_multiplier = 1.0
        
        adjusted_size = base_size * volatility_multiplier
        
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
            'total_signals': self.mean_reversion_performance['total_signals'],
            'successful_reversions': self.mean_reversion_performance['successful_reversions'],
            'success_rate': (
                self.mean_reversion_performance['successful_reversions'] / 
                max(self.mean_reversion_performance['total_signals'], 1)
            ),
            'avg_reversion_time': self.mean_reversion_performance['avg_reversion_time'],
            'avg_profit_per_reversion': self.mean_reversion_performance['avg_profit_per_reversion'],
            'current_z_score': self.z_scores[-1] if self.z_scores else 0,
            'volatility_regime': self.detect_volatility_regime(),
            'mean_deviation': self.calculate_mean_deviation()
        } 