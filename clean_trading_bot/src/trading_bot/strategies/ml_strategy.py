"""
Machine Learning Trading Strategy implementation.

This module implements a sophisticated ML-based trading strategy using
multiple models for price prediction, signal generation, and risk management.
"""

import asyncio
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from .base_strategy import BaseStrategy, Signal, SignalType, StrategyType
from ..models.market_data import MarketData
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelPrediction:
    """Model prediction with confidence"""
    model_name: str
    prediction: Union[float, int]
    confidence: float
    features: Dict
    timestamp: datetime


class MLStrategy(BaseStrategy):
    """
    Advanced Machine Learning Trading Strategy.
    
    This strategy uses multiple ML models for:
    - Price prediction
    - Signal generation
    - Risk assessment
    - Portfolio optimization
    """
    
    def __init__(
        self,
        strategy_id: str,
        symbol: str,
        models_config: Optional[Dict] = None,
        feature_lookback: int = 50,
        prediction_horizon: int = 5,
        ensemble_method: str = 'voting',
        retrain_frequency: int = 1000,
        **kwargs
    ):
        """
        Initialize ML strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            symbol: Trading symbol
            models_config: Configuration for ML models
            feature_lookback: Number of historical data points for features
            prediction_horizon: Prediction horizon in periods
            ensemble_method: Ensemble method ('voting', 'weighted', 'stacking')
            retrain_frequency: Retrain models every N trades
        """
        super().__init__(
            strategy_id=strategy_id,
            symbol=symbol,
            strategy_type=StrategyType.ML,
            **kwargs
        )
        
        self.feature_lookback = feature_lookback
        self.prediction_horizon = prediction_horizon
        self.ensemble_method = ensemble_method
        self.retrain_frequency = retrain_frequency
        
        # Model configuration
        self.models_config = models_config or self._get_default_models_config()
        
        # ML models
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        
        # Feature engineering
        self.feature_columns = [
            'price', 'volume', 'rsi', 'macd', 'bollinger_position',
            'volatility', 'trend_strength', 'support_resistance',
            'price_momentum', 'volume_momentum', 'price_acceleration',
            'volatility_regime', 'market_regime', 'correlation'
        ]
        
        # Data storage
        self.historical_data = pd.DataFrame()
        self.feature_data = pd.DataFrame()
        self.predictions_history = []
        
        # Performance tracking
        self.model_accuracy = {}
        self.prediction_errors = []
        
        logger.info(f"Initialized ML Strategy with {len(self.models_config)} models")
    
    def _get_default_models_config(self) -> Dict:
        """Get default configuration for ML models."""
        return {
            'price_predictor': {
                'type': 'regression',
                'model': 'xgboost',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'signal_classifier': {
                'type': 'classification',
                'model': 'random_forest',
                'params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'random_state': 42
                }
            },
            'risk_assessor': {
                'type': 'regression',
                'model': 'gradient_boosting',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 4,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'lstm_predictor': {
                'type': 'sequence',
                'model': 'lstm',
                'params': {
                    'units': 50,
                    'layers': 2,
                    'dropout': 0.2,
                    'epochs': 50,
                    'batch_size': 32
                }
            }
        }
    
    async def initialize_models(self):
        """Initialize all ML models."""
        for model_name, config in self.models_config.items():
            await self._create_model(model_name, config)
    
    async def _create_model(self, model_name: str, config: Dict):
        """Create and initialize a specific model."""
        try:
            model_type = config['type']
            model_class = config['model']
            
            if model_type == 'regression':
                if model_class == 'xgboost':
                    model = xgb.XGBRegressor(**config['params'])
                elif model_class == 'gradient_boosting':
                    model = GradientBoostingRegressor(**config['params'])
                else:
                    model = RandomForestRegressor(**config['params'])
            
            elif model_type == 'classification':
                if model_class == 'random_forest':
                    model = RandomForestClassifier(**config['params'])
                elif model_class == 'logistic':
                    model = LogisticRegression(**config['params'])
                else:
                    model = RandomForestClassifier(**config['params'])
            
            elif model_type == 'sequence':
                model = self._create_lstm_model(config['params'])
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.models[model_name] = model
            self.scalers[model_name] = StandardScaler()
            self.model_performance[model_name] = {
                'accuracy': 0.0,
                'predictions': 0,
                'last_retrain': datetime.now()
            }
            
            logger.info(f"Created {model_name} model ({model_type})")
            
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {e}")
    
    def _create_lstm_model(self, params: Dict) -> keras.Model:
        """Create LSTM model for sequence prediction."""
        model = Sequential([
            LSTM(params['units'], return_sequences=True, input_shape=(self.feature_lookback, len(self.feature_columns))),
            Dropout(params['dropout']),
            LSTM(params['units'], return_sequences=False),
            Dropout(params['dropout']),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    async def prepare_features(self, market_data: MarketData) -> pd.DataFrame:
        """
        Prepare features for ML models.
        
        Args:
            market_data: Current market data
            
        Returns:
            Feature DataFrame
        """
        # Add current data to history
        new_data = pd.DataFrame([{
            'timestamp': market_data.timestamp,
            'price': market_data.current_price,
            'volume': market_data.volume,
            'rsi': getattr(market_data, 'rsi', 50),
            'macd': getattr(market_data, 'macd', 0),
            'bollinger_position': getattr(market_data, 'bollinger_position', 0.5),
            'volatility': self.calculate_volatility(),
            'trend_strength': self.calculate_trend_strength(),
            'support_resistance': self.calculate_support_resistance(),
            'price_momentum': self.calculate_price_momentum(),
            'volume_momentum': self.calculate_volume_momentum(),
            'price_acceleration': self.calculate_price_acceleration(),
            'volatility_regime': self.calculate_volatility_regime(),
            'market_regime': self.calculate_market_regime(),
            'correlation': self.calculate_correlation()
        }])
        
        self.historical_data = pd.concat([self.historical_data, new_data], ignore_index=True)
        
        # Keep only recent data
        if len(self.historical_data) > self.feature_lookback * 2:
            self.historical_data = self.historical_data.tail(self.feature_lookback * 2)
        
        # Prepare features
        if len(self.historical_data) >= self.feature_lookback:
            features = self.historical_data[self.feature_columns].tail(self.feature_lookback)
            return features
        
        return pd.DataFrame()
    
    def calculate_price_momentum(self) -> float:
        """Calculate price momentum."""
        if len(self.price_history) < 10:
            return 0.0
        
        prices = np.array(self.price_history[-10:])
        return (prices[-1] - prices[0]) / prices[0]
    
    def calculate_volume_momentum(self) -> float:
        """Calculate volume momentum."""
        if len(self.volume_history) < 10:
            return 0.0
        
        volumes = np.array(self.volume_history[-10:])
        return (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0.0
    
    def calculate_price_acceleration(self) -> float:
        """Calculate price acceleration."""
        if len(self.price_history) < 20:
            return 0.0
        
        prices = np.array(self.price_history[-20:])
        momentum_1 = (prices[-10] - prices[0]) / prices[0]
        momentum_2 = (prices[-1] - prices[-10]) / prices[-10]
        
        return momentum_2 - momentum_1
    
    def calculate_volatility(self) -> float:
        """Calculate current volatility."""
        if len(self.price_history) < 20:
            return 0.0
        
        returns = pd.Series(self.price_history).pct_change().dropna()
        return returns.tail(20).std()
    
    def calculate_volatility_regime(self) -> float:
        """Calculate volatility regime."""
        if len(self.price_history) < 50:
            return 0.5
        
        returns = pd.Series(self.price_history).pct_change().dropna()
        current_vol = returns.tail(20).std()
        historical_vol = returns.std()
        
        return current_vol / historical_vol if historical_vol > 0 else 1.0
    
    def calculate_market_regime(self) -> float:
        """Calculate market regime."""
        if len(self.price_history) < 50:
            return 0.5
        
        prices = np.array(self.price_history[-50:])
        trend = np.polyfit(range(len(prices)), prices, 1)[0]
        
        # Normalize trend
        return np.tanh(trend / prices.mean() * 100)
    
    def calculate_trend_strength(self) -> float:
        """Calculate trend strength."""
        if len(self.price_history) < 20:
            return 0.0
        
        prices = np.array(self.price_history[-20:])
        trend = np.polyfit(range(len(prices)), prices, 1)[0]
        
        # Normalize trend strength
        return np.tanh(trend / prices.mean() * 100)
    
    def calculate_support_resistance(self) -> float:
        """Calculate support/resistance level."""
        if len(self.price_history) < 20:
            return 0.5
        
        current_price = self.price_history[-1]
        recent_prices = np.array(self.price_history[-20:])
        
        # Simple support/resistance calculation
        resistance = np.percentile(recent_prices, 80)
        support = np.percentile(recent_prices, 20)
        
        if current_price > resistance:
            return 1.0  # Above resistance
        elif current_price < support:
            return 0.0  # Below support
        else:
            return (current_price - support) / (resistance - support) if resistance > support else 0.5
    
    def calculate_correlation(self) -> float:
        """Calculate correlation with market."""
        # Simplified correlation calculation
        if len(self.price_history) < 20:
            return 0.0
        
        prices = np.array(self.price_history[-20:])
        returns = np.diff(prices) / prices[:-1]
        
        # Auto-correlation
        if len(returns) >= 2:
            return np.corrcoef(returns[:-1], returns[1:])[0, 1]
        
        return 0.0
    
    async def train_models(self, historical_data: pd.DataFrame):
        """
        Train all ML models with historical data.
        
        Args:
            historical_data: Historical market data
        """
        if len(historical_data) < 100:
            logger.warning("Insufficient data for model training")
            return
        
        for model_name, config in self.models_config.items():
            await self._train_model(model_name, config, historical_data)
    
    async def _train_model(self, model_name: str, config: Dict, data: pd.DataFrame):
        """Train a specific model."""
        try:
            model = self.models[model_name]
            model_type = config['type']
            
            if model_type == 'regression':
                # Price prediction
                X = data[self.feature_columns].iloc[:-self.prediction_horizon]
                y = data['price'].iloc[self.prediction_horizon:].values
                
                # Align lengths
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y[:min_len]
                
                # Scale features
                X_scaled = self.scalers[model_name].fit_transform(X)
                
                # Train model
                model.fit(X_scaled, y)
                
                # Evaluate
                y_pred = model.predict(X_scaled)
                mse = np.mean((y - y_pred) ** 2)
                self.model_performance[model_name]['accuracy'] = 1 / (1 + mse)
            
            elif model_type == 'classification':
                # Signal classification
                X = data[self.feature_columns].iloc[:-1]
                
                # Create labels: 1 for price increase, 0 for decrease
                price_changes = data['price'].pct_change().iloc[1:]
                y = (price_changes > 0).astype(int)
                
                # Align lengths
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]
                
                # Scale features
                X_scaled = self.scalers[model_name].fit_transform(X)
                
                # Train model
                model.fit(X_scaled, y)
                
                # Evaluate
                y_pred = model.predict(X_scaled)
                accuracy = accuracy_score(y, y_pred)
                self.model_performance[model_name]['accuracy'] = accuracy
            
            elif model_type == 'sequence':
                # LSTM sequence prediction
                X, y = self._prepare_sequence_data(data)
                
                if len(X) > 0:
                    # Train LSTM
                    model.fit(X, y, epochs=config['params']['epochs'], 
                             batch_size=config['params']['batch_size'], verbose=0)
                    
                    # Evaluate
                    y_pred = model.predict(X, verbose=0)
                    mse = np.mean((y - y_pred.flatten()) ** 2)
                    self.model_performance[model_name]['accuracy'] = 1 / (1 + mse)
            
            self.model_performance[model_name]['last_retrain'] = datetime.now()
            logger.info(f"Trained {model_name} model (accuracy: {self.model_performance[model_name]['accuracy']:.3f})")
            
        except Exception as e:
            logger.error(f"Error training {model_name} model: {e}")
    
    def _prepare_sequence_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM."""
        if len(data) < self.feature_lookback + self.prediction_horizon:
            return np.array([]), np.array([])
        
        X, y = [], []
        features = data[self.feature_columns].values
        
        for i in range(self.feature_lookback, len(data) - self.prediction_horizon):
            X.append(features[i-self.feature_lookback:i])
            y.append(data['price'].iloc[i + self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    async def generate_predictions(self, features: pd.DataFrame) -> List[ModelPrediction]:
        """
        Generate predictions from all models.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            List of model predictions
        """
        predictions = []
        
        for model_name, model in self.models.items():
            try:
                if len(features) == 0:
                    continue
                
                model_type = self.models_config[model_name]['type']
                
                if model_type in ['regression', 'classification']:
                    # Standard ML models
                    X_scaled = self.scalers[model_name].transform(features)
                    prediction = model.predict(X_scaled)[-1]
                    confidence = self.model_performance[model_name]['accuracy']
                
                elif model_type == 'sequence':
                    # LSTM model
                    X = features.values.reshape(1, self.feature_lookback, len(self.feature_columns))
                    prediction = model.predict(X, verbose=0)[0][0]
                    confidence = self.model_performance[model_name]['accuracy']
                
                else:
                    continue
                
                predictions.append(ModelPrediction(
                    model_name=model_name,
                    prediction=prediction,
                    confidence=confidence,
                    features=features.iloc[-1].to_dict(),
                    timestamp=datetime.now()
                ))
                
            except Exception as e:
                logger.error(f"Error generating prediction for {model_name}: {e}")
        
        return predictions
    
    async def ensemble_predict(self, predictions: List[ModelPrediction]) -> Tuple[float, float]:
        """
        Combine predictions using ensemble method.
        
        Args:
            predictions: List of model predictions
            
        Returns:
            Tuple of (ensemble_prediction, confidence)
        """
        if not predictions:
            return 0.0, 0.0
        
        if self.ensemble_method == 'voting':
            # Simple voting
            if len(predictions) == 1:
                pred = predictions[0]
                return pred.prediction, pred.confidence
            
            # For classification models
            classification_preds = [p for p in predictions if self.models_config[p.model_name]['type'] == 'classification']
            if classification_preds:
                votes = [1 if p.prediction > 0.5 else 0 for p in classification_preds]
                ensemble_pred = np.mean(votes)
                confidence = np.mean([p.confidence for p in classification_preds])
                return ensemble_pred, confidence
            
            # For regression models
            regression_preds = [p for p in predictions if self.models_config[p.model_name]['type'] == 'regression']
            if regression_preds:
                ensemble_pred = np.mean([p.prediction for p in regression_preds])
                confidence = np.mean([p.confidence for p in regression_preds])
                return ensemble_pred, confidence
        
        elif self.ensemble_method == 'weighted':
            # Weighted average based on model performance
            total_weight = sum(p.confidence for p in predictions)
            if total_weight > 0:
                ensemble_pred = sum(p.prediction * p.confidence for p in predictions) / total_weight
                confidence = np.mean([p.confidence for p in predictions])
                return ensemble_pred, confidence
        
        # Default: simple average
        ensemble_pred = np.mean([p.prediction for p in predictions])
        confidence = np.mean([p.confidence for p in predictions])
        return ensemble_pred, confidence
    
    async def generate_signal(self, market_data: MarketData) -> Optional[Signal]:
        """
        Generate trading signal based on ML predictions.
        
        Args:
            market_data: Current market data
            
        Returns:
            Trading signal or None
        """
        # Prepare features
        features = await self.prepare_features(market_data)
        
        if len(features) == 0:
            return None
        
        # Generate predictions
        predictions = await self.generate_predictions(features)
        
        if not predictions:
            return None
        
        # Store predictions
        self.predictions_history.extend(predictions)
        
        # Ensemble prediction
        ensemble_pred, confidence = await self.ensemble_predict(predictions)
        
        # Generate signal based on prediction
        current_price = market_data.current_price
        
        # Signal generation logic
        signal = None
        
        # Price prediction model
        price_pred = next((p for p in predictions if 'price' in p.model_name), None)
        if price_pred:
            price_change_pct = (price_pred.prediction - current_price) / current_price
            
            if price_change_pct > 0.02 and confidence > 0.6:  # 2% increase predicted
                signal = Signal(
                    signal_type=SignalType.BUY,
                    symbol=self.symbol,
                    price=current_price,
                    quantity=0,  # Will be calculated
                    confidence=confidence,
                    timestamp=market_data.timestamp,
                    reasoning=f"ML predicts {price_change_pct:.2%} price increase",
                    stop_loss=current_price * (1 - self.stop_loss_pct),
                    take_profit=current_price * (1 + self.take_profit_pct)
                )
            
            elif price_change_pct < -0.02 and confidence > 0.6:  # 2% decrease predicted
                signal = Signal(
                    signal_type=SignalType.SELL,
                    symbol=self.symbol,
                    price=current_price,
                    quantity=0,  # Will be calculated
                    confidence=confidence,
                    timestamp=market_data.timestamp,
                    reasoning=f"ML predicts {price_change_pct:.2%} price decrease",
                    stop_loss=current_price * (1 + self.stop_loss_pct),
                    take_profit=current_price * (1 - self.take_profit_pct)
                )
        
        # Retrain models periodically
        if len(self.trades) > 0 and len(self.trades) % self.retrain_frequency == 0:
            await self.train_models(self.historical_data)
        
        return signal
    
    async def calculate_position_size(self, signal: Signal, capital: float) -> float:
        """
        Calculate position size based on ML confidence and risk assessment.
        
        Args:
            signal: Trading signal
            capital: Available capital
            
        Returns:
            Position size
        """
        # Base position size
        base_position = capital * self.risk_per_trade
        
        # Adjust based on ML confidence
        confidence_multiplier = signal.confidence
        
        # Risk assessment
        risk_model = self.models.get('risk_assessor')
        if risk_model and len(self.historical_data) > 0:
            features = self.historical_data[self.feature_columns].iloc[-1:]
            X_scaled = self.scalers['risk_assessor'].transform(features)
            risk_score = risk_model.predict(X_scaled)[0]
            
            # Adjust position based on risk
            risk_multiplier = 1 - risk_score  # Lower risk = larger position
            confidence_multiplier *= risk_multiplier
        
        # Calculate final position size
        position_value = base_position * confidence_multiplier
        position_value = min(position_value, capital * self.max_position_size)
        
        return position_value / signal.price
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all models."""
        return {
            model_name: {
                'accuracy': metrics['accuracy'],
                'predictions': metrics['predictions'],
                'last_retrain': metrics['last_retrain'].isoformat(),
                'model_type': self.models_config[model_name]['type']
            }
            for model_name, metrics in self.model_performance.items()
        }
    
    def get_strategy_summary(self) -> Dict:
        """Get comprehensive ML strategy summary."""
        performance = self.calculate_performance()
        model_performance = self.get_model_performance()
        
        return {
            'strategy_id': self.strategy_id,
            'strategy_type': self.strategy_type.value,
            'symbol': self.symbol,
            'performance': performance.dict(),
            'model_performance': model_performance,
            'ensemble_method': self.ensemble_method,
            'total_predictions': len(self.predictions_history),
            'models_count': len(self.models),
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital
        } 