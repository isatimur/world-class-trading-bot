"""
Configuration settings for the World-Class Trading Bot.

This module contains all configuration classes and settings for the trading bot,
including trading parameters, API configurations, and model settings.
"""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent


class TradingSettings(BaseSettings):
    """Core trading application settings"""
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "World-Class Trading Bot")
    PROJECT_DESCRIPTION: str = os.getenv("PROJECT_DESCRIPTION", "Advanced AI-Powered Trading System")
    BASE_DIR: Path = BASE_DIR
    APP_NAME: str = "trading-bot"
    
    # Trading Parameters
    RISK_TOLERANCE: str = os.getenv("RISK_TOLERANCE", "MODERATE")  # LOW, MODERATE, HIGH
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.2"))  # 20% max per position
    STOP_LOSS_PERCENTAGE: float = float(os.getenv("STOP_LOSS_PERCENTAGE", "0.05"))  # 5% stop loss
    TAKE_PROFIT_PERCENTAGE: float = float(os.getenv("TAKE_PROFIT_PERCENTAGE", "0.15"))  # 15% take profit
    
    # Trading Mode
    TRADING_MODE: str = os.getenv("TRADING_MODE", "PAPER")  # PAPER, LIVE, BACKTEST
    PAPER_TRADING_BALANCE: float = float(os.getenv("PAPER_TRADING_BALANCE", "100000"))  # USD
    
    # Risk Management
    MAX_PORTFOLIO_RISK: float = float(os.getenv("MAX_PORTFOLIO_RISK", "0.02"))  # 2% max portfolio risk
    MAX_CORRELATION: float = float(os.getenv("MAX_CORRELATION", "0.7"))  # Maximum correlation between positions
    MIN_DIVERSIFICATION: int = int(os.getenv("MIN_DIVERSIFICATION", "5"))  # Minimum number of positions
    MAX_DRAWDOWN_THRESHOLD: float = float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.20"))  # 20%
    
    # Performance Configuration
    PERFORMANCE_MONITORING: bool = os.getenv("PERFORMANCE_MONITORING", "true").lower() == "true"
    PERFORMANCE_LOG_INTERVAL: int = int(os.getenv("PERFORMANCE_LOG_INTERVAL", "300"))  # 5 minutes
    
    # API Configuration
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "30"))  # seconds
    API_RETRY_ATTEMPTS: int = int(os.getenv("API_RETRY_ATTEMPTS", "3"))
    API_RETRY_DELAY: float = float(os.getenv("API_RETRY_DELAY", "1.0"))  # seconds
    
    # Market Data Configuration
    MARKET_DATA_CACHE_DURATION: int = int(os.getenv("MARKET_DATA_CACHE_DURATION", "60"))  # seconds
    TECHNICAL_ANALYSIS_PERIOD: str = os.getenv("TECHNICAL_ANALYSIS_PERIOD", "6mo")  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    
    # Technical Analysis Configuration
    RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
    MACD_FAST: int = int(os.getenv("MACD_FAST", "12"))
    MACD_SLOW: int = int(os.getenv("MACD_SLOW", "26"))
    MACD_SIGNAL: int = int(os.getenv("MACD_SIGNAL", "9"))
    BOLLINGER_PERIOD: int = int(os.getenv("BOLLINGER_PERIOD", "20"))
    BOLLINGER_STD: int = int(os.getenv("BOLLINGER_STD", "2"))
    
    # Portfolio Analysis Configuration
    RISK_FREE_RATE: float = float(os.getenv("RISK_FREE_RATE", "0.02"))  # 2%
    VAR_CONFIDENCE: float = float(os.getenv("VAR_CONFIDENCE", "0.95"))  # 95%
    SHARPE_RATIO_THRESHOLD: float = float(os.getenv("SHARPE_RATIO_THRESHOLD", "1.0"))
    
    # ML Model Configuration
    ML_MODEL_RETRAIN_FREQUENCY: int = int(os.getenv("ML_MODEL_RETRAIN_FREQUENCY", "1000"))  # trades
    ML_FEATURE_LOOKBACK: int = int(os.getenv("ML_FEATURE_LOOKBACK", "50"))  # periods
    ML_PREDICTION_HORIZON: int = int(os.getenv("ML_PREDICTION_HORIZON", "5"))  # periods
    
    # Strategy Configuration
    GRID_LEVELS: int = int(os.getenv("GRID_LEVELS", "15"))
    GRID_SPACING_PCT: float = float(os.getenv("GRID_SPACING_PCT", "0.015"))  # 1.5%
    MEAN_REVERSION_LOOKBACK: int = int(os.getenv("MEAN_REVERSION_LOOKBACK", "50"))
    MEAN_REVERSION_Z_SCORE: float = float(os.getenv("MEAN_REVERSION_Z_SCORE", "2.0"))
    MOMENTUM_SHORT_PERIOD: int = int(os.getenv("MOMENTUM_SHORT_PERIOD", "10"))
    MOMENTUM_LONG_PERIOD: int = int(os.getenv("MOMENTUM_LONG_PERIOD", "30"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "trading_bot.log")
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///trading_bot.db")
    
    # Security Configuration
    ENCRYPT_API_KEYS: bool = os.getenv("ENCRYPT_API_KEYS", "true").lower() == "true"
    API_KEY_ENCRYPTION_KEY: str = os.getenv("API_KEY_ENCRYPTION_KEY", "your_encryption_key_here")
    
    # Notification Configuration
    ENABLE_NOTIFICATIONS: bool = os.getenv("ENABLE_NOTIFICATIONS", "false").lower() == "true"
    NOTIFICATION_EMAIL: str = os.getenv("NOTIFICATION_EMAIL", "")
    NOTIFICATION_WEBHOOK: str = os.getenv("NOTIFICATION_WEBHOOK", "")


class BybitSettings(BaseSettings):
    """Bybit API settings"""
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")
    BYBIT_TESTNET: bool = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
    
    # Bybit Trading Parameters
    BYBIT_MAX_LEVERAGE: int = int(os.getenv("BYBIT_MAX_LEVERAGE", "10"))
    BYBIT_DEFAULT_LEVERAGE: int = int(os.getenv("BYBIT_DEFAULT_LEVERAGE", "5"))
    BYBIT_MIN_ORDER_SIZE: float = float(os.getenv("BYBIT_MIN_ORDER_SIZE", "10"))  # USD
    BYBIT_MAX_ORDER_SIZE: float = float(os.getenv("BYBIT_MAX_ORDER_SIZE", "10000"))  # USD


class GoogleSettings(BaseSettings):
    """Google AI settings"""
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    USE_AI: bool = os.getenv("USE_AI", "false").lower() == "true"
    
    # AI Model Configuration
    MARKET_ANALYST_MODEL: str = os.getenv("MARKET_ANALYST_MODEL", "gemini-2.0-flash")
    TECHNICAL_ANALYST_MODEL: str = os.getenv("TECHNICAL_ANALYST_MODEL", "gemini-2.0-flash")
    PORTFOLIO_MANAGER_MODEL: str = os.getenv("PORTFOLIO_MANAGER_MODEL", "gemini-2.0-flash")
    CRYPTO_TRADER_MODEL: str = os.getenv("CRYPTO_TRADER_MODEL", "gemini-2.0-flash")
    STRATEGY_COORDINATOR_MODEL: str = os.getenv("STRATEGY_COORDINATOR_MODEL", "gemini-2.0-flash")
    
    # Model Performance Settings
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    MODEL_TOP_P: float = float(os.getenv("MODEL_TOP_P", "0.9"))
    MODEL_TOP_K: int = int(os.getenv("MODEL_TOP_K", "40"))
    MODEL_MAX_OUTPUT_TOKENS: int = int(os.getenv("MODEL_MAX_OUTPUT_TOKENS", "8192"))


class TelegramSettings(BaseSettings):
    """Telegram Bot settings"""
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_WEBHOOK_URL: str = os.getenv("TELEGRAM_WEBHOOK_URL", "")
    TELEGRAM_WEBHOOK_ENABLED: bool = os.getenv("TELEGRAM_WEBHOOK_ENABLED", "false").lower() == "true"
    
    # Rate Limiting
    TELEGRAM_RATE_LIMIT_PER_MINUTE: int = int(os.getenv("TELEGRAM_RATE_LIMIT_PER_MINUTE", "10"))
    TELEGRAM_RATE_LIMIT_WINDOW: int = int(os.getenv("TELEGRAM_RATE_LIMIT_WINDOW", "60"))  # seconds


class SettingsModel(
    TradingSettings,
    BybitSettings,
    GoogleSettings,
    TelegramSettings
):
    """Combined settings model for the entire application"""
    
    class Config:
        env_file = BASE_DIR / ".env"
        extra = "ignore"


# Global settings instance
Settings = SettingsModel() 