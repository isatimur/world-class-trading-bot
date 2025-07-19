"""
Market data models for stocks, cryptocurrencies, and market sentiment.

This module contains Pydantic models for representing market data
structures used throughout the trading bot application.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MarketData(BaseModel):
    """Base market data model"""
    symbol: str = Field(..., description="Trading symbol (e.g., AAPL, BTC)")
    current_price: float = Field(..., description="Current market price")
    price_change: float = Field(..., description="Price change from previous close")
    price_change_pct: float = Field(..., description="Percentage price change")
    volume: int = Field(..., description="Trading volume")
    market_cap: float = Field(..., description="Market capitalization")
    timestamp: datetime = Field(default_factory=datetime.now, description="Data timestamp")
    
    # Technical indicators
    rsi: Optional[float] = Field(None, description="Relative Strength Index")
    macd: Optional[float] = Field(None, description="MACD line")
    macd_signal: Optional[float] = Field(None, description="MACD signal line")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram")
    bb_upper: Optional[float] = Field(None, description="Bollinger Bands upper")
    bb_middle: Optional[float] = Field(None, description="Bollinger Bands middle")
    bb_lower: Optional[float] = Field(None, description="Bollinger Bands lower")
    bb_position: Optional[float] = Field(None, description="Bollinger Bands position")
    sma_20: Optional[float] = Field(None, description="20-period Simple Moving Average")
    sma_50: Optional[float] = Field(None, description="50-period Simple Moving Average")
    ema_12: Optional[float] = Field(None, description="12-period Exponential Moving Average")
    ema_26: Optional[float] = Field(None, description="26-period Exponential Moving Average")
    atr: Optional[float] = Field(None, description="Average True Range")
    
    # Additional fields for strategies
    open_price: Optional[float] = Field(None, description="Opening price")
    high_price: Optional[float] = Field(None, description="High price")
    low_price: Optional[float] = Field(None, description="Low price")
    close_price: Optional[float] = Field(None, description="Closing price")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StockData(MarketData):
    """Stock-specific market data model"""
    pe_ratio: float = Field(..., description="Price-to-Earnings ratio")
    dividend_yield: float = Field(..., description="Dividend yield percentage")
    beta: float = Field(..., description="Stock beta (volatility relative to market)")
    sector: str = Field(..., description="Stock sector")
    industry: str = Field(..., description="Stock industry")
    company_name: str = Field(..., description="Company name")
    exchange: str = Field(..., description="Stock exchange")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CryptoData(MarketData):
    """Cryptocurrency-specific market data model"""
    circulating_supply: float = Field(..., description="Circulating supply")
    total_supply: float = Field(..., description="Total supply")
    max_supply: Optional[float] = Field(None, description="Maximum supply")
    rank: int = Field(..., description="Market cap rank")
    price_change_24h: float = Field(..., description="24-hour price change")
    volume_change_24h: float = Field(..., description="24-hour volume change")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MarketSentiment(BaseModel):
    """Market sentiment data model"""
    overall_sentiment: str = Field(..., description="Overall market sentiment (Bullish/Bearish/Neutral)")
    fear_greed_index: Optional[int] = Field(None, description="Fear & Greed Index (0-100)")
    vix_level: Optional[float] = Field(None, description="VIX volatility index level")
    indices: Dict[str, Dict[str, Union[float, str]]] = Field(..., description="Major market indices data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Sentiment timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MarketOverview(BaseModel):
    """Comprehensive market overview model"""
    sentiment: MarketSentiment = Field(..., description="Market sentiment data")
    top_gainers: List[MarketData] = Field(..., description="Top gaining assets")
    top_losers: List[MarketData] = Field(..., description="Top losing assets")
    most_active: List[MarketData] = Field(..., description="Most actively traded assets")
    timestamp: datetime = Field(default_factory=datetime.now, description="Overview timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 