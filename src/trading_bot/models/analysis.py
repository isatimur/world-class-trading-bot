"""
Analysis models for technical indicators and AI analysis results.

This module contains Pydantic models for representing analysis results
from technical indicators and AI-powered analysis.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class TechnicalAnalysis(BaseModel):
    """Technical analysis results model"""
    symbol: str = Field(..., description="Trading symbol")
    rsi: float = Field(..., description="Relative Strength Index")
    macd: float = Field(..., description="MACD line value")
    macd_signal: float = Field(..., description="MACD signal line")
    macd_histogram: float = Field(..., description="MACD histogram")
    bollinger_upper: float = Field(..., description="Bollinger Bands upper")
    bollinger_middle: float = Field(..., description="Bollinger Bands middle")
    bollinger_lower: float = Field(..., description="Bollinger Bands lower")
    sma_20: float = Field(..., description="20-day Simple Moving Average")
    sma_50: float = Field(..., description="50-day Simple Moving Average")
    sma_200: float = Field(..., description="200-day Simple Moving Average")
    signals: List[str] = Field(..., description="Technical analysis signals")
    strength: str = Field(..., description="Signal strength (Strong/Moderate/Weak)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AIAnalysis(BaseModel):
    """AI-powered analysis results model"""
    query: str = Field(..., description="Original analysis query")
    analysis: str = Field(..., description="AI-generated analysis text")
    sentiment: str = Field(..., description="AI sentiment analysis (Bullish/Bearish/Neutral)")
    confidence: float = Field(..., description="AI confidence score (0-1)")
    key_points: List[str] = Field(..., description="Key points from analysis")
    recommendations: List[str] = Field(..., description="Trading recommendations")
    risk_level: str = Field(..., description="Risk assessment level")
    source: str = Field(..., description="Analysis source (AI/Model)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TradingRecommendation(BaseModel):
    """Trading recommendation model"""
    symbol: str = Field(..., description="Trading symbol")
    action: str = Field(..., description="Recommended action (BUY/SELL/HOLD)")
    confidence: float = Field(..., description="Recommendation confidence (0-1)")
    target_price: Optional[float] = Field(None, description="Target price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    reasoning: str = Field(..., description="Reasoning for recommendation")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    time_horizon: str = Field(..., description="Investment time horizon")
    source: str = Field(..., description="Recommendation source")
    timestamp: datetime = Field(default_factory=datetime.now, description="Recommendation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PortfolioAnalysis(BaseModel):
    """Portfolio analysis results model"""
    total_value: float = Field(..., description="Total portfolio value")
    total_return: float = Field(..., description="Total return percentage")
    daily_return: float = Field(..., description="Daily return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    volatility: float = Field(..., description="Portfolio volatility")
    beta: float = Field(..., description="Portfolio beta")
    alpha: float = Field(..., description="Portfolio alpha")
    var_95: float = Field(..., description="95% Value at Risk")
    diversification_score: float = Field(..., description="Diversification score (0-1)")
    recommendations: List[str] = Field(..., description="Portfolio recommendations")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MarketAnalysis(BaseModel):
    """Comprehensive market analysis model"""
    technical: TechnicalAnalysis = Field(..., description="Technical analysis results")
    ai_analysis: Optional[AIAnalysis] = Field(None, description="AI analysis results")
    recommendation: Optional[TradingRecommendation] = Field(None, description="Trading recommendation")
    market_data: Dict[str, Union[str, float, int]] = Field(..., description="Market data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 