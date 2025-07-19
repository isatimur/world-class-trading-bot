"""
Portfolio models for positions and performance metrics.

This module contains Pydantic models for representing portfolio data,
positions, and performance metrics.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Position(BaseModel):
    """Individual trading position model"""
    symbol: str = Field(..., description="Trading symbol")
    quantity: float = Field(..., description="Position quantity")
    entry_price: float = Field(..., description="Entry price")
    current_price: float = Field(..., description="Current market price")
    market_value: float = Field(..., description="Current market value")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss")
    unrealized_pnl_pct: float = Field(..., description="Unrealized P&L percentage")
    realized_pnl: float = Field(..., description="Realized profit/loss")
    total_pnl: float = Field(..., description="Total profit/loss")
    weight: float = Field(..., description="Position weight in portfolio")
    entry_date: datetime = Field(..., description="Position entry date")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics model"""
    total_return: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    volatility: float = Field(..., description="Portfolio volatility")
    beta: float = Field(..., description="Portfolio beta")
    alpha: float = Field(..., description="Portfolio alpha")
    information_ratio: float = Field(..., description="Information ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    var_95: float = Field(..., description="95% Value at Risk")
    cvar_95: float = Field(..., description="95% Conditional Value at Risk")
    win_rate: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    average_win: float = Field(..., description="Average winning trade")
    average_loss: float = Field(..., description="Average losing trade")
    largest_win: float = Field(..., description="Largest winning trade")
    largest_loss: float = Field(..., description="Largest losing trade")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Portfolio(BaseModel):
    """Portfolio model"""
    name: str = Field(..., description="Portfolio name")
    total_value: float = Field(..., description="Total portfolio value")
    cash_balance: float = Field(..., description="Available cash balance")
    invested_value: float = Field(..., description="Total invested value")
    positions: List[Position] = Field(..., description="Portfolio positions")
    performance: PerformanceMetrics = Field(..., description="Performance metrics")
    risk_metrics: Dict[str, float] = Field(..., description="Risk metrics")
    allocation: Dict[str, float] = Field(..., description="Asset allocation")
    last_rebalance: Optional[datetime] = Field(None, description="Last rebalance date")
    created_date: datetime = Field(..., description="Portfolio creation date")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Trade(BaseModel):
    """Individual trade model"""
    id: str = Field(..., description="Unique trade ID")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Trade side (BUY/SELL)")
    quantity: float = Field(..., description="Trade quantity")
    price: float = Field(..., description="Trade price")
    value: float = Field(..., description="Trade value")
    commission: float = Field(..., description="Trade commission")
    net_value: float = Field(..., description="Net trade value")
    timestamp: datetime = Field(..., description="Trade timestamp")
    order_type: str = Field(..., description="Order type (MARKET/LIMIT/STOP)")
    status: str = Field(..., description="Trade status")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PortfolioSummary(BaseModel):
    """Portfolio summary model"""
    total_value: float = Field(..., description="Total portfolio value")
    total_return: float = Field(..., description="Total return percentage")
    daily_return: float = Field(..., description="Daily return percentage")
    number_of_positions: int = Field(..., description="Number of positions")
    top_performer: Optional[str] = Field(None, description="Top performing position")
    worst_performer: Optional[str] = Field(None, description="Worst performing position")
    cash_percentage: float = Field(..., description="Cash percentage of portfolio")
    timestamp: datetime = Field(default_factory=datetime.now, description="Summary timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 