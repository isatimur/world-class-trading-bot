"""
Strategy Manager for coordinating multiple trading strategies.

This module provides a unified interface for managing multiple trading strategies,
portfolio allocation, and performance analysis.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from .base_strategy import BaseStrategy, StrategyType, Signal
from ..models.market_data import MarketData
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AllocationMethod(Enum):
    """Portfolio allocation methods"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    SHARPE_OPTIMIZATION = "sharpe_optimization"
    KELLY_CRITERION = "kelly_criterion"
    CUSTOM = "custom"


@dataclass
class StrategyAllocation:
    """Strategy allocation configuration"""
    strategy_id: str
    weight: float
    max_allocation: float
    min_allocation: float
    risk_budget: float
    is_active: bool = True


class PortfolioMetrics(BaseModel):
    """Portfolio performance metrics"""
    total_return: float = Field(..., description="Total portfolio return")
    sharpe_ratio: float = Field(..., description="Portfolio Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum portfolio drawdown")
    volatility: float = Field(..., description="Portfolio volatility")
    beta: float = Field(..., description="Portfolio beta")
    alpha: float = Field(..., description="Portfolio alpha")
    information_ratio: float = Field(..., description="Information ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(..., description="Strategy correlation matrix")
    allocation_weights: Dict[str, float] = Field(..., description="Current allocation weights")
    timestamp: datetime = Field(default_factory=datetime.now)


class StrategyManager:
    """
    Strategy Manager for coordinating multiple trading strategies.
    
    This class provides portfolio management, risk allocation, and
    unified performance analysis across multiple strategies.
    """
    
    def __init__(
        self,
        manager_id: str,
        total_capital: float = 100000.0,
        allocation_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
        rebalance_frequency: int = 30,  # days
        risk_free_rate: float = 0.02,
        max_portfolio_risk: float = 0.15,
        **kwargs
    ):
        """
        Initialize strategy manager.
        
        Args:
            manager_id: Unique manager identifier
            total_capital: Total portfolio capital
            allocation_method: Portfolio allocation method
            rebalance_frequency: Rebalancing frequency in days
            risk_free_rate: Risk-free rate for calculations
            max_portfolio_risk: Maximum portfolio risk (volatility)
        """
        self.manager_id = manager_id
        self.total_capital = total_capital
        self.current_capital = total_capital
        self.allocation_method = allocation_method
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate
        self.max_portfolio_risk = max_portfolio_risk
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.allocations: Dict[str, StrategyAllocation] = {}
        self.performance_history: List[PortfolioMetrics] = []
        
        # Portfolio tracking
        self.portfolio_equity: List[float] = []
        self.portfolio_weights: List[Dict[str, float]] = []
        self.rebalance_dates: List[datetime] = []
        
        # Risk management
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.volatility_targets: Dict[str, float] = {}
        
        logger.info(f"Initialized Strategy Manager: {manager_id}")
    
    def add_strategy(self, strategy: BaseStrategy, allocation: StrategyAllocation):
        """
        Add a strategy to the manager.
        
        Args:
            strategy: Trading strategy instance
            allocation: Strategy allocation configuration
        """
        self.strategies[strategy.strategy_id] = strategy
        self.allocations[strategy.strategy_id] = allocation
        
        logger.info(f"Added strategy: {strategy.strategy_id} with weight: {allocation.weight}")
    
    def remove_strategy(self, strategy_id: str):
        """
        Remove a strategy from the manager.
        
        Args:
            strategy_id: Strategy identifier to remove
        """
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            del self.allocations[strategy_id]
            logger.info(f"Removed strategy: {strategy_id}")
    
    def calculate_equal_weight_allocation(self) -> Dict[str, float]:
        """
        Calculate equal weight allocation.
        
        Returns:
            Dictionary of strategy weights
        """
        active_strategies = [s for s in self.strategies.keys() 
                           if self.allocations[s].is_active]
        
        if not active_strategies:
            return {}
        
        weight = 1.0 / len(active_strategies)
        return {strategy_id: weight for strategy_id in active_strategies}
    
    def calculate_risk_parity_allocation(self) -> Dict[str, float]:
        """
        Calculate risk parity allocation.
        
        Returns:
            Dictionary of strategy weights
        """
        active_strategies = [s for s in self.strategies.keys() 
                           if self.allocations[s].is_active]
        
        if len(active_strategies) < 2:
            return self.calculate_equal_weight_allocation()
        
        # Calculate strategy volatilities
        volatilities = {}
        for strategy_id in active_strategies:
            strategy = self.strategies[strategy_id]
            if hasattr(strategy, 'calculate_volatility'):
                volatilities[strategy_id] = strategy.calculate_volatility()
            else:
                volatilities[strategy_id] = 0.15  # Default volatility
        
        # Calculate inverse volatility weights
        total_inverse_vol = sum(1 / vol for vol in volatilities.values())
        weights = {strategy_id: (1 / volatilities[strategy_id]) / total_inverse_vol 
                  for strategy_id in active_strategies}
        
        return weights
    
    def calculate_sharpe_optimization_allocation(self) -> Dict[str, float]:
        """
        Calculate Sharpe ratio optimized allocation.
        
        Returns:
            Dictionary of strategy weights
        """
        active_strategies = [s for s in self.strategies.keys() 
                           if self.allocations[s].is_active]
        
        if len(active_strategies) < 2:
            return self.calculate_equal_weight_allocation()
        
        # Calculate strategy Sharpe ratios
        sharpe_ratios = {}
        for strategy_id in active_strategies:
            strategy = self.strategies[strategy_id]
            performance = strategy.get_performance()
            if performance and performance.sharpe_ratio > 0:
                sharpe_ratios[strategy_id] = performance.sharpe_ratio
            else:
                sharpe_ratios[strategy_id] = 0.5  # Default Sharpe ratio
        
        # Calculate Sharpe-weighted allocation
        total_sharpe = sum(sharpe_ratios.values())
        weights = {strategy_id: sharpe_ratios[strategy_id] / total_sharpe 
                  for strategy_id in active_strategies}
        
        return weights
    
    def calculate_kelly_criterion_allocation(self) -> Dict[str, float]:
        """
        Calculate Kelly Criterion allocation.
        
        Returns:
            Dictionary of strategy weights
        """
        active_strategies = [s for s in self.strategies.keys() 
                           if self.allocations[s].is_active]
        
        if not active_strategies:
            return {}
        
        kelly_weights = {}
        for strategy_id in active_strategies:
            strategy = self.strategies[strategy_id]
            performance = strategy.get_performance()
            
            if performance and performance.win_rate > 0:
                # Kelly Criterion: f = (bp - q) / b
                # where b = odds received, p = win probability, q = loss probability
                win_rate = performance.win_rate / 100
                avg_win = performance.best_trade if performance.best_trade > 0 else 1.0
                avg_loss = abs(performance.worst_trade) if performance.worst_trade < 0 else 1.0
                
                if avg_loss > 0:
                    b = avg_win / avg_loss
                    p = win_rate
                    q = 1 - win_rate
                    kelly_fraction = (b * p - q) / b
                    kelly_weights[strategy_id] = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                else:
                    kelly_weights[strategy_id] = 0.1  # Default weight
            else:
                kelly_weights[strategy_id] = 0.1  # Default weight
        
        # Normalize weights
        total_weight = sum(kelly_weights.values())
        if total_weight > 0:
            normalized_weights = {strategy_id: weight / total_weight 
                                for strategy_id, weight in kelly_weights.items()}
        else:
            normalized_weights = self.calculate_equal_weight_allocation()
        
        return normalized_weights
    
    def calculate_allocation(self) -> Dict[str, float]:
        """
        Calculate portfolio allocation based on selected method.
        
        Returns:
            Dictionary of strategy weights
        """
        if self.allocation_method == AllocationMethod.EQUAL_WEIGHT:
            return self.calculate_equal_weight_allocation()
        elif self.allocation_method == AllocationMethod.RISK_PARITY:
            return self.calculate_risk_parity_allocation()
        elif self.allocation_method == AllocationMethod.SHARPE_OPTIMIZATION:
            return self.calculate_sharpe_optimization_allocation()
        elif self.allocation_method == AllocationMethod.KELLY_CRITERION:
            return self.calculate_kelly_criterion_allocation()
        else:
            return self.calculate_equal_weight_allocation()
    
    def update_correlation_matrix(self):
        """Update strategy correlation matrix."""
        if len(self.strategies) < 2:
            return
        
        # Get strategy equity curves
        equity_data = {}
        for strategy_id, strategy in self.strategies.items():
            if hasattr(strategy, 'get_equity_curve'):
                equity_curve = strategy.get_equity_curve()
                if equity_curve is not None and not equity_curve.empty:
                    equity_data[strategy_id] = equity_curve['equity']
        
        if len(equity_data) >= 2:
            equity_df = pd.DataFrame(equity_data)
            self.correlation_matrix = equity_df.corr()
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics.
        
        Returns:
            Portfolio performance metrics
        """
        # Get current allocation
        current_allocation = self.calculate_allocation()
        
        # Calculate portfolio returns
        portfolio_returns = []
        strategy_returns = {}
        
        for strategy_id, strategy in self.strategies.items():
            if strategy_id in current_allocation:
                performance = strategy.get_performance()
                if performance:
                    strategy_returns[strategy_id] = performance.total_return / 100
                    portfolio_returns.append(performance.total_return / 100 * current_allocation[strategy_id])
        
        if not portfolio_returns:
            return PortfolioMetrics(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                beta=0.0,
                alpha=0.0,
                information_ratio=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                correlation_matrix={},
                allocation_weights=current_allocation
            )
        
        # Calculate portfolio metrics
        total_return = sum(portfolio_returns)
        
        # Calculate volatility (simplified)
        volatility = np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 1 else 0.15
        
        # Calculate Sharpe ratio
        sharpe_ratio = (total_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Calculate other metrics (simplified)
        max_drawdown = min(portfolio_returns) if portfolio_returns else 0
        beta = 1.0  # Simplified beta calculation
        alpha = total_return - (self.risk_free_rate + beta * 0.08)  # Assuming 8% market return
        information_ratio = alpha / volatility if volatility > 0 else 0
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sortino_ratio = (total_return - self.risk_free_rate) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Update correlation matrix
        self.update_correlation_matrix()
        correlation_dict = self.correlation_matrix.to_dict() if not self.correlation_matrix.empty else {}
        
        return PortfolioMetrics(
            total_return=total_return * 100,  # Convert to percentage
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown * 100,  # Convert to percentage
            volatility=volatility * 100,  # Convert to percentage
            beta=beta,
            alpha=alpha * 100,  # Convert to percentage
            information_ratio=information_ratio,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            correlation_matrix=correlation_dict,
            allocation_weights=current_allocation
        )
    
    async def rebalance_portfolio(self):
        """Rebalance portfolio based on current allocation method."""
        logger.info("🔄 Rebalancing portfolio...")
        
        # Calculate new allocation
        new_allocation = self.calculate_allocation()
        
        # Update strategy capital allocations
        for strategy_id, weight in new_allocation.items():
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                new_capital = self.current_capital * weight
                strategy.current_capital = new_capital
                
                logger.info(f"Allocated {new_capital:.2f} to {strategy_id} (weight: {weight:.2%})")
        
        # Record rebalance
        self.rebalance_dates.append(datetime.now())
        self.portfolio_weights.append(new_allocation)
        
        logger.info("✅ Portfolio rebalancing completed")
    
    async def run_portfolio(self, market_data_stream):
        """
        Run the entire portfolio of strategies.
        
        Args:
            market_data_stream: Stream of market data
        """
        logger.info("🚀 Starting portfolio execution...")
        
        # Initial rebalancing
        await self.rebalance_portfolio()
        
        # Run all strategies
        tasks = []
        for strategy_id, strategy in self.strategies.items():
            if self.allocations[strategy_id].is_active:
                task = asyncio.create_task(strategy.run(market_data_stream))
                tasks.append(task)
        
        # Wait for all strategies to complete
        await asyncio.gather(*tasks)
        
        # Calculate final portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics()
        self.performance_history.append(portfolio_metrics)
        
        logger.info("✅ Portfolio execution completed")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.
        
        Returns:
            Portfolio summary dictionary
        """
        portfolio_metrics = self.calculate_portfolio_metrics()
        
        summary = {
            'manager_id': self.manager_id,
            'total_capital': self.total_capital,
            'current_capital': self.current_capital,
            'allocation_method': self.allocation_method.value,
            'num_strategies': len(self.strategies),
            'active_strategies': len([s for s in self.strategies.keys() 
                                    if self.allocations[s].is_active]),
            'portfolio_metrics': portfolio_metrics.dict(),
            'strategy_summaries': {}
        }
        
        # Add individual strategy summaries
        for strategy_id, strategy in self.strategies.items():
            if hasattr(strategy, 'get_strategy_summary'):
                summary['strategy_summaries'][strategy_id] = strategy.get_strategy_summary()
        
        return summary
    
    def get_performance_history(self) -> List[PortfolioMetrics]:
        """
        Get portfolio performance history.
        
        Returns:
            List of portfolio performance metrics
        """
        return self.performance_history
    
    def save_portfolio_report(self, filename: str):
        """
        Save comprehensive portfolio report.
        
        Args:
            filename: Output filename
        """
        import json
        
        report = {
            'portfolio_summary': self.get_portfolio_summary(),
            'performance_history': [metrics.dict() for metrics in self.performance_history],
            'rebalance_dates': [date.isoformat() for date in self.rebalance_dates],
            'correlation_matrix': self.correlation_matrix.to_dict() if not self.correlation_matrix.empty else {}
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Portfolio report saved: {filename}") 