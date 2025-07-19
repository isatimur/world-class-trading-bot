#!/usr/bin/env python3
"""
Portfolio Manager Demo - Advanced Multi-Strategy Trading System

This script demonstrates the advanced portfolio management capabilities with:
1. Multiple trading strategies (Grid, ML, Mean Reversion, Momentum)
2. Different portfolio allocation methods
3. Risk management and correlation analysis
4. Portfolio rebalancing and optimization
5. Comprehensive performance reporting

Usage:
    python examples/portfolio_manager_demo.py
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from trading_bot.strategies import (
    GridStrategy, MLStrategy, MeanReversionStrategy, MomentumStrategy,
    StrategyManager, StrategyAllocation, AllocationMethod
)
from trading_bot.backtesting import BacktestEngine, BacktestConfig
from trading_bot.pine_scripts import (
    PineScriptConfig, MeanReversionPineGenerator, MomentumPineGenerator
)
from trading_bot.config.settings import Settings
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


async def main():
    """Main function demonstrating the portfolio manager."""
    
    logger.info("🚀 Starting Portfolio Manager Demo")
    
    # 1. Initialize Backtesting Engine
    logger.info("📊 Setting up backtesting engine...")
    
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        initial_capital=100000.0,
        commission=0.001,  # 0.1% commission
        slippage=0.0005,   # 0.05% slippage
        symbols=["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"],
        benchmark_symbol="^GSPC"  # S&P 500
    )
    
    engine = BacktestEngine(config)
    
    # 2. Load Historical Data
    logger.info("📈 Loading historical data...")
    await engine.load_data()
    
    # 3. Create Trading Strategies
    logger.info("🔧 Creating trading strategies...")
    
    # Grid Strategy
    grid_strategy = GridStrategy(
        strategy_id="grid_advanced",
        symbol="AAPL",
        grid_levels=15,
        grid_spacing_pct=0.015,
        ml_enabled=True,
        adaptive_grid=True,
        initial_capital=25000.0,
        risk_per_trade=0.02
    )
    
    # ML Strategy
    ml_strategy = MLStrategy(
        strategy_id="ml_ensemble",
        symbol="AAPL",
        feature_lookback=50,
        prediction_horizon=5,
        ensemble_method='weighted',
        retrain_frequency=500,
        initial_capital=25000.0,
        risk_per_trade=0.015
    )
    
    # Mean Reversion Strategy
    mean_reversion_strategy = MeanReversionStrategy(
        strategy_id="mean_reversion",
        symbol="AAPL",
        lookback_period=50,
        z_score_threshold=2.0,
        volatility_lookback=20,
        mean_reversion_strength=0.7,
        volatility_adjustment=True,
        ml_enabled=True,
        initial_capital=25000.0,
        risk_per_trade=0.02
    )
    
    # Momentum Strategy
    momentum_strategy = MomentumStrategy(
        strategy_id="momentum",
        symbol="AAPL",
        short_period=10,
        long_period=30,
        momentum_threshold=0.02,
        trend_confirmation_periods=3,
        volatility_lookback=20,
        momentum_strength=0.8,
        dynamic_sizing=True,
        ml_enabled=True,
        initial_capital=25000.0,
        risk_per_trade=0.02
    )
    
    # 4. Initialize Strategy Manager
    logger.info("💼 Initializing strategy manager...")
    
    strategy_manager = StrategyManager(
        manager_id="advanced_portfolio",
        total_capital=100000.0,
        allocation_method=AllocationMethod.SHARPE_OPTIMIZATION,
        rebalance_frequency=30,
        risk_free_rate=0.02,
        max_portfolio_risk=0.15
    )
    
    # 5. Add Strategies with Allocations
    logger.info("📊 Adding strategies to portfolio...")
    
    # Equal weight allocations (will be overridden by allocation method)
    allocations = [
        StrategyAllocation(
            strategy_id="grid_advanced",
            weight=0.25,
            max_allocation=0.4,
            min_allocation=0.1,
            risk_budget=0.25,
            is_active=True
        ),
        StrategyAllocation(
            strategy_id="ml_ensemble",
            weight=0.25,
            max_allocation=0.4,
            min_allocation=0.1,
            risk_budget=0.25,
            is_active=True
        ),
        StrategyAllocation(
            strategy_id="mean_reversion",
            weight=0.25,
            max_allocation=0.4,
            min_allocation=0.1,
            risk_budget=0.25,
            is_active=True
        ),
        StrategyAllocation(
            strategy_id="momentum",
            weight=0.25,
            max_allocation=0.4,
            min_allocation=0.1,
            risk_budget=0.25,
            is_active=True
        )
    ]
    
    strategies = [grid_strategy, ml_strategy, mean_reversion_strategy, momentum_strategy]
    
    for strategy, allocation in zip(strategies, allocations):
        strategy_manager.add_strategy(strategy, allocation)
    
    # 6. Train ML Models
    logger.info("🧠 Training ML models...")
    if "AAPL" in engine.data:
        historical_data = engine.data["AAPL"]
        await ml_strategy.train_models(historical_data)
        await mean_reversion_strategy.train_ml_model(historical_data)
        await momentum_strategy.train_ml_model(historical_data)
    
    # 7. Add Strategies to Backtesting Engine
    logger.info("⚡ Adding strategies to backtesting engine...")
    for strategy in strategies:
        engine.add_strategy(strategy)
    
    # 8. Run Backtests
    logger.info("📊 Running comprehensive backtests...")
    results = await engine.run_all_backtests()
    
    # 9. Analyze Individual Strategy Results
    logger.info("📈 Analyzing individual strategy results...")
    
    summary = engine.get_results_summary()
    print("\n" + "="*80)
    print("🎯 INDIVIDUAL STRATEGY RESULTS")
    print("="*80)
    print(summary.to_string(index=False))
    
    # 10. Portfolio Analysis
    logger.info("💼 Analyzing portfolio performance...")
    
    # Calculate portfolio metrics
    portfolio_metrics = strategy_manager.calculate_portfolio_metrics()
    
    print("\n" + "="*80)
    print("💼 PORTFOLIO PERFORMANCE METRICS")
    print("="*80)
    print(f"Total Return: {portfolio_metrics.total_return:.2f}%")
    print(f"Sharpe Ratio: {portfolio_metrics.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {portfolio_metrics.max_drawdown:.2f}%")
    print(f"Volatility: {portfolio_metrics.volatility:.2f}%")
    print(f"Beta: {portfolio_metrics.beta:.3f}")
    print(f"Alpha: {portfolio_metrics.alpha:.2f}%")
    print(f"Information Ratio: {portfolio_metrics.information_ratio:.3f}")
    print(f"Calmar Ratio: {portfolio_metrics.calmar_ratio:.3f}")
    print(f"Sortino Ratio: {portfolio_metrics.sortino_ratio:.3f}")
    
    # 11. Allocation Analysis
    print("\n" + "="*80)
    print("📊 PORTFOLIO ALLOCATION ANALYSIS")
    print("="*80)
    print(f"Allocation Method: {strategy_manager.allocation_method.value}")
    print("\nCurrent Allocation Weights:")
    for strategy_id, weight in portfolio_metrics.allocation_weights.items():
        print(f"  {strategy_id}: {weight:.2%}")
    
    # 12. Correlation Analysis
    if portfolio_metrics.correlation_matrix:
        print("\nStrategy Correlation Matrix:")
        for strategy1, correlations in portfolio_metrics.correlation_matrix.items():
            for strategy2, correlation in correlations.items():
                if strategy1 != strategy2:
                    print(f"  {strategy1} vs {strategy2}: {correlation:.3f}")
    
    # 13. Compare Different Allocation Methods
    logger.info("🔄 Comparing different allocation methods...")
    
    allocation_methods = [
        AllocationMethod.EQUAL_WEIGHT,
        AllocationMethod.RISK_PARITY,
        AllocationMethod.SHARPE_OPTIMIZATION,
        AllocationMethod.KELLY_CRITERION
    ]
    
    print("\n" + "="*80)
    print("🔄 ALLOCATION METHOD COMPARISON")
    print("="*80)
    
    allocation_results = {}
    
    for method in allocation_methods:
        strategy_manager.allocation_method = method
        metrics = strategy_manager.calculate_portfolio_metrics()
        allocation_results[method.value] = metrics
        
        print(f"\n{method.value.upper().replace('_', ' ')}:")
        print(f"  Total Return: {metrics.total_return:.2f}%")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")
        print(f"  Volatility: {metrics.volatility:.2f}%")
        
        print("  Allocation Weights:")
        for strategy_id, weight in metrics.allocation_weights.items():
            print(f"    {strategy_id}: {weight:.2%}")
    
    # 14. Generate Pine Scripts
    logger.info("📝 Generating Pine Scripts for TradingView...")
    
    # Mean Reversion Strategy Pine Script
    mean_reversion_config = PineScriptConfig(
        strategy_name="Mean Reversion Strategy",
        symbol="AAPL",
        timeframe="1D",
        parameters={
            "lookback_period": 50,
            "z_score_threshold": 2.0,
            "volatility_lookback": 20,
            "mean_reversion_strength": 0.7
        }
    )
    
    mean_reversion_generator = MeanReversionPineGenerator(mean_reversion_config)
    mean_reversion_script = mean_reversion_generator.generate_script()
    
    # Momentum Strategy Pine Script
    momentum_config = PineScriptConfig(
        strategy_name="Momentum Strategy",
        symbol="AAPL",
        timeframe="1D",
        parameters={
            "short_period": 10,
            "long_period": 30,
            "momentum_threshold": 0.02,
            "trend_confirmation_periods": 3,
            "volatility_lookback": 20,
            "momentum_strength": 0.8
        }
    )
    
    momentum_generator = MomentumPineGenerator(momentum_config)
    momentum_script = momentum_generator.generate_script()
    
    # Save Pine Scripts
    with open("mean_reversion_strategy.pine", "w") as f:
        f.write(mean_reversion_script)
    
    with open("momentum_strategy.pine", "w") as f:
        f.write(momentum_script)
    
    logger.info("✅ Pine Scripts saved: mean_reversion_strategy.pine, momentum_strategy.pine")
    
    # 15. Portfolio Summary
    logger.info("📋 Generating portfolio summary...")
    
    portfolio_summary = strategy_manager.get_portfolio_summary()
    
    print("\n" + "="*80)
    print("📋 PORTFOLIO SUMMARY")
    print("="*80)
    print(f"Manager ID: {portfolio_summary['manager_id']}")
    print(f"Total Capital: ${portfolio_summary['total_capital']:,.2f}")
    print(f"Current Capital: ${portfolio_summary['current_capital']:,.2f}")
    print(f"Number of Strategies: {portfolio_summary['num_strategies']}")
    print(f"Active Strategies: {portfolio_summary['active_strategies']}")
    
    # 16. Strategy Summaries
    print("\n" + "="*80)
    print("🔧 INDIVIDUAL STRATEGY SUMMARIES")
    print("="*80)
    
    for strategy_id, summary in portfolio_summary['strategy_summaries'].items():
        print(f"\n{strategy_id.upper()}:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    # 17. Save Comprehensive Reports
    logger.info("💾 Saving comprehensive reports...")
    
    # Save portfolio report
    strategy_manager.save_portfolio_report("portfolio_report.json")
    
    # Save backtest results
    engine.save_results("portfolio_backtest_results.csv")
    
    # Save allocation comparison
    import json
    allocation_comparison = {
        method: {
            'total_return': metrics.total_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'volatility': metrics.volatility,
            'allocation_weights': metrics.allocation_weights
        }
        for method, metrics in allocation_results.items()
    }
    
    with open("allocation_comparison.json", "w") as f:
        json.dump(allocation_comparison, f, indent=2, default=str)
    
    # 18. Trading Recommendations
    logger.info("💡 Generating trading recommendations...")
    
    print("\n" + "="*80)
    print("💡 TRADING RECOMMENDATIONS")
    print("="*80)
    
    # Find best allocation method
    best_method = max(allocation_results.keys(), 
                     key=lambda x: allocation_results[x].sharpe_ratio)
    best_metrics = allocation_results[best_method]
    
    print(f"🏆 Best Allocation Method: {best_method}")
    print(f"   Sharpe Ratio: {best_metrics.sharpe_ratio:.3f}")
    print(f"   Total Return: {best_metrics.total_return:.2f}%")
    print(f"   Max Drawdown: {best_metrics.max_drawdown:.2f}%")
    
    print("\nRecommended Portfolio Allocation:")
    for strategy_id, weight in best_metrics.allocation_weights.items():
        print(f"  {strategy_id}: {weight:.2%}")
    
    # Strategy-specific recommendations
    print("\nStrategy Recommendations:")
    for strategy_id, strategy in strategy_manager.strategies.items():
        performance = strategy.get_performance()
        if performance:
            if performance.sharpe_ratio > 1.0 and performance.win_rate > 50:
                recommendation = "STRONG BUY"
                reason = f"Excellent performance (Sharpe: {performance.sharpe_ratio:.3f}, Win Rate: {performance.win_rate:.1f}%)"
            elif performance.sharpe_ratio > 0.5 and performance.win_rate > 45:
                recommendation = "BUY"
                reason = f"Good performance (Sharpe: {performance.sharpe_ratio:.3f}, Win Rate: {performance.win_rate:.1f}%)"
            elif performance.max_drawdown > 20:
                recommendation = "AVOID"
                reason = f"High risk (Max Drawdown: {performance.max_drawdown:.1f}%)"
            else:
                recommendation = "HOLD"
                reason = "Mixed performance metrics"
            
            print(f"  {strategy_id}: {recommendation} - {reason}")
    
    logger.info("✅ Portfolio Manager Demo Completed!")
    
    # 19. Next Steps
    print("\n" + "="*80)
    print("🚀 NEXT STEPS FOR PRODUCTION DEPLOYMENT")
    print("="*80)
    print("1. 📊 Analyze allocation method performance")
    print("2. 🔧 Fine-tune strategy parameters")
    print("3. 🧠 Retrain ML models with more data")
    print("4. 📝 Generate Pine Scripts for all strategies")
    print("5. 🛡️ Implement additional risk management")
    print("6. 🔄 Set up automated rebalancing")
    print("7. 📈 Monitor correlation changes")
    print("8. 🚀 Scale up with additional capital")
    print("9. 📊 Set up real-time monitoring dashboard")
    print("10. 🔄 Implement dynamic allocation adjustments")
    print("="*80)


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    Path("examples").mkdir(exist_ok=True)
    
    # Run the demo
    asyncio.run(main()) 