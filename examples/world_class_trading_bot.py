#!/usr/bin/env python3
"""
World-Class Trading Bot - Complete Example

This script demonstrates the complete world-class trading bot system with:
1. ML Models trained on historical data
2. Real Trading Strategies (Grid, ML, Mean Reversion)
3. Advanced Technical Indicators
4. Pine Script Generation for TradingView
5. Comprehensive Backtesting Framework
6. Risk Management & Portfolio Optimization

Usage:
    python examples/world_class_trading_bot.py
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from trading_bot.strategies import GridStrategy, MLStrategy, MeanReversionStrategy, MomentumStrategy
from trading_bot.backtesting import BacktestEngine, BacktestConfig
from trading_bot.pine_scripts import PineScriptConfig, MeanReversionPineGenerator, MomentumPineGenerator
from trading_bot.config.settings import Settings
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


async def main():
    """Main function demonstrating the world-class trading bot."""
    
    logger.info("🚀 Starting World-Class Trading Bot Demo")
    
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
    
    # Grid Strategy
    logger.info("🔧 Creating Grid Strategy...")
    grid_strategy = GridStrategy(
        strategy_id="grid_advanced",
        symbol="AAPL",
        grid_levels=15,
        grid_spacing_pct=0.015,  # 1.5% spacing
        ml_enabled=True,
        adaptive_grid=True,
        initial_capital=20000.0,
        risk_per_trade=0.02
    )
    
    # ML Strategy
    logger.info("🤖 Creating ML Strategy...")
    ml_strategy = MLStrategy(
        strategy_id="ml_ensemble",
        symbol="AAPL",
        feature_lookback=50,
        prediction_horizon=5,
        ensemble_method='weighted',
        retrain_frequency=500,
        initial_capital=20000.0,
        risk_per_trade=0.015
    )
    
    # Mean Reversion Strategy
    logger.info("🔄 Creating Mean Reversion Strategy...")
    mean_reversion_strategy = MeanReversionStrategy(
        strategy_id="mean_reversion",
        symbol="AAPL",
        lookback_period=50,
        z_score_threshold=2.0,
        volatility_lookback=20,
        mean_reversion_strength=0.7,
        volatility_adjustment=True,
        ml_enabled=True,
        initial_capital=20000.0,
        risk_per_trade=0.02
    )
    
    # Momentum Strategy
    logger.info("📈 Creating Momentum Strategy...")
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
        initial_capital=20000.0,
        risk_per_trade=0.02
    )
    
    # Add strategies to engine
    engine.add_strategy(grid_strategy)
    engine.add_strategy(ml_strategy)
    engine.add_strategy(mean_reversion_strategy)
    engine.add_strategy(momentum_strategy)
    
    # 4. Train ML Models
    logger.info("🧠 Training ML models...")
    if "AAPL" in engine.data:
        historical_data = engine.data["AAPL"]
        await ml_strategy.train_models(historical_data)
        await mean_reversion_strategy.train_ml_model(historical_data)
        await momentum_strategy.train_ml_model(historical_data)
    
    # 5. Run Backtests
    logger.info("⚡ Running comprehensive backtests...")
    
    results = await engine.run_all_backtests()
    
    # 6. Analyze Results
    logger.info("📊 Analyzing results...")
    
    summary = engine.get_results_summary()
    print("\n" + "="*80)
    print("🎯 BACKTEST RESULTS SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    
    # 7. Detailed Analysis
    for key, result in results.items():
        print(f"\n📈 {key} - Detailed Analysis:")
        print(f"   Total Return: {result.total_return:.2f}%")
        print(f"   Annualized Return: {result.annualized_return:.2f}%")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"   Win Rate: {result.win_rate:.2f}%")
        print(f"   Profit Factor: {result.profit_factor:.3f}")
        print(f"   Total Trades: {result.total_trades}")
        print(f"   Volatility: {result.volatility:.2f}%")
        print(f"   Beta: {result.beta:.3f}")
        print(f"   Alpha: {result.alpha:.2f}%")
    
    # 8. Generate Pine Scripts
    logger.info("📝 Generating Pine Scripts for TradingView...")
    
    # Grid Strategy Pine Script
    grid_config = PineScriptConfig(
        strategy_name="Advanced Grid Strategy",
        symbol="AAPL",
        timeframe="1D",
        parameters={
            "grid_levels": 15,
            "grid_spacing": 0.015,
            "adaptive_grid": True
        }
    )
    
    # ML Strategy Pine Script
    ml_config = PineScriptConfig(
        strategy_name="ML Ensemble Strategy",
        symbol="AAPL",
        timeframe="1D",
        parameters={
            "feature_lookback": 50,
            "prediction_horizon": 5,
            "ensemble_method": "weighted"
        }
    )
    
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
    
    # Save results
    engine.save_results("backtest_results.csv")
    
    # 9. Strategy Performance Comparison
    logger.info("🏆 Comparing strategy performance...")
    
    best_strategy = None
    best_sharpe = -999
    
    for key, result in results.items():
        if result.sharpe_ratio > best_sharpe:
            best_sharpe = result.sharpe_ratio
            best_strategy = key
    
    print(f"\n🏆 Best Strategy: {best_strategy}")
    print(f"   Sharpe Ratio: {best_sharpe:.3f}")
    
    # 10. Risk Analysis
    logger.info("🛡️ Performing risk analysis...")
    
    for key, result in results.items():
        print(f"\n🛡️ {key} - Risk Analysis:")
        print(f"   Sortino Ratio: {result.sortino_ratio:.3f}")
        print(f"   Calmar Ratio: {result.calmar_ratio:.3f}")
        print(f"   Information Ratio: {result.information_ratio:.3f}")
        print(f"   Average Win: ${result.avg_win:.2f}")
        print(f"   Average Loss: ${result.avg_loss:.2f}")
        print(f"   Best Trade: ${result.best_trade:.2f}")
        print(f"   Worst Trade: ${result.worst_trade:.2f}")
    
    # 11. Portfolio Optimization
    logger.info("💼 Portfolio optimization analysis...")
    
    # Calculate correlation matrix
    equity_curves = {}
    for key, result in results.items():
        equity_curves[key] = result.equity_curve['equity']
    
    if len(equity_curves) > 1:
        import pandas as pd
        import numpy as np
        
        # Create correlation matrix
        equity_df = pd.DataFrame(equity_curves)
        correlation_matrix = equity_df.corr()
        
        print("\n📊 Strategy Correlation Matrix:")
        print(correlation_matrix.round(3))
        
        # Calculate portfolio metrics
        portfolio_returns = equity_df.pct_change().mean(axis=1)
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        portfolio_sharpe = (portfolio_returns.mean() * 252) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        print(f"\n💼 Portfolio Metrics:")
        print(f"   Portfolio Volatility: {portfolio_volatility*100:.2f}%")
        print(f"   Portfolio Sharpe Ratio: {portfolio_sharpe:.3f}")
    
    # 12. Generate Trading Recommendations
    logger.info("💡 Generating trading recommendations...")
    
    recommendations = []
    
    for key, result in results.items():
        if result.sharpe_ratio > 1.0 and result.win_rate > 50:
            recommendations.append({
                'strategy': key,
                'recommendation': 'STRONG BUY',
                'reason': f'High Sharpe ({result.sharpe_ratio:.3f}) and good win rate ({result.win_rate:.1f}%)'
            })
        elif result.sharpe_ratio > 0.5 and result.win_rate > 45:
            recommendations.append({
                'strategy': key,
                'recommendation': 'BUY',
                'reason': f'Moderate Sharpe ({result.sharpe_ratio:.3f}) and acceptable win rate ({result.win_rate:.1f}%)'
            })
        elif result.max_drawdown > 20:
            recommendations.append({
                'strategy': key,
                'recommendation': 'AVOID',
                'reason': f'High drawdown ({result.max_drawdown:.1f}%) indicates high risk'
            })
        else:
            recommendations.append({
                'strategy': key,
                'recommendation': 'HOLD',
                'reason': 'Mixed performance metrics'
            })
    
    print("\n💡 Trading Recommendations:")
    for rec in recommendations:
        print(f"   {rec['strategy']}: {rec['recommendation']} - {rec['reason']}")
    
    # 13. Save Detailed Reports
    logger.info("💾 Saving detailed reports...")
    
    # Save equity curves
    for key, result in results.items():
        filename = f"equity_curve_{key.replace(' ', '_')}.csv"
        result.equity_curve.to_csv(filename, index=False)
        logger.info(f"Saved equity curve: {filename}")
    
    # Save strategy performance
    for key, result in results.items():
        strategy = engine.strategies.get(result.strategy_name)
        if strategy:
            summary = strategy.get_strategy_summary()
            filename = f"strategy_summary_{key.replace(' ', '_')}.json"
            import json
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Saved strategy summary: {filename}")
    
    logger.info("✅ World-Class Trading Bot Demo Completed!")
    
    # 14. Next Steps
    print("\n" + "="*80)
    print("🚀 NEXT STEPS FOR PRODUCTION DEPLOYMENT")
    print("="*80)
    print("1. 📊 Analyze results and select best strategies")
    print("2. 🔧 Fine-tune strategy parameters")
    print("3. 🧠 Retrain ML models with more data")
    print("4. 📝 Generate Pine Scripts for TradingView")
    print("5. 🛡️ Implement additional risk management")
    print("6. 🔄 Set up live trading with paper trading first")
    print("7. 📈 Monitor performance and rebalance portfolio")
    print("8. 🚀 Scale up with additional capital")
    print("="*80)


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    Path("examples").mkdir(exist_ok=True)
    
    # Run the demo
    asyncio.run(main()) 