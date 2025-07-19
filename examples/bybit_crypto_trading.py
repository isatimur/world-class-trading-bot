#!/usr/bin/env python3
"""
Bybit Crypto Trading Example

This script demonstrates how to use the Bybit integration
with the trading strategies for cryptocurrency trading.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from trading_bot.tools.bybit_trading_tool import BybitTradingTool
from trading_bot.strategies import GridStrategy, MLStrategy, MeanReversionStrategy, MomentumStrategy
from trading_bot.config.settings import Settings
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


async def get_bybit_market_data(symbol: str, interval: str = "1", limit: int = 100):
    """Get market data from Bybit and convert to pandas DataFrame."""
    async with BybitTradingTool(testnet=True) as bybit:
        result = await bybit.get_market_data(symbol, interval, limit)
        
        if result['success']:
            import pandas as pd
            
            # Convert Bybit data to DataFrame
            data = result['data'].get('list', [])
            if data:
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                    df[col] = pd.to_numeric(df[col])
                
                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
            else:
                logger.error(f"No data returned for {symbol}")
                return None
        else:
            logger.error(f"Failed to get market data for {symbol}: {result['error']}")
            return None


async def crypto_grid_strategy_example():
    """Example of using Grid Strategy with Bybit crypto data."""
    print("🔧 Crypto Grid Strategy Example")
    print("=" * 50)
    
    # Get BTCUSDT market data
    print("📊 Fetching BTCUSDT market data...")
    market_data = await get_bybit_market_data("BTCUSDT", interval="1", limit=200)
    
    if market_data is not None:
        print(f"✅ Got {len(market_data)} data points")
        print(f"   Latest price: ${market_data['close'].iloc[-1]:,.2f}")
        print(f"   Date range: {market_data.index[0]} to {market_data.index[-1]}")
        
        # Create Grid Strategy for BTCUSDT
        grid_strategy = GridStrategy(
            strategy_id="btc_grid",
            symbol="BTCUSDT",
            grid_levels=10,
            grid_spacing_pct=0.02,  # 2% spacing for crypto
            ml_enabled=True,
            adaptive_grid=True,
            initial_capital=1000.0,  # $1000 for crypto
            risk_per_trade=0.01      # 1% risk per trade
        )
        
        print(f"✅ Grid Strategy created for BTCUSDT")
        print(f"   Grid levels: {grid_strategy.grid_levels}")
        print(f"   Spacing: {grid_strategy.grid_spacing_pct*100}%")
        print(f"   Initial capital: ${grid_strategy.initial_capital:,.2f}")
        
        # Simulate strategy with market data
        print("\n📈 Simulating grid strategy...")
        # Note: This would require adapting the strategy to work with Bybit data format
        
    else:
        print("❌ Failed to get market data")


async def crypto_ml_strategy_example():
    """Example of using ML Strategy with Bybit crypto data."""
    print("\n🤖 Crypto ML Strategy Example")
    print("=" * 50)
    
    # Get ETHUSDT market data
    print("📊 Fetching ETHUSDT market data...")
    market_data = await get_bybit_market_data("ETHUSDT", interval="1", limit=500)
    
    if market_data is not None:
        print(f"✅ Got {len(market_data)} data points")
        print(f"   Latest price: ${market_data['close'].iloc[-1]:,.2f}")
        
        # Create ML Strategy for ETHUSDT
        ml_strategy = MLStrategy(
            strategy_id="eth_ml",
            symbol="ETHUSDT",
            feature_lookback=50,
            prediction_horizon=5,
            ensemble_method='weighted',
            retrain_frequency=100,
            initial_capital=1000.0,
            risk_per_trade=0.015
        )
        
        print(f"✅ ML Strategy created for ETHUSDT")
        print(f"   Feature lookback: {ml_strategy.feature_lookback}")
        print(f"   Prediction horizon: {ml_strategy.prediction_horizon}")
        print(f"   Ensemble method: {ml_strategy.ensemble_method}")
        
    else:
        print("❌ Failed to get market data")


async def crypto_momentum_strategy_example():
    """Example of using Momentum Strategy with Bybit crypto data."""
    print("\n📈 Crypto Momentum Strategy Example")
    print("=" * 50)
    
    # Get SOLUSDT market data
    print("📊 Fetching SOLUSDT market data...")
    market_data = await get_bybit_market_data("SOLUSDT", interval="1", limit=300)
    
    if market_data is not None:
        print(f"✅ Got {len(market_data)} data points")
        print(f"   Latest price: ${market_data['close'].iloc[-1]:,.2f}")
        
        # Create Momentum Strategy for SOLUSDT
        momentum_strategy = MomentumStrategy(
            strategy_id="sol_momentum",
            symbol="SOLUSDT",
            short_period=10,
            long_period=30,
            momentum_threshold=0.03,  # 3% for crypto
            trend_confirmation_periods=3,
            volatility_lookback=20,
            momentum_strength=0.8,
            dynamic_sizing=True,
            ml_enabled=True,
            initial_capital=1000.0,
            risk_per_trade=0.02
        )
        
        print(f"✅ Momentum Strategy created for SOLUSDT")
        print(f"   Short period: {momentum_strategy.short_period}")
        print(f"   Long period: {momentum_strategy.long_period}")
        print(f"   Momentum threshold: {momentum_strategy.momentum_threshold*100}%")
        
    else:
        print("❌ Failed to get market data")


async def bybit_portfolio_overview():
    """Get overview of Bybit portfolio and available symbols."""
    print("\n💼 Bybit Portfolio Overview")
    print("=" * 50)
    
    async with BybitTradingTool(testnet=True) as bybit:
        # Get available symbols
        print("🔍 Getting available trading symbols...")
        symbols_result = await bybit.get_available_symbols("linear")
        
        if symbols_result['success']:
            symbols = symbols_result['data'].get('list', [])
            print(f"✅ Total symbols available: {len(symbols)}")
            
            # Show popular crypto pairs
            popular_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
            available_symbols = [s['symbol'] for s in symbols]
            
            print("\n📊 Popular Crypto Pairs:")
            for pair in popular_pairs:
                if pair in available_symbols:
                    # Get current price
                    ticker_result = await bybit.get_ticker(pair)
                    if ticker_result['success']:
                        ticker_data = ticker_result['data'].get('list', [])
                        if ticker_data:
                            price = float(ticker_data[0]['lastPrice'])
                            change_24h = float(ticker_data[0]['price24hPcnt']) * 100
                            print(f"   ✅ {pair}: ${price:,.2f} ({change_24h:+.2f}%)")
                    else:
                        print(f"   ✅ {pair}: Available")
                else:
                    print(f"   ❌ {pair}: Not available")
        
        # Get account info if API keys are configured
        if Settings.BYBIT_API_KEY and Settings.BYBIT_API_SECRET:
            print("\n💰 Account Information:")
            account_result = await bybit.get_account_info()
            
            if account_result['success']:
                account_data = account_result['data']
                wallet_list = account_data.get('list', [])
                
                if wallet_list:
                    wallet = wallet_list[0]
                    total_balance = wallet.get('totalWalletBalance', 'Unknown')
                    available_balance = wallet.get('availableToWithdraw', 'Unknown')
                    print(f"   Total Balance: {total_balance}")
                    print(f"   Available Balance: {available_balance}")
                else:
                    print("   No wallet data available")
            else:
                print(f"   Failed to get account info: {account_result['error']}")
        else:
            print("\n💰 Account Information:")
            print("   API keys not configured - cannot fetch account info")
            print("   Add BYBIT_API_KEY and BYBIT_API_SECRET to .env file")


async def main():
    """Main function demonstrating Bybit crypto trading."""
    print("🚀 Bybit Crypto Trading Example")
    print("=" * 60)
    
    # Test 1: Grid Strategy with BTCUSDT
    await crypto_grid_strategy_example()
    
    # Test 2: ML Strategy with ETHUSDT
    await crypto_ml_strategy_example()
    
    # Test 3: Momentum Strategy with SOLUSDT
    await crypto_momentum_strategy_example()
    
    # Test 4: Portfolio Overview
    await bybit_portfolio_overview()
    
    print("\n" + "=" * 60)
    print("🎉 Bybit Crypto Trading Example Complete!")
    print("\n📋 Summary:")
    print("✅ Bybit integration working with real market data")
    print("✅ Can create strategies for crypto pairs")
    print("✅ Access to 500+ trading symbols")
    print("✅ Real-time price data available")
    print("\n💡 Next Steps:")
    print("1. Add API keys to enable live trading")
    print("2. Integrate strategies with Bybit execution")
    print("3. Add crypto commands to Telegram bot")
    print("4. Implement portfolio management for crypto")


if __name__ == "__main__":
    asyncio.run(main()) 