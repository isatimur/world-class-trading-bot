#!/usr/bin/env python3
"""
Complete Integration Test

This script tests all components working together:
- Bybit integration
- Natural language translation
- Trading strategies
- Configuration management
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
from trading_bot.tools.bybit_trading_tool import BybitTradingTool
from trading_bot.tools.natural_language_translator import (
    NaturalLanguageTranslator,
    translate_trading_signal,
    get_market_advice
)
from trading_bot.config.settings import Settings
from trading_bot.strategies import GridStrategy, MLStrategy, MomentumStrategy
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


@pytest.mark.integration
async def test_bybit_connection():
    """Test Bybit connection and market data."""
    print("🔗 Testing Bybit Connection...")
    
    try:
        async with BybitTradingTool(testnet=True) as bybit:
            # Test market data
            ticker = await bybit.get_ticker("BTCUSDT")
            if ticker['success']:
                price = float(ticker['data']['list'][0]['lastPrice'])
                print(f"✅ Bybit connected - BTCUSDT: ${price:,.2f}")
                return True
            else:
                print(f"❌ Bybit connection failed: {ticker['error']}")
                return False
    except Exception as e:
        print(f"❌ Bybit connection error: {e}")
        return False


@pytest.mark.integration
async def test_natural_language_translation():
    """Test natural language translation."""
    print("\n🗣️ Testing Natural Language Translation...")
    
    try:
        # Test signal translation
        response = await translate_trading_signal(
            symbol="BTCUSDT",
            signal_type="BUY",
            price=62000.0,
            confidence=0.85,
            strategy="Grid Strategy",
            technical_indicators={"rsi": 35.2, "macd": "bullish"},
            market_context={"sentiment": "bullish"},
            reasoning="Price at support level with oversold RSI"
        )
        
        print(f"✅ Signal translation working: {response.summary}")
        return True
        
    except Exception as e:
        print(f"❌ Natural language translation error: {e}")
        return False


@pytest.mark.integration
async def test_trading_strategies():
    """Test trading strategies creation."""
    print("\n📊 Testing Trading Strategies...")
    
    try:
        # Test Grid Strategy
        grid_strategy = GridStrategy(
            strategy_id="test_grid",
            symbol="BTCUSDT",
            grid_levels=10,
            grid_spacing_pct=0.02,
            initial_capital=1000.0,
            risk_per_trade=0.01
        )
        print(f"✅ Grid Strategy created: {grid_strategy.strategy_id}")
        
        # Test ML Strategy
        ml_strategy = MLStrategy(
            strategy_id="test_ml",
            symbol="ETHUSDT",
            feature_lookback=50,
            prediction_horizon=5,
            initial_capital=1000.0,
            risk_per_trade=0.015
        )
        print(f"✅ ML Strategy created: {ml_strategy.strategy_id}")
        
        # Test Momentum Strategy
        momentum_strategy = MomentumStrategy(
            strategy_id="test_momentum",
            symbol="SOLUSDT",
            short_period=10,
            long_period=30,
            initial_capital=1000.0,
            risk_per_trade=0.02
        )
        print(f"✅ Momentum Strategy created: {momentum_strategy.strategy_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading strategies error: {e}")
        return False


@pytest.mark.integration
async def test_configuration():
    """Test configuration management."""
    print("\n⚙️ Testing Configuration...")
    
    try:
        # Test settings
        print(f"✅ Google API Key: {Settings.GOOGLE_API_KEY[:10]}...")
        print(f"✅ Bybit API Key: {Settings.BYBIT_API_KEY[:10]}...")
        print(f"✅ Telegram Token: {Settings.TELEGRAM_BOT_TOKEN[:10]}...")
        print(f"✅ Trading Mode: {Settings.TRADING_MODE}")
        print(f"✅ Risk Tolerance: {Settings.RISK_TOLERANCE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


@pytest.mark.integration
async def test_combined_workflow():
    """Test complete workflow from market data to natural language."""
    print("\n🔄 Testing Combined Workflow...")
    
    try:
        # Step 1: Get market data from Bybit
        async with BybitTradingTool(testnet=True) as bybit:
            ticker = await bybit.get_ticker("BTCUSDT")
            if ticker['success']:
                price = float(ticker['data']['list'][0]['lastPrice'])
                change_24h = float(ticker['data']['list'][0]['price24hPcnt']) * 100
                
                print(f"📊 Market Data: BTCUSDT ${price:,.2f} ({change_24h:+.2f}%)")
                
                # Step 2: Generate trading signal
                if change_24h > 5:
                    signal_type = "BUY"
                    confidence = 0.75
                    reasoning = f"Strong positive momentum with {change_24h:.1f}% gain"
                elif change_24h < -5:
                    signal_type = "SELL"
                    confidence = 0.70
                    reasoning = f"Negative momentum with {change_24h:.1f}% loss"
                else:
                    signal_type = "HOLD"
                    confidence = 0.60
                    reasoning = f"Sideways movement with {change_24h:.1f}% change"
                
                # Step 3: Translate to natural language
                response = await translate_trading_signal(
                    symbol="BTCUSDT",
                    signal_type=signal_type,
                    price=price,
                    confidence=confidence,
                    strategy="Momentum Analysis",
                    technical_indicators={
                        "rsi": 50.0 + (change_24h * 2),
                        "macd": "bullish" if change_24h > 0 else "bearish",
                        "trend": "uptrend" if change_24h > 0 else "downtrend"
                    },
                    market_context={
                        "market_sentiment": "bullish" if change_24h > 0 else "bearish",
                        "volatility": "high" if abs(change_24h) > 5 else "moderate"
                    },
                    reasoning=reasoning,
                    risk_level="HIGH" if abs(change_24h) > 10 else "MODERATE"
                )
                
                print(f"🎯 Signal: {response.summary}")
                print(f"💡 Action: {response.action_recommendation}")
                print(f"⚠️ Risk: {response.risk_assessment}")
                
                return True
            else:
                print(f"❌ Failed to get market data: {ticker['error']}")
                return False
                
    except Exception as e:
        print(f"❌ Combined workflow error: {e}")
        return False


@pytest.mark.integration
async def test_market_advice():
    """Test market advice generation."""
    print("\n💡 Testing Market Advice Generation...")
    
    try:
        # Create market context
        market_context = {
            "current_prices": {
                "BTCUSDT": 120000.0,
                "ETHUSDT": 3500.0,
                "SOLUSDT": 150.0
            },
            "market_sentiment": "bullish",
            "volatility": "moderate",
            "trend": "uptrend",
            "key_events": [
                "Bitcoin halving approaching",
                "Institutional adoption increasing",
                "Regulatory clarity improving"
            ],
            "risk_factors": [
                "Geopolitical tensions",
                "Fed policy uncertainty",
                "Market volatility"
            ]
        }
        
        # Generate market advice
        advice = await get_market_advice(market_context)
        
        print("✅ Market advice generated successfully!")
        print(f"📝 Advice length: {len(advice)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Market advice error: {e}")
        return False


@pytest.mark.integration
async def test_complete_integration():
    """Run complete integration test."""
    print("🚀 Complete Integration Test")
    print("=" * 60)
    
    # Run all tests
    tests = [
        test_configuration,
        test_bybit_connection,
        test_natural_language_translation,
        test_trading_strategies,
        test_combined_workflow,
        test_market_advice
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            results.append(False)
    
    # Print results
    print("\n" + "=" * 60)
    print("🎉 Complete Integration Test Results")
    print()
    
    test_names = [
        "Configuration",
        "Bybit Connection", 
        "Natural Language",
        "Trading Strategies",
        "Combined Workflow",
        "Market Advice"
    ]
    
    print("📋 Summary:")
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASS" if result else "FAIL"
        print(f"{'✅' if result else '❌'} {name}: {status}")
    
    print()
    if all(results):
        print("🎯 ALL SYSTEMS WORKING TOGETHER PERFECTLY!")
        print("✅ Bybit integration: Real-time market data")
        print("✅ Google Agent SDK: Natural language translation")
        print("✅ Trading strategies: Grid, ML, Momentum")
        print("✅ Configuration: All settings loaded")
        print("✅ Combined workflow: End-to-end functionality")
        print("✅ Market advice: AI-powered recommendations")
        print()
        print("🚀 READY FOR PRODUCTION DEPLOYMENT!")
    else:
        print("❌ Some systems need attention")
        print("🔧 Check the failed components above")
    
    return all(results)


# For direct execution
if __name__ == "__main__":
    asyncio.run(test_complete_integration()) 