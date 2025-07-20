#!/usr/bin/env python3
"""
Natural Language Translation Test

This script tests the natural language translation functionality:
- Trading signal translation
- Market advice generation
- Portfolio translation
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from trading_bot.tools.natural_language_translator import (
    NaturalLanguageTranslator,
    translate_trading_signal,
    get_market_advice
)
from trading_bot.config.settings import Settings

async def test_signal_translation():
    """Test translating a trading signal into natural language."""
    print("🧪 Testing Signal Translation...")
    
    try:
        # Create a sample trading signal
        signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type="BUY",
            price=63076.60,
            confidence=0.85,
            strategy="Grid Strategy",
            timestamp=asyncio.get_event_loop().time(),
            technical_indicators={
                "rsi": 35.2,
                "macd": "bullish",
                "bollinger_position": "lower_band",
                "volume": "above_average",
                "trend": "uptrend"
            },
            market_context={
                "market_sentiment": "bullish",
                "support_level": 62000,
                "resistance_level": 65000,
                "volatility": "moderate",
                "volume_trend": "increasing"
            },
            reasoning="Price at support level with oversold RSI and bullish MACD crossover",
            risk_level="MODERATE",
            position_size=0.1,
            stop_loss=61000,
            take_profit=66000
        )
        
        # Translate the signal
        translator = NaturalLanguageTranslator()
        response = await translator.translate_signal(signal)
        
        print("✅ Signal translation successful!")
        print(f"\n📊 Summary: {response.summary}")
        print(f"🎯 Action: {response.action_recommendation}")
        print(f"🧠 Reasoning: {response.reasoning}")
        print(f"⚠️ Risk: {response.risk_assessment}")
        print(f"📈 Context: {response.market_context}")
        print(f"🎯 Confidence: {response.confidence_explanation}")
        print(f"📋 Next Steps: {response.next_steps}")
        
        return True
        
    except Exception as e:
        print(f"❌ Signal translation failed: {e}")
        return False

async def test_market_advice():
    """Test generating trading advice based on market context."""
    print("\n🧪 Testing Market Advice Generation...")
    
    try:
        # Create sample market context
        market_context = {
            "current_prices": {
                "BTCUSDT": 63076.60,
                "ETHUSDT": 3150.25,
                "SOLUSDT": 159.79
            },
            "market_sentiment": "bullish",
            "volatility": "moderate",
            "trend": "uptrend",
            "key_events": [
                "Fed meeting this week",
                "Bitcoin halving approaching",
                "Institutional adoption increasing"
            ],
            "risk_factors": [
                "Geopolitical tensions",
                "Regulatory uncertainty",
                "Market volatility"
            ]
        }
        
        # Generate trading advice
        advice = await get_market_advice(market_context)
        
        print("✅ Market advice generation successful!")
        print(f"\n💡 Trading Advice:")
        print(advice)
        
        return True
        
    except Exception as e:
        print(f"❌ Market advice generation failed: {e}")
        return False

async def test_portfolio_translation():
    """Test translating portfolio status into natural language."""
    print("\n🧪 Testing Portfolio Translation...")
    
    try:
        # Create sample portfolio data
        portfolio_data = {
            "total_value": 125000.0,
            "total_return": 0.25,
            "daily_pnl": 1250.0,
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "quantity": 0.5,
                    "entry_price": 60000,
                    "current_price": 63076.60,
                    "unrealized_pnl": 1538.30,
                    "weight": 0.4
                },
                {
                    "symbol": "ETHUSDT",
                    "quantity": 5.0,
                    "entry_price": 3000,
                    "current_price": 3150.25,
                    "unrealized_pnl": 751.25,
                    "weight": 0.3
                },
                {
                    "symbol": "SOLUSDT",
                    "quantity": 50.0,
                    "entry_price": 140,
                    "current_price": 159.79,
                    "unrealized_pnl": 989.50,
                    "weight": 0.3
                }
            ],
            "risk_metrics": {
                "sharpe_ratio": 1.8,
                "max_drawdown": -0.08,
                "volatility": 0.15,
                "var_95": -0.02
            }
        }
        
        # Translate portfolio status
        summary = await NaturalLanguageTranslator.translate_portfolio_status(portfolio_data)
        
        print("✅ Portfolio translation successful!")
        print(f"\n💼 Portfolio Summary:")
        print(summary)
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio translation failed: {e}")
        return False

async def test_convenience_functions():
    """Test the convenience functions for easy integration."""
    print("\n🧪 Testing Convenience Functions...")
    
    try:
        # Test translate_trading_signal convenience function
        response = await translate_trading_signal(
            symbol="ETHUSDT",
            signal_type="SELL",
            price=3150.25,
            confidence=0.78,
            strategy="Momentum Strategy",
            technical_indicators={
                "rsi": 72.5,
                "macd": "bearish",
                "bollinger_position": "upper_band"
            },
            market_context={
                "market_sentiment": "neutral",
                "resistance_level": 3200
            },
            reasoning="Price at resistance with overbought RSI and bearish MACD",
            risk_level="HIGH"
        )
        
        print("✅ Convenience function test successful!")
        print(f"\n📊 Quick Signal Translation:")
        print(f"Summary: {response.summary}")
        print(f"Action: {response.action_recommendation}")
        print(f"Risk: {response.risk_assessment}")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🚀 Testing Natural Language Translation")
    print("=" * 60)
    
    # Check if Google API key is configured
    if not Settings.GOOGLE_API_KEY:
        print("❌ Google API key not configured!")
        print("Please add GOOGLE_API_KEY to your .env file")
        return
    
    print(f"✅ Google API key configured: {Settings.GOOGLE_API_KEY[:10]}...")
    
    # Test 1: Signal translation
    signal_test = await test_signal_translation()
    
    # Test 2: Market advice
    advice_test = await test_market_advice()
    
    # Test 3: Portfolio translation
    portfolio_test = await test_portfolio_translation()
    
    # Test 4: Convenience functions
    convenience_test = await test_convenience_functions()
    
    print("\n" + "=" * 60)
    print("🎉 Natural Language Translation Test Results")
    print("\n📋 Summary:")
    print(f"✅ Signal translation: {'PASS' if signal_test else 'FAIL'}")
    print(f"✅ Market advice: {'PASS' if advice_test else 'FAIL'}")
    print(f"✅ Portfolio translation: {'PASS' if portfolio_test else 'FAIL'}")
    print(f"✅ Convenience functions: {'PASS' if convenience_test else 'FAIL'}")
    
    if all([signal_test, advice_test, portfolio_test, convenience_test]):
        print("\n🎯 Natural language translation is working perfectly!")
        print("📊 Can translate trading signals into human language")
        print("💡 Can generate trading advice based on market context")
        print("💼 Can translate portfolio status into natural language")
        print("🔧 Ready for integration with Telegram bot and strategies")
    else:
        print("\n❌ Some tests failed - check configuration and API key")

if __name__ == "__main__":
    asyncio.run(main()) 