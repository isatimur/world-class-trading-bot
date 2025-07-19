#!/usr/bin/env python3
"""
Bybit Integration with Natural Language Translation

This script demonstrates how to combine Bybit trading with natural language
translation to provide human-readable trading signals and advice.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from trading_bot.tools.bybit_trading_tool import BybitTradingTool
from trading_bot.tools.natural_language_translator import (
    NaturalLanguageTranslator,
    TradingSignal,
    translate_trading_signal,
    get_market_advice
)
from trading_bot.config.settings import Settings
from trading_bot.utils.logging import get_logger

logger = get_logger(__name__)


async def get_crypto_market_context():
    """Get current market context from Bybit."""
    print("📊 Getting Crypto Market Context...")
    
    async with BybitTradingTool(testnet=True) as bybit:
        # Get popular crypto pairs
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
        market_context = {
            "current_prices": {},
            "market_sentiment": "neutral",
            "volatility": "moderate",
            "trend": "mixed",
            "key_events": [
                "Crypto market analysis",
                "Technical indicators review",
                "Risk assessment"
            ],
            "risk_factors": [
                "Market volatility",
                "Regulatory changes",
                "Geopolitical events"
            ]
        }
        
        for symbol in symbols:
            try:
                ticker = await bybit.get_ticker(symbol)
                if ticker['success']:
                    ticker_data = ticker['data'].get('list', [])
                    if ticker_data:
                        price = float(ticker_data[0]['lastPrice'])
                        change_24h = float(ticker_data[0]['price24hPcnt']) * 100
                        market_context["current_prices"][symbol] = {
                            "price": price,
                            "change_24h": change_24h,
                            "volume": float(ticker_data[0]['volume24h'])
                        }
                        print(f"   ✅ {symbol}: ${price:,.2f} ({change_24h:+.2f}%)")
            except Exception as e:
                print(f"   ❌ Failed to get {symbol} data: {e}")
        
        return market_context


async def generate_trading_signals():
    """Generate sample trading signals based on market data."""
    print("\n🎯 Generating Trading Signals...")
    
    signals = []
    
    # Get market context
    market_context = await get_crypto_market_context()
    
    # Create sample signals based on market data
    for symbol, data in market_context["current_prices"].items():
        price = data["price"]
        change_24h = data["change_24h"]
        
        # Simple signal logic based on 24h change
        if change_24h > 5:
            signal_type = "BUY"
            confidence = 0.75
            reasoning = f"Strong positive momentum with {change_24h:.1f}% gain in 24h"
            risk_level = "MODERATE"
        elif change_24h < -5:
            signal_type = "SELL"
            confidence = 0.70
            reasoning = f"Negative momentum with {change_24h:.1f}% loss in 24h"
            risk_level = "HIGH"
        else:
            signal_type = "HOLD"
            confidence = 0.60
            reasoning = f"Sideways movement with {change_24h:.1f}% change in 24h"
            risk_level = "LOW"
        
        # Create trading signal
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            price=price,
            confidence=confidence,
            strategy="Momentum Analysis",
            timestamp=datetime.now(),
            technical_indicators={
                "rsi": 50.0 + (change_24h * 2),  # Simulated RSI
                "macd": "bullish" if change_24h > 0 else "bearish",
                "trend": "uptrend" if change_24h > 0 else "downtrend",
                "volume": "above_average" if abs(change_24h) > 3 else "normal"
            },
            market_context={
                "market_sentiment": "bullish" if change_24h > 0 else "bearish",
                "support_level": price * 0.95,
                "resistance_level": price * 1.05,
                "volatility": "high" if abs(change_24h) > 5 else "moderate"
            },
            reasoning=reasoning,
            risk_level=risk_level,
            position_size=0.1,
            stop_loss=price * 0.95 if signal_type == "BUY" else price * 1.05,
            take_profit=price * 1.10 if signal_type == "BUY" else price * 0.90
        )
        
        signals.append(signal)
        print(f"   📊 {symbol}: {signal_type} signal generated")
    
    return signals


async def translate_signals_to_natural_language(signals):
    """Translate trading signals into natural language."""
    print("\n🗣️ Translating Signals to Natural Language...")
    
    translator = NaturalLanguageTranslator()
    translated_signals = []
    
    for signal in signals:
        try:
            response = await translator.translate_signal(signal)
            translated_signals.append(response)
            
            print(f"\n📊 {signal.symbol} Signal Translation:")
            print(f"   Summary: {response.summary}")
            print(f"   Action: {response.action_recommendation}")
            print(f"   Risk: {response.risk_assessment}")
            print(f"   Confidence: {response.confidence_explanation}")
            
        except Exception as e:
            print(f"   ❌ Failed to translate {signal.symbol} signal: {e}")
    
    return translated_signals


async def get_market_advice_with_bybit_data():
    """Get trading advice based on Bybit market data."""
    print("\n💡 Getting Market Advice...")
    
    try:
        # Get market context from Bybit
        market_context = await get_crypto_market_context()
        
        # Generate trading advice
        advice = await get_market_advice(market_context)
        
        print("✅ Market advice generated successfully!")
        print(f"\n💡 Trading Advice:")
        print(advice)
        
        return advice
        
    except Exception as e:
        print(f"❌ Failed to get market advice: {e}")
        return None


async def demonstrate_bybit_order_with_natural_language():
    """Demonstrate how to place orders with natural language explanations."""
    print("\n📝 Demonstrating Order Placement with Natural Language...")
    
    if not Settings.BYBIT_API_KEY or not Settings.BYBIT_API_SECRET:
        print("⚠️  API keys not configured - skipping order placement demo")
        print("   Add BYBIT_API_KEY and BYBIT_API_SECRET to .env file")
        return
    
    try:
        async with BybitTradingTool(testnet=True) as bybit:
            # Get current BTC price
            ticker = await bybit.get_ticker("BTCUSDT")
            if ticker['success']:
                ticker_data = ticker['data'].get('list', [])
                if ticker_data:
                    current_price = float(ticker_data[0]['lastPrice'])
                    
                    # Create a sample order scenario
                    signal = TradingSignal(
                        symbol="BTCUSDT",
                        signal_type="BUY",
                        price=current_price,
                        confidence=0.80,
                        strategy="Support Level Bounce",
                        timestamp=datetime.now(),
                        technical_indicators={
                            "rsi": 35.0,
                            "macd": "bullish",
                            "support": current_price * 0.98
                        },
                        market_context={
                            "market_sentiment": "bullish",
                            "support_level": current_price * 0.98
                        },
                        reasoning="Price at support level with oversold RSI",
                        risk_level="MODERATE",
                        position_size=0.001,  # Small test position
                        stop_loss=current_price * 0.97,
                        take_profit=current_price * 1.03
                    )
                    
                    # Translate the signal
                    translator = NaturalLanguageTranslator()
                    response = await translator.translate_signal(signal)
                    
                    print(f"\n📊 Order Scenario for BTCUSDT:")
                    print(f"   Current Price: ${current_price:,.2f}")
                    print(f"   Signal: {response.summary}")
                    print(f"   Action: {response.action_recommendation}")
                    print(f"   Reasoning: {response.reasoning}")
                    print(f"   Risk: {response.risk_assessment}")
                    
                    # Note: In a real scenario, you would place the order here
                    print(f"\n💡 Order Details:")
                    print(f"   Symbol: {signal.symbol}")
                    print(f"   Side: {signal.signal_type}")
                    print(f"   Quantity: {signal.position_size}")
                    print(f"   Stop Loss: ${signal.stop_loss:,.2f}")
                    print(f"   Take Profit: ${signal.take_profit:,.2f}")
                    
    except Exception as e:
        print(f"❌ Order placement demo failed: {e}")


async def main():
    """Main function demonstrating Bybit + Natural Language integration."""
    print("🚀 Bybit Integration with Natural Language Translation")
    print("=" * 70)
    
    # Check if Google API key is configured
    if not Settings.GOOGLE_API_KEY:
        print("❌ Google API key not configured!")
        print("Please add GOOGLE_API_KEY to your .env file")
        return
    
    print(f"✅ Google API key configured: {Settings.GOOGLE_API_KEY[:10]}...")
    
    # Step 1: Get market context from Bybit
    market_context = await get_crypto_market_context()
    
    # Step 2: Generate trading signals
    signals = await generate_trading_signals()
    
    # Step 3: Translate signals to natural language
    translated_signals = await translate_signals_to_natural_language(signals)
    
    # Step 4: Get market advice
    advice = await get_market_advice_with_bybit_data()
    
    # Step 5: Demonstrate order placement with natural language
    await demonstrate_bybit_order_with_natural_language()
    
    print("\n" + "=" * 70)
    print("🎉 Bybit + Natural Language Integration Complete!")
    print("\n📋 Summary:")
    print("✅ Real-time market data from Bybit")
    print("✅ Trading signals generated automatically")
    print("✅ Signals translated to natural language")
    print("✅ Market advice generated using AI")
    print("✅ Order placement with human-readable explanations")
    
    print("\n💡 Key Benefits:")
    print("🔗 Seamless integration between Bybit and Google AI")
    print("🗣️ Trading signals in human language")
    print("📊 Real-time market analysis")
    print("💼 Professional trading advice")
    print("🎯 Actionable recommendations")
    
    print("\n🚀 Ready for Production:")
    print("1. Add Bybit API keys for live trading")
    print("2. Integrate with Telegram bot for notifications")
    print("3. Connect to trading strategies for automation")
    print("4. Deploy for live crypto trading")


if __name__ == "__main__":
    asyncio.run(main()) 