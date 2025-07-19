"""
Market Data Tools
Tools for retrieving and analyzing market data
"""
import asyncio
import yfinance as yf
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

async def get_stock_data(symbol: str, data_type: str, period: str) -> Dict[str, Any]:
    """Get stock data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        
        if data_type == "price":
            data = ticker.history(period=period)
            if data.empty:
                return {
                    "success": False,
                    "error": f"No data available for {symbol}"
                }
            
            # Get current price
            current_price = data['Close'].iloc[-1]
            previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100
            
            return {
                "success": True,
                "symbol": symbol,
                "current_price": float(current_price),
                "previous_price": float(previous_price),
                "price_change": float(price_change),
                "price_change_pct": float(price_change_pct),
                "volume": int(data['Volume'].iloc[-1]),
                "high": float(data['High'].iloc[-1]),
                "low": float(data['Low'].iloc[-1]),
                "open": float(data['Open'].iloc[-1]),
                "timestamp": datetime.now().isoformat()
            }
        
        elif data_type == "info":
            info = ticker.info
            return {
                "success": True,
                "symbol": symbol,
                "info": info,
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            return {
                "success": False,
                "error": f"Unknown data type: {data_type}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get data for {symbol}: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

async def get_market_overview() -> Dict[str, Any]:
    """Get market overview for major indices"""
    try:
        indices = ["^GSPC", "^DJI", "^IXIC", "^RUT"]  # S&P 500, Dow Jones, NASDAQ, Russell 2000
        overview = {}
        
        for index in indices:
            try:
                data = await get_stock_data(index, "price", "1d")
                if data['success']:
                    overview[index] = data
            except Exception as e:
                overview[index] = {
                    "success": False,
                    "error": str(e)
                }
        
        return {
            "success": True,
            "overview": overview,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get market overview: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

async def get_sector_performance() -> Dict[str, Any]:
    """Get sector performance data"""
    try:
        # Major sector ETFs
        sectors = {
            "XLK": "Technology",
            "XLF": "Financials", 
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLI": "Industrials",
            "XLP": "Consumer Staples",
            "XLY": "Consumer Discretionary",
            "XLU": "Utilities",
            "XLB": "Materials",
            "XLRE": "Real Estate"
        }
        
        performance = {}
        
        for etf, sector_name in sectors.items():
            try:
                data = await get_stock_data(etf, "price", "5d")
                if data['success']:
                    performance[sector_name] = {
                        "etf": etf,
                        "price": data['current_price'],
                        "change_pct": data['price_change_pct']
                    }
            except Exception as e:
                performance[sector_name] = {
                    "etf": etf,
                    "error": str(e)
                }
        
        return {
            "success": True,
            "sectors": performance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get sector performance: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

async def calculate_rsi(symbol: str, period: int = 14) -> Dict[str, Any]:
    """Calculate RSI for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{period + 20}d")  # Get extra data for calculation
        
        if data.empty:
            return {
                "success": False,
                "error": f"No data available for {symbol}"
            }
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        # Determine RSI signal
        if current_rsi > 70:
            signal = "Overbought"
        elif current_rsi < 30:
            signal = "Oversold"
        else:
            signal = "Neutral"
        
        return {
            "success": True,
            "symbol": symbol,
            "rsi": float(current_rsi),
            "signal": signal,
            "period": period,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to calculate RSI for {symbol}: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

async def calculate_macd(symbol: str, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
    """Calculate MACD for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{slow + 20}d")  # Get extra data for calculation
        
        if data.empty:
            return {
                "success": False,
                "error": f"No data available for {symbol}"
            }
        
        # Calculate MACD
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # Determine MACD signal
        if current_macd > current_signal and current_histogram > 0:
            signal_type = "Bullish"
        elif current_macd < current_signal and current_histogram < 0:
            signal_type = "Bearish"
        else:
            signal_type = "Neutral"
        
        return {
            "success": True,
            "symbol": symbol,
            "macd_line": float(current_macd),
            "signal_line": float(current_signal),
            "histogram": float(current_histogram),
            "signal": signal_type,
            "fast_period": fast,
            "slow_period": slow,
            "signal_period": signal,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to calculate MACD for {symbol}: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        } 