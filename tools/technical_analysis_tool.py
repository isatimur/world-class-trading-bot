"""
Technical Analysis Tools
Tools for analyzing technical indicators and generating trading signals
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

async def analyze_technical_indicators(symbol: str, analysis_type: str) -> Dict[str, Any]:
    """Analyze technical indicators for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="6mo")
        
        if data.empty:
            return {
                "success": False,
                "error": f"No data available for {symbol}"
            }
        
        # Calculate basic indicators
        trend_indicators = await calculate_trend_indicators(data)
        momentum_indicators = await calculate_momentum_indicators(data)
        volatility_indicators = await calculate_volatility_indicators(data)
        volume_indicators = await calculate_volume_indicators(data)
        
        # Generate trading signals
        signals = await generate_trading_signals(data)
        
        # Calculate support and resistance
        support_resistance = await calculate_support_resistance(data)
        
        # Generate summary
        summary = await generate_analysis_summary(
            trend_indicators, momentum_indicators, 
            volatility_indicators, volume_indicators, signals
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "analysis_type": analysis_type,
            "trend_indicators": trend_indicators,
            "momentum_indicators": momentum_indicators,
            "volatility_indicators": volatility_indicators,
            "volume_indicators": volume_indicators,
            "trading_signals": signals,
            "support_resistance": support_resistance,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to analyze technical indicators for {symbol}: {str(e)}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }

async def calculate_trend_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate trend indicators"""
    try:
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # Current values
        current_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        sma_200 = data['SMA_200'].iloc[-1]
        ema_12 = data['EMA_12'].iloc[-1]
        ema_26 = data['EMA_26'].iloc[-1]
        
        # Trend analysis
        trend_signals = {
            "price_above_sma_20": bool(current_price > sma_20),
            "price_above_sma_50": bool(current_price > sma_50),
            "price_above_sma_200": bool(current_price > sma_200),
            "sma_20_above_sma_50": bool(sma_20 > sma_50),
            "sma_50_above_sma_200": bool(sma_50 > sma_200),
            "ema_12_above_ema_26": bool(ema_12 > ema_26)
        }
        
        # Overall trend
        bullish_signals = sum(trend_signals.values())
        if bullish_signals >= 4:
            overall_trend = "Strong Bullish"
        elif bullish_signals >= 2:
            overall_trend = "Bullish"
        elif bullish_signals >= 1:
            overall_trend = "Weak Bullish"
        else:
            overall_trend = "Bearish"
        
        return {
            "sma_20": float(sma_20) if not pd.isna(sma_20) else None,
            "sma_50": float(sma_50) if not pd.isna(sma_50) else None,
            "sma_200": float(sma_200) if not pd.isna(sma_200) else None,
            "ema_12": float(ema_12) if not pd.isna(ema_12) else None,
            "ema_26": float(ema_26) if not pd.isna(ema_26) else None,
            "trend_signals": trend_signals,
            "overall_trend": overall_trend,
            "bullish_signals": bullish_signals
        }
        
    except Exception as e:
        return {
            "error": f"Failed to calculate trend indicators: {str(e)}"
        }

async def calculate_momentum_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate momentum indicators"""
    try:
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        k_percent = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        d_percent = k_percent.rolling(window=3).mean()
        
        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]
        
        # Momentum signals
        rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
        macd_signal = "Bullish" if current_macd > current_signal else "Bearish"
        stoch_signal = "Overbought" if current_k > 80 else "Oversold" if current_k < 20 else "Neutral"
        
        return {
            "rsi": {
                "value": float(current_rsi),
                "signal": rsi_signal
            },
            "macd": {
                "macd_line": float(current_macd),
                "signal_line": float(current_signal),
                "histogram": float(current_histogram),
                "signal": macd_signal
            },
            "stochastic": {
                "k_percent": float(current_k),
                "d_percent": float(current_d),
                "signal": stoch_signal
            }
        }
        
    except Exception as e:
        return {
            "error": f"Failed to calculate momentum indicators: {str(e)}"
        }

async def calculate_volatility_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate volatility indicators"""
    try:
        # Bollinger Bands
        sma_20 = data['Close'].rolling(window=20).mean()
        std_20 = data['Close'].rolling(window=20).std()
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        
        current_price = data['Close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_sma = sma_20.iloc[-1]
        
        # Bollinger Band position
        bb_position = (current_price - current_lower) / (current_upper - current_lower)
        
        # Average True Range (ATR)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=14).mean()
        current_atr = atr.iloc[-1]
        
        return {
            "bollinger_bands": {
                "upper_band": float(current_upper),
                "middle_band": float(current_sma),
                "lower_band": float(current_lower),
                "position": float(bb_position),
                "signal": "Overbought" if bb_position > 0.8 else "Oversold" if bb_position < 0.2 else "Neutral"
            },
            "atr": {
                "value": float(current_atr),
                "volatility_level": "High" if current_atr > current_price * 0.03 else "Medium" if current_atr > current_price * 0.015 else "Low"
            }
        }
        
    except Exception as e:
        return {
            "error": f"Failed to calculate volatility indicators: {str(e)}"
        }

async def calculate_volume_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate volume indicators"""
    try:
        # Volume SMA
        volume_sma = data['Volume'].rolling(window=20).mean()
        current_volume = data['Volume'].iloc[-1]
        current_volume_sma = volume_sma.iloc[-1]
        
        # Volume ratio
        volume_ratio = current_volume / current_volume_sma if current_volume_sma > 0 else 1
        
        # On-Balance Volume (OBV)
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        current_obv = obv.iloc[-1]
        
        # Volume signal
        volume_signal = "High" if volume_ratio > 1.5 else "Normal" if volume_ratio > 0.5 else "Low"
        
        return {
            "volume_sma": float(current_volume_sma),
            "current_volume": int(current_volume),
            "volume_ratio": float(volume_ratio),
            "volume_signal": volume_signal,
            "obv": float(current_obv)
        }
        
    except Exception as e:
        return {
            "error": f"Failed to calculate volume indicators: {str(e)}"
        }

async def generate_trading_signals(data: pd.DataFrame) -> Dict[str, Any]:
    """Generate trading signals based on technical analysis"""
    try:
        signals = []
        
        # Price action signals
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2]
        
        if current_price > previous_price:
            signals.append({"type": "price_action", "signal": "bullish", "strength": "medium"})
        else:
            signals.append({"type": "price_action", "signal": "bearish", "strength": "medium"})
        
        # Moving average signals
        sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
        
        if current_price > sma_20 and sma_20 > sma_50:
            signals.append({"type": "moving_average", "signal": "bullish", "strength": "strong"})
        elif current_price < sma_20 and sma_20 < sma_50:
            signals.append({"type": "moving_average", "signal": "bearish", "strength": "strong"})
        
        # RSI signals
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < 30:
            signals.append({"type": "rsi", "signal": "bullish", "strength": "strong"})
        elif current_rsi > 70:
            signals.append({"type": "rsi", "signal": "bearish", "strength": "strong"})
        
        # MACD signals
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            signals.append({"type": "macd", "signal": "bullish", "strength": "medium"})
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            signals.append({"type": "macd", "signal": "bearish", "strength": "medium"})
        
        # Overall signal
        bullish_count = sum(1 for s in signals if s['signal'] == 'bullish')
        bearish_count = sum(1 for s in signals if s['signal'] == 'bearish')
        
        if bullish_count > bearish_count:
            overall_signal = "bullish"
            confidence = min(bullish_count / len(signals), 1.0) if signals else 0.5
        elif bearish_count > bullish_count:
            overall_signal = "bearish"
            confidence = min(bearish_count / len(signals), 1.0) if signals else 0.5
        else:
            overall_signal = "neutral"
            confidence = 0.5
        
        return {
            "signals": signals,
            "overall_signal": overall_signal,
            "confidence": float(confidence),
            "signal_count": len(signals)
        }
        
    except Exception as e:
        return {
            "error": f"Failed to generate trading signals: {str(e)}"
        }

async def calculate_support_resistance(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate support and resistance levels"""
    try:
        # Simple support and resistance using recent highs and lows
        recent_highs = data['High'].tail(20).nlargest(3)
        recent_lows = data['Low'].tail(20).nsmallest(3)
        
        current_price = data['Close'].iloc[-1]
        
        # Find nearest support and resistance
        resistance_levels = sorted(recent_highs.values, reverse=True)
        support_levels = sorted(recent_lows.values)
        
        nearest_resistance = next((r for r in resistance_levels if r > current_price), None)
        nearest_support = next((s for s in support_levels if s < current_price), None)
        
        return {
            "nearest_resistance": float(nearest_resistance) if nearest_resistance else None,
            "nearest_support": float(nearest_support) if nearest_support else None,
            "resistance_levels": [float(r) for r in resistance_levels],
            "support_levels": [float(s) for s in support_levels],
            "current_price": float(current_price)
        }
        
    except Exception as e:
        return {
            "error": f"Failed to calculate support and resistance: {str(e)}"
        }

async def generate_analysis_summary(trend_indicators: Dict, momentum_indicators: Dict, 
                                  volatility_indicators: Dict, volume_indicators: Dict, 
                                  signals: Dict) -> Dict[str, Any]:
    """Generate a summary of the technical analysis"""
    try:
        summary = {
            "trend": trend_indicators.get("overall_trend", "Unknown"),
            "momentum": "Mixed",
            "volatility": "Medium",
            "volume": volume_indicators.get("volume_signal", "Normal"),
            "overall_signal": signals.get("overall_signal", "neutral"),
            "confidence": signals.get("confidence", 0.5),
            "key_points": []
        }
        
        # Add key points based on indicators
        if "trend_indicators" in trend_indicators:
            bullish_signals = trend_indicators.get("bullish_signals", 0)
            if bullish_signals >= 4:
                summary["key_points"].append("Strong bullish trend with multiple moving average confirmations")
            elif bullish_signals >= 2:
                summary["key_points"].append("Moderate bullish trend")
            else:
                summary["key_points"].append("Bearish or neutral trend")
        
        if "rsi" in momentum_indicators:
            rsi_signal = momentum_indicators["rsi"]["signal"]
            if rsi_signal == "Overbought":
                summary["key_points"].append("RSI indicates overbought conditions")
            elif rsi_signal == "Oversold":
                summary["key_points"].append("RSI indicates oversold conditions")
        
        if "bollinger_bands" in volatility_indicators:
            bb_signal = volatility_indicators["bollinger_bands"]["signal"]
            if bb_signal == "Overbought":
                summary["key_points"].append("Price near upper Bollinger Band")
            elif bb_signal == "Oversold":
                summary["key_points"].append("Price near lower Bollinger Band")
        
        if volume_indicators.get("volume_signal") == "High":
            summary["key_points"].append("High volume supporting price action")
        
        return summary
        
    except Exception as e:
        return {
            "error": f"Failed to generate analysis summary: {str(e)}"
        }

async def detect_double_top(data: pd.DataFrame) -> Dict[str, Any]:
    """Detect double top pattern"""
    try:
        # Simple double top detection
        highs = data['High'].tail(50)
        peaks = []
        
        for i in range(1, len(highs) - 1):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                peaks.append((i, highs.iloc[i]))
        
        if len(peaks) >= 2:
            # Check if last two peaks are similar in height
            last_two_peaks = peaks[-2:]
            height_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1])
            avg_height = (last_two_peaks[0][1] + last_two_peaks[1][1]) / 2
            
            if height_diff / avg_height < 0.02:  # Within 2%
                return {
                    "detected": True,
                    "pattern": "double_top",
                    "confidence": "medium",
                    "resistance_level": float(avg_height)
                }
        
        return {
            "detected": False,
            "pattern": "double_top"
        }
        
    except Exception as e:
        return {
            "error": f"Failed to detect double top: {str(e)}"
        }

async def detect_double_bottom(data: pd.DataFrame) -> Dict[str, Any]:
    """Detect double bottom pattern"""
    try:
        # Simple double bottom detection
        lows = data['Low'].tail(50)
        troughs = []
        
        for i in range(1, len(lows) - 1):
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                troughs.append((i, lows.iloc[i]))
        
        if len(troughs) >= 2:
            # Check if last two troughs are similar in height
            last_two_troughs = troughs[-2:]
            height_diff = abs(last_two_troughs[0][1] - last_two_troughs[1][1])
            avg_height = (last_two_troughs[0][1] + last_two_troughs[1][1]) / 2
            
            if height_diff / avg_height < 0.02:  # Within 2%
                return {
                    "detected": True,
                    "pattern": "double_bottom",
                    "confidence": "medium",
                    "support_level": float(avg_height)
                }
        
        return {
            "detected": False,
            "pattern": "double_bottom"
        }
        
    except Exception as e:
        return {
            "error": f"Failed to detect double bottom: {str(e)}"
        }

async def detect_head_shoulders(data: pd.DataFrame) -> Dict[str, Any]:
    """Detect head and shoulders pattern"""
    try:
        # Simplified head and shoulders detection
        highs = data['High'].tail(100)
        peaks = []
        
        for i in range(2, len(highs) - 2):
            if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and
                highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]):
                peaks.append((i, highs.iloc[i]))
        
        if len(peaks) >= 3:
            # Check for head and shoulders pattern
            last_three_peaks = peaks[-3:]
            left_shoulder = last_three_peaks[0][1]
            head = last_three_peaks[1][1]
            right_shoulder = last_three_peaks[2][1]
            
            # Head should be higher than shoulders
            if head > left_shoulder and head > right_shoulder:
                # Shoulders should be roughly equal
                shoulder_diff = abs(left_shoulder - right_shoulder)
                avg_shoulder = (left_shoulder + right_shoulder) / 2
                
                if shoulder_diff / avg_shoulder < 0.05:  # Within 5%
                    return {
                        "detected": True,
                        "pattern": "head_and_shoulders",
                        "confidence": "medium",
                        "neckline_level": float(avg_shoulder)
                    }
        
        return {
            "detected": False,
            "pattern": "head_and_shoulders"
        }
        
    except Exception as e:
        return {
            "error": f"Failed to detect head and shoulders: {str(e)}"
        }

async def detect_inverse_head_shoulders(data: pd.DataFrame) -> Dict[str, Any]:
    """Detect inverse head and shoulders pattern"""
    try:
        # Simplified inverse head and shoulders detection
        lows = data['Low'].tail(100)
        troughs = []
        
        for i in range(2, len(lows) - 2):
            if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and
                lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]):
                troughs.append((i, lows.iloc[i]))
        
        if len(troughs) >= 3:
            # Check for inverse head and shoulders pattern
            last_three_troughs = troughs[-3:]
            left_shoulder = last_three_troughs[0][1]
            head = last_three_troughs[1][1]
            right_shoulder = last_three_troughs[2][1]
            
            # Head should be lower than shoulders
            if head < left_shoulder and head < right_shoulder:
                # Shoulders should be roughly equal
                shoulder_diff = abs(left_shoulder - right_shoulder)
                avg_shoulder = (left_shoulder + right_shoulder) / 2
                
                if shoulder_diff / avg_shoulder < 0.05:  # Within 5%
                    return {
                        "detected": True,
                        "pattern": "inverse_head_and_shoulders",
                        "confidence": "medium",
                        "neckline_level": float(avg_shoulder)
                    }
        
        return {
            "detected": False,
            "pattern": "inverse_head_and_shoulders"
        }
        
    except Exception as e:
        return {
            "error": f"Failed to detect inverse head and shoulders: {str(e)}"
        }