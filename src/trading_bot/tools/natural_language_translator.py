"""
Natural Language Translator for Trading Signals

This module uses Google Agent SDK to translate trading signals and technical analysis
into natural human language, making trading decisions more understandable.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ..config.settings import Settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TradingSignal:
    """Trading signal with all relevant information"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    price: float
    confidence: float
    strategy: str
    timestamp: datetime
    technical_indicators: Dict[str, Any]
    market_context: Dict[str, Any]
    reasoning: str
    risk_level: str  # LOW, MODERATE, HIGH
    position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class NaturalLanguageResponse:
    """Natural language response from the translator"""
    summary: str
    action_recommendation: str
    reasoning: str
    risk_assessment: str
    market_context: str
    confidence_explanation: str
    next_steps: str


class NaturalLanguageTranslator:
    """
    Translates trading signals and technical analysis into natural human language
    using Google Agent SDK.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the natural language translator."""
        self.api_key = api_key or Settings.GOOGLE_API_KEY
        
        if not self.api_key:
            raise ValueError("Google API key is required for natural language translation")
        
        # Configure Google Generative AI
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=Settings.MARKET_ANALYST_MODEL or "gemini-2.0-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                max_output_tokens=8192,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        logger.info("Natural Language Translator initialized")
    
    async def translate_signal(self, signal: TradingSignal) -> NaturalLanguageResponse:
        """
        Translate a trading signal into natural human language.
        
        Args:
            signal: Trading signal to translate
            
        Returns:
            Natural language response
        """
        try:
            # Create the prompt for signal translation
            prompt = self._create_signal_prompt(signal)
            
            # Generate response
            response = await self._generate_response(prompt)
            
            # Parse the response
            return self._parse_signal_response(response, signal)
            
        except Exception as e:
            logger.error(f"Error translating signal: {e}")
            return self._create_fallback_response(signal)
    
    async def translate_market_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Translate market analysis into natural language.
        
        Args:
            analysis: Market analysis data
            
        Returns:
            Natural language market analysis
        """
        try:
            prompt = self._create_market_analysis_prompt(analysis)
            response = await self._generate_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error translating market analysis: {e}")
            return "Unable to translate market analysis at this time."
    
    async def translate_portfolio_status(self, portfolio: Dict[str, Any]) -> str:
        """
        Translate portfolio status into natural language.
        
        Args:
            portfolio: Portfolio data
            
        Returns:
            Natural language portfolio status
        """
        try:
            prompt = self._create_portfolio_prompt(portfolio)
            response = await self._generate_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error translating portfolio status: {e}")
            return "Unable to translate portfolio status at this time."
    
    async def generate_trading_advice(self, context: Dict[str, Any]) -> str:
        """
        Generate trading advice based on current market context.
        
        Args:
            context: Market and trading context
            
        Returns:
            Natural language trading advice
        """
        try:
            prompt = self._create_advice_prompt(context)
            response = await self._generate_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating trading advice: {e}")
            return "Unable to generate trading advice at this time."
    
    def _create_signal_prompt(self, signal: TradingSignal) -> str:
        """Create a prompt for signal translation."""
        return f"""
You are an expert trading analyst. Translate this trading signal into clear, actionable natural language that a human trader can understand.

Trading Signal:
- Symbol: {signal.symbol}
- Signal Type: {signal.signal_type}
- Price: ${signal.price:,.2f}
- Confidence: {signal.confidence:.1%}
- Strategy: {signal.strategy}
- Risk Level: {signal.risk_level}
- Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Technical Indicators:
{json.dumps(signal.technical_indicators, indent=2)}

Market Context:
{json.dumps(signal.market_context, indent=2)}

Reasoning: {signal.reasoning}

Position Details:
- Position Size: {signal.position_size or 'Not specified'}
- Stop Loss: {f'${signal.stop_loss:,.2f}' if signal.stop_loss else 'Not specified'}
- Take Profit: {f'${signal.take_profit:,.2f}' if signal.take_profit else 'Not specified'}

Please provide a natural language response that includes:
1. A clear summary of what this signal means
2. A specific action recommendation in plain English
3. The reasoning behind the signal in simple terms
4. Risk assessment and considerations
5. Market context explanation
6. Confidence level explanation
7. Next steps or what to watch for

Make it conversational and easy to understand for a human trader.
"""
    
    def _create_market_analysis_prompt(self, analysis: Dict[str, Any]) -> str:
        """Create a prompt for market analysis translation."""
        return f"""
You are an expert market analyst. Translate this market analysis into clear, understandable natural language.

Market Analysis Data:
{json.dumps(analysis, indent=2)}

Please provide a natural language summary that includes:
1. Overall market sentiment
2. Key trends and patterns
3. Important support/resistance levels
4. Risk factors to watch
5. Opportunities in the market

Make it conversational and actionable for traders.
"""
    
    def _create_portfolio_prompt(self, portfolio: Dict[str, Any]) -> str:
        """Create a prompt for portfolio status translation."""
        return f"""
You are an expert portfolio manager. Translate this portfolio status into clear, understandable natural language.

Portfolio Data:
{json.dumps(portfolio, indent=2)}

Please provide a natural language summary that includes:
1. Overall portfolio performance
2. Key positions and their status
3. Risk assessment
4. Recommendations for rebalancing
5. Areas of concern or opportunity

Make it conversational and actionable for portfolio management.
"""
    
    def _create_advice_prompt(self, context: Dict[str, Any]) -> str:
        """Create a prompt for trading advice generation."""
        return f"""
You are an expert trading advisor. Based on the current market context, provide actionable trading advice.

Market Context:
{json.dumps(context, indent=2)}

Please provide natural language trading advice that includes:
1. Current market conditions assessment
2. Specific trading opportunities
3. Risk management recommendations
4. Position sizing suggestions
5. What to watch for in the coming hours/days

Make it conversational, actionable, and easy to understand.
"""
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using Google Generative AI."""
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            raise
    
    def _parse_signal_response(self, response: str, signal: TradingSignal) -> NaturalLanguageResponse:
        """Parse the AI response into structured format."""
        try:
            # For now, return a simple structured response
            # In a more advanced implementation, you could use structured output
            return NaturalLanguageResponse(
                summary=f"Signal for {signal.symbol}: {signal.signal_type} at ${signal.price:,.2f}",
                action_recommendation=f"Consider {signal.signal_type.lower()}ing {signal.symbol} at current market price",
                reasoning=signal.reasoning,
                risk_assessment=f"Risk level: {signal.risk_level}",
                market_context="Market conditions support this signal",
                confidence_explanation=f"Confidence: {signal.confidence:.1%}",
                next_steps="Monitor the position and adjust stop loss as needed"
            )
        except Exception as e:
            logger.error(f"Error parsing signal response: {e}")
            return self._create_fallback_response(signal)
    
    def _create_fallback_response(self, signal: TradingSignal) -> NaturalLanguageResponse:
        """Create a fallback response when AI translation fails."""
        return NaturalLanguageResponse(
            summary=f"Trading signal for {signal.symbol}",
            action_recommendation=f"Signal: {signal.signal_type} {signal.symbol}",
            reasoning=signal.reasoning,
            risk_assessment=f"Risk: {signal.risk_level}",
            market_context="Market analysis available",
            confidence_explanation=f"Confidence: {signal.confidence:.1%}",
            next_steps="Review signal details and execute accordingly"
        )


# Convenience functions for easy integration
async def translate_trading_signal(
    symbol: str,
    signal_type: str,
    price: float,
    confidence: float,
    strategy: str,
    technical_indicators: Dict[str, Any],
    market_context: Dict[str, Any],
    reasoning: str,
    risk_level: str = "MODERATE"
) -> NaturalLanguageResponse:
    """Translate a trading signal into natural language."""
    translator = NaturalLanguageTranslator()
    
    signal = TradingSignal(
        symbol=symbol,
        signal_type=signal_type,
        price=price,
        confidence=confidence,
        strategy=strategy,
        timestamp=datetime.now(),
        technical_indicators=technical_indicators,
        market_context=market_context,
        reasoning=reasoning,
        risk_level=risk_level
    )
    
    return await translator.translate_signal(signal)


async def get_market_advice(market_data: Dict[str, Any]) -> str:
    """Get natural language trading advice based on market data."""
    translator = NaturalLanguageTranslator()
    return await translator.generate_trading_advice(market_data)


async def translate_portfolio_summary(portfolio_data: Dict[str, Any]) -> str:
    """Translate portfolio data into natural language summary."""
    translator = NaturalLanguageTranslator()
    return await translator.translate_portfolio_status(portfolio_data) 