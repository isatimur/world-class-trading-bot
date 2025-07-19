"""
Trading Agent Tools Package
Comprehensive tools for market data, technical analysis, portfolio analysis, and cryptocurrency trading
"""

# Market Data Tools
from .market_data_tool import (
    get_stock_data,
    get_market_overview,
    get_sector_performance,
    calculate_rsi,
    calculate_macd
)

# Technical Analysis Tools
from .technical_analysis_tool import (
    analyze_technical_indicators,
    calculate_trend_indicators,
    calculate_momentum_indicators,
    calculate_volatility_indicators,
    calculate_volume_indicators,
    generate_trading_signals,
    calculate_support_resistance,
    generate_analysis_summary,
    detect_double_top,
    detect_double_bottom,
    detect_head_shoulders,
    detect_inverse_head_shoulders
)

# Portfolio Analysis Tools
from .portfolio_analysis_tool import (
    analyze_portfolio,
    calculate_performance_metrics,
    calculate_risk_metrics,
    calculate_optimization_suggestions,
    calculate_additional_metrics,
    calculate_portfolio_risk_score,
    get_risk_level
)

# Bybit Trading Tools
from .bybit_trading_tool import (
    BybitTradingTool,
    get_bybit_market_data,
    get_bybit_account_info,
    get_bybit_positions,
    place_bybit_order,
    get_bybit_portfolio_summary
)

# Natural Language Translation Tools
from .natural_language_translator import (
    NaturalLanguageTranslator,
    TradingSignal,
    NaturalLanguageResponse,
    translate_trading_signal,
    get_market_advice,
    translate_portfolio_summary
)

__all__ = [
    # Market Data
    'get_stock_data',
    'get_market_overview',
    'get_sector_performance',
    'calculate_rsi',
    'calculate_macd',
    
    # Technical Analysis
    'analyze_technical_indicators',
    'calculate_trend_indicators',
    'calculate_momentum_indicators',
    'calculate_volatility_indicators',
    'calculate_volume_indicators',
    'generate_trading_signals',
    'calculate_support_resistance',
    'generate_analysis_summary',
    'detect_double_top',
    'detect_double_bottom',
    'detect_head_shoulders',
    'detect_inverse_head_shoulders',
    
    # Portfolio Analysis
    'analyze_portfolio',
    'calculate_performance_metrics',
    'calculate_risk_metrics',
    'calculate_optimization_suggestions',
    'calculate_additional_metrics',
    'calculate_portfolio_risk_score',
    'get_risk_level',
    
    # Bybit Trading
    'BybitTradingTool',
    'get_bybit_market_data',
    'get_bybit_account_info',
    'get_bybit_positions',
    'place_bybit_order',
    'get_bybit_portfolio_summary',
    
    # Natural Language Translation
    'NaturalLanguageTranslator',
    'TradingSignal',
    'NaturalLanguageResponse',
    'translate_trading_signal',
    'get_market_advice',
    'translate_portfolio_summary'
] 