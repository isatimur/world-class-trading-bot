"""
Portfolio Analysis Tools
Tools for analyzing portfolio performance and risk metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

async def analyze_portfolio(portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze portfolio performance and metrics"""
    try:
        if not portfolio:
            return {
                "success": False,
                "error": "Portfolio is empty"
            }
        
        # Calculate basic metrics
        performance_metrics = await calculate_performance_metrics(portfolio)
        risk_metrics = await calculate_risk_metrics(portfolio)
        optimization_suggestions = await calculate_optimization_suggestions(portfolio)
        additional_metrics = await calculate_additional_metrics(portfolio)
        
        return {
            "success": True,
            "portfolio_size": len(portfolio),
            "performance_metrics": performance_metrics,
            "risk_metrics": risk_metrics,
            "optimization_suggestions": optimization_suggestions,
            "additional_metrics": additional_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to analyze portfolio: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

async def calculate_performance_metrics(portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate portfolio performance metrics"""
    try:
        total_value = sum(position.get('value', 0) for position in portfolio)
        total_cost = sum(position.get('cost', position.get('value', 0)) for position in portfolio)
        
        # Calculate returns
        total_return = total_value - total_cost
        total_return_pct = (total_return / total_cost * 100) if total_cost > 0 else 0
        
        # Calculate weighted average return
        weighted_returns = []
        for position in portfolio:
            value = position.get('value', 0)
            cost = position.get('cost', value)
            if cost > 0:
                position_return = (value - cost) / cost
                weight = value / total_value if total_value > 0 else 0
                weighted_returns.append(position_return * weight)
        
        weighted_avg_return = sum(weighted_returns) * 100 if weighted_returns else 0
        
        # Top performers
        performers = []
        for position in portfolio:
            value = position.get('value', 0)
            cost = position.get('cost', value)
            if cost > 0:
                return_pct = (value - cost) / cost * 100
                performers.append({
                    "symbol": position.get('symbol', 'Unknown'),
                    "return_pct": return_pct,
                    "value": value
                })
        
        performers.sort(key=lambda x: x['return_pct'], reverse=True)
        top_performers = performers[:3]
        worst_performers = performers[-3:] if len(performers) >= 3 else performers
        
        return {
            "total_value": float(total_value),
            "total_cost": float(total_cost),
            "total_return": float(total_return),
            "total_return_pct": float(total_return_pct),
            "weighted_avg_return": float(weighted_avg_return),
            "top_performers": top_performers,
            "worst_performers": worst_performers
        }
        
    except Exception as e:
        return {
            "error": f"Failed to calculate performance metrics: {str(e)}"
        }

async def calculate_risk_metrics(portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate portfolio risk metrics"""
    try:
        total_value = sum(position.get('value', 0) for position in portfolio)
        
        # Calculate concentration risk
        concentration_risk = {}
        for position in portfolio:
            symbol = position.get('symbol', 'Unknown')
            value = position.get('value', 0)
            weight = value / total_value if total_value > 0 else 0
            concentration_risk[symbol] = float(weight)
        
        # Top concentrated positions
        sorted_concentration = sorted(concentration_risk.items(), key=lambda x: x[1], reverse=True)
        top_concentrated = sorted_concentration[:5]
        
        # Calculate diversification score
        weights = list(concentration_risk.values())
        if weights:
            # Herfindahl-Hirschman Index (HHI) for concentration
            hhi = sum(w**2 for w in weights)
            # Convert to diversification score (0-100, higher is better)
            diversification_score = max(0, 100 - (hhi * 100))
        else:
            diversification_score = 0
        
        # Risk assessment
        if diversification_score >= 80:
            risk_level = "Low"
        elif diversification_score >= 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            "concentration_risk": concentration_risk,
            "top_concentrated": top_concentrated,
            "diversification_score": float(diversification_score),
            "risk_level": risk_level,
            "herfindahl_index": float(hhi) if 'hhi' in locals() else 0
        }
        
    except Exception as e:
        return {
            "error": f"Failed to calculate risk metrics: {str(e)}"
        }

async def calculate_optimization_suggestions(portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate portfolio optimization suggestions"""
    try:
        suggestions = []
        
        # Check for over-concentration
        total_value = sum(position.get('value', 0) for position in portfolio)
        for position in portfolio:
            symbol = position.get('symbol', 'Unknown')
            value = position.get('value', 0)
            weight = value / total_value if total_value > 0 else 0
            
            if weight > 0.2:  # More than 20% in single position
                suggestions.append({
                    "type": "concentration",
                    "symbol": symbol,
                    "weight": float(weight),
                    "suggestion": f"Consider reducing position in {symbol} (currently {weight:.1%})"
                })
        
        # Check for under-diversification
        if len(portfolio) < 5:
            suggestions.append({
                "type": "diversification",
                "suggestion": f"Consider adding more positions (currently {len(portfolio)} positions)"
            })
        
        # Check for sector concentration (if sector data available)
        sectors = {}
        for position in portfolio:
            sector = position.get('sector', 'Unknown')
            value = position.get('value', 0)
            sectors[sector] = sectors.get(sector, 0) + value
        
        for sector, value in sectors.items():
            sector_weight = value / total_value if total_value > 0 else 0
            if sector_weight > 0.4:  # More than 40% in single sector
                suggestions.append({
                    "type": "sector_concentration",
                    "sector": sector,
                    "weight": float(sector_weight),
                    "suggestion": f"Consider diversifying away from {sector} sector (currently {sector_weight:.1%})"
                })
        
        return {
            "suggestions": suggestions,
            "suggestion_count": len(suggestions)
        }
        
    except Exception as e:
        return {
            "error": f"Failed to calculate optimization suggestions: {str(e)}"
        }

async def calculate_additional_metrics(portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate additional portfolio metrics"""
    try:
        total_value = sum(position.get('value', 0) for position in portfolio)
        
        # Sector breakdown
        sectors = {}
        for position in portfolio:
            sector = position.get('sector', 'Unknown')
            value = position.get('value', 0)
            sectors[sector] = sectors.get(sector, 0) + value
        
        sector_breakdown = {
            sector: {
                "value": float(value),
                "weight": float(value / total_value) if total_value > 0 else 0
            }
            for sector, value in sectors.items()
        }
        
        # Market cap breakdown (if available)
        market_caps = {"large": 0, "mid": 0, "small": 0, "unknown": 0}
        for position in portfolio:
            market_cap = position.get('market_cap', 'unknown')
            value = position.get('value', 0)
            
            if market_cap == 'large':
                market_caps['large'] += value
            elif market_cap == 'mid':
                market_caps['mid'] += value
            elif market_cap == 'small':
                market_caps['small'] += value
            else:
                market_caps['unknown'] += value
        
        market_cap_breakdown = {
            cap: {
                "value": float(value),
                "weight": float(value / total_value) if total_value > 0 else 0
            }
            for cap, value in market_caps.items()
        }
        
        # Liquidity analysis
        liquid_positions = sum(1 for position in portfolio if position.get('liquid', True))
        illiquid_positions = len(portfolio) - liquid_positions
        
        return {
            "sector_breakdown": sector_breakdown,
            "market_cap_breakdown": market_cap_breakdown,
            "liquidity": {
                "liquid_positions": liquid_positions,
                "illiquid_positions": illiquid_positions,
                "liquidity_ratio": float(liquid_positions / len(portfolio)) if portfolio else 0
            }
        }
        
    except Exception as e:
        return {
            "error": f"Failed to calculate additional metrics: {str(e)}"
        }

async def calculate_portfolio_risk_score(portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive portfolio risk score"""
    try:
        if not portfolio:
            return {
                "success": False,
                "error": "Portfolio is empty"
            }
        
        total_value = sum(position.get('value', 0) for position in portfolio)
        
        # Calculate various risk factors
        risk_factors = {}
        
        # Concentration risk
        max_position_weight = max(
            position.get('value', 0) / total_value 
            for position in portfolio
        ) if total_value > 0 else 0
        risk_factors['concentration'] = min(max_position_weight * 5, 1.0)  # Scale 0-1
        
        # Diversification risk
        diversification_score = len(portfolio) / 20  # Normalize to 0-1, 20+ positions = 1.0
        risk_factors['diversification'] = 1.0 - diversification_score
        
        # Volatility risk (simplified)
        volatility_scores = []
        for position in portfolio:
            volatility = position.get('volatility', 0.2)  # Default 20%
            volatility_scores.append(volatility)
        
        avg_volatility = sum(volatility_scores) / len(volatility_scores) if volatility_scores else 0.2
        risk_factors['volatility'] = min(avg_volatility, 1.0)
        
        # Liquidity risk
        liquid_positions = sum(1 for position in portfolio if position.get('liquid', True))
        liquidity_ratio = liquid_positions / len(portfolio) if portfolio else 0
        risk_factors['liquidity'] = 1.0 - liquidity_ratio
        
        # Calculate overall risk score
        risk_weights = {
            'concentration': 0.3,
            'diversification': 0.25,
            'volatility': 0.25,
            'liquidity': 0.2
        }
        
        overall_risk_score = sum(
            risk_factors[factor] * risk_weights[factor]
            for factor in risk_weights
        )
        
        # Determine risk level
        risk_level = get_risk_level(overall_risk_score)
        
        return {
            "success": True,
            "overall_risk_score": float(overall_risk_score),
            "risk_level": risk_level,
            "risk_factors": {
                factor: {
                    "score": float(score),
                    "weight": float(risk_weights[factor])
                }
                for factor, score in risk_factors.items()
            },
            "risk_weights": risk_weights,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to calculate portfolio risk score: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def get_risk_level(risk_score: float) -> str:
    """Get risk level based on risk score"""
    if risk_score < 0.3:
        return "Low"
    elif risk_score < 0.6:
        return "Medium"
    else:
        return "High" 