"""
Base Pine Script generator for TradingView integration.

This module provides the foundation for generating Pine Scripts
that can be used in TradingView for backtesting and visualization.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PineScriptConfig:
    """Configuration for Pine Script generation"""
    strategy_name: str
    symbol: str
    timeframe: str = "1D"
    version: str = "5"
    overlay: bool = False
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class BasePineGenerator(ABC):
    """
    Base class for Pine Script generators.
    
    This class provides the foundation for generating Pine Scripts
    that can be used in TradingView for backtesting and visualization.
    """
    
    def __init__(self, config: PineScriptConfig):
        """
        Initialize Pine Script generator.
        
        Args:
            config: Pine Script configuration
        """
        self.config = config
        self.script_parts = []
        
    @abstractmethod
    def generate_script(self) -> str:
        """
        Generate complete Pine Script.
        
        Returns:
            Complete Pine Script as string
        """
        pass
    
    def add_header(self):
        """Add Pine Script header."""
        header = f'''//@version={self.config.version}
// {self.config.strategy_name} Strategy
// Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
// Symbol: {self.config.symbol}
// Timeframe: {self.config.timeframe}

strategy("{self.config.strategy_name}", overlay={str(self.config.overlay).lower()}, 
         default_qty_type=strategy.percent_of_equity, default_qty_value=10,
         commission_type=strategy.commission.percent, commission_value=0.1)'''
        
        self.script_parts.append(header)
    
    def add_parameters(self):
        """Add strategy parameters."""
        if not self.config.parameters:
            return
        
        params = []
        for param_name, param_value in self.config.parameters.items():
            if isinstance(param_value, bool):
                params.append(f'{param_name} = input.bool({str(param_value).lower()}, "{param_name}")')
            elif isinstance(param_value, int):
                params.append(f'{param_name} = input.int({param_value}, "{param_name}")')
            elif isinstance(param_value, float):
                params.append(f'{param_name} = input.float({param_value}, "{param_name}")')
            elif isinstance(param_value, str):
                params.append(f'{param_name} = input.string("{param_value}", "{param_name}")')
            else:
                params.append(f'{param_name} = input.float({param_value}, "{param_name}")')
        
        if params:
            self.script_parts.append('\n// Strategy Parameters')
            self.script_parts.extend(params)
    
    def add_indicators(self):
        """Add technical indicators."""
        indicators = '''
// Technical Indicators
rsi = ta.rsi(close, 14)
macd_line = ta.ema(close, 12) - ta.ema(close, 26)
macd_signal = ta.ema(macd_line, 9)
macd_histogram = macd_line - macd_signal

[bb_upper, bb_middle, bb_lower] = ta.bb(close, 20, 2)
bb_position = (close - bb_lower) / (bb_upper - bb_lower)

sma_20 = ta.sma(close, 20)
sma_50 = ta.sma(close, 50)
sma_200 = ta.sma(close, 200)

atr = ta.atr(14)
volatility = ta.stdev(close, 20) / close * 100'''
        
        self.script_parts.append(indicators)
    
    def add_plotting(self):
        """Add plotting functions."""
        plotting = '''
// Plotting
plot(sma_20, "SMA 20", color=color.blue)
plot(sma_50, "SMA 50", color=color.orange)
plot(sma_200, "SMA 200", color=color.red)

plot(bb_upper, "BB Upper", color=color.gray)
plot(bb_lower, "BB Lower", color=color.gray)
fill(plot(bb_upper), plot(bb_lower), color=color.new(color.gray, 90))'''
        
        self.script_parts.append(plotting)
    
    def add_strategy_logic(self):
        """Add strategy-specific logic (to be implemented by subclasses)."""
        pass
    
    def add_entry_exit_rules(self):
        """Add entry and exit rules."""
        rules = '''
// Entry and Exit Rules
long_condition = false
short_condition = false
exit_long = false
exit_short = false

// Strategy-specific conditions will be added by subclasses'''
        
        self.script_parts.append(rules)
    
    def add_execution(self):
        """Add strategy execution."""
        execution = '''
// Strategy Execution
if long_condition
    strategy.entry("Long", strategy.long)

if short_condition
    strategy.entry("Short", strategy.short)

if exit_long
    strategy.close("Long")

if exit_short
    strategy.close("Short")'''
        
        self.script_parts.append(execution)
    
    def add_plotting_signals(self):
        """Add signal plotting."""
        signals = '''
// Plot Signals
plotshape(long_condition, "Long Signal", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(short_condition, "Short Signal", shape.triangledown, location.abovebar, color.red, size=size.small)
plotshape(exit_long or exit_short, "Exit Signal", shape.circle, location.abovebar, color.yellow, size=size.tiny)'''
        
        self.script_parts.append(signals)
    
    def add_alert_conditions(self):
        """Add alert conditions."""
        alerts = '''
// Alert Conditions
alertcondition(long_condition, "Long Entry", "Long entry signal")
alertcondition(short_condition, "Short Entry", "Short entry signal")
alertcondition(exit_long, "Exit Long", "Exit long position")
alertcondition(exit_short, "Exit Short", "Exit short position")'''
        
        self.script_parts.append(alerts)
    
    def add_performance_metrics(self):
        """Add performance metrics display."""
        metrics = '''
// Performance Metrics
var table perf_table = table.new(position.top_right, 2, 4, bgcolor=color.white, border_width=1)
if barstate.islast
    table.cell(perf_table, 0, 0, "Metric", text_color=color.black, bgcolor=color.gray)
    table.cell(perf_table, 1, 0, "Value", text_color=color.black, bgcolor=color.gray)
    table.cell(perf_table, 0, 1, "Total Return", text_color=color.black)
    table.cell(perf_table, 1, 1, str.tostring(strategy.netprofit, "#.##"), text_color=color.black)
    table.cell(perf_table, 0, 2, "Win Rate", text_color=color.black)
    table.cell(perf_table, 1, 2, str.tostring(strategy.wintrades / strategy.closedtrades * 100, "#.##") + "%", text_color=color.black)
    table.cell(perf_table, 0, 3, "Max Drawdown", text_color=color.black)
    table.cell(perf_table, 1, 3, str.tostring(strategy.max_drawdown, "#.##"), text_color=color.black)'''
        
        self.script_parts.append(metrics)
    
    def build_script(self) -> str:
        """
        Build complete Pine Script.
        
        Returns:
            Complete Pine Script as string
        """
        # Add all components
        self.add_header()
        self.add_parameters()
        self.add_indicators()
        self.add_plotting()
        self.add_strategy_logic()
        self.add_entry_exit_rules()
        self.add_execution()
        self.add_plotting_signals()
        self.add_alert_conditions()
        self.add_performance_metrics()
        
        # Join all parts
        return '\n'.join(self.script_parts)
    
    def save_script(self, filename: str):
        """
        Save Pine Script to file.
        
        Args:
            filename: Output filename
        """
        script = self.build_script()
        
        try:
            with open(filename, 'w') as f:
                f.write(script)
            logger.info(f"Pine Script saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving Pine Script: {e}")
    
    def get_script(self) -> str:
        """
        Get the generated Pine Script.
        
        Returns:
            Pine Script as string
        """
        return self.build_script()
    
    def add_custom_function(self, function_name: str, function_code: str):
        """
        Add custom Pine Script function.
        
        Args:
            function_name: Name of the function
            function_code: Function code
        """
        custom_function = f'''
// Custom Function: {function_name}
{function_code}'''
        
        self.script_parts.append(custom_function)
    
    def add_variable(self, var_name: str, var_value: str):
        """
        Add variable to Pine Script.
        
        Args:
            var_name: Variable name
            var_value: Variable value/expression
        """
        variable = f'{var_name} = {var_value}'
        self.script_parts.append(variable)
    
    def add_comment(self, comment: str):
        """
        Add comment to Pine Script.
        
        Args:
            comment: Comment text
        """
        self.script_parts.append(f'// {comment}')
    
    def add_condition(self, condition_name: str, condition_code: str):
        """
        Add condition to Pine Script.
        
        Args:
            condition_name: Name of the condition
            condition_code: Condition code
        """
        condition = f'{condition_name} = {condition_code}'
        self.script_parts.append(condition) 