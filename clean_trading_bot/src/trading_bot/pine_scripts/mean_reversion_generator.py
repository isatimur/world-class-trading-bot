"""
Mean Reversion Strategy Pine Script Generator.

This module generates Pine Scripts for mean reversion strategies
that can be used in TradingView for backtesting and visualization.
"""

from typing import Dict, Any
from .base_generator import BasePineGenerator, PineScriptConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MeanReversionPineGenerator(BasePineGenerator):
    """
    Pine Script generator for Mean Reversion Strategy.
    
    This generator creates Pine Scripts that implement mean reversion
    logic with statistical arbitrage and volatility adjustment.
    """
    
    def __init__(self, config: PineScriptConfig):
        """
        Initialize Mean Reversion Pine Script generator.
        
        Args:
            config: Pine Script configuration
        """
        super().__init__(config)
        
        # Set default parameters if not provided
        if 'lookback_period' not in self.config.parameters:
            self.config.parameters['lookback_period'] = 50
        if 'z_score_threshold' not in self.config.parameters:
            self.config.parameters['z_score_threshold'] = 2.0
        if 'volatility_lookback' not in self.config.parameters:
            self.config.parameters['volatility_lookback'] = 20
        if 'mean_reversion_strength' not in self.config.parameters:
            self.config.parameters['mean_reversion_strength'] = 0.7
    
    def generate_script(self) -> str:
        """
        Generate complete Pine Script for mean reversion strategy.
        
        Returns:
            Complete Pine Script as string
        """
        self.add_header()
        self.add_parameters()
        self.add_indicators()
        self.add_mean_reversion_logic()
        self.add_entry_exit_rules()
        self.add_execution()
        self.add_plotting()
        self.add_plotting_signals()
        self.add_alert_conditions()
        self.add_performance_metrics()
        
        return self.build_script()
    
    def add_mean_reversion_logic(self):
        """Add mean reversion specific logic."""
        logic = f'''
// Mean Reversion Logic
lookback_period = {self.config.parameters.get('lookback_period', 50)}
z_score_threshold = {self.config.parameters.get('z_score_threshold', 2.0)}
volatility_lookback = {self.config.parameters.get('volatility_lookback', 20)}
mean_reversion_strength = {self.config.parameters.get('mean_reversion_strength', 0.7)}

// Calculate rolling statistics
rolling_mean = ta.sma(close, lookback_period)
rolling_std = ta.stdev(close, lookback_period)
z_score = (close - rolling_mean) / rolling_std

// Calculate volatility regime
volatility = ta.stdev(close, volatility_lookback) / close * 100
volatility_avg = ta.sma(volatility, 10)
volatility_regime = volatility > volatility_avg * 1.3 ? "high" : 
                   volatility < volatility_avg * 0.7 ? "low" : "medium"

// Calculate trend strength
trend_strength = math.abs(ta.linreg(close, 10, 0))

// Calculate support and resistance
support = ta.lowest(close, 20)
resistance = ta.highest(close, 20)
support_resistance_position = (close - support) / (resistance - support)

// Mean reversion signals
overbought = z_score > z_score_threshold
oversold = z_score < -z_score_threshold

// Adjust confidence based on volatility regime
volatility_multiplier = volatility_regime == "high" ? 0.8 : 
                       volatility_regime == "low" ? 1.2 : 1.0

// Calculate signal confidence
base_confidence = math.abs(z_score) / z_score_threshold
adjusted_confidence = base_confidence * volatility_multiplier * mean_reversion_strength

// Signal conditions
long_condition = oversold and adjusted_confidence > 0.3 and trend_strength < 0.7
short_condition = overbought and adjusted_confidence > 0.3 and trend_strength < 0.7

// Exit conditions
exit_long = z_score > 0 or close > rolling_mean * 1.02
exit_short = z_score < 0 or close < rolling_mean * 0.98

// Stop loss and take profit
stop_loss_pct = 0.05
take_profit_pct = 0.15

long_stop_loss = strategy.position_avg_price * (1 - stop_loss_pct)
long_take_profit = strategy.position_avg_price * (1 + take_profit_pct)

short_stop_loss = strategy.position_avg_price * (1 + stop_loss_pct)
short_take_profit = strategy.position_avg_price * (1 - take_profit_pct)

// Additional exit conditions
exit_long_stop = strategy.position_size > 0 and (close <= long_stop_loss or close >= long_take_profit)
exit_short_stop = strategy.position_size < 0 and (close >= short_stop_loss or close <= short_take_profit)'''
        
        self.script_parts.append(logic)
    
    def add_plotting(self):
        """Add mean reversion specific plotting."""
        plotting = '''
// Mean Reversion Plotting
plot(rolling_mean, "Rolling Mean", color=color.blue, linewidth=2)
plot(rolling_mean + rolling_std * z_score_threshold, "Upper Band", color=color.red, linewidth=1)
plot(rolling_mean - rolling_std * z_score_threshold, "Lower Band", color=color.red, linewidth=1)

// Z-score plotting
hline(0, "Zero Line", color=color.gray, linestyle=hline.style_dashed)
hline(z_score_threshold, "Upper Threshold", color=color.red, linestyle=hline.style_dotted)
hline(-z_score_threshold, "Lower Threshold", color=color.red, linestyle=hline.style_dotted)

// Volatility regime background
bgcolor(volatility_regime == "high" ? color.new(color.red, 95) : 
        volatility_regime == "low" ? color.new(color.green, 95) : na)

// Support and resistance levels
plot(support, "Support", color=color.green, linewidth=1)
plot(resistance, "Resistance", color=color.red, linewidth=1)'''
        
        self.script_parts.append(plotting)
    
    def add_entry_exit_rules(self):
        """Add mean reversion entry and exit rules."""
        rules = '''
// Mean Reversion Entry and Exit Rules
// Entry conditions are already defined in the logic section
// Exit conditions include both signal-based and stop-loss/take-profit exits

// Combine all exit conditions
final_exit_long = exit_long or exit_long_stop
final_exit_short = exit_short or exit_short_stop'''
        
        self.script_parts.append(rules)
    
    def add_execution(self):
        """Add strategy execution with mean reversion specific logic."""
        execution = '''
// Mean Reversion Strategy Execution
if long_condition and strategy.position_size == 0
    strategy.entry("Mean Reversion Long", strategy.long)

if short_condition and strategy.position_size == 0
    strategy.entry("Mean Reversion Short", strategy.short)

if final_exit_long and strategy.position_size > 0
    strategy.close("Mean Reversion Long")

if final_exit_short and strategy.position_size < 0
    strategy.close("Mean Reversion Short")'''
        
        self.script_parts.append(execution)
    
    def add_plotting_signals(self):
        """Add mean reversion signal plotting."""
        signals = '''
// Mean Reversion Signal Plotting
plotshape(long_condition, "Long Signal", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(short_condition, "Short Signal", shape.triangledown, location.abovebar, color.red, size=size.small)
plotshape(final_exit_long, "Exit Long", shape.circle, location.abovebar, color.yellow, size=size.tiny)
plotshape(final_exit_short, "Exit Short", shape.circle, location.belowbar, color.yellow, size=size.tiny)

// Z-score indicator
plot(z_score, "Z-Score", color=color.purple, display=display.pane)'''
        
        self.script_parts.append(signals)
    
    def add_alert_conditions(self):
        """Add mean reversion alert conditions."""
        alerts = '''
// Mean Reversion Alert Conditions
alertcondition(long_condition, "Mean Reversion Long Entry", 
              "Mean reversion long entry signal - Z-Score: " + str.tostring(z_score, "#.##"))
alertcondition(short_condition, "Mean Reversion Short Entry", 
              "Mean reversion short entry signal - Z-Score: " + str.tostring(z_score, "#.##"))
alertcondition(final_exit_long, "Mean Reversion Exit Long", "Mean reversion exit long position")
alertcondition(final_exit_short, "Mean Reversion Exit Short", "Mean reversion exit short position")
alertcondition(volatility_regime == "high", "High Volatility Regime", "High volatility regime detected")
alertcondition(volatility_regime == "low", "Low Volatility Regime", "Low volatility regime detected")'''
        
        self.script_parts.append(alerts)
    
    def add_performance_metrics(self):
        """Add mean reversion specific performance metrics."""
        metrics = '''
// Mean Reversion Performance Metrics
var table mean_rev_table = table.new(position.top_right, 2, 6, bgcolor=color.white, border_width=1)
if barstate.islast
    table.cell(mean_rev_table, 0, 0, "Mean Reversion Metrics", text_color=color.black, bgcolor=color.gray)
    table.cell(mean_rev_table, 1, 0, "Value", text_color=color.black, bgcolor=color.gray)
    table.cell(mean_rev_table, 0, 1, "Total Return", text_color=color.black)
    table.cell(mean_rev_table, 1, 1, str.tostring(strategy.netprofit, "#.##"), text_color=color.black)
    table.cell(mean_rev_table, 0, 2, "Win Rate", text_color=color.black)
    table.cell(mean_rev_table, 1, 2, str.tostring(strategy.wintrades / strategy.closedtrades * 100, "#.##") + "%", text_color=color.black)
    table.cell(mean_rev_table, 0, 3, "Max Drawdown", text_color=color.black)
    table.cell(mean_rev_table, 1, 3, str.tostring(strategy.max_drawdown, "#.##"), text_color=color.black)
    table.cell(mean_rev_table, 0, 4, "Current Z-Score", text_color=color.black)
    table.cell(mean_rev_table, 1, 4, str.tostring(z_score, "#.##"), text_color=color.black)
    table.cell(mean_rev_table, 0, 5, "Volatility Regime", text_color=color.black)
    table.cell(mean_rev_table, 1, 5, volatility_regime, text_color=color.black)'''
        
        self.script_parts.append(metrics) 