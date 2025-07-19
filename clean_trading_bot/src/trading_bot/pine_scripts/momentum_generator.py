"""
Momentum Strategy Pine Script Generator.

This module generates Pine Scripts for momentum strategies
that can be used in TradingView for backtesting and visualization.
"""

from typing import Dict, Any
from .base_generator import BasePineGenerator, PineScriptConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MomentumPineGenerator(BasePineGenerator):
    """
    Pine Script generator for Momentum Strategy.
    
    This generator creates Pine Scripts that implement momentum
    logic with trend following and dynamic position sizing.
    """
    
    def __init__(self, config: PineScriptConfig):
        """
        Initialize Momentum Pine Script generator.
        
        Args:
            config: Pine Script configuration
        """
        super().__init__(config)
        
        # Set default parameters if not provided
        if 'short_period' not in self.config.parameters:
            self.config.parameters['short_period'] = 10
        if 'long_period' not in self.config.parameters:
            self.config.parameters['long_period'] = 30
        if 'momentum_threshold' not in self.config.parameters:
            self.config.parameters['momentum_threshold'] = 0.02
        if 'trend_confirmation_periods' not in self.config.parameters:
            self.config.parameters['trend_confirmation_periods'] = 3
        if 'volatility_lookback' not in self.config.parameters:
            self.config.parameters['volatility_lookback'] = 20
        if 'momentum_strength' not in self.config.parameters:
            self.config.parameters['momentum_strength'] = 0.8
    
    def generate_script(self) -> str:
        """
        Generate complete Pine Script for momentum strategy.
        
        Returns:
            Complete Pine Script as string
        """
        self.add_header()
        self.add_parameters()
        self.add_indicators()
        self.add_momentum_logic()
        self.add_entry_exit_rules()
        self.add_execution()
        self.add_plotting()
        self.add_plotting_signals()
        self.add_alert_conditions()
        self.add_performance_metrics()
        
        return self.build_script()
    
    def add_momentum_logic(self):
        """Add momentum specific logic."""
        logic = f'''
// Momentum Strategy Logic
short_period = {self.config.parameters.get('short_period', 10)}
long_period = {self.config.parameters.get('long_period', 30)}
momentum_threshold = {self.config.parameters.get('momentum_threshold', 0.02)}
trend_confirmation_periods = {self.config.parameters.get('trend_confirmation_periods', 3)}
volatility_lookback = {self.config.parameters.get('volatility_lookback', 20)}
momentum_strength = {self.config.parameters.get('momentum_strength', 0.8)}

// Calculate momentum indicators
short_momentum = (close - close[short_period - 1]) / close[short_period - 1]
long_momentum = (close - close[long_period - 1]) / close[long_period - 1]

// Calculate trend direction
trend_direction = short_momentum > 0 and long_momentum > 0 ? "uptrend" : 
                 short_momentum < 0 and long_momentum < 0 ? "downtrend" : "sideways"

// Calculate trend strength using linear regression
trend_strength = math.abs(ta.linreg(close, 10, 0))

// Calculate momentum consistency
short_momentum_consistency = ta.stdev(short_momentum, 5)
long_momentum_consistency = ta.stdev(long_momentum, 5)

// Calculate momentum ratio
momentum_ratio = math.abs(short_momentum) / math.max(math.abs(long_momentum), 0.001)

// Calculate trend consistency
trend_consistency = 0.0
if trend_direction == "uptrend"
    trend_consistency := trend_consistency + 1
else if trend_direction == "downtrend"
    trend_consistency := trend_consistency + 1
else
    trend_consistency := 0

// Calculate overall momentum score
momentum_score = (short_momentum * 0.6 + long_momentum * 0.4)

// Calculate volatility regime
volatility = ta.stdev(close, volatility_lookback) / close * 100
volatility_avg = ta.sma(volatility, 10)
volatility_regime = volatility > volatility_avg * 1.3 ? "high" : 
                   volatility < volatility_avg * 0.7 ? "low" : "medium"

// Calculate volume momentum
volume_momentum = (volume - ta.sma(volume, long_period)) / ta.sma(volume, long_period)

// Calculate price acceleration (second derivative)
price_acceleration = ta.change(ta.change(close))

// Momentum signals
strong_uptrend = momentum_score > momentum_threshold and trend_consistency >= trend_confirmation_periods
strong_downtrend = momentum_score < -momentum_threshold and trend_consistency >= trend_confirmation_periods

// Adjust confidence based on volatility regime
volatility_multiplier = volatility_regime == "high" ? 0.8 : 
                       volatility_regime == "low" ? 1.1 : 1.0

// Calculate signal confidence
base_confidence = math.abs(momentum_score) / momentum_threshold
trend_strength_factor = math.min(trend_strength, 1.0)
adjusted_confidence = base_confidence * trend_strength_factor * volatility_multiplier * momentum_strength

// Signal conditions
long_condition = strong_uptrend and adjusted_confidence > 0.4 and trend_strength < 0.8
short_condition = strong_downtrend and adjusted_confidence > 0.4 and trend_strength < 0.8

// Exit conditions
exit_long = momentum_score < 0 or trend_direction == "downtrend"
exit_short = momentum_score > 0 or trend_direction == "uptrend"

// Dynamic stop loss and take profit based on momentum strength
momentum_multiplier = 1 + math.abs(momentum_score)
stop_loss_pct = 0.05
take_profit_pct = 0.15 * momentum_multiplier

long_stop_loss = strategy.position_avg_price * (1 - stop_loss_pct)
long_take_profit = strategy.position_avg_price * (1 + take_profit_pct)

short_stop_loss = strategy.position_avg_price * (1 + stop_loss_pct)
short_take_profit = strategy.position_avg_price * (1 - take_profit_pct)

// Additional exit conditions
exit_long_stop = strategy.position_size > 0 and (close <= long_stop_loss or close >= long_take_profit)
exit_short_stop = strategy.position_size < 0 and (close >= short_stop_loss or close <= short_take_profit)'''
        
        self.script_parts.append(logic)
    
    def add_plotting(self):
        """Add momentum specific plotting."""
        plotting = '''
// Momentum Strategy Plotting
// Plot moving averages
sma_short = ta.sma(close, short_period)
sma_long = ta.sma(close, long_period)
plot(sma_short, "Short SMA", color=color.blue, linewidth=2)
plot(sma_long, "Long SMA", color=color.orange, linewidth=2)

// Plot momentum bands
upper_momentum_band = close * (1 + momentum_threshold)
lower_momentum_band = close * (1 - momentum_threshold)
plot(upper_momentum_band, "Upper Momentum Band", color=color.green, linewidth=1)
plot(lower_momentum_band, "Lower Momentum Band", color=color.red, linewidth=1)

// Plot trend strength
plot(trend_strength, "Trend Strength", color=color.purple, display=display.pane)

// Volatility regime background
bgcolor(volatility_regime == "high" ? color.new(color.red, 95) : 
        volatility_regime == "low" ? color.new(color.green, 95) : na)

// Momentum score indicator
plot(momentum_score, "Momentum Score", color=color.blue, display=display.pane)
hline(0, "Zero Line", color=color.gray, linestyle=hline.style_dashed)
hline(momentum_threshold, "Upper Threshold", color=color.green, linestyle=hline.style_dotted)
hline(-momentum_threshold, "Lower Threshold", color=color.red, linestyle=hline.style_dotted)'''
        
        self.script_parts.append(plotting)
    
    def add_entry_exit_rules(self):
        """Add momentum entry and exit rules."""
        rules = '''
// Momentum Strategy Entry and Exit Rules
// Entry conditions are already defined in the logic section
// Exit conditions include both signal-based and stop-loss/take-profit exits

// Combine all exit conditions
final_exit_long = exit_long or exit_long_stop
final_exit_short = exit_short or exit_short_stop'''
        
        self.script_parts.append(rules)
    
    def add_execution(self):
        """Add strategy execution with momentum specific logic."""
        execution = '''
// Momentum Strategy Execution
if long_condition and strategy.position_size == 0
    strategy.entry("Momentum Long", strategy.long)

if short_condition and strategy.position_size == 0
    strategy.entry("Momentum Short", strategy.short)

if final_exit_long and strategy.position_size > 0
    strategy.close("Momentum Long")

if final_exit_short and strategy.position_size < 0
    strategy.close("Momentum Short")'''
        
        self.script_parts.append(execution)
    
    def add_plotting_signals(self):
        """Add momentum signal plotting."""
        signals = '''
// Momentum Signal Plotting
plotshape(long_condition, "Long Signal", shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(short_condition, "Short Signal", shape.triangledown, location.abovebar, color.red, size=size.small)
plotshape(final_exit_long, "Exit Long", shape.circle, location.abovebar, color.yellow, size=size.tiny)
plotshape(final_exit_short, "Exit Short", shape.circle, location.belowbar, color.yellow, size=size.tiny)

// Trend direction indicator
plotshape(trend_direction == "uptrend", "Uptrend", shape.flag, location.top, color.green, size=size.tiny)
plotshape(trend_direction == "downtrend", "Downtrend", shape.flag, location.bottom, color.red, size=size.tiny)'''
        
        self.script_parts.append(signals)
    
    def add_alert_conditions(self):
        """Add momentum alert conditions."""
        alerts = '''
// Momentum Strategy Alert Conditions
alertcondition(long_condition, "Momentum Long Entry", 
              "Momentum long entry signal - Score: " + str.tostring(momentum_score, "#.##"))
alertcondition(short_condition, "Momentum Short Entry", 
              "Momentum short entry signal - Score: " + str.tostring(momentum_score, "#.##"))
alertcondition(final_exit_long, "Momentum Exit Long", "Momentum exit long position")
alertcondition(final_exit_short, "Momentum Exit Short", "Momentum exit short position")
alertcondition(trend_direction == "uptrend", "Uptrend Detected", "Uptrend detected")
alertcondition(trend_direction == "downtrend", "Downtrend Detected", "Downtrend detected")
alertcondition(volatility_regime == "high", "High Volatility Regime", "High volatility regime detected")
alertcondition(volatility_regime == "low", "Low Volatility Regime", "Low volatility regime detected")'''
        
        self.script_parts.append(alerts)
    
    def add_performance_metrics(self):
        """Add momentum specific performance metrics."""
        metrics = '''
// Momentum Strategy Performance Metrics
var table momentum_table = table.new(position.top_right, 2, 7, bgcolor=color.white, border_width=1)
if barstate.islast
    table.cell(momentum_table, 0, 0, "Momentum Metrics", text_color=color.black, bgcolor=color.gray)
    table.cell(momentum_table, 1, 0, "Value", text_color=color.black, bgcolor=color.gray)
    table.cell(momentum_table, 0, 1, "Total Return", text_color=color.black)
    table.cell(momentum_table, 1, 1, str.tostring(strategy.netprofit, "#.##"), text_color=color.black)
    table.cell(momentum_table, 0, 2, "Win Rate", text_color=color.black)
    table.cell(momentum_table, 1, 2, str.tostring(strategy.wintrades / strategy.closedtrades * 100, "#.##") + "%", text_color=color.black)
    table.cell(momentum_table, 0, 3, "Max Drawdown", text_color=color.black)
    table.cell(momentum_table, 1, 3, str.tostring(strategy.max_drawdown, "#.##"), text_color=color.black)
    table.cell(momentum_table, 0, 4, "Momentum Score", text_color=color.black)
    table.cell(momentum_table, 1, 4, str.tostring(momentum_score, "#.##"), text_color=color.black)
    table.cell(momentum_table, 0, 5, "Trend Direction", text_color=color.black)
    table.cell(momentum_table, 1, 5, trend_direction, text_color=color.black)
    table.cell(momentum_table, 0, 6, "Volatility Regime", text_color=color.black)
    table.cell(momentum_table, 1, 6, volatility_regime, text_color=color.black)'''
        
        self.script_parts.append(metrics) 