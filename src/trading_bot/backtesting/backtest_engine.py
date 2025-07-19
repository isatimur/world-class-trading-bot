"""
Backtesting engine for trading strategies.

This module provides a comprehensive backtesting engine that can test
any trading strategy with historical data and provide detailed analysis.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path

from ..strategies.base_strategy import BaseStrategy, Signal, StrategyPerformance
from ..models.market_data import MarketData
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005   # 0.05% slippage
    data_source: str = "yahoo"
    timeframe: str = "1D"
    symbols: List[str] = field(default_factory=list)
    
    # Risk management
    max_position_size: float = 0.1  # 10% max position
    stop_loss_pct: float = 0.05     # 5% stop loss
    take_profit_pct: float = 0.15   # 15% take profit
    
    # Performance tracking
    benchmark_symbol: str = "^GSPC"  # S&P 500 as benchmark
    risk_free_rate: float = 0.02    # 2% risk-free rate


@dataclass
class BacktestResult:
    """Results of backtesting"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_trade_duration: float
    volatility: float
    beta: float
    alpha: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    trades: List[Dict] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)
    timestamp: datetime = field(default_factory=datetime.now)


class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies.
    
    This engine can test any trading strategy with historical data
    and provide detailed performance analysis.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.data = {}
        self.strategies = {}
        self.results = {}
        
        # Performance tracking
        self.equity_curves = {}
        self.trade_logs = {}
        self.performance_metrics = {}
        
        logger.info(f"Initialized BacktestEngine for {config.symbols}")
    
    async def load_data(self, symbols: Optional[List[str]] = None):
        """
        Load historical data for backtesting.
        
        Args:
            symbols: List of symbols to load (uses config if None)
        """
        symbols = symbols or self.config.symbols
        
        try:
            import yfinance as yf
            
            for symbol in symbols:
                logger.info(f"Loading data for {symbol}")
                
                # Download data
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval=self.config.timeframe
                )
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Clean and prepare data
                data = self._prepare_data(data)
                self.data[symbol] = data
                
                logger.info(f"Loaded {len(data)} records for {symbol}")
            
            # Load benchmark data
            if self.config.benchmark_symbol:
                benchmark = yf.Ticker(self.config.benchmark_symbol)
                benchmark_data = benchmark.history(
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval=self.config.timeframe
                )
                self.data[self.config.benchmark_symbol] = self._prepare_data(benchmark_data)
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for backtesting.
        
        Args:
            data: Raw data from data source
            
        Returns:
            Prepared data
        """
        # Ensure required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add technical indicators
        data = self._add_technical_indicators(data)
        
        # Add datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to data.
        
        Args:
            data: Price data
            
        Returns:
            Data with technical indicators
        """
        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # MACD
        data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = self._calculate_macd(data['Close'])
        
        # Bollinger Bands
        data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = self._calculate_bollinger_bands(data['Close'])
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Moving Averages
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['SMA_200'] = data['Close'].rolling(200).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # ATR
        data['ATR'] = self._calculate_atr(data)
        
        # Volatility
        data['Volatility'] = data['Close'].rolling(20).std() / data['Close'] * 100
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return atr
    
    def add_strategy(self, strategy: BaseStrategy):
        """
        Add strategy to backtest.
        
        Args:
            strategy: Trading strategy to test
        """
        self.strategies[strategy.strategy_id] = strategy
        logger.info(f"Added strategy: {strategy.strategy_id}")
    
    async def run_backtest(self, strategy_id: str, symbol: str) -> BacktestResult:
        """
        Run backtest for a specific strategy and symbol.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            
        Returns:
            Backtest results
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        if symbol not in self.data:
            raise ValueError(f"Data for {symbol} not found")
        
        strategy = self.strategies[strategy_id]
        data = self.data[symbol]
        
        logger.info(f"Running backtest for {strategy_id} on {symbol}")
        
        # Initialize strategy
        strategy.current_capital = self.config.initial_capital
        strategy.trades = []
        strategy.signals = []
        
        # Track equity curve
        equity_curve = []
        current_equity = self.config.initial_capital
        
        # Run simulation
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Create market data object with all fields
            market_data = MarketData(
                symbol=symbol,
                current_price=row['Close'],
                price_change=row['Close'] - row['Open'],
                price_change_pct=(row['Close'] - row['Open']) / row['Open'],
                volume=row['Volume'],
                market_cap=row['Close'] * row['Volume'],  # Approximate
                timestamp=timestamp,
                # Technical indicators
                rsi=row.get('RSI', 50),
                macd=row.get('MACD', 0),
                macd_signal=row.get('MACD_Signal', 0),
                macd_histogram=row.get('MACD_Histogram', 0),
                bb_upper=row.get('BB_Upper', row['Close']),
                bb_middle=row.get('BB_Middle', row['Close']),
                bb_lower=row.get('BB_Lower', row['Close']),
                bb_position=row.get('BB_Position', 0.5),
                sma_20=row.get('SMA_20', row['Close']),
                sma_50=row.get('SMA_50', row['Close']),
                ema_12=row.get('EMA_12', row['Close']),
                ema_26=row.get('EMA_26', row['Close']),
                atr=row.get('ATR', 0),
                # Price data
                open_price=row['Open'],
                high_price=row['High'],
                low_price=row['Low'],
                close_price=row['Close']
            )
            
            # Update strategy position
            await strategy.update_position(row['Close'], timestamp)
            
            # Generate signal
            signal = await strategy.generate_signal(market_data)
            
            if signal:
                # Apply commission and slippage
                if signal.signal_type.value == 'BUY':
                    execution_price = row['Close'] * (1 + self.config.slippage)
                else:
                    execution_price = row['Close'] * (1 - self.config.slippage)
                
                signal.price = execution_price
                
                # Execute signal
                success = await strategy.execute_signal(signal)
                
                if success:
                    # Calculate commission
                    trade_value = signal.quantity * execution_price
                    commission = trade_value * self.config.commission
                    
                    # Update equity
                    if signal.signal_type.value == 'BUY':
                        current_equity -= trade_value + commission
                    else:
                        current_equity += trade_value - commission
            
            # Record equity
            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'price': row['Close']
            })
        
        # Calculate performance metrics
        result = self._calculate_performance(strategy, equity_curve, symbol)
        
        # Store results
        self.results[f"{strategy_id}_{symbol}"] = result
        self.equity_curves[f"{strategy_id}_{symbol}"] = pd.DataFrame(equity_curve)
        
        logger.info(f"Backtest completed: {result.total_return:.2f}% return")
        
        return result
    
    def _calculate_performance(self, strategy: BaseStrategy, equity_curve: List[Dict], symbol: str) -> BacktestResult:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            strategy: Trading strategy
            equity_curve: Equity curve data
            symbol: Trading symbol
            
        Returns:
            Performance results
        """
        # Basic metrics
        initial_capital = self.config.initial_capital
        final_capital = equity_curve[-1]['equity'] if equity_curve else initial_capital
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        # Calculate annualized return
        days = (self.config.end_date - self.config.start_date).days
        annualized_return = ((final_capital / initial_capital) ** (365 / days) - 1) * 100 if days > 0 else 0
        
        # Calculate max drawdown
        equity_series = pd.Series([e['equity'] for e in equity_curve])
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Trade statistics
        trades = strategy.trades
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.net_value > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profit metrics
        if trades:
            trade_values = [t.net_value for t in trades]
            gross_profit = sum([v for v in trade_values if v > 0])
            gross_loss = abs(sum([v for v in trade_values if v < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
            best_trade = max(trade_values) if trade_values else 0
            worst_trade = min(trade_values) if trade_values else 0
        else:
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            best_trade = 0
            worst_trade = 0
        
        # Calculate Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series([e['equity'] for e in equity_curve]).pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            excess_return = (annualized_return - self.config.risk_free_rate * 100) / 100
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        else:
            sharpe_ratio = 0
            volatility = 0
        
        # Calculate beta and alpha
        if self.config.benchmark_symbol in self.data:
            benchmark_data = self.data[self.config.benchmark_symbol]
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()
            
            if len(returns) > 0 and len(benchmark_returns) > 0:
                min_len = min(len(returns), len(benchmark_returns))
                returns_aligned = returns.iloc[:min_len]
                benchmark_aligned = benchmark_returns.iloc[:min_len]
                
                covariance = np.cov(returns_aligned, benchmark_aligned)[0, 1]
                benchmark_variance = np.var(benchmark_aligned)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
                
                alpha = (annualized_return - self.config.risk_free_rate * 100) - beta * (benchmark_aligned.mean() * 252 * 100 - self.config.risk_free_rate * 100)
            else:
                beta = 1
                alpha = 0
        else:
            beta = 1
            alpha = 0
        
        # Calculate Sortino ratio
        if len(returns) > 0:
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        else:
            sortino_ratio = 0
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Calculate Information ratio
        if self.config.benchmark_symbol in self.data and len(returns) > 0:
            benchmark_returns = self.data[self.config.benchmark_symbol]['Close'].pct_change().dropna()
            min_len = min(len(returns), len(benchmark_returns))
            returns_aligned = returns.iloc[:min_len]
            benchmark_aligned = benchmark_returns.iloc[:min_len]
            
            active_returns = returns_aligned - benchmark_aligned
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        else:
            information_ratio = 0
        
        # Calculate average trade duration
        if len(trades) >= 2:
            durations = []
            for i in range(1, len(trades)):
                # Handle timezone-aware and naive timestamps
                ts1 = trades[i].timestamp
                ts2 = trades[i-1].timestamp
                
                # Convert to naive if needed
                if ts1.tzinfo is not None and ts2.tzinfo is None:
                    ts1 = ts1.replace(tzinfo=None)
                elif ts2.tzinfo is not None and ts1.tzinfo is None:
                    ts2 = ts2.replace(tzinfo=None)
                
                duration = (ts1 - ts2).total_seconds() / 3600
                durations.append(duration)
            avg_trade_duration = np.mean(durations) if durations else 0
        else:
            avg_trade_duration = 0
        
        return BacktestResult(
            strategy_name=strategy.strategy_id,
            symbol=symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade_duration=avg_trade_duration,
            volatility=volatility * 100,
            beta=beta,
            alpha=alpha,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            trades=[t.__dict__ for t in trades],
            equity_curve=pd.DataFrame(equity_curve)
        )
    
    async def run_all_backtests(self) -> Dict[str, BacktestResult]:
        """
        Run backtests for all strategies and symbols.
        
        Returns:
            Dictionary of backtest results
        """
        results = {}
        
        for strategy_id in self.strategies:
            for symbol in self.data:
                if symbol != self.config.benchmark_symbol:
                    result = await self.run_backtest(strategy_id, symbol)
                    results[f"{strategy_id}_{symbol}"] = result
        
        return results
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of all backtest results.
        
        Returns:
            DataFrame with results summary
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for key, result in self.results.items():
            summary_data.append({
                'Strategy': result.strategy_name,
                'Symbol': result.symbol,
                'Total Return (%)': round(result.total_return, 2),
                'Annualized Return (%)': round(result.annualized_return, 2),
                'Sharpe Ratio': round(result.sharpe_ratio, 3),
                'Max Drawdown (%)': round(result.max_drawdown, 2),
                'Win Rate (%)': round(result.win_rate, 2),
                'Profit Factor': round(result.profit_factor, 3),
                'Total Trades': result.total_trades,
                'Volatility (%)': round(result.volatility, 2),
                'Beta': round(result.beta, 3),
                'Alpha (%)': round(result.alpha, 2)
            })
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, filename: str):
        """
        Save backtest results to file.
        
        Args:
            filename: Output filename
        """
        try:
            summary = self.get_results_summary()
            summary.to_csv(filename, index=False)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def get_equity_curve(self, strategy_id: str, symbol: str) -> pd.DataFrame:
        """
        Get equity curve for a specific strategy and symbol.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            
        Returns:
            Equity curve DataFrame
        """
        key = f"{strategy_id}_{symbol}"
        return self.equity_curves.get(key, pd.DataFrame()) 