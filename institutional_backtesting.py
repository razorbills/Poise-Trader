#!/usr/bin/env python3
"""
🎯 INSTITUTIONAL-GRADE BACKTESTING FRAMEWORK
Enterprise-level strategy validation and optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import concurrent.futures
from pathlib import Path
import logging
import os

@dataclass
class BacktestResult:
    """Comprehensive backtesting results"""
    strategy_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    total_trades: int
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional VaR

class InstitutionalBacktester:
    """Professional backtesting with walk-forward analysis"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% per trade
        self.slippage = 0.0005   # 0.05% slippage

        try:
            fee_rate = float(os.getenv('PAPER_FEE_RATE', '') or 0.0)
            if fee_rate > 0:
                self.commission = float(fee_rate)
        except Exception:
            pass

        try:
            spread_bps = float(os.getenv('PAPER_SPREAD_BPS', '') or 0.0)
        except Exception:
            spread_bps = 0.0
        try:
            slip_bps = float(os.getenv('PAPER_SLIPPAGE_BPS', '') or 0.0)
        except Exception:
            slip_bps = 0.0

        try:
            if spread_bps > 0 or slip_bps > 0:
                self.slippage = float((max(0.0, spread_bps) / 2.0 + max(0.0, slip_bps)) / 10000.0)
        except Exception:
            pass
        
    async def run_comprehensive_backtest(self, 
                                       strategy_func: callable = None,
                                       price_data: pd.DataFrame = None,
                                       parameters: Dict = None,
                                       start_date: str = None,
                                       end_date: str = None,
                                       **kwargs) -> BacktestResult:
        """Run comprehensive institutional-grade backtest"""

        # Compatibility path for integrated_trading_orchestrator.py
        if strategy_func is None and price_data is None and (kwargs.get('strategy_config') is not None or kwargs.get('historical_data') is not None):
            strategy_config = kwargs.get('strategy_config') or {}
            historical_data = kwargs.get('historical_data') or {}
            try:
                initial_capital = float(kwargs.get('initial_capital', self.initial_capital) or self.initial_capital)
            except Exception:
                initial_capital = float(self.initial_capital)

            try:
                walk_forward_window = int(kwargs.get('walk_forward_window', 7) or 7)
            except Exception:
                walk_forward_window = 7

            try:
                monte_carlo_runs = int(kwargs.get('monte_carlo_runs', 1000) or 1000)
            except Exception:
                monte_carlo_runs = 1000

            self.initial_capital = float(initial_capital)

            # Pick a representative symbol dataset (prefer BTC when available).
            chosen_df = None
            try:
                if isinstance(historical_data, dict):
                    for _sym, _df in (historical_data or {}).items():
                        if _df is not None and 'BTC' in str(_sym).upper():
                            chosen_df = _df
                            break
                if chosen_df is None:
                    for _sym, _df in (historical_data or {}).items():
                        if _df is not None:
                            chosen_df = _df
                            break
            except Exception:
                chosen_df = None

            if chosen_df is None:
                return self._empty_backtest_result(strategy_name=str(strategy_config.get('name') or 'Empty'))

            try:
                if isinstance(chosen_df, pd.DataFrame) and 'timestamp' in chosen_df.columns:
                    chosen_df = chosen_df.copy()
                    chosen_df['timestamp'] = pd.to_datetime(chosen_df['timestamp'], errors='coerce')
                    chosen_df = chosen_df.set_index('timestamp')
            except Exception:
                pass

            @dataclass
            class _BasicSignal:
                action: str
                position_size: float

            async def _default_strategy_func(training_data: pd.DataFrame, _params: Dict) -> Any:
                try:
                    close = training_data['close']
                    if close is None or len(close) < 60:
                        return _BasicSignal(action='HOLD', position_size=0.0)

                    sma_fast = float(close.rolling(20).mean().iloc[-1])
                    sma_slow = float(close.rolling(50).mean().iloc[-1])
                    if sma_fast <= 0 or sma_slow <= 0:
                        return _BasicSignal(action='HOLD', position_size=0.0)

                    try:
                        rpt = float(strategy_config.get('risk_per_trade', 0.02) or 0.02)
                    except Exception:
                        rpt = 0.02
                    pos_size = max(0.05, min(0.30, rpt * 5.0))

                    if sma_fast > sma_slow * 1.001:
                        return _BasicSignal(action='BUY', position_size=pos_size)
                    if sma_fast < sma_slow * 0.999:
                        return _BasicSignal(action='SELL', position_size=pos_size)
                    return _BasicSignal(action='HOLD', position_size=0.0)
                except Exception:
                    return _BasicSignal(action='HOLD', position_size=0.0)

            parameters = dict(strategy_config or {})
            parameters['walk_forward_window'] = walk_forward_window
            parameters['monte_carlo_runs'] = monte_carlo_runs
            parameters['strategy_name'] = str(strategy_config.get('name') or 'Strategy')
            strategy_func = _default_strategy_func
            price_data = chosen_df
        
        # 1. Data preparation and validation
        validated_data = self._validate_and_prepare_data(price_data, start_date, end_date)

        # 2. Walk-forward analysis
        walk_forward_results = await self._walk_forward_analysis(
            strategy_func, validated_data, parameters
        )
        
        # 3. Monte Carlo simulation
        try:
            n_sims = int((parameters or {}).get('monte_carlo_runs', 1000) or 1000)
        except Exception:
            n_sims = 1000
        monte_carlo_results = await self._monte_carlo_simulation(
            walk_forward_results, n_simulations=n_sims
        )
        
        # 4. Calculate comprehensive metrics
        final_results = self._calculate_institutional_metrics(
            walk_forward_results, monte_carlo_results
        )
        
        return final_results
    
    def _validate_and_prepare_data(self, data: pd.DataFrame, 
                                 start_date: str = None, 
                                 end_date: str = None) -> pd.DataFrame:
        """Validate and prepare data for backtesting"""
        
        # Data quality checks
        if data.isnull().any().any():
            data = data.fillna(method='ffill')
        
        # Date filtering
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Add required columns
        if 'returns' not in data.columns:
            data['returns'] = data['close'].pct_change()
        
        return data
    
    async def _walk_forward_analysis(self, strategy_func: callable,
                                   data: pd.DataFrame, 
                                   parameters: Dict) -> Dict[str, Any]:
        """Walk-forward analysis with expanding window"""
        
        results = {
            'trades': [],
            'equity_curve': [],
            'performance_by_period': []
        }

        try:
            results['strategy_name'] = str((parameters or {}).get('strategy_name') or 'Strategy')
        except Exception:
            results['strategy_name'] = 'Strategy'
        
        # Parameters for walk-forward
        min_training_days = 30
        rebalance_frequency = 7  # Days
        try:
            rebalance_frequency = int((parameters or {}).get('walk_forward_window', rebalance_frequency) or rebalance_frequency)
        except Exception:
            rebalance_frequency = rebalance_frequency

        try:
            min_training_days = int((parameters or {}).get('min_training_days', min_training_days) or min_training_days)
        except Exception:
            min_training_days = min_training_days

        bars_per_day = 1
        try:
            if isinstance(data.index, pd.DatetimeIndex) and len(data.index) >= 5:
                deltas = (data.index[1:] - data.index[:-1])
                delta_s = []
                for d in deltas:
                    try:
                        s = float(d.total_seconds())
                        if s > 0:
                            delta_s.append(s)
                    except Exception:
                        continue
                if delta_s:
                    median_s = float(np.median(np.array(delta_s)))
                    if median_s > 0:
                        bars_per_day = int(round(86400.0 / median_s))
                        if bars_per_day < 1:
                            bars_per_day = 1
                        if bars_per_day > 1440:
                            bars_per_day = 1440
        except Exception:
            bars_per_day = 1

        min_training_bars = int(max(5, min_training_days * bars_per_day))
        rebalance_bars = int(max(1, rebalance_frequency * bars_per_day))
        
        current_capital = self.initial_capital
        position = 0
        
        for i in range(min_training_bars, len(data), rebalance_bars):
            # Training period
            training_data = data.iloc[:i]
            
            # Generate trading signal
            signal = await strategy_func(training_data, parameters)
            
            # Execute trade if signal
            if signal and signal.action != 'HOLD':
                trade_result = self._execute_backtest_trade(
                    signal, data.iloc[i], current_capital, position
                )
                
                results['trades'].append(trade_result)
                current_capital = trade_result['new_capital']
                position = trade_result['new_position']
            
            # Record equity
            results['equity_curve'].append({
                'date': data.index[i],
                'capital': current_capital,
                'position': position
            })
        
        return results
    
    def _execute_backtest_trade(self, signal: Any, current_data: pd.Series,
                              capital: float, current_position: float) -> Dict:
        """Execute trade in backtest with realistic costs"""
        
        price = current_data['close']
        
        # Calculate position size
        if signal.action == 'BUY' and current_position <= 0:
            position_value = capital * signal.position_size
            shares = position_value / price
            cost = shares * price * (1 + self.commission + self.slippage)
            
            new_capital = capital - cost
            new_position = shares
            
        elif signal.action == 'SELL' and current_position > 0:
            proceeds = current_position * price * (1 - self.commission - self.slippage)
            new_capital = capital + proceeds
            new_position = 0
            
        else:
            new_capital = capital
            new_position = current_position
        
        return {
            'timestamp': current_data.name,
            'action': signal.action,
            'price': price,
            'quantity': abs(new_position - current_position),
            'capital': capital,
            'new_capital': new_capital,
            'new_position': new_position,
            'pnl': new_capital - capital
        }
    
    async def _monte_carlo_simulation(self, backtest_results: Dict,
                                    n_simulations: int = 1000) -> Dict[str, Any]:
        """Monte Carlo simulation for robustness testing"""
        
        trades = backtest_results['trades']
        if not trades:
            return {'confidence_intervals': {}, 'risk_metrics': {}}
        
        # Extract returns
        returns = [trade['pnl'] / trade['capital'] for trade in trades]
        
        # Monte Carlo simulation
        simulation_results = []
        
        for _ in range(n_simulations):
            # Randomly resample trades with replacement
            simulated_returns = np.random.choice(returns, len(returns), replace=True)
            simulated_total_return = (1 + pd.Series(simulated_returns)).prod() - 1
            simulation_results.append(simulated_total_return)
        
        # Calculate confidence intervals
        simulation_results = np.array(simulation_results)
        
        confidence_intervals = {
            '95th_percentile': np.percentile(simulation_results, 95),
            '5th_percentile': np.percentile(simulation_results, 5),
            'median': np.percentile(simulation_results, 50),
            'probability_of_loss': np.sum(simulation_results < 0) / n_simulations
        }
        
        return {
            'confidence_intervals': confidence_intervals,
            'simulation_results': simulation_results
        }
    
    def _calculate_institutional_metrics(self, backtest_results: Dict,
                                       monte_carlo_results: Dict) -> BacktestResult:
        """Calculate comprehensive institutional metrics"""
        
        trades = backtest_results['trades']
        equity_curve = backtest_results['equity_curve']
        
        strategy_name = "Strategy"
        try:
            strategy_name = str((backtest_results or {}).get('strategy_name') or "Strategy")
        except Exception:
            strategy_name = "Strategy"

        if not trades or not equity_curve:
            return self._empty_backtest_result(strategy_name=strategy_name)
        
        # Calculate returns
        capital_series = pd.Series([eq['capital'] for eq in equity_curve])
        returns = capital_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (capital_series.iloc[-1] / capital_series.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(capital_series)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Downside deviation for Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        sortino_ratio = annualized_return / downside_deviation
        
        # Maximum drawdown
        peak = capital_series.expanding().max()
        drawdown = (capital_series - peak) / peak
        max_drawdown = abs(float(drawdown.min()))
        
        # Trade analysis
        trade_pnls = [trade['pnl'] for trade in trades]
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        
        # Advanced metrics
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        return BacktestResult(
            strategy_name=str(strategy_name),
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            total_trades=len(trades),
            avg_trade_duration=0,  # Would calculate from actual trade duration
            best_trade=max(trade_pnls) if trade_pnls else 0,
            worst_trade=min(trade_pnls) if trade_pnls else 0,
            consecutive_wins=0,  # Would calculate consecutive streaks
            consecutive_losses=0,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def _empty_backtest_result(self, strategy_name: str = "Empty") -> BacktestResult:
        """Return empty result for error cases"""
        return BacktestResult(
            strategy_name=str(strategy_name or "Empty"),
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            avg_trade_duration=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            var_95=0.0,
            cvar_95=0.0
        )

# Global backtesting engine
institutional_backtester = InstitutionalBacktester()
