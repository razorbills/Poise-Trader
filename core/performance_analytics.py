#!/usr/bin/env python3
"""
📊 COMPREHENSIVE PERFORMANCE ATTRIBUTION & ANALYTICS SYSTEM
Advanced P&L analysis, risk metrics, and strategy attribution
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Individual trade record for analysis"""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_time: datetime
    exit_time: Optional[datetime]
    strategy: str
    confidence: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    hold_time_minutes: float = 0.0
    exit_reason: str = ""

@dataclass
class PerformanceMetrics:
    """Core performance metrics"""
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    volatility: float = 0.0

class PerformanceAnalyzer:
    """Main performance analysis engine"""
    
    def __init__(self, data_dir: str = "data/analytics"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.trade_records: List[TradeRecord] = []
        self.equity_curve: List[Dict] = []
        self._lock = threading.Lock()
        
    def add_trade(self, trade: TradeRecord):
        """Add trade for analysis"""
        with self._lock:
            self.trade_records.append(trade)
            
            # Calculate derived metrics
            if trade.exit_price and trade.exit_time:
                trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
                if trade.side.lower() == 'sell':
                    trade.pnl = -trade.pnl
                
                trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)
                trade.hold_time_minutes = (trade.exit_time - trade.entry_time).total_seconds() / 60
        
        self._save_trade_record(trade)
        logger.debug(f"📊 Trade added: {trade.symbol} {trade.pnl:+.2f}")
    
    def add_portfolio_update(self, timestamp: datetime, portfolio_value: float):
        """Add portfolio update"""
        equity_point = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value
        }
        self.equity_curve.append(equity_point)
    
    def get_strategy_attribution(self) -> Dict[str, Dict[str, float]]:
        """Get performance attribution by strategy"""
        strategy_stats = defaultdict(lambda: {
            'total_pnl': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0
        })
        
        closed_trades = [t for t in self.trade_records if t.exit_time]
        
        for trade in closed_trades:
            strategy = trade.strategy
            stats = strategy_stats[strategy]
            
            stats['total_pnl'] += trade.pnl
            stats['total_trades'] += 1
            
            if trade.pnl > 0:
                stats['winning_trades'] += 1
        
        # Calculate derived metrics
        for strategy, stats in strategy_stats.items():
            if stats['total_trades'] > 0:
                stats['avg_pnl'] = stats['total_pnl'] / stats['total_trades']
                stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
        
        return dict(strategy_stats)
    
    def get_symbol_attribution(self) -> Dict[str, Dict[str, float]]:
        """Get performance attribution by symbol"""
        symbol_stats = defaultdict(lambda: {
            'total_pnl': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0
        })
        
        closed_trades = [t for t in self.trade_records if t.exit_time]
        
        for trade in closed_trades:
            symbol = trade.symbol
            stats = symbol_stats[symbol]
            
            stats['total_pnl'] += trade.pnl
            stats['total_trades'] += 1
            
            if trade.pnl > 0:
                stats['win_rate'] += 1
        
        # Calculate derived metrics
        for symbol, stats in symbol_stats.items():
            if stats['total_trades'] > 0:
                stats['avg_pnl'] = stats['total_pnl'] / stats['total_trades']
                stats['win_rate'] = stats['win_rate'] / stats['total_trades']
        
        return dict(symbol_stats)
    
    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        closed_trades = [t for t in self.trade_records if t.exit_time]
        
        if not closed_trades:
            return self._get_empty_metrics()
        
        # Basic calculations
        total_pnl = sum(t.pnl for t in closed_trades)
        total_fees = sum(t.fees for t in closed_trades)
        net_pnl = total_pnl - total_fees
        
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Risk metrics
        returns = [t.pnl_pct for t in closed_trades]
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Sharpe ratio
        risk_free_rate = 0.02
        excess_returns = np.array(returns) - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Time-based metrics
        if len(self.equity_curve) > 1:
            start_value = self.equity_curve[0]['portfolio_value']
            end_value = self.equity_curve[-1]['portfolio_value']
            total_return_pct = (end_value - start_value) / start_value * 100 if start_value > 0 else 0
        else:
            total_return_pct = 0
        
        return {
            'performance_metrics': PerformanceMetrics(
                total_return=net_pnl,
                total_return_pct=total_return_pct,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                volatility=volatility
            ),
            'trade_statistics': {
                'total_trades': len(closed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'total_pnl': total_pnl,
                'total_fees': total_fees,
                'net_pnl': net_pnl
            },
            'attribution': {
                'by_strategy': self.get_strategy_attribution(),
                'by_symbol': self.get_symbol_attribution()
            }
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        values = [point['portfolio_value'] for point in self.equity_curve]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'performance_metrics': PerformanceMetrics(),
            'trade_statistics': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'total_fees': 0.0,
                'net_pnl': 0.0
            },
            'attribution': {
                'by_strategy': {},
                'by_symbol': {}
            }
        }
    
    def _save_trade_record(self, trade: TradeRecord):
        """Save trade record to file"""
        try:
            trades_file = self.data_dir / "trades.jsonl"
            
            trade_dict = {
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'entry_time': trade.entry_time.isoformat(),
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'strategy': trade.strategy,
                'confidence': trade.confidence,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'fees': trade.fees,
                'exit_reason': trade.exit_reason
            }
            
            with open(trades_file, 'a') as f:
                f.write(json.dumps(trade_dict) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save trade record: {e}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return self.calculate_comprehensive_metrics()

# Global performance analyzer
performance_analyzer = PerformanceAnalyzer()
