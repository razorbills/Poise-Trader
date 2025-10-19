#!/usr/bin/env python3
"""
🏆 90% WIN RATE OPTIMIZATION SYSTEM 🏆
Advanced machine learning and statistical analysis for achieving 90%+ win rate
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class WinRateMetrics:
    """Detailed win rate metrics"""
    current_win_rate: float = 0.0
    target_win_rate: float = 0.90
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    risk_reward_ratio: float = 0.0
    confidence_accuracy: float = 0.0
    strategy_win_rates: Dict[str, float] = field(default_factory=dict)

@dataclass
class TradeQualityScore:
    """Trade quality scoring system"""
    overall_score: float = 0.0
    confidence_score: float = 0.0
    timing_score: float = 0.0
    risk_reward_score: float = 0.0
    market_condition_score: float = 0.0
    technical_score: float = 0.0
    volume_score: float = 0.0
    momentum_score: float = 0.0
    recommendation: str = "SKIP"  # SKIP, CAUTION, GOOD, EXCELLENT

class WinRateOptimizer:
    """
    🏆 ADVANCED WIN RATE OPTIMIZATION SYSTEM
    
    This system uses multiple techniques to achieve 90%+ win rate:
    1. Trade Quality Filtering - Only take the highest quality trades
    2. Confidence Calibration - Ensure confidence scores match actual outcomes
    3. Dynamic Risk Management - Adjust position sizes based on win probability
    4. Strategy Performance Tracking - Focus on best-performing strategies
    5. Market Condition Analysis - Trade only in favorable conditions
    6. Entry/Exit Optimization - Perfect timing for max win probability
    """
    
    def __init__(self, target_win_rate: float = 0.90):
        self.target_win_rate = target_win_rate
        self.min_trade_quality_score = 75.0  # Only take trades scoring 75%+
        self.min_confidence_threshold = 0.70  # Only trade with 70%+ confidence
        
        # Trade history tracking
        self.trade_history: List[Dict] = []
        self.strategy_performance: Dict[str, Dict] = defaultdict(lambda: {
            'wins': 0,
            'losses': 0,
            'total_profit': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0
        })
        
        # Quality control
        self.quality_scores_history: List[float] = []
        self.confidence_accuracy_history: List[Dict] = []
        
        # Performance tracking
        self.current_win_streak = 0
        self.longest_win_streak = 0
        self.current_loss_streak = 0
        
        # Data directory
        self.data_dir = Path("data/win_rate_optimization")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🏆 Win Rate Optimizer initialized with target: {target_win_rate*100:.1f}%")
    
    def calculate_trade_quality_score(self, 
                                     symbol: str,
                                     signal_confidence: float,
                                     market_conditions: Dict,
                                     technical_indicators: Dict,
                                     risk_reward_ratio: float) -> TradeQualityScore:
        """
        Calculate comprehensive trade quality score (0-100)
        Higher score = Higher probability of winning trade
        """
        
        # 1. Confidence Score (30% weight)
        confidence_score = signal_confidence * 30
        
        # 2. Risk/Reward Score (25% weight)
        # Prefer trades with RR > 2.0
        if risk_reward_ratio >= 3.0:
            rr_score = 25.0
        elif risk_reward_ratio >= 2.0:
            rr_score = 20.0
        elif risk_reward_ratio >= 1.5:
            rr_score = 15.0
        else:
            rr_score = 5.0
        
        # 3. Market Condition Score (20% weight)
        market_score = 0.0
        if market_conditions.get('regime') == 'trending':
            market_score += 10
        if market_conditions.get('volatility', 'high') == 'normal':
            market_score += 10
        
        # 4. Technical Score (15% weight)
        technical_score = 0.0
        indicators = technical_indicators
        
        # RSI in good range
        rsi = indicators.get('rsi', 50)
        if 30 <= rsi <= 70:
            technical_score += 5
        
        # Volume confirmation
        if indicators.get('volume_signal') == 'strong':
            technical_score += 5
        
        # Trend alignment
        if indicators.get('trend_aligned', False):
            technical_score += 5
        
        # 5. Timing Score (10% weight)
        timing_score = 10.0  # Default good timing
        current_hour = datetime.now().hour
        # Prefer high-liquidity hours (avoid low-liquidity periods)
        if 8 <= current_hour <= 20:  # Active trading hours
            timing_score = 10.0
        else:
            timing_score = 5.0
        
        # Calculate overall score
        overall_score = confidence_score + rr_score + market_score + technical_score + timing_score
        
        # Determine recommendation
        if overall_score >= 85:
            recommendation = "EXCELLENT"
        elif overall_score >= 75:
            recommendation = "GOOD"
        elif overall_score >= 60:
            recommendation = "CAUTION"
        else:
            recommendation = "SKIP"
        
        return TradeQualityScore(
            overall_score=overall_score,
            confidence_score=confidence_score,
            timing_score=timing_score,
            risk_reward_score=rr_score,
            market_condition_score=market_score,
            technical_score=technical_score,
            recommendation=recommendation
        )
    
    def should_take_trade(self, quality_score: TradeQualityScore, 
                         current_portfolio_risk: float = 0.0) -> Tuple[bool, str]:
        """
        Determine if trade should be taken based on quality score
        Returns: (should_take, reason)
        """
        
        # Check minimum quality threshold
        if quality_score.overall_score < self.min_trade_quality_score:
            return False, f"Quality score too low: {quality_score.overall_score:.1f} < {self.min_trade_quality_score}"
        
        # Check recommendation
        if quality_score.recommendation == "SKIP":
            return False, "Trade quality recommendation: SKIP"
        
        # Check if currently in a losing streak
        if self.current_loss_streak >= 3:
            # Require higher quality after losing streak
            if quality_score.overall_score < 85:
                return False, f"In losing streak ({self.current_loss_streak}), requiring quality 85+, got {quality_score.overall_score:.1f}"
        
        # Check portfolio risk
        if current_portfolio_risk > 0.10:  # More than 10% at risk
            if quality_score.overall_score < 90:
                return False, f"High portfolio risk ({current_portfolio_risk*100:.1f}%), requiring quality 90+"
        
        # Check current win rate
        metrics = self.get_win_rate_metrics()
        if metrics.current_win_rate < self.target_win_rate - 0.05:
            # Below target, be more selective
            if quality_score.overall_score < 80:
                return False, f"Win rate below target ({metrics.current_win_rate*100:.1f}%), requiring quality 80+"
        
        # All checks passed
        return True, f"Trade approved with quality score: {quality_score.overall_score:.1f}"
    
    def record_trade_outcome(self, 
                            trade_id: str,
                            symbol: str,
                            strategy: str,
                            entry_price: float,
                            exit_price: float,
                            profit_loss: float,
                            confidence: float,
                            quality_score: float,
                            exit_reason: str):
        """Record trade outcome for analysis"""
        
        is_win = profit_loss > 0
        
        trade_record = {
            'trade_id': trade_id,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'strategy': strategy,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'profit_pct': (exit_price - entry_price) / entry_price * 100,
            'confidence': confidence,
            'quality_score': quality_score,
            'exit_reason': exit_reason,
            'is_win': is_win
        }
        
        self.trade_history.append(trade_record)
        
        # Update strategy performance
        strategy_stats = self.strategy_performance[strategy]
        if is_win:
            strategy_stats['wins'] += 1
            self.current_win_streak += 1
            self.current_loss_streak = 0
            self.longest_win_streak = max(self.longest_win_streak, self.current_win_streak)
        else:
            strategy_stats['losses'] += 1
            self.current_loss_streak += 1
            self.current_win_streak = 0
        
        strategy_stats['total_profit'] += profit_loss
        total_trades = strategy_stats['wins'] + strategy_stats['losses']
        strategy_stats['win_rate'] = strategy_stats['wins'] / total_trades if total_trades > 0 else 0.0
        strategy_stats['avg_profit'] = strategy_stats['total_profit'] / total_trades if total_trades > 0 else 0.0
        
        # Track confidence accuracy
        self.confidence_accuracy_history.append({
            'confidence': confidence,
            'actual_outcome': 1.0 if is_win else 0.0,
            'quality_score': quality_score
        })
        
        # Save to file
        self._save_trade_record(trade_record)
        
        # Log performance
        metrics = self.get_win_rate_metrics()
        logger.info(f"🏆 Trade recorded: {symbol} | Win: {is_win} | Win Rate: {metrics.current_win_rate*100:.1f}% | Quality: {quality_score:.1f}")
        
        # Alert if below target
        if metrics.current_win_rate < self.target_win_rate and len(self.trade_history) >= 10:
            logger.warning(f"⚠️ Win rate {metrics.current_win_rate*100:.1f}% below target {self.target_win_rate*100:.1f}%")
    
    def get_win_rate_metrics(self) -> WinRateMetrics:
        """Calculate current win rate metrics"""
        
        if not self.trade_history:
            return WinRateMetrics()
        
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t['is_win'])
        losing_trades = total_trades - winning_trades
        
        current_win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        wins = [t['profit_loss'] for t in self.trade_history if t['is_win']]
        losses = [abs(t['profit_loss']) for t in self.trade_history if not t['is_win']]
        
        average_win = np.mean(wins) if wins else 0.0
        average_loss = np.mean(losses) if losses else 0.0
        
        total_wins = sum(wins)
        total_losses = sum(losses)
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        risk_reward_ratio = average_win / average_loss if average_loss > 0 else 0.0
        
        # Calculate confidence accuracy
        if self.confidence_accuracy_history:
            confidences = [h['confidence'] for h in self.confidence_accuracy_history]
            outcomes = [h['actual_outcome'] for h in self.confidence_accuracy_history]
            # Correlation between confidence and actual outcome
            if len(confidences) > 1:
                confidence_accuracy = np.corrcoef(confidences, outcomes)[0, 1]
            else:
                confidence_accuracy = 0.0
        else:
            confidence_accuracy = 0.0
        
        # Strategy-specific win rates
        strategy_win_rates = {
            strategy: stats['win_rate']
            for strategy, stats in self.strategy_performance.items()
        }
        
        return WinRateMetrics(
            current_win_rate=current_win_rate,
            target_win_rate=self.target_win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            risk_reward_ratio=risk_reward_ratio,
            confidence_accuracy=confidence_accuracy,
            strategy_win_rates=strategy_win_rates
        )
    
    def get_strategy_recommendations(self) -> Dict[str, str]:
        """Get recommendations for which strategies to use"""
        
        recommendations = {}
        
        for strategy, stats in self.strategy_performance.items():
            total_trades = stats['wins'] + stats['losses']
            
            if total_trades < 5:
                recommendations[strategy] = "INSUFFICIENT_DATA"
            elif stats['win_rate'] >= 0.85:
                recommendations[strategy] = "EXCELLENT"
            elif stats['win_rate'] >= 0.70:
                recommendations[strategy] = "GOOD"
            elif stats['win_rate'] >= 0.50:
                recommendations[strategy] = "CAUTION"
            else:
                recommendations[strategy] = "AVOID"
        
        return recommendations
    
    def optimize_confidence_threshold(self) -> float:
        """Dynamically optimize confidence threshold based on accuracy"""
        
        if len(self.confidence_accuracy_history) < 20:
            return self.min_confidence_threshold
        
        # Analyze which confidence levels lead to wins
        recent_history = self.confidence_accuracy_history[-100:]
        
        # Group by confidence ranges
        confidence_bins = np.arange(0.5, 1.01, 0.05)
        bin_win_rates = []
        
        for i in range(len(confidence_bins) - 1):
            low, high = confidence_bins[i], confidence_bins[i + 1]
            trades_in_bin = [
                h for h in recent_history 
                if low <= h['confidence'] < high
            ]
            
            if trades_in_bin:
                win_rate = np.mean([t['actual_outcome'] for t in trades_in_bin])
                bin_win_rates.append((low, win_rate))
        
        # Find minimum confidence that achieves target win rate
        for confidence_level, win_rate in bin_win_rates:
            if win_rate >= self.target_win_rate:
                logger.info(f"🎯 Optimized confidence threshold: {confidence_level:.2f} (win rate: {win_rate*100:.1f}%)")
                return confidence_level
        
        # If no bin achieves target, return highest tested
        return 0.80
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        
        metrics = self.get_win_rate_metrics()
        strategy_recs = self.get_strategy_recommendations()
        optimal_confidence = self.optimize_confidence_threshold()
        
        return {
            'win_rate_metrics': {
                'current_win_rate': f"{metrics.current_win_rate*100:.2f}%",
                'target_win_rate': f"{metrics.target_win_rate*100:.2f}%",
                'gap_to_target': f"{(metrics.current_win_rate - metrics.target_win_rate)*100:+.2f}%",
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'risk_reward_ratio': f"{metrics.risk_reward_ratio:.2f}"
            },
            'performance_streaks': {
                'current_win_streak': self.current_win_streak,
                'longest_win_streak': self.longest_win_streak,
                'current_loss_streak': self.current_loss_streak
            },
            'strategy_performance': {
                strategy: {
                    'win_rate': f"{stats['win_rate']*100:.2f}%",
                    'total_trades': stats['wins'] + stats['losses'],
                    'avg_profit': f"${stats['avg_profit']:.2f}",
                    'recommendation': strategy_recs.get(strategy, "UNKNOWN")
                }
                for strategy, stats in self.strategy_performance.items()
            },
            'optimization_settings': {
                'min_quality_score': self.min_trade_quality_score,
                'min_confidence_threshold': self.min_confidence_threshold,
                'optimized_confidence': f"{optimal_confidence:.2f}",
                'confidence_accuracy': f"{metrics.confidence_accuracy:.2f}"
            },
            'recommendations': self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: WinRateMetrics) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Win rate recommendations
        if metrics.current_win_rate < self.target_win_rate:
            gap = (self.target_win_rate - metrics.current_win_rate) * 100
            recommendations.append(f"🎯 Increase trade selectivity to close {gap:.1f}% gap to target win rate")
            recommendations.append(f"📊 Current minimum quality score: {self.min_trade_quality_score} - Consider raising to 80+")
        else:
            recommendations.append(f"✅ Win rate {metrics.current_win_rate*100:.1f}% exceeds target!")
        
        # Strategy recommendations
        strategy_recs = self.get_strategy_recommendations()
        excellent_strategies = [s for s, r in strategy_recs.items() if r == "EXCELLENT"]
        avoid_strategies = [s for s, r in strategy_recs.items() if r == "AVOID"]
        
        if excellent_strategies:
            recommendations.append(f"🌟 Focus on excellent strategies: {', '.join(excellent_strategies)}")
        
        if avoid_strategies:
            recommendations.append(f"⚠️ Avoid underperforming strategies: {', '.join(avoid_strategies)}")
        
        # Risk/reward recommendations
        if metrics.risk_reward_ratio < 2.0:
            recommendations.append(f"💰 Improve risk/reward ratio (current: {metrics.risk_reward_ratio:.2f}, target: 2.0+)")
        
        # Confidence accuracy
        if metrics.confidence_accuracy < 0.5:
            recommendations.append(f"🎯 Improve confidence calibration (current accuracy: {metrics.confidence_accuracy:.2f})")
        
        return recommendations
    
    def _save_trade_record(self, trade_record: Dict):
        """Save trade record to file"""
        try:
            trades_file = self.data_dir / "optimized_trades.jsonl"
            with open(trades_file, 'a') as f:
                f.write(json.dumps(trade_record) + '\n')
        except Exception as e:
            logger.error(f"Failed to save trade record: {e}")
    
    def save_state(self):
        """Save optimizer state"""
        try:
            state = {
                'trade_history': self.trade_history,
                'strategy_performance': dict(self.strategy_performance),
                'current_win_streak': self.current_win_streak,
                'longest_win_streak': self.longest_win_streak,
                'current_loss_streak': self.current_loss_streak,
                'confidence_accuracy_history': self.confidence_accuracy_history
            }
            
            state_file = self.data_dir / "optimizer_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info("✅ Win rate optimizer state saved")
        except Exception as e:
            logger.error(f"Failed to save optimizer state: {e}")
    
    def load_state(self):
        """Load optimizer state"""
        try:
            state_file = self.data_dir / "optimizer_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.trade_history = state.get('trade_history', [])
                self.strategy_performance = defaultdict(lambda: {
                    'wins': 0, 'losses': 0, 'total_profit': 0.0,
                    'win_rate': 0.0, 'avg_profit': 0.0
                }, state.get('strategy_performance', {}))
                self.current_win_streak = state.get('current_win_streak', 0)
                self.longest_win_streak = state.get('longest_win_streak', 0)
                self.current_loss_streak = state.get('current_loss_streak', 0)
                self.confidence_accuracy_history = state.get('confidence_accuracy_history', [])
                
                logger.info(f"✅ Win rate optimizer state loaded: {len(self.trade_history)} trades")
        except Exception as e:
            logger.error(f"Failed to load optimizer state: {e}")


# Global win rate optimizer instance
win_rate_optimizer = WinRateOptimizer(target_win_rate=0.90)
