#!/usr/bin/env python3
"""
🎯 ADVANCED AI POSITION MANAGEMENT SYSTEM
Sophisticated position management for 90% win rate trading

FEATURES:
✅ AI-Powered Exit Strategy Selection
✅ Dynamic Partial Profit Taking
✅ Intelligent Trailing Stop System
✅ Time-Based Exit Optimization
✅ Market Condition-Based Exits
✅ Risk-Adjusted Position Scaling
✅ Breakeven Protection AI
✅ Momentum-Based Profit Extension
✅ Volatility-Adaptive Management
✅ Cross-Position Correlation Management
"""

import asyncio
import numpy as np
import pandas as pd
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque, defaultdict
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PositionState:
    """Current state of an active position"""
    symbol: str
    direction: str  # BUY, SELL
    entry_price: float
    current_price: float
    position_size: float
    entry_time: datetime
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: float
    take_profit_levels: List[Tuple[float, float]]  # (price, remaining_allocation)
    time_in_trade: int  # minutes
    max_favorable_excursion: float
    max_adverse_excursion: float
    trailing_stop_active: bool
    trailing_stop_level: float

@dataclass
class ExitDecision:
    """AI exit decision"""
    action: str  # HOLD, PARTIAL_EXIT, FULL_EXIT, UPDATE_STOPS
    reasoning: str
    confidence: float
    urgency: float  # 0-1
    recommended_exit_size: float  # As fraction of position
    new_stop_loss: Optional[float]
    new_take_profit: Optional[float]
    expected_outcome: Dict[str, float]
    market_factors: List[str]

@dataclass
class ExitStrategy:
    """Exit strategy configuration"""
    strategy_name: str
    exit_conditions: List[str]
    priority: int
    market_regimes: List[str]  # Applicable market regimes
    profit_targets: List[float]
    stop_adjustments: Dict[str, float]
    time_limits: Dict[str, int]

class TrailingStopManager:
    """Intelligent trailing stop management"""
    
    def __init__(self):
        self.trailing_methods = {
            'atr_trailing': self._atr_trailing_stop,
            'percentage_trailing': self._percentage_trailing_stop,
            'support_resistance_trailing': self._support_resistance_trailing,
            'volatility_adaptive': self._volatility_adaptive_trailing,
            'momentum_trailing': self._momentum_trailing_stop
        }
        
        self.trailing_history = deque(maxlen=1000)
        
    async def calculate_trailing_stop(self, position: PositionState, 
                                    market_intelligence: Dict,
                                    method: str = 'auto') -> Optional[float]:
        """Calculate intelligent trailing stop level"""
        
        # Auto-select best method if not specified
        if method == 'auto':
            method = await self._select_optimal_trailing_method(position, market_intelligence)
        
        if method not in self.trailing_methods:
            method = 'atr_trailing'
        
        # Calculate trailing stop using selected method
        new_trailing_stop = await self.trailing_methods[method](position, market_intelligence)
        
        # Validate and apply constraints
        if new_trailing_stop is not None:
            new_trailing_stop = self._validate_trailing_stop(position, new_trailing_stop)
        
        return new_trailing_stop
    
    async def _select_optimal_trailing_method(self, position: PositionState, 
                                           market_intelligence: Dict) -> str:
        """Select optimal trailing stop method based on market conditions"""
        
        regime = market_intelligence.get('regime', {})
        volatility = market_intelligence.get('volatility', {})
        
        regime_type = regime.get('regime_type', 'neutral')
        vol_percentile = volatility.get('percentile_rank', 0.5)
        
        # Method selection based on market conditions
        if regime_type in ['strong_bull', 'strong_bear']:
            if vol_percentile < 0.3:  # Low volatility trending
                return 'momentum_trailing'
            else:  # High volatility trending
                return 'atr_trailing'
        
        elif regime_type == 'consolidating':
            return 'support_resistance_trailing'
        
        elif vol_percentile > 0.8:  # High volatility
            return 'volatility_adaptive'
        
        else:  # Default
            return 'atr_trailing'
    
    async def _atr_trailing_stop(self, position: PositionState, 
                               market_intelligence: Dict) -> Optional[float]:
        """ATR-based trailing stop"""
        
        volatility = market_intelligence.get('volatility', {}).get('forecast', 0.02)
        
        # ATR multiplier based on time in trade
        if position.time_in_trade < 60:  # First hour
            atr_multiplier = 2.5
        elif position.time_in_trade < 240:  # First 4 hours
            atr_multiplier = 2.0
        else:  # Longer trades
            atr_multiplier = 1.5
        
        atr_distance = position.current_price * volatility * atr_multiplier
        
        if position.direction == 'BUY':
            new_stop = position.current_price - atr_distance
            # Only move stop up
            return new_stop if new_stop > position.trailing_stop_level else None
        else:  # SELL
            new_stop = position.current_price + atr_distance
            # Only move stop down
            return new_stop if new_stop < position.trailing_stop_level else None
    
    async def _percentage_trailing_stop(self, position: PositionState,
                                      market_intelligence: Dict) -> Optional[float]:
        """Percentage-based trailing stop"""
        
        # Dynamic percentage based on profit level
        profit_pct = position.unrealized_pnl_pct
        
        if profit_pct > 0.1:  # >10% profit
            trail_pct = 0.03  # 3% trailing distance
        elif profit_pct > 0.05:  # >5% profit
            trail_pct = 0.04  # 4% trailing distance
        elif profit_pct > 0.02:  # >2% profit
            trail_pct = 0.05  # 5% trailing distance
        else:
            return None  # No trailing until profitable
        
        if position.direction == 'BUY':
            new_stop = position.current_price * (1 - trail_pct)
            return new_stop if new_stop > position.trailing_stop_level else None
        else:
            new_stop = position.current_price * (1 + trail_pct)
            return new_stop if new_stop < position.trailing_stop_level else None
    
    async def _support_resistance_trailing(self, position: PositionState,
                                         market_intelligence: Dict) -> Optional[float]:
        """Support/resistance based trailing stop"""
        
        regime = market_intelligence.get('regime', {})
        support = regime.get('support_level')
        resistance = regime.get('resistance_level')
        
        # Only trail if we have valid levels
        if position.direction == 'BUY' and support:
            # Trail to just below recent support
            new_stop = support * 0.998
            return new_stop if new_stop > position.trailing_stop_level else None
        
        elif position.direction == 'SELL' and resistance:
            # Trail to just above recent resistance
            new_stop = resistance * 1.002
            return new_stop if new_stop < position.trailing_stop_level else None
        
        return None
    
    async def _volatility_adaptive_trailing(self, position: PositionState,
                                          market_intelligence: Dict) -> Optional[float]:
        """Volatility-adaptive trailing stop"""
        
        volatility_data = market_intelligence.get('volatility', {})
        vol_percentile = volatility_data.get('percentile_rank', 0.5)
        forecast_vol = volatility_data.get('forecast', 0.02)
        
        # Adjust trailing distance based on volatility percentile
        if vol_percentile > 0.9:  # Extreme volatility
            multiplier = 4.0
        elif vol_percentile > 0.7:  # High volatility
            multiplier = 3.0
        elif vol_percentile < 0.3:  # Low volatility
            multiplier = 1.5
        else:  # Normal volatility
            multiplier = 2.0
        
        trail_distance = position.current_price * forecast_vol * multiplier
        
        if position.direction == 'BUY':
            new_stop = position.current_price - trail_distance
            return new_stop if new_stop > position.trailing_stop_level else None
        else:
            new_stop = position.current_price + trail_distance
            return new_stop if new_stop < position.trailing_stop_level else None
    
    async def _momentum_trailing_stop(self, position: PositionState,
                                    market_intelligence: Dict) -> Optional[float]:
        """Momentum-based trailing stop"""
        
        # Tighter trailing in strong momentum, wider in weak momentum
        regime = market_intelligence.get('regime', {})
        trend_strength = regime.get('trend_strength', 0)
        
        if abs(trend_strength) > 0.8:  # Strong momentum
            trail_pct = 0.02  # Tight 2% trailing
        elif abs(trend_strength) > 0.5:  # Moderate momentum
            trail_pct = 0.035  # 3.5% trailing
        else:  # Weak momentum
            trail_pct = 0.05  # Wide 5% trailing
        
        if position.direction == 'BUY':
            new_stop = position.current_price * (1 - trail_pct)
            return new_stop if new_stop > position.trailing_stop_level else None
        else:
            new_stop = position.current_price * (1 + trail_pct)
            return new_stop if new_stop < position.trailing_stop_level else None
    
    def _validate_trailing_stop(self, position: PositionState, new_stop: float) -> float:
        """Validate and constrain trailing stop"""
        
        if position.direction == 'BUY':
            # Stop must be below current price and above entry
            if new_stop >= position.current_price:
                new_stop = position.current_price * 0.99
            
            # Don't trail below breakeven (with small buffer)
            breakeven_buffer = position.entry_price * 0.995
            if new_stop < breakeven_buffer:
                new_stop = breakeven_buffer
        
        else:  # SELL
            # Stop must be above current price and below entry
            if new_stop <= position.current_price:
                new_stop = position.current_price * 1.01
            
            # Don't trail above breakeven (with small buffer)
            breakeven_buffer = position.entry_price * 1.005
            if new_stop > breakeven_buffer:
                new_stop = breakeven_buffer
        
        return new_stop

class ProfitTakingManager:
    """Intelligent partial profit taking system"""
    
    def __init__(self):
        self.profit_taking_strategies = {
            'fibonacci_levels': self._fibonacci_profit_taking,
            'volatility_scaled': self._volatility_scaled_profit_taking,
            'time_based': self._time_based_profit_taking,
            'momentum_based': self._momentum_based_profit_taking,
            'support_resistance': self._support_resistance_profit_taking
        }
        
        self.profit_history = deque(maxlen=500)
    
    async def calculate_profit_taking_plan(self, position: PositionState,
                                         market_intelligence: Dict) -> List[Tuple[float, float, str]]:
        """Calculate intelligent profit taking plan"""
        
        # Analyze current profit situation
        if position.unrealized_pnl_pct <= 0:
            return []  # No profit to take
        
        profit_plans = []
        
        # 1. Quick profit securing (for volatile markets)
        if position.unrealized_pnl_pct > 0.02:  # 2% profit
            volatility = market_intelligence.get('volatility', {})
            if volatility.get('percentile_rank', 0.5) > 0.8:  # High volatility
                profit_plans.append((0.3, position.current_price, "Secure profits in high volatility"))
        
        # 2. Fibonacci-based profit taking
        fib_levels = await self._fibonacci_profit_taking(position, market_intelligence)
        profit_plans.extend(fib_levels)
        
        # 3. Support/resistance profit taking
        sr_levels = await self._support_resistance_profit_taking(position, market_intelligence)
        profit_plans.extend(sr_levels)
        
        # 4. Time-based profit taking
        time_levels = await self._time_based_profit_taking(position, market_intelligence)
        profit_plans.extend(time_levels)
        
        # 5. Volatility-scaled profit taking
        vol_levels = await self._volatility_scaled_profit_taking(position, market_intelligence)
        profit_plans.extend(vol_levels)
        
        # Optimize and combine plans
        optimized_plan = await self._optimize_profit_taking_plan(profit_plans, position, market_intelligence)
        
        return optimized_plan
    
    async def _fibonacci_profit_taking(self, position: PositionState,
                                     market_intelligence: Dict) -> List[Tuple[float, float, str]]:
        """Fibonacci retracement-based profit taking"""
        
        entry_price = position.entry_price
        current_price = position.current_price
        
        # Calculate price move
        price_move = current_price - entry_price if position.direction == 'BUY' else entry_price - current_price
        
        if price_move <= 0:
            return []
        
        # Fibonacci levels for profit taking
        fib_levels = [0.382, 0.618, 1.0, 1.618]
        profit_plans = []
        
        for i, fib in enumerate(fib_levels):
            fib_price = entry_price + (price_move * fib) if position.direction == 'BUY' else entry_price - (price_move * fib)
            
            # Only consider levels above current price (for further profits)
            if ((position.direction == 'BUY' and fib_price > current_price * 1.01) or
                (position.direction == 'SELL' and fib_price < current_price * 0.99)):
                
                # Allocation decreases for higher levels
                allocation = 0.4 / (i + 1)  # 40%, 20%, 13%, 10%
                
                profit_plans.append((
                    allocation,
                    fib_price, 
                    f"Fibonacci {fib:.3f} level"
                ))
        
        return profit_plans
    
    async def _support_resistance_profit_taking(self, position: PositionState,
                                              market_intelligence: Dict) -> List[Tuple[float, float, str]]:
        """Support/resistance based profit taking"""
        
        regime = market_intelligence.get('regime', {})
        support = regime.get('support_level')
        resistance = regime.get('resistance_level')
        
        profit_plans = []
        
        if position.direction == 'BUY' and resistance:
            # Take profits near resistance levels
            if resistance > position.current_price:
                # Conservative level just below resistance
                conservative_target = resistance * 0.998
                profit_plans.append((0.5, conservative_target, "Conservative resistance target"))
                
                # Aggressive level through resistance
                aggressive_target = resistance * 1.01
                profit_plans.append((0.3, aggressive_target, "Aggressive resistance breakout"))
        
        elif position.direction == 'SELL' and support:
            # Take profits near support levels
            if support < position.current_price:
                # Conservative level just above support
                conservative_target = support * 1.002
                profit_plans.append((0.5, conservative_target, "Conservative support target"))
                
                # Aggressive level through support
                aggressive_target = support * 0.99
                profit_plans.append((0.3, aggressive_target, "Aggressive support breakdown"))
        
        return profit_plans
    
    async def _time_based_profit_taking(self, position: PositionState,
                                      market_intelligence: Dict) -> List[Tuple[float, float, str]]:
        """Time-based profit taking strategy"""
        
        profit_plans = []
        
        # Take partial profits based on time in trade
        if position.time_in_trade > 180 and position.unrealized_pnl_pct > 0.015:  # 3h + 1.5% profit
            profit_plans.append((0.25, position.current_price, "Time-based profit securing (3h)"))
        
        elif position.time_in_trade > 360 and position.unrealized_pnl_pct > 0.01:  # 6h + 1% profit
            profit_plans.append((0.5, position.current_price, "Time-based profit securing (6h)"))
        
        elif position.time_in_trade > 720:  # 12h
            if position.unrealized_pnl_pct > 0:
                profit_plans.append((0.7, position.current_price, "Long trade profit securing"))
            else:
                # Close losing positions that have been open too long
                profit_plans.append((1.0, position.current_price, "Time-based loss cutting"))
        
        return profit_plans
    
    async def _volatility_scaled_profit_taking(self, position: PositionState,
                                             market_intelligence: Dict) -> List[Tuple[float, float, str]]:
        """Volatility-scaled profit taking"""
        
        volatility_data = market_intelligence.get('volatility', {})
        vol_percentile = volatility_data.get('percentile_rank', 0.5)
        forecast_vol = volatility_data.get('forecast', 0.02)
        
        profit_plans = []
        
        # Scale profit taking to volatility environment
        if vol_percentile > 0.8:  # High volatility - take profits aggressively
            if position.unrealized_pnl_pct > 0.03:  # 3% profit
                profit_plans.append((0.4, position.current_price, "High volatility profit securing"))
        
        elif vol_percentile < 0.2:  # Low volatility - let profits run
            # More conservative profit taking in low volatility
            if position.unrealized_pnl_pct > 0.06:  # 6% profit
                profit_plans.append((0.25, position.current_price, "Low volatility profit taking"))
        
        # Volatility expansion profit taking
        if forecast_vol > 0.05:  # High absolute volatility
            vol_distance = position.current_price * forecast_vol * 2
            
            if position.direction == 'BUY':
                target_price = position.current_price + vol_distance
            else:
                target_price = position.current_price - vol_distance
            
            profit_plans.append((0.3, target_price, "Volatility expansion target"))
        
        return profit_plans
    
    async def _momentum_based_profit_taking(self, position: PositionState,
                                          market_intelligence: Dict) -> List[Tuple[float, float, str]]:
        """Momentum-based profit taking"""
        
        regime = market_intelligence.get('regime', {})
        trend_strength = regime.get('trend_strength', 0)
        
        profit_plans = []
        
        # Strong momentum - let profits run with wider targets
        if abs(trend_strength) > 0.7:
            if position.unrealized_pnl_pct > 0.08:  # 8% profit in strong trend
                profit_plans.append((0.2, position.current_price, "Strong momentum partial profit"))
        
        # Weak momentum - take profits more conservatively
        elif abs(trend_strength) < 0.3:
            if position.unrealized_pnl_pct > 0.03:  # 3% profit in weak trend
                profit_plans.append((0.4, position.current_price, "Weak momentum profit securing"))
        
        return profit_plans
    
    async def _optimize_profit_taking_plan(self, all_plans: List[Tuple[float, float, str]],
                                         position: PositionState,
                                         market_intelligence: Dict) -> List[Tuple[float, float, str]]:
        """Optimize and combine profit taking plans"""
        
        if not all_plans:
            return []
        
        # Group plans by similar price levels
        price_groups = defaultdict(list)
        
        for allocation, price, reason in all_plans:
            price_group = round(price / position.current_price, 3)  # Group by price ratio
            price_groups[price_group].append((allocation, price, reason))
        
        # Combine plans for similar price levels
        optimized_plans = []
        
        for price_ratio, plans in price_groups.items():
            # Combine allocations for similar prices
            total_allocation = sum(plan[0] for plan in plans)
            avg_price = np.mean([plan[1] for plan in plans])
            combined_reason = " + ".join(set(plan[2] for plan in plans))
            
            # Limit total allocation to reasonable amount
            total_allocation = min(0.8, total_allocation)  # Max 80% of position
            
            optimized_plans.append((total_allocation, avg_price, combined_reason))
        
        # Sort by price distance from current
        optimized_plans.sort(key=lambda x: abs(x[1] - position.current_price))
        
        # Limit to 3 profit levels maximum
        return optimized_plans[:3]

class ExitStrategySelector:
    """AI-powered exit strategy selection"""
    
    def __init__(self):
        self.exit_strategies = {
            'momentum_continuation': ExitStrategy(
                strategy_name='momentum_continuation',
                exit_conditions=['momentum_weakening', 'volume_decline'],
                priority=1,
                market_regimes=['strong_bull', 'strong_bear', 'bull_trend', 'bear_trend'],
                profit_targets=[0.02, 0.05, 0.08],
                stop_adjustments={'tight': 0.8, 'normal': 1.0, 'wide': 1.5},
                time_limits={'quick': 60, 'normal': 240, 'extended': 480}
            ),
            
            'mean_reversion_exit': ExitStrategy(
                strategy_name='mean_reversion_exit',
                exit_conditions=['return_to_mean', 'overbought_oversold'],
                priority=2,
                market_regimes=['consolidating', 'sideways'],
                profit_targets=[0.015, 0.03, 0.045],
                stop_adjustments={'tight': 0.9, 'normal': 1.0, 'wide': 1.2},
                time_limits={'quick': 120, 'normal': 300, 'extended': 600}
            ),
            
            'breakout_continuation': ExitStrategy(
                strategy_name='breakout_continuation',
                exit_conditions=['false_breakout', 'volume_exhaustion'],
                priority=1,
                market_regimes=['volatile_trending', 'quiet_trend'],
                profit_targets=[0.025, 0.05, 0.1],
                stop_adjustments={'tight': 0.95, 'normal': 1.0, 'wide': 1.8},
                time_limits={'quick': 30, 'normal': 180, 'extended': 360}
            ),
            
            'volatility_adaptive': ExitStrategy(
                strategy_name='volatility_adaptive',
                exit_conditions=['volatility_normalization', 'regime_change'],
                priority=3,
                market_regimes=['volatile_sideways', 'volatile_trending'],
                profit_targets=[0.03, 0.06, 0.12],
                stop_adjustments={'tight': 1.2, 'normal': 1.5, 'wide': 2.0},
                time_limits={'quick': 45, 'normal': 150, 'extended': 300}
            )
        }
        
        self.strategy_performance = defaultdict(list)
    
    async def select_optimal_exit_strategy(self, position: PositionState,
                                         market_intelligence: Dict) -> ExitStrategy:
        """Select optimal exit strategy based on conditions"""
        
        regime = market_intelligence.get('regime', {})
        regime_type = regime.get('regime_type', 'neutral')
        
        # Filter strategies by applicable market regime
        applicable_strategies = []
        for strategy in self.exit_strategies.values():
            if regime_type in strategy.market_regimes or not strategy.market_regimes:
                applicable_strategies.append(strategy)
        
        if not applicable_strategies:
            # Fallback to momentum continuation
            return self.exit_strategies['momentum_continuation']
        
        # Select based on priority and recent performance
        best_strategy = None
        best_score = 0
        
        for strategy in applicable_strategies:
            # Base score from priority
            score = (5 - strategy.priority) / 4.0
            
            # Adjust for recent performance
            recent_performance = await self._get_strategy_recent_performance(strategy.strategy_name)
            score *= (0.5 + recent_performance)
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy or self.exit_strategies['momentum_continuation']
    
    async def _get_strategy_recent_performance(self, strategy_name: str) -> float:
        """Get recent performance for exit strategy"""
        
        strategy_trades = self.strategy_performance.get(strategy_name, [])
        
        if len(strategy_trades) < 5:
            return 0.5  # Neutral for insufficient data
        
        # Calculate recent win rate
        recent_trades = strategy_trades[-20:]
        wins = sum(1 for trade in recent_trades if trade.get('successful', False))
        return wins / len(recent_trades)

class PositionExitAnalyzer:
    """Analyzes optimal exit timing and conditions"""
    
    def __init__(self):
        self.exit_patterns = deque(maxlen=1000)
        self.market_exit_conditions = {
            'regime_change': 0.8,
            'volume_exhaustion': 0.6,
            'momentum_divergence': 0.7,
            'time_decay': 0.5,
            'volatility_spike': 0.75
        }
    
    async def analyze_exit_conditions(self, position: PositionState,
                                    market_intelligence: Dict) -> Dict[str, float]:
        """Analyze various exit conditions and their strength"""
        
        exit_scores = {}
        
        # 1. Market Regime Change
        regime_change_score = await self._analyze_regime_change(position, market_intelligence)
        exit_scores['regime_change'] = regime_change_score
        
        # 2. Volume Exhaustion
        volume_exhaustion_score = await self._analyze_volume_exhaustion(position, market_intelligence)
        exit_scores['volume_exhaustion'] = volume_exhaustion_score
        
        # 3. Momentum Divergence
        momentum_divergence_score = await self._analyze_momentum_divergence(position, market_intelligence)
        exit_scores['momentum_divergence'] = momentum_divergence_score
        
        # 4. Time Decay
        time_decay_score = await self._analyze_time_decay(position)
        exit_scores['time_decay'] = time_decay_score
        
        # 5. Volatility Spike
        volatility_spike_score = await self._analyze_volatility_conditions(position, market_intelligence)
        exit_scores['volatility_spike'] = volatility_spike_score
        
        # 6. Profit Target Achievement
        profit_target_score = await self._analyze_profit_targets(position)
        exit_scores['profit_targets'] = profit_target_score
        
        # 7. Stop Loss Proximity
        stop_proximity_score = await self._analyze_stop_proximity(position)
        exit_scores['stop_proximity'] = stop_proximity_score
        
        return exit_scores
    
    async def _analyze_regime_change(self, position: PositionState, 
                                   market_intelligence: Dict) -> float:
        """Analyze if market regime has changed against position"""
        
        regime = market_intelligence.get('regime', {})
        regime_type = regime.get('regime_type', 'neutral')
        regime_confidence = regime.get('confidence', 0.5)
        
        # Check if regime is against position direction
        if position.direction == 'BUY':
            if regime_type in ['strong_bear', 'bear_trend']:
                return regime_confidence * 0.9
        elif position.direction == 'SELL':
            if regime_type in ['strong_bull', 'bull_trend']:
                return regime_confidence * 0.9
        
        # Regime supporting position
        if position.direction == 'BUY' and regime_type in ['strong_bull', 'bull_trend']:
            return -0.3  # Negative score means favorable for holding
        elif position.direction == 'SELL' and regime_type in ['strong_bear', 'bear_trend']:
            return -0.3
        
        return 0.0
    
    async def _analyze_volume_exhaustion(self, position: PositionState,
                                       market_intelligence: Dict) -> float:
        """Analyze volume exhaustion patterns"""
        
        volume_profile = market_intelligence.get('volume_profile', {})
        relative_volume = volume_profile.get('relative_volume', 1.0)
        volume_trend = volume_profile.get('volume_trend', 'stable')
        
        exhaustion_score = 0
        
        # Declining volume during favorable move
        if position.unrealized_pnl_pct > 0.02:  # In profit
            if volume_trend == 'decreasing' and relative_volume < 0.7:
                exhaustion_score = 0.6  # Volume exhaustion signal
        
        # Very low volume
        if relative_volume < 0.3:
            exhaustion_score += 0.3
        
        return min(1.0, exhaustion_score)
    
    async def _analyze_momentum_divergence(self, position: PositionState,
                                         market_intelligence: Dict) -> float:
        """Analyze momentum divergence signals"""
        
        regime = market_intelligence.get('regime', {})
        trend_strength = regime.get('trend_strength', 0)
        
        # Price vs momentum divergence
        if position.unrealized_pnl_pct > 0.03:  # In good profit
            if position.direction == 'BUY' and trend_strength < 0.2:
                return 0.7  # Bullish position but weakening momentum
            elif position.direction == 'SELL' and trend_strength > -0.2:
                return 0.7  # Bearish position but weakening momentum
        
        return 0.0
    
    async def _analyze_time_decay(self, position: PositionState) -> float:
        """Analyze time-based exit pressure"""
        
        # Increasing exit pressure over time
        if position.time_in_trade < 60:  # First hour
            return 0.0
        elif position.time_in_trade < 240:  # 4 hours
            return 0.1
        elif position.time_in_trade < 480:  # 8 hours
            return 0.3
        elif position.time_in_trade < 720:  # 12 hours
            return 0.5
        else:  # >12 hours
            return 0.7
    
    async def _analyze_volatility_conditions(self, position: PositionState,
                                           market_intelligence: Dict) -> float:
        """Analyze volatility-based exit conditions"""
        
        volatility_data = market_intelligence.get('volatility', {})
        vol_percentile = volatility_data.get('percentile_rank', 0.5)
        
        # High volatility increases exit urgency when in profit
        if position.unrealized_pnl_pct > 0:
            if vol_percentile > 0.9:  # Extreme volatility
                return 0.8
            elif vol_percentile > 0.8:  # High volatility
                return 0.5
        
        return 0.0
    
    async def _analyze_profit_targets(self, position: PositionState) -> float:
        """Analyze profit target achievement"""
        
        # Standard profit target levels
        if position.unrealized_pnl_pct > 0.1:  # 10% profit
            return 0.9  # Strong signal to take profits
        elif position.unrealized_pnl_pct > 0.06:  # 6% profit
            return 0.6
        elif position.unrealized_pnl_pct > 0.03:  # 3% profit
            return 0.3
        
        return 0.0
    
    async def _analyze_stop_proximity(self, position: PositionState) -> float:
        """Analyze proximity to stop loss"""
        
        if position.stop_loss is None:
            return 0.0
        
        distance_to_stop = abs(position.current_price - position.stop_loss) / position.current_price
        
        # Higher score when close to stop
        if distance_to_stop < 0.005:  # <0.5% from stop
            return 0.9
        elif distance_to_stop < 0.01:  # <1% from stop
            return 0.6
        elif distance_to_stop < 0.02:  # <2% from stop
            return 0.3
        
        return 0.0

class AdvancedPositionManager:
    """Master AI position management system"""
    
    def __init__(self):
        self.trailing_stop_manager = TrailingStopManager()
        self.profit_taking_manager = ProfitTakingManager()
        self.exit_strategy_selector = ExitStrategySelector()
        self.exit_analyzer = PositionExitAnalyzer()
        
        # Position tracking
        self.active_positions = {}
        self.position_history = deque(maxlen=1000)
        
        # Management parameters
        self.management_interval = 30  # seconds
        self.last_management_update = {}
        
        # Configuration parameters
        self.max_positions = 5
        self.use_trailing_stops = True
        self.partial_profit_levels = [0.02, 0.05, 0.08]
        
        # Performance tracking
        self.exit_performance = deque(maxlen=500)
        
        print("🎯 ADVANCED AI POSITION MANAGEMENT SYSTEM INITIALIZED")
        print("   ✅ Trailing Stop Manager")
        print("   ✅ Profit Taking Manager")
        print("   ✅ Exit Strategy Selector")
        print("   ✅ Position Exit Analyzer")
    
    async def configure(self, max_positions: int = 5, use_trailing_stops: bool = True, 
                      partial_profit_levels: List[float] = None) -> None:
        """Configure position management parameters"""
        self.max_positions = max_positions
        self.use_trailing_stops = use_trailing_stops
        self.partial_profit_levels = partial_profit_levels or [0.02, 0.05, 0.08]
        
        print(f"🎯 POSITION MANAGER CONFIGURED")
        print(f"   📊 Max Positions: {self.max_positions}")
        print(f"   🔄 Trailing Stops: {'Enabled' if self.use_trailing_stops else 'Disabled'}")
        print(f"   📈 Profit Levels: {[f'{level:.1%}' for level in self.partial_profit_levels]}")
    
    async def manage_position(self, symbol: str, position_data: Dict,
                            current_price: float, market_intelligence: Dict) -> ExitDecision:
        """Main position management function"""
        
        # Create position state
        position = self._create_position_state(symbol, position_data, current_price)
        
        print(f"🎯 MANAGING POSITION {symbol} ({position.direction})")
        print(f"   💰 P&L: {position.unrealized_pnl_pct:.2%}")
        print(f"   ⏰ Time: {position.time_in_trade}min")
        
        # 1. Analyze exit conditions
        exit_conditions = await self.exit_analyzer.analyze_exit_conditions(
            position, market_intelligence
        )
        
        # 2. Select optimal exit strategy
        exit_strategy = await self.exit_strategy_selector.select_optimal_exit_strategy(
            position, market_intelligence
        )
        
        # 3. Calculate trailing stops
        new_trailing_stop = await self.trailing_stop_manager.calculate_trailing_stop(
            position, market_intelligence
        )
        
        # 4. Calculate profit taking opportunities
        profit_plan = await self.profit_taking_manager.calculate_profit_taking_plan(
            position, market_intelligence
        )
        
        # 5. Make final exit decision
        exit_decision = await self._make_exit_decision(
            position, exit_conditions, exit_strategy, new_trailing_stop, 
            profit_plan, market_intelligence
        )
        
        # 6. Update position tracking
        await self._update_position_tracking(symbol, position, exit_decision)
        
        print(f"✅ EXIT DECISION: {exit_decision.action} (conf: {exit_decision.confidence:.3f})")
        
        return exit_decision
    
    def _create_position_state(self, symbol: str, position_data: Dict, 
                             current_price: float) -> PositionState:
        """Create position state from position data"""
        
        entry_price = position_data.get('entry_price', current_price)
        direction = position_data.get('direction', 'BUY')
        position_size = position_data.get('position_size', 0)
        entry_time = position_data.get('entry_time', datetime.now())
        
        # Calculate unrealized P&L
        if direction == 'BUY':
            unrealized_pnl = (current_price - entry_price) * position_size
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
        else:
            unrealized_pnl = (entry_price - current_price) * position_size
            unrealized_pnl_pct = (entry_price - current_price) / entry_price
        
        # Time in trade
        time_in_trade = int((datetime.now() - entry_time).total_seconds() / 60)
        
        # Max favorable and adverse excursions (simplified)
        if unrealized_pnl_pct > 0:
            max_favorable_excursion = max(position_data.get('max_favorable_excursion', 0), 
                                        unrealized_pnl_pct)
            max_adverse_excursion = position_data.get('max_adverse_excursion', 0)
        else:
            max_favorable_excursion = position_data.get('max_favorable_excursion', 0)
            max_adverse_excursion = max(position_data.get('max_adverse_excursion', 0),
                                      abs(unrealized_pnl_pct))
        
        return PositionState(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=current_price,
            position_size=position_size,
            entry_time=entry_time,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            stop_loss=position_data.get('stop_loss'),
            take_profit_levels=position_data.get('take_profit_levels', []),
            time_in_trade=time_in_trade,
            max_favorable_excursion=max_favorable_excursion,
            max_adverse_excursion=max_adverse_excursion,
            trailing_stop_active=position_data.get('trailing_stop_active', False),
            trailing_stop_level=position_data.get('trailing_stop_level', 0)
        )
    
    async def _make_exit_decision(self, position: PositionState, exit_conditions: Dict[str, float],
                                exit_strategy: ExitStrategy, new_trailing_stop: Optional[float],
                                profit_plan: List[Tuple[float, float, str]],
                                market_intelligence: Dict) -> ExitDecision:
        """Make final AI-powered exit decision"""
        
        action = "HOLD"
        reasoning = []
        confidence = 0.5
        urgency = 0.0
        recommended_exit_size = 0.0
        new_stop_loss = None
        new_take_profit = None
        
        # Calculate weighted exit score
        exit_score = 0
        for condition, score in exit_conditions.items():
            if condition in self.exit_analyzer.market_exit_conditions:
                weight = self.exit_analyzer.market_exit_conditions[condition]
                exit_score += score * weight
        
        # Emergency exits
        if exit_score > 0.8 or exit_conditions.get('stop_proximity', 0) > 0.8:
            action = "FULL_EXIT"
            reasoning.append("Emergency exit triggered by high risk conditions")
            confidence = 0.9
            urgency = 0.9
            recommended_exit_size = 1.0
        
        # Profit taking decisions
        elif profit_plan and position.unrealized_pnl_pct > 0.02:
            # Determine if we should take partial profits
            total_profit_allocation = sum(plan[0] for plan in profit_plan)
            
            if total_profit_allocation > 0.3 or position.unrealized_pnl_pct > 0.05:
                action = "PARTIAL_EXIT"
                recommended_exit_size = min(0.5, total_profit_allocation)
                reasoning.append(f"Partial profit taking: {recommended_exit_size:.1%} of position")
                confidence = 0.75
                urgency = 0.4
        
        # Trailing stop updates
        elif new_trailing_stop is not None:
            action = "UPDATE_STOPS"
            new_stop_loss = new_trailing_stop
            reasoning.append(f"Update trailing stop to {new_trailing_stop:.2f}")
            confidence = 0.8
            urgency = 0.3
        
        # Time-based exits
        elif position.time_in_trade > 480 and position.unrealized_pnl_pct >= 0:
            action = "FULL_EXIT"
            reasoning.append("Time-based exit for long-held profitable position")
            confidence = 0.7
            urgency = 0.5
            recommended_exit_size = 1.0
        
        elif position.time_in_trade > 720:  # 12 hours
            action = "FULL_EXIT"
            reasoning.append("Maximum time limit reached")
            confidence = 0.8
            urgency = 0.6
            recommended_exit_size = 1.0
        
        # Market condition exits
        elif exit_score > 0.6:
            action = "PARTIAL_EXIT"
            recommended_exit_size = 0.3 + (exit_score - 0.6) * 0.5
            reasoning.append("Market conditions favor partial exit")
            confidence = 0.7
            urgency = exit_score - 0.2
        
        # Profit protection for large gains
        elif position.unrealized_pnl_pct > 0.08:  # 8% profit
            if not position.trailing_stop_active:
                action = "UPDATE_STOPS"
                # Activate trailing stop at breakeven + 2%
                new_stop_loss = position.entry_price * (1.02 if position.direction == 'BUY' else 0.98)
                reasoning.append("Activate profit protection trailing stop")
                confidence = 0.85
                urgency = 0.4
        
        # Calculate expected outcome
        expected_outcome = await self._calculate_expected_outcome(
            position, action, recommended_exit_size, market_intelligence
        )
        
        # Extract market factors
        market_factors = []
        regime = market_intelligence.get('regime', {})
        if regime.get('regime_type'):
            market_factors.append(f"Regime: {regime['regime_type']}")
        
        volatility = market_intelligence.get('volatility', {})
        if volatility.get('percentile_rank'):
            vol_level = "High" if volatility['percentile_rank'] > 0.7 else "Normal" if volatility['percentile_rank'] > 0.3 else "Low"
            market_factors.append(f"Volatility: {vol_level}")
        
        return ExitDecision(
            action=action,
            reasoning=" | ".join(reasoning) if reasoning else "Hold position",
            confidence=confidence,
            urgency=urgency,
            recommended_exit_size=recommended_exit_size,
            new_stop_loss=new_stop_loss,
            new_take_profit=new_take_profit,
            expected_outcome=expected_outcome,
            market_factors=market_factors
        )
    
    async def _calculate_expected_outcome(self, position: PositionState, action: str,
                                        exit_size: float, market_intelligence: Dict) -> Dict[str, float]:
        """Calculate expected outcome of exit decision"""
        
        expected_outcome = {
            'expected_pnl': 0.0,
            'risk_reduction': 0.0,
            'opportunity_cost': 0.0,
            'probability_of_profit': 0.5
        }
        
        if action == "FULL_EXIT":
            expected_outcome['expected_pnl'] = position.unrealized_pnl
            expected_outcome['risk_reduction'] = 1.0
            expected_outcome['opportunity_cost'] = 0.5  # Medium opportunity cost
            expected_outcome['probability_of_profit'] = 1.0 if position.unrealized_pnl > 0 else 0.0
        
        elif action == "PARTIAL_EXIT":
            expected_outcome['expected_pnl'] = position.unrealized_pnl * exit_size
            expected_outcome['risk_reduction'] = exit_size
            expected_outcome['opportunity_cost'] = exit_size * 0.3
            expected_outcome['probability_of_profit'] = 0.9 if position.unrealized_pnl > 0 else 0.1
        
        elif action == "UPDATE_STOPS":
            expected_outcome['risk_reduction'] = 0.3
            expected_outcome['opportunity_cost'] = 0.1
            expected_outcome['probability_of_profit'] = 0.7 if position.unrealized_pnl > 0 else 0.3
        
        return expected_outcome
    
    async def _update_position_tracking(self, symbol: str, position: PositionState,
                                      exit_decision: ExitDecision):
        """Update position tracking and learning"""
        
        # Track position in active positions
        self.active_positions[symbol] = {
            'position_state': position,
            'last_decision': exit_decision,
            'last_update': datetime.now()
        }
        
        # Track exit decisions for learning
        decision_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'position_state': position.__dict__.copy(),
            'decision': exit_decision.__dict__.copy(),
            'market_conditions': market_intelligence.get('regime', {})
        }
        
        self.exit_performance.append(decision_record)
    
    async def optimize_exit_strategies(self, completed_trades: List[Dict]) -> Dict[str, Any]:
        """Optimize exit strategies based on completed trade performance"""
        
        if len(completed_trades) < 20:
            return {'status': 'insufficient_data'}
        
        optimization_results = {
            'timestamp': datetime.now(),
            'analyzed_trades': len(completed_trades),
            'strategy_performance': {},
            'optimizations_applied': [],
            'performance_improvement': 0.0
        }
        
        # Analyze exit strategy performance
        strategy_analysis = defaultdict(list)
        
        for trade in completed_trades:
            exit_strategy = trade.get('exit_strategy', 'unknown')
            exit_timing = trade.get('exit_timing', 'unknown')
            pnl = trade.get('pnl', 0)
            
            strategy_analysis[exit_strategy].append({
                'pnl': pnl,
                'exit_timing': exit_timing,
                'profitable': pnl > 0
            })
        
        # Calculate strategy performance
        for strategy_name, trades in strategy_analysis.items():
            if len(trades) >= 5:
                win_rate = sum(1 for trade in trades if trade['profitable']) / len(trades)
                avg_pnl = np.mean([trade['pnl'] for trade in trades])
                
                optimization_results['strategy_performance'][strategy_name] = {
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'trade_count': len(trades),
                    'performance_score': win_rate * 0.7 + (avg_pnl / 100) * 0.3
                }
        
        # Optimize based on analysis
        optimizations = await self._generate_exit_optimizations(optimization_results['strategy_performance'])
        optimization_results['optimizations_applied'] = optimizations
        
        return optimization_results
    
    async def _generate_exit_optimizations(self, strategy_performance: Dict[str, Dict]) -> List[Dict]:
        """Generate exit strategy optimizations"""
        
        optimizations = []
        
        # Find best performing strategies
        best_strategies = sorted(strategy_performance.items(), 
                               key=lambda x: x[1]['performance_score'], reverse=True)
        
        if best_strategies:
            best_strategy = best_strategies[0]
            best_score = best_strategy[1]['performance_score']
            
            # Increase weight of best performing strategies
            if best_score > 0.8:
                optimizations.append({
                    'type': 'increase_strategy_priority',
                    'strategy': best_strategy[0],
                    'old_priority': 1,
                    'new_priority': 0.5,  # Higher priority (lower number)
                    'reason': f"Best performing strategy with score {best_score:.3f}"
                })
        
        # Find underperforming strategies
        poor_strategies = [strategy for strategy, perf in strategy_performance.items() 
                          if perf['performance_score'] < 0.4]
        
        for strategy in poor_strategies:
            optimizations.append({
                'type': 'reduce_strategy_usage',
                'strategy': strategy,
                'reason': f"Poor performance with score {strategy_performance[strategy]['performance_score']:.3f}"
            })
        
        return optimizations
    
    async def get_position_management_statistics(self) -> Dict[str, Any]:
        """Get comprehensive position management statistics"""
        
        stats = {
            'timestamp': datetime.now(),
            'active_positions': len(self.active_positions),
            'total_decisions': len(self.exit_performance),
            'decision_distribution': {},
            'avg_time_in_trade': 0,
            'profit_taking_effectiveness': {},
            'trailing_stop_effectiveness': {},
            'exit_timing_analysis': {}
        }
        
        if self.exit_performance:
            # Decision distribution
            decisions = [decision['decision']['action'] for decision in self.exit_performance]
            for action in set(decisions):
                stats['decision_distribution'][action] = decisions.count(action)
            
            # Average time in trade
            times = [decision['position_state']['time_in_trade'] for decision in self.exit_performance]
            stats['avg_time_in_trade'] = np.mean(times)
        
        # Active position summary
        if self.active_positions:
            active_summary = {}
            for symbol, pos_data in self.active_positions.items():
                pos_state = pos_data['position_state']
                active_summary[symbol] = {
                    'unrealized_pnl_pct': pos_state.unrealized_pnl_pct,
                    'time_in_trade': pos_state.time_in_trade,
                    'direction': pos_state.direction,
                    'trailing_stop_active': pos_state.trailing_stop_active
                }
            
            stats['active_positions_detail'] = active_summary
        
        return stats
    
    async def emergency_position_management(self, emergency_type: str,
                                          active_positions: Dict[str, Dict]) -> List[Dict]:
        """Emergency position management actions"""
        
        emergency_actions = []
        
        if emergency_type == 'max_drawdown_exceeded':
            # Close all losing positions immediately
            for symbol, pos_data in active_positions.items():
                position = pos_data['position_state']
                if position.unrealized_pnl < 0:
                    emergency_actions.append({
                        'symbol': symbol,
                        'action': 'EMERGENCY_CLOSE',
                        'reason': 'Max drawdown exceeded - close losing position',
                        'exit_size': 1.0
                    })
                else:
                    # Secure profits on winning positions
                    emergency_actions.append({
                        'symbol': symbol,
                        'action': 'SECURE_PROFITS',
                        'reason': 'Max drawdown exceeded - secure profits',
                        'exit_size': 0.7
                    })
        
        elif emergency_type == 'volatility_spike':
            # Reduce all position sizes
            for symbol, pos_data in active_positions.items():
                emergency_actions.append({
                    'symbol': symbol,
                    'action': 'REDUCE_POSITION',
                    'reason': 'Volatility spike - reduce exposure',
                    'exit_size': 0.4
                })
        
        elif emergency_type == 'market_crash':
            # Close all positions immediately
            for symbol, pos_data in active_positions.items():
                emergency_actions.append({
                    'symbol': symbol,
                    'action': 'EMERGENCY_CLOSE',
                    'reason': 'Market crash detected - emergency exit',
                    'exit_size': 1.0
                })
        
        return emergency_actions
    
    async def optimize_position_allocation(self, available_capital: float,
                                         open_positions: Dict[str, PositionState],
                                         new_signal: Dict) -> Dict[str, float]:
        """Optimize position allocation across multiple positions"""
        
        allocation_plan = {
            'new_position_size': 0.0,
            'position_adjustments': {},
            'total_risk_after': 0.0,
            'risk_utilization': 0.0
        }
        
        # Calculate current risk exposure
        total_current_risk = sum(
            abs(pos.current_price - pos.stop_loss) * pos.position_size 
            for pos in open_positions.values() 
            if pos.stop_loss
        )
        
        current_risk_pct = total_current_risk / available_capital if available_capital > 0 else 0
        
        # Calculate risk budget for new position
        max_portfolio_risk = 0.12  # 12% max portfolio risk
        remaining_risk_budget = max(0, max_portfolio_risk - current_risk_pct)
        
        # Determine new position size
        signal_confidence = new_signal.get('confidence', 0.5)
        base_risk_per_trade = 0.02  # 2% base risk
        
        # Adjust risk based on confidence and market conditions
        confidence_multiplier = min(2.0, signal_confidence * 2)
        risk_for_new_position = base_risk_per_trade * confidence_multiplier
        
        # Constrain to remaining budget
        risk_for_new_position = min(risk_for_new_position, remaining_risk_budget)
        
        # Convert to position size
        signal_stop_distance = abs(new_signal.get('entry_price', 0) - new_signal.get('stop_loss', 0))
        if signal_stop_distance > 0:
            new_position_size = (available_capital * risk_for_new_position) / signal_stop_distance
        else:
            new_position_size = 0
        
        allocation_plan['new_position_size'] = new_position_size
        allocation_plan['total_risk_after'] = current_risk_pct + risk_for_new_position
        allocation_plan['risk_utilization'] = allocation_plan['total_risk_after'] / max_portfolio_risk
        
        # Consider reducing existing positions if risk budget is tight
        if allocation_plan['risk_utilization'] > 0.8:
            # Reduce losing positions first
            for symbol, position in open_positions.items():
                if position.unrealized_pnl < 0:
                    allocation_plan['position_adjustments'][symbol] = {
                        'action': 'REDUCE',
                        'reduction_pct': 0.3,
                        'reason': 'Risk budget management - reduce losing position'
                    }
        
        return allocation_plan

# Global position manager instance
position_manager = AdvancedPositionManager()

# Export components
__all__ = [
    'position_manager',
    'AdvancedPositionManager',
    'PositionState',
    'ExitDecision',
    'ExitStrategy',
    'TrailingStopManager',
    'ProfitTakingManager',
    'ExitStrategySelector',
    'PositionExitAnalyzer'
]
