#!/usr/bin/env python3
"""
🎯 ADVANCED ENTRY/EXIT OPTIMIZATION SYSTEM
Perfect timing for maximum win probability
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EntrySignal:
    """Optimized entry signal"""
    symbol: str
    entry_price: float
    optimal_entry_price: float
    confidence: float
    timing_score: float
    wait_for_better_entry: bool
    reason: str
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float

@dataclass
class ExitSignal:
    """Optimized exit signal"""
    symbol: str
    current_price: float
    optimal_exit_price: float
    should_exit: bool
    exit_type: str
    confidence: float
    reason: str
    expected_profit_pct: float

class AdvancedEntryExitOptimizer:
    """Advanced entry/exit optimization for maximum win probability"""
    
    def __init__(self):
        self.min_confidence_for_entry = 0.75
        self.min_risk_reward_ratio = 2.0
        self.optimal_rr_ratio = 3.0
        self.entry_patience_threshold = 0.5
        self.max_wait_time_minutes = 15
        self.trailing_stop_activation = 1.02
        self.trailing_stop_percentage = 0.005
        
        logger.info("🎯 Advanced Entry/Exit Optimizer initialized")
    
    def optimize_entry(self, symbol: str, signal_price: float, 
                      signal_confidence: float, market_data: Dict,
                      technical_indicators: Dict) -> EntrySignal:
        """Optimize entry point for maximum win probability"""
        
        current_price = market_data.get('price', signal_price)
        volatility = market_data.get('volatility', 0.02)
        
        support_levels = self._calculate_support_levels(market_data)
        resistance_levels = self._calculate_resistance_levels(market_data)
        
        optimal_entry = self._find_optimal_entry_price(
            current_price, support_levels, resistance_levels, technical_indicators
        )
        
        stop_loss = self._calculate_dynamic_stop_loss(optimal_entry, support_levels, volatility)
        take_profit = self._calculate_dynamic_take_profit(optimal_entry, resistance_levels, self.optimal_rr_ratio)
        
        risk = optimal_entry - stop_loss
        reward = take_profit - optimal_entry
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        timing_score = self._calculate_timing_score(current_price, optimal_entry, technical_indicators, market_data)
        
        price_difference_pct = abs(current_price - optimal_entry) / current_price
        wait_for_better_entry = (
            price_difference_pct > self.entry_patience_threshold / 100 and
            timing_score < 0.9 and signal_confidence > 0.80
        )
        
        if wait_for_better_entry:
            reason = f"Wait for {optimal_entry:.2f} for optimal R/R: {risk_reward_ratio:.2f}"
        elif timing_score >= 0.9:
            reason = f"Excellent timing {timing_score:.2f} - Enter now"
        else:
            reason = f"Good entry, R/R: {risk_reward_ratio:.2f}"
        
        return EntrySignal(
            symbol=symbol, entry_price=current_price, optimal_entry_price=optimal_entry,
            confidence=signal_confidence * timing_score, timing_score=timing_score,
            wait_for_better_entry=wait_for_better_entry, reason=reason,
            stop_loss=stop_loss, take_profit=take_profit, risk_reward_ratio=risk_reward_ratio
        )
    
    def optimize_exit(self, symbol: str, entry_price: float, current_price: float,
                     stop_loss: float, take_profit: float, market_data: Dict,
                     technical_indicators: Dict, hold_time_minutes: float) -> ExitSignal:
        """Optimize exit point to lock in maximum profit"""
        
        profit_pct = (current_price - entry_price) / entry_price
        should_exit = False
        exit_type = "HOLD"
        exit_reason = ""
        
        if current_price <= stop_loss:
            should_exit, exit_type = True, "STOP_LOSS"
            exit_reason = f"Stop loss triggered at {current_price:.2f}"
        elif current_price >= take_profit:
            should_exit, exit_type = True, "TAKE_PROFIT"
            exit_reason = f"Take profit reached: {profit_pct*100:.2f}%"
        elif profit_pct >= (self.trailing_stop_activation - 1):
            highest_price = market_data.get('highest_since_entry', current_price)
            trailing_stop_price = highest_price * (1 - self.trailing_stop_percentage)
            if current_price <= trailing_stop_price:
                should_exit, exit_type = True, "TRAILING_STOP"
                exit_reason = f"Trailing stop, profit: {profit_pct*100:.2f}%"
        
        optimal_exit_price = current_price if should_exit else take_profit
        confidence = 0.95 if exit_type in ["STOP_LOSS", "TAKE_PROFIT"] else 0.5
        
        if not should_exit:
            exit_reason = f"Hold, current: {profit_pct*100:+.2f}%"
        
        return ExitSignal(
            symbol=symbol, current_price=current_price, optimal_exit_price=optimal_exit_price,
            should_exit=should_exit, exit_type=exit_type, confidence=confidence,
            reason=exit_reason, expected_profit_pct=profit_pct * 100
        )
    
    def _calculate_support_levels(self, market_data: Dict) -> List[float]:
        price_history = market_data.get('price_history', [])
        if not price_history or len(price_history) < 20:
            current_price = market_data.get('price', 0)
            return [current_price * 0.98, current_price * 0.95]
        
        supports = []
        prices = np.array(price_history)
        for i in range(2, len(prices) - 2):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                supports.append(prices[i])
        return sorted(supports[-5:]) if supports else [prices[-1] * 0.98]
    
    def _calculate_resistance_levels(self, market_data: Dict) -> List[float]:
        price_history = market_data.get('price_history', [])
        if not price_history or len(price_history) < 20:
            current_price = market_data.get('price', 0)
            return [current_price * 1.02, current_price * 1.05]
        
        resistances = []
        prices = np.array(price_history)
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                resistances.append(prices[i])
        return sorted(resistances[-5:]) if resistances else [prices[-1] * 1.02]
    
    def _find_optimal_entry_price(self, current_price: float, support_levels: List[float],
                                  resistance_levels: List[float], technical_indicators: Dict) -> float:
        supports_below = [s for s in support_levels if s < current_price]
        if supports_below:
            return max(supports_below) * 1.001
        return current_price
    
    def _calculate_dynamic_stop_loss(self, entry_price: float, support_levels: List[float],
                                    volatility: float) -> float:
        supports_below = [s for s in support_levels if s < entry_price]
        if supports_below:
            stop_loss = max(supports_below) * 0.999
        else:
            stop_loss = entry_price * (1 - (2 * volatility))
        return min(stop_loss, entry_price * 0.99)
    
    def _calculate_dynamic_take_profit(self, entry_price: float, resistance_levels: List[float],
                                       target_rr_ratio: float) -> float:
        resistances_above = [r for r in resistance_levels if r > entry_price]
        if resistances_above:
            return min(resistances_above) * 0.999
        return entry_price * 1.03
    
    def _calculate_timing_score(self, current_price: float, optimal_entry: float,
                               technical_indicators: Dict, market_data: Dict) -> float:
        score = 0.5
        if abs(current_price - optimal_entry) / current_price < 0.002:
            score += 0.3
        if technical_indicators.get('rsi', 50) >= 30 and technical_indicators.get('rsi', 50) <= 70:
            score += 0.1
        if technical_indicators.get('trend_aligned', False):
            score += 0.1
        return min(score, 1.0)
