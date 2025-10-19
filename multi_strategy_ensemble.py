#!/usr/bin/env python3
"""
🎯 MULTI-STRATEGY ENSEMBLE TRADING SYSTEM
Advanced ensemble of multiple AI strategies for 90% win rate trading

FEATURES:
✅ Multiple Trading Strategy Models
✅ Dynamic Strategy Weighting
✅ Market Condition-Based Strategy Selection
✅ Performance-Based Strategy Allocation
✅ Real-time Strategy Optimization
✅ Cross-Strategy Signal Validation
✅ Adaptive Strategy Parameters
✅ Strategy Performance Tracking
✅ Ensemble Confidence Calculation
✅ Meta-Strategy Intelligence
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
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StrategySignal:
    """Individual strategy signal"""
    strategy_name: str
    direction: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    strength: float  # Signal strength 0-1
    timestamp: datetime
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    market_conditions: Dict[str, Any]
    risk_score: float  # 0-1

@dataclass
class EnsembleSignal:
    """Combined ensemble signal"""
    direction: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    strength: float  # 0-1
    contributing_strategies: List[str]
    strategy_weights: Dict[str, float]
    consensus_level: float  # How much strategies agree
    risk_adjusted_confidence: float
    market_regime_alignment: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: List[str]
    timestamp: datetime

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: float
    recent_performance: float  # Last 20 trades
    market_condition_performance: Dict[str, float]
    confidence_calibration: float
    current_weight: float

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.performance_history = deque(maxlen=500)
        self.signal_history = deque(maxlen=1000)
        self.parameters = {}
        self.is_active = True
        self.weight = 1.0
        
    @abstractmethod
    async def generate_signal(self, symbol: str, price_data: List[float], 
                            market_intelligence: Dict, current_price: float) -> StrategySignal:
        """Generate trading signal"""
        pass
    
    @abstractmethod
    def update_parameters(self, performance_data: Dict):
        """Update strategy parameters based on performance"""
        pass
    
    def calculate_performance_metrics(self) -> StrategyPerformance:
        """Calculate strategy performance metrics"""
        
        if not self.performance_history:
            return StrategyPerformance(
                strategy_name=self.name,
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_trade_duration=0.0,
                recent_performance=0.0,
                market_condition_performance={},
                confidence_calibration=0.5,
                current_weight=self.weight
            )
        
        trades = list(self.performance_history)
        total_trades = len(trades)
        
        # Win rate
        wins = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(max(0, trade.get('pnl', 0)) for trade in trades)
        gross_loss = abs(sum(min(0, trade.get('pnl', 0)) for trade in trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Recent performance (last 20 trades)
        recent_trades = trades[-20:] if len(trades) >= 20 else trades
        recent_wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
        recent_performance = recent_wins / len(recent_trades) if recent_trades else 0
        
        # Average trade duration
        durations = [trade.get('duration', 60) for trade in trades if 'duration' in trade]
        avg_duration = np.mean(durations) if durations else 60
        
        # Sharpe ratio (simplified)
        returns = [trade.get('return_pct', 0) for trade in trades]
        if returns and len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        running_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        
        for trade in trades:
            running_pnl += trade.get('pnl', 0)
            peak_pnl = max(peak_pnl, running_pnl)
            drawdown = peak_pnl - running_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        # Confidence calibration
        confidence_accuracy = []
        for trade in trades:
            predicted_conf = trade.get('predicted_confidence', 0.5)
            actual_success = 1 if trade.get('pnl', 0) > 0 else 0
            confidence_accuracy.append(abs(predicted_conf - actual_success))
        
        confidence_calibration = 1.0 - np.mean(confidence_accuracy) if confidence_accuracy else 0.5
        
        return StrategyPerformance(
            strategy_name=self.name,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_trade_duration=avg_duration,
            recent_performance=recent_performance,
            market_condition_performance={},
            confidence_calibration=confidence_calibration,
            current_weight=self.weight
        )

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(self):
        super().__init__("Momentum_Master")
        self.parameters = {
            'short_period': 12,
            'long_period': 26,
            'signal_period': 9,
            'momentum_threshold': 0.02,
            'confirmation_period': 3
        }
    
    async def generate_signal(self, symbol: str, price_data: List[float], 
                            market_intelligence: Dict, current_price: float) -> StrategySignal:
        """Generate momentum-based signal"""
        
        if len(price_data) < self.parameters['long_period']:
            return self._no_signal(symbol, current_price)
        
        prices = np.array(price_data)
        
        # 1. MACD Calculation
        ema_short = self._calculate_ema(prices, self.parameters['short_period'])
        ema_long = self._calculate_ema(prices, self.parameters['long_period'])
        macd_line = ema_short - ema_long
        signal_line = self._calculate_ema(macd_line, self.parameters['signal_period'])
        macd_histogram = macd_line - signal_line
        
        # 2. Rate of Change
        roc_short = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        roc_long = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        
        # 3. Momentum Confirmation
        momentum_score = 0
        confidence = 0.5
        direction = "HOLD"
        
        # MACD signals
        if macd_line[-1] > signal_line[-1] and macd_histogram[-1] > 0:
            momentum_score += 0.3
        elif macd_line[-1] < signal_line[-1] and macd_histogram[-1] < 0:
            momentum_score -= 0.3
        
        # ROC signals
        if roc_short > self.parameters['momentum_threshold']:
            momentum_score += 0.25
        elif roc_short < -self.parameters['momentum_threshold']:
            momentum_score -= 0.25
        
        if roc_long > self.parameters['momentum_threshold']:
            momentum_score += 0.2
        elif roc_long < -self.parameters['momentum_threshold']:
            momentum_score -= 0.2
        
        # Price momentum
        recent_trend = (prices[-1] - prices[-self.parameters['confirmation_period']]) / prices[-self.parameters['confirmation_period']]
        if abs(recent_trend) > 0.01:  # 1% move
            momentum_score += 0.25 if recent_trend > 0 else -0.25
        
        # Signal determination
        if momentum_score > 0.4:
            direction = "BUY"
            confidence = min(0.95, 0.5 + momentum_score)
        elif momentum_score < -0.4:
            direction = "SELL"
            confidence = min(0.95, 0.5 + abs(momentum_score))
        
        # Market regime adjustment
        regime = market_intelligence.get('regime', {})
        if regime.get('regime_type') in ['strong_bull', 'strong_bear']:
            confidence *= 1.2  # Boost confidence in trending markets
        elif regime.get('regime_type') == 'consolidating':
            confidence *= 0.7  # Reduce confidence in consolidation
        
        # Risk assessment
        volatility = market_intelligence.get('volatility', {}).get('forecast', 0.02)
        risk_score = min(1.0, volatility / 0.05)  # Higher volatility = higher risk
        
        # Stop loss and take profit
        atr_distance = current_price * volatility * 2
        if direction == "BUY":
            stop_loss = current_price - atr_distance
            take_profit = current_price + (atr_distance * 2)
        elif direction == "SELL":
            stop_loss = current_price + atr_distance
            take_profit = current_price - (atr_distance * 2)
        else:
            stop_loss = None
            take_profit = None
        
        signal = StrategySignal(
            strategy_name=self.name,
            direction=direction,
            confidence=confidence,
            strength=abs(momentum_score),
            timestamp=datetime.now(),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"Momentum score: {momentum_score:.3f}, MACD: {macd_line[-1]:.6f}, ROC: {roc_short:.3f}",
            market_conditions=market_intelligence.get('regime', {}),
            risk_score=risk_score
        )
        
        self.signal_history.append(signal)
        return signal
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2.0 / (period + 1.0)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _no_signal(self, symbol: str, current_price: float) -> StrategySignal:
        """Return neutral signal when insufficient data"""
        return StrategySignal(
            strategy_name=self.name,
            direction="HOLD",
            confidence=0.0,
            strength=0.0,
            timestamp=datetime.now(),
            entry_price=current_price,
            stop_loss=None,
            take_profit=None,
            reasoning="Insufficient data for momentum analysis",
            market_conditions={},
            risk_score=0.5
        )
    
    def update_parameters(self, performance_data: Dict):
        """Update momentum strategy parameters based on performance"""
        
        win_rate = performance_data.get('win_rate', 0.5)
        profit_factor = performance_data.get('profit_factor', 1.0)
        
        # Adjust periods based on performance
        if win_rate < 0.4:
            # Poor performance - try longer periods for stability
            self.parameters['short_period'] = min(15, self.parameters['short_period'] + 1)
            self.parameters['long_period'] = min(30, self.parameters['long_period'] + 1)
        elif win_rate > 0.7 and profit_factor > 1.5:
            # Good performance - try shorter periods for more signals
            self.parameters['short_period'] = max(8, self.parameters['short_period'] - 1)
            self.parameters['long_period'] = max(20, self.parameters['long_period'] - 1)

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(self):
        super().__init__("MeanReversion_Pro")
        self.parameters = {
            'lookback_period': 20,
            'std_threshold': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'mean_reversion_strength': 0.6
        }
    
    async def generate_signal(self, symbol: str, price_data: List[float], 
                            market_intelligence: Dict, current_price: float) -> StrategySignal:
        """Generate mean reversion signal"""
        
        if len(price_data) < self.parameters['lookback_period']:
            return self._no_signal(symbol, current_price)
        
        prices = np.array(price_data)
        
        # 1. Bollinger Bands
        sma = np.mean(prices[-self.parameters['lookback_period']:])
        std = np.std(prices[-self.parameters['lookback_period']:])
        
        upper_band = sma + (self.parameters['std_threshold'] * std)
        lower_band = sma - (self.parameters['std_threshold'] * std)
        
        # 2. RSI Calculation
        rsi = self._calculate_rsi(prices, self.parameters['rsi_period'])
        current_rsi = rsi[-1] if len(rsi) > 0 else 50
        
        # 3. Distance from mean
        price_deviation = (current_price - sma) / sma
        
        # 4. Mean reversion signals
        reversion_score = 0
        confidence = 0.5
        direction = "HOLD"
        
        # Bollinger Band signals
        if current_price > upper_band:
            reversion_score -= 0.4  # Overextended up - sell signal
        elif current_price < lower_band:
            reversion_score += 0.4  # Overextended down - buy signal
        
        # RSI signals
        if current_rsi < self.parameters['rsi_oversold']:
            reversion_score += 0.3  # Oversold - buy signal
        elif current_rsi > self.parameters['rsi_overbought']:
            reversion_score -= 0.3  # Overbought - sell signal
        
        # Price deviation signals
        if abs(price_deviation) > 0.03:  # 3% deviation from mean
            if price_deviation > 0:
                reversion_score -= 0.2  # Above mean - sell
            else:
                reversion_score += 0.2  # Below mean - buy
        
        # Signal determination
        if reversion_score > 0.5:
            direction = "BUY"
            confidence = min(0.95, 0.5 + (reversion_score * self.parameters['mean_reversion_strength']))
        elif reversion_score < -0.5:
            direction = "SELL"
            confidence = min(0.95, 0.5 + (abs(reversion_score) * self.parameters['mean_reversion_strength']))
        
        # Market regime adjustment
        regime = market_intelligence.get('regime', {})
        if regime.get('regime_type') == 'consolidating':
            confidence *= 1.3  # Mean reversion works better in consolidation
        elif regime.get('regime_type') in ['strong_bull', 'strong_bear']:
            confidence *= 0.6  # Mean reversion less effective in strong trends
        
        # Risk assessment
        volatility = market_intelligence.get('volatility', {}).get('forecast', 0.02)
        risk_score = min(1.0, volatility / 0.04)  # Adjust for volatility
        
        # Stop loss and take profit
        if direction == "BUY":
            stop_loss = lower_band * 0.995  # Just below lower band
            take_profit = sma + (std * 0.5)    # Partial reversion to mean
        elif direction == "SELL":
            stop_loss = upper_band * 1.005  # Just above upper band
            take_profit = sma - (std * 0.5)    # Partial reversion to mean
        else:
            stop_loss = None
            take_profit = None
        
        signal = StrategySignal(
            strategy_name=self.name,
            direction=direction,
            confidence=confidence,
            strength=abs(reversion_score),
            timestamp=datetime.now(),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"Mean reversion score: {reversion_score:.3f}, RSI: {current_rsi:.1f}, Price vs SMA: {price_deviation:.3f}",
            market_conditions=market_intelligence.get('regime', {}),
            risk_score=risk_score
        )
        
        self.signal_history.append(signal)
        return signal
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Relative Strength Index"""
        
        if len(prices) < period + 1:
            return np.array([50])
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        # Avoid division by zero
        avg_losses = np.where(avg_losses == 0, 0.0001, avg_losses)
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _no_signal(self, symbol: str, current_price: float) -> StrategySignal:
        """Return neutral signal when insufficient data"""
        return StrategySignal(
            strategy_name=self.name,
            direction="HOLD",
            confidence=0.0,
            strength=0.0,
            timestamp=datetime.now(),
            entry_price=current_price,
            stop_loss=None,
            take_profit=None,
            reasoning="Insufficient data for mean reversion analysis",
            market_conditions={},
            risk_score=0.5
        )
    
    def update_parameters(self, performance_data: Dict):
        """Update mean reversion parameters based on performance"""
        
        win_rate = performance_data.get('win_rate', 0.5)
        
        # Adjust RSI thresholds based on performance
        if win_rate < 0.4:
            # Tighten thresholds for more selective signals
            self.parameters['rsi_oversold'] = max(25, self.parameters['rsi_oversold'] - 2)
            self.parameters['rsi_overbought'] = min(75, self.parameters['rsi_overbought'] + 2)
        elif win_rate > 0.7:
            # Widen thresholds for more signals
            self.parameters['rsi_oversold'] = min(35, self.parameters['rsi_oversold'] + 1)
            self.parameters['rsi_overbought'] = max(65, self.parameters['rsi_overbought'] - 1)

class BreakoutStrategy(BaseStrategy):
    """Breakout trading strategy"""
    
    def __init__(self):
        super().__init__("Breakout_Hunter")
        self.parameters = {
            'consolidation_period': 15,
            'breakout_threshold': 0.02,
            'volume_confirmation': 1.5,
            'false_breakout_filter': 0.005,
            'consolidation_range_min': 0.01
        }
    
    async def generate_signal(self, symbol: str, price_data: List[float], 
                            market_intelligence: Dict, current_price: float) -> StrategySignal:
        """Generate breakout signal"""
        
        if len(price_data) < self.parameters['consolidation_period'] + 5:
            return self._no_signal(symbol, current_price)
        
        prices = np.array(price_data)
        
        # 1. Identify consolidation range
        recent_prices = prices[-self.parameters['consolidation_period']:]
        range_high = np.max(recent_prices)
        range_low = np.min(recent_prices)
        range_size = (range_high - range_low) / range_low
        
        # 2. Check if we have a valid consolidation
        if range_size < self.parameters['consolidation_range_min']:
            return self._no_signal(symbol, current_price)
        
        # 3. Detect breakout
        breakout_score = 0
        confidence = 0.5
        direction = "HOLD"
        
        # Price breakout
        if current_price > range_high * (1 + self.parameters['false_breakout_filter']):
            breakout_score += 0.4  # Upward breakout
        elif current_price < range_low * (1 - self.parameters['false_breakout_filter']):
            breakout_score -= 0.4  # Downward breakout
        
        # Volume confirmation (simulated)
        volume_multiplier = self._simulate_volume_confirmation(prices)
        if volume_multiplier > self.parameters['volume_confirmation']:
            breakout_score += 0.2 if breakout_score > 0 else -0.2
        
        # Momentum confirmation
        recent_momentum = (current_price - prices[-3]) / prices[-3]
        if abs(recent_momentum) > 0.01:  # 1% move
            breakout_score += 0.2 if recent_momentum > 0 else -0.2
        
        # Time since consolidation start
        consolidation_strength = min(1.0, self.parameters['consolidation_period'] / 20)
        breakout_score *= (1 + consolidation_strength * 0.3)
        
        # Signal determination
        if breakout_score > 0.5:
            direction = "BUY"
            confidence = min(0.95, 0.6 + (breakout_score * 0.4))
        elif breakout_score < -0.5:
            direction = "SELL"
            confidence = min(0.95, 0.6 + (abs(breakout_score) * 0.4))
        
        # Market regime adjustment
        regime = market_intelligence.get('regime', {})
        if regime.get('regime_type') in ['strong_bull', 'strong_bear']:
            confidence *= 1.1  # Breakouts work well in trending markets
        elif regime.get('volatility_level') == 'low':
            confidence *= 1.2  # Breakouts more significant in low volatility
        
        # Risk assessment
        volatility = market_intelligence.get('volatility', {}).get('forecast', 0.02)
        risk_score = min(1.0, volatility / 0.03)
        
        # Stop loss and take profit
        if direction == "BUY":
            stop_loss = range_high * 0.995  # Just below breakout level
            take_profit = current_price + ((range_high - range_low) * 2)  # Range projection
        elif direction == "SELL":
            stop_loss = range_low * 1.005  # Just above breakdown level
            take_profit = current_price - ((range_high - range_low) * 2)  # Range projection
        else:
            stop_loss = None
            take_profit = None
        
        signal = StrategySignal(
            strategy_name=self.name,
            direction=direction,
            confidence=confidence,
            strength=abs(breakout_score),
            timestamp=datetime.now(),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"Breakout score: {breakout_score:.3f}, Range: {range_size:.3f}, Volume conf: {volume_multiplier:.2f}",
            market_conditions=market_intelligence.get('regime', {}),
            risk_score=risk_score
        )
        
        self.signal_history.append(signal)
        return signal
    
    def _simulate_volume_confirmation(self, prices: np.ndarray) -> float:
        """Simulate volume confirmation for breakout"""
        
        # Simulate volume based on price action
        recent_volatility = np.std(prices[-10:]) if len(prices) >= 10 else 0.02
        volume_multiplier = 1.0 + (recent_volatility / 0.02) * 2.0
        
        # Add some randomness
        volume_multiplier *= np.random.uniform(0.8, 2.0)
        
        return volume_multiplier
    
    def _no_signal(self, symbol: str, current_price: float) -> StrategySignal:
        """Return neutral signal when insufficient data"""
        return StrategySignal(
            strategy_name=self.name,
            direction="HOLD",
            confidence=0.0,
            strength=0.0,
            timestamp=datetime.now(),
            entry_price=current_price,
            stop_loss=None,
            take_profit=None,
            reasoning="Insufficient data or no valid consolidation for breakout analysis",
            market_conditions={},
            risk_score=0.5
        )
    
    def update_parameters(self, performance_data: Dict):
        """Update breakout parameters based on performance"""
        
        win_rate = performance_data.get('win_rate', 0.5)
        
        # Adjust breakout threshold based on performance
        if win_rate < 0.4:
            # More selective - increase threshold
            self.parameters['breakout_threshold'] = min(0.035, self.parameters['breakout_threshold'] + 0.002)
        elif win_rate > 0.7:
            # Less selective - decrease threshold
            self.parameters['breakout_threshold'] = max(0.015, self.parameters['breakout_threshold'] - 0.001)

class PatternRecognitionStrategy(BaseStrategy):
    """Pattern recognition trading strategy"""
    
    def __init__(self):
        super().__init__("Pattern_Master")
        self.parameters = {
            'min_pattern_bars': 8,
            'pattern_tolerance': 0.02,
            'confirmation_bars': 2,
            'pattern_strength_threshold': 0.6
        }
        
        # Pattern definitions
        self.patterns = {
            'double_bottom': {'bullish': True, 'reliability': 0.75},
            'double_top': {'bullish': False, 'reliability': 0.75},
            'head_shoulders': {'bullish': False, 'reliability': 0.8},
            'inverse_head_shoulders': {'bullish': True, 'reliability': 0.8},
            'ascending_triangle': {'bullish': True, 'reliability': 0.7},
            'descending_triangle': {'bullish': False, 'reliability': 0.7},
            'cup_handle': {'bullish': True, 'reliability': 0.65}
        }
    
    async def generate_signal(self, symbol: str, price_data: List[float], 
                            market_intelligence: Dict, current_price: float) -> StrategySignal:
        """Generate pattern-based signal"""
        
        if len(price_data) < self.parameters['min_pattern_bars'] * 2:
            return self._no_signal(symbol, current_price)
        
        prices = np.array(price_data)
        
        # 1. Detect patterns
        detected_patterns = []
        
        # Double Bottom/Top
        double_pattern = self._detect_double_pattern(prices)
        if double_pattern:
            detected_patterns.append(double_pattern)
        
        # Head and Shoulders
        hs_pattern = self._detect_head_shoulders(prices)
        if hs_pattern:
            detected_patterns.append(hs_pattern)
        
        # Triangle patterns
        triangle_pattern = self._detect_triangle_patterns(prices)
        if triangle_pattern:
            detected_patterns.append(triangle_pattern)
        
        # Cup and Handle
        cup_handle_pattern = self._detect_cup_handle(prices)
        if cup_handle_pattern:
            detected_patterns.append(cup_handle_pattern)
        
        # 2. Evaluate patterns
        if not detected_patterns:
            return self._no_signal(symbol, current_price)
        
        # Select strongest pattern
        best_pattern = max(detected_patterns, key=lambda p: p['strength'])
        
        pattern_score = best_pattern['strength']
        is_bullish = self.patterns[best_pattern['name']]['bullish']
        reliability = self.patterns[best_pattern['name']]['reliability']
        
        # 3. Generate signal
        direction = "BUY" if is_bullish else "SELL"
        confidence = min(0.95, reliability * pattern_score)
        
        # Market regime adjustment
        regime = market_intelligence.get('regime', {})
        regime_type = regime.get('regime_type', 'neutral')
        
        if ((is_bullish and regime_type in ['bull_trend', 'strong_bull']) or
            (not is_bullish and regime_type in ['bear_trend', 'strong_bear'])):
            confidence *= 1.2  # Pattern aligns with regime
        elif ((is_bullish and regime_type in ['bear_trend', 'strong_bear']) or
              (not is_bullish and regime_type in ['bull_trend', 'strong_bull'])):
            confidence *= 0.7  # Pattern against regime
        
        # Risk assessment
        volatility = market_intelligence.get('volatility', {}).get('forecast', 0.02)
        risk_score = min(1.0, volatility / 0.025)
        
        # Pattern-specific stop loss and take profit
        pattern_range = best_pattern.get('range', current_price * 0.05)
        
        if direction == "BUY":
            stop_loss = best_pattern.get('support', current_price * 0.98)
            take_profit = current_price + pattern_range
        else:
            stop_loss = best_pattern.get('resistance', current_price * 1.02)
            take_profit = current_price - pattern_range
        
        signal = StrategySignal(
            strategy_name=self.name,
            direction=direction,
            confidence=confidence,
            strength=pattern_score,
            timestamp=datetime.now(),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"Detected {best_pattern['name']} pattern with strength {pattern_score:.3f}",
            market_conditions=market_intelligence.get('regime', {}),
            risk_score=risk_score
        )
        
        self.signal_history.append(signal)
        return signal
    
    def _detect_double_pattern(self, prices: np.ndarray) -> Optional[Dict]:
        """Detect double top/bottom patterns"""
        
        if len(prices) < 20:
            return None
        
        # Find peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(2, len(prices) - 2):
            if (prices[i] > prices[i-1] and prices[i] > prices[i+1] and 
                prices[i] > prices[i-2] and prices[i] > prices[i+2]):
                peaks.append((i, prices[i]))
            elif (prices[i] < prices[i-1] and prices[i] < prices[i+1] and
                  prices[i] < prices[i-2] and prices[i] < prices[i+2]):
                troughs.append((i, prices[i]))
        
        # Check for double bottom
        if len(troughs) >= 2:
            last_two_troughs = troughs[-2:]
            price_diff = abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1]
            
            if price_diff < self.parameters['pattern_tolerance']:
                return {
                    'name': 'double_bottom',
                    'strength': max(0.3, 0.8 - price_diff),
                    'support': min(last_two_troughs[0][1], last_two_troughs[1][1]),
                    'range': prices[-1] * 0.04
                }
        
        # Check for double top
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            price_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1]
            
            if price_diff < self.parameters['pattern_tolerance']:
                return {
                    'name': 'double_top',
                    'strength': max(0.3, 0.8 - price_diff),
                    'resistance': max(last_two_peaks[0][1], last_two_peaks[1][1]),
                    'range': prices[-1] * 0.04
                }
        
        return None
    
    def _detect_head_shoulders(self, prices: np.ndarray) -> Optional[Dict]:
        """Detect head and shoulders patterns"""
        
        if len(prices) < 30:
            return None
        
        # Simplified head and shoulders detection
        third = len(prices) // 3
        
        left_shoulder = np.max(prices[:third])
        head = np.max(prices[third:2*third])
        right_shoulder = np.max(prices[2*third:])
        
        # Check if head is higher than shoulders
        if (head > left_shoulder * 1.02 and head > right_shoulder * 1.02 and
            abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
            
            strength = min(1.0, (head - max(left_shoulder, right_shoulder)) / head)
            
            return {
                'name': 'head_shoulders',
                'strength': strength,
                'resistance': head,
                'range': head - min(left_shoulder, right_shoulder)
            }
        
        # Check for inverse head and shoulders
        left_trough = np.min(prices[:third])
        head_trough = np.min(prices[third:2*third])
        right_trough = np.min(prices[2*third:])
        
        if (head_trough < left_trough * 0.98 and head_trough < right_trough * 0.98 and
            abs(left_trough - right_trough) / left_trough < 0.05):
            
            strength = min(1.0, (min(left_trough, right_trough) - head_trough) / left_trough)
            
            return {
                'name': 'inverse_head_shoulders',
                'strength': strength,
                'support': head_trough,
                'range': max(left_trough, right_trough) - head_trough
            }
        
        return None
    
    def _detect_triangle_patterns(self, prices: np.ndarray) -> Optional[Dict]:
        """Detect triangle patterns"""
        
        if len(prices) < 20:
            return None
        
        # Calculate trend lines for highs and lows
        recent_prices = prices[-20:]
        
        # Find highs and lows
        highs = []
        lows = []
        
        for i in range(1, len(recent_prices) - 1):
            if (recent_prices[i] > recent_prices[i-1] and 
                recent_prices[i] > recent_prices[i+1]):
                highs.append((i, recent_prices[i]))
            elif (recent_prices[i] < recent_prices[i-1] and 
                  recent_prices[i] < recent_prices[i+1]):
                lows.append((i, recent_prices[i]))
        
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        # Calculate trend line slopes
        high_slope = (highs[-1][1] - highs[0][1]) / (highs[-1][0] - highs[0][0]) if len(highs) > 1 else 0
        low_slope = (lows[-1][1] - lows[0][1]) / (lows[-1][0] - lows[0][0]) if len(lows) > 1 else 0
        
        # Ascending triangle (flat resistance, rising support)
        if abs(high_slope) < 0.001 and low_slope > 0.001:
            return {
                'name': 'ascending_triangle',
                'strength': min(1.0, low_slope * 1000),
                'resistance': highs[-1][1],
                'range': recent_prices[-1] * 0.03
            }
        
        # Descending triangle (declining resistance, flat support)
        if high_slope < -0.001 and abs(low_slope) < 0.001:
            return {
                'name': 'descending_triangle',
                'strength': min(1.0, abs(high_slope) * 1000),
                'support': lows[-1][1],
                'range': recent_prices[-1] * 0.03
            }
        
        return None
    
    def _detect_cup_handle(self, prices: np.ndarray) -> Optional[Dict]:
        """Detect cup and handle pattern"""
        
        if len(prices) < 40:
            return None
        
        # Simplified cup and handle detection
        # Cup: U-shaped recovery
        # Handle: Small consolidation after cup
        
        cup_start = len(prices) - 30
        cup_middle = len(prices) - 20
        handle_start = len(prices) - 10
        
        if cup_start < 0:
            return None
        
        cup_left = prices[cup_start]
        cup_bottom = np.min(prices[cup_start:cup_middle])
        cup_right = prices[cup_middle]
        handle_high = np.max(prices[handle_start:])
        current_price = prices[-1]
        
        # Check cup shape
        if (cup_bottom < cup_left * 0.9 and cup_bottom < cup_right * 0.9 and
            abs(cup_left - cup_right) / cup_left < 0.05):
            
            # Check handle
            if (handle_high < cup_right * 1.02 and current_price < handle_high):
                
                strength = (cup_left - cup_bottom) / cup_left
                
                return {
                    'name': 'cup_handle',
                    'strength': min(1.0, strength),
                    'support': cup_bottom,
                    'range': cup_left - cup_bottom
                }
        
        return None
    
    def _no_signal(self, symbol: str, current_price: float) -> StrategySignal:
        """Return neutral signal when no patterns detected"""
        return StrategySignal(
            strategy_name=self.name,
            direction="HOLD",
            confidence=0.0,
            strength=0.0,
            timestamp=datetime.now(),
            entry_price=current_price,
            stop_loss=None,
            take_profit=None,
            reasoning="No significant patterns detected",
            market_conditions={},
            risk_score=0.5
        )
    
    def update_parameters(self, performance_data: Dict):
        """Update pattern recognition parameters"""
        
        win_rate = performance_data.get('win_rate', 0.5)
        
        # Adjust pattern strength threshold
        if win_rate < 0.4:
            self.parameters['pattern_strength_threshold'] = min(0.8, 
                self.parameters['pattern_strength_threshold'] + 0.05)
        elif win_rate > 0.7:
            self.parameters['pattern_strength_threshold'] = max(0.4,
                self.parameters['pattern_strength_threshold'] - 0.02)

class VolatilityStrategy(BaseStrategy):
    """Volatility-based trading strategy"""
    
    def __init__(self):
        super().__init__("Volatility_Rider")
        self.parameters = {
            'vol_lookback': 20,
            'vol_threshold_high': 0.04,
            'vol_threshold_low': 0.015,
            'vol_expansion_multiplier': 1.5,
            'vol_contraction_threshold': 0.7
        }
    
    async def generate_signal(self, symbol: str, price_data: List[float], 
                            market_intelligence: Dict, current_price: float) -> StrategySignal:
        """Generate volatility-based signal"""
        
        if len(price_data) < self.parameters['vol_lookback'] + 10:
            return self._no_signal(symbol, current_price)
        
        prices = np.array(price_data)
        returns = np.diff(prices) / prices[:-1]
        
        # 1. Current volatility
        current_vol = np.std(returns[-self.parameters['vol_lookback']:])
        historical_vol = np.std(returns[-40:-20]) if len(returns) >= 40 else current_vol
        
        # 2. Volatility expansion/contraction
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        # 3. Volatility regime
        vol_percentile = market_intelligence.get('volatility', {}).get('percentile_rank', 0.5)
        
        volatility_score = 0
        confidence = 0.5
        direction = "HOLD"
        
        # Volatility expansion signals
        if vol_ratio > self.parameters['vol_expansion_multiplier']:
            if vol_percentile > 0.8:  # High volatility environment
                # Expect mean reversion
                recent_return = (current_price - prices[-5]) / prices[-5]
                if recent_return > 0.02:
                    volatility_score -= 0.3  # Sell after big up move
                elif recent_return < -0.02:
                    volatility_score += 0.3  # Buy after big down move
            else:
                # Volatility breakout - follow momentum
                recent_momentum = (current_price - prices[-3]) / prices[-3]
                volatility_score += 0.2 if recent_momentum > 0 else -0.2
        
        # Volatility contraction signals
        elif vol_ratio < self.parameters['vol_contraction_threshold']:
            # Prepare for volatility expansion
            if vol_percentile < 0.3:  # Low volatility environment
                # Look for direction signals
                ma_short = np.mean(prices[-5:])
                ma_long = np.mean(prices[-15:])
                
                if ma_short > ma_long * 1.005:
                    volatility_score += 0.25  # Bullish setup
                elif ma_short < ma_long * 0.995:
                    volatility_score -= 0.25  # Bearish setup
        
        # Market structure analysis
        regime = market_intelligence.get('regime', {})
        if regime.get('regime_type') in ['volatile_trending', 'volatile_sideways']:
            volatility_score *= 1.3  # Boost signals in volatile regimes
        
        # Signal determination
        if volatility_score > 0.3:
            direction = "BUY"
            confidence = min(0.9, 0.5 + abs(volatility_score))
        elif volatility_score < -0.3:
            direction = "SELL"
            confidence = min(0.9, 0.5 + abs(volatility_score))
        
        # Risk assessment
        risk_score = min(1.0, current_vol / 0.05)
        
        # Volatility-adjusted stops and targets
        vol_distance = current_price * current_vol * 2
        
        if direction == "BUY":
            stop_loss = current_price - vol_distance
            take_profit = current_price + (vol_distance * 1.5)
        elif direction == "SELL":
            stop_loss = current_price + vol_distance
            take_profit = current_price - (vol_distance * 1.5)
        else:
            stop_loss = None
            take_profit = None
        
        signal = StrategySignal(
            strategy_name=self.name,
            direction=direction,
            confidence=confidence,
            strength=abs(volatility_score),
            timestamp=datetime.now(),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"Volatility score: {volatility_score:.3f}, Vol ratio: {vol_ratio:.2f}, Vol percentile: {vol_percentile:.2f}",
            market_conditions=market_intelligence.get('regime', {}),
            risk_score=risk_score
        )
        
        self.signal_history.append(signal)
        return signal
    
    def _no_signal(self, symbol: str, current_price: float) -> StrategySignal:
        """Return neutral signal when insufficient data"""
        return StrategySignal(
            strategy_name=self.name,
            direction="HOLD",
            confidence=0.0,
            strength=0.0,
            timestamp=datetime.now(),
            entry_price=current_price,
            stop_loss=None,
            take_profit=None,
            reasoning="Insufficient data for volatility analysis",
            market_conditions={},
            risk_score=0.5
        )
    
    def update_parameters(self, performance_data: Dict):
        """Update volatility strategy parameters"""
        
        win_rate = performance_data.get('win_rate', 0.5)
        
        # Adjust volatility thresholds
        if win_rate < 0.4:
            # More conservative - higher thresholds
            self.parameters['vol_threshold_high'] = min(0.06, 
                self.parameters['vol_threshold_high'] + 0.005)
        elif win_rate > 0.7:
            # More aggressive - lower thresholds
            self.parameters['vol_threshold_high'] = max(0.025,
                self.parameters['vol_threshold_high'] - 0.002)

class StrategyEnsemble:
    """Multi-strategy ensemble system"""
    
    def __init__(self):
        # Initialize all strategies
        self.strategies = {
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': BreakoutStrategy(),
            'pattern': PatternRecognitionStrategy(),
            'volatility': VolatilityStrategy()
        }
        
        # Strategy weights (sum should be 1.0)
        self.strategy_weights = {
            'momentum': 0.25,
            'mean_reversion': 0.20,
            'breakout': 0.20,
            'pattern': 0.20,
            'volatility': 0.15
        }
        
        # Performance tracking
        self.ensemble_history = deque(maxlen=1000)
        self.strategy_performance = {}
        
        # Ensemble parameters
        self.min_strategies_agree = 2
        self.consensus_threshold = 0.6
        self.confidence_boost_threshold = 0.8
        # Minimum required confidence for ensemble to consider a trade (set by controller)
        self.min_confidence_threshold = 0.5
        
        print("🎯 MULTI-STRATEGY ENSEMBLE SYSTEM INITIALIZED")
        print(f"   ✅ {len(self.strategies)} Trading Strategies Loaded")
        print("   ✅ Dynamic Weighting System")
        print("   ✅ Consensus-Based Signal Generation")
        print("   ✅ Performance-Based Optimization")
    
    async def generate_ensemble_signal(self, symbol: str, price_data: List[float],
                                     market_intelligence: Dict, current_price: float) -> EnsembleSignal:
        """Generate ensemble signal from all strategies"""
        
        print(f"🎯 GENERATING ENSEMBLE SIGNAL FOR {symbol}...")
        
        # 1. Get signals from all active strategies
        strategy_signals = {}
        for name, strategy in self.strategies.items():
            if strategy.is_active:
                try:
                    signal = await strategy.generate_signal(
                        symbol, price_data, market_intelligence, current_price
                    )
                    strategy_signals[name] = signal
                    print(f"   📊 {name}: {signal.direction} (conf: {signal.confidence:.3f})")
                except Exception as e:
                    print(f"   ⚠️ Error in {name} strategy: {e}")
        
        if not strategy_signals:
            return self._no_ensemble_signal(symbol, current_price)
        
        # 2. Apply market condition filters
        filtered_signals = await self._apply_market_filters(
            strategy_signals, market_intelligence
        )
        
        # 3. Calculate ensemble signal
        ensemble_signal = await self._calculate_ensemble_signal(
            symbol, filtered_signals, current_price, market_intelligence
        )
        
        # 4. Track ensemble signal
        self.ensemble_history.append({
            'timestamp': datetime.now(),
            'signal': ensemble_signal,
            'contributing_strategies': list(filtered_signals.keys()),
            'market_intelligence': market_intelligence
        })
        
        print(f"✅ ENSEMBLE SIGNAL: {ensemble_signal.direction} (conf: {ensemble_signal.confidence:.3f})")
        
        return ensemble_signal
    
    async def _apply_market_filters(self, strategy_signals: Dict[str, StrategySignal],
                                  market_intelligence: Dict) -> Dict[str, StrategySignal]:
        """Apply market condition filters to strategy signals"""
        
        regime = market_intelligence.get('regime', {})
        regime_type = regime.get('regime_type', 'neutral')
        volatility = market_intelligence.get('volatility', {})
        
        filtered_signals = {}
        
        for name, signal in strategy_signals.items():
            # Skip weak signals
            if signal.confidence < 0.3:
                continue
            
            # Market regime compatibility
            regime_compatible = True
            
            if regime_type == 'consolidating':
                # Favor mean reversion and breakout in consolidation
                if name in ['momentum']:
                    signal.confidence *= 0.7
            elif regime_type in ['strong_bull', 'strong_bear']:
                # Favor momentum and breakout in trending
                if name == 'mean_reversion':
                    signal.confidence *= 0.7
                elif name in ['momentum', 'breakout']:
                    signal.confidence *= 1.2
            
            # Volatility regime compatibility
            vol_percentile = volatility.get('percentile_rank', 0.5)
            
            if vol_percentile > 0.8:  # High volatility
                if name == 'volatility':
                    signal.confidence *= 1.3
                elif name in ['pattern', 'breakout']:
                    signal.confidence *= 0.8
            elif vol_percentile < 0.2:  # Low volatility
                if name in ['pattern', 'breakout']:
                    signal.confidence *= 1.2
                elif name == 'volatility':
                    signal.confidence *= 0.8
            
            # Only include signals above minimum confidence after filtering
            if signal.confidence >= 0.4:
                filtered_signals[name] = signal
        
        return filtered_signals
    
    async def _calculate_ensemble_signal(self, symbol: str, strategy_signals: Dict[str, StrategySignal],
                                       current_price: float, market_intelligence: Dict) -> EnsembleSignal:
        """Calculate final ensemble signal"""
        
        if not strategy_signals:
            return self._no_ensemble_signal(symbol, current_price)
        
        # 1. Separate signals by direction
        buy_signals = []
        sell_signals = []
        hold_signals = []
        
        for name, signal in strategy_signals.items():
            if signal.direction == "BUY":
                buy_signals.append((name, signal))
            elif signal.direction == "SELL":
                sell_signals.append((name, signal))
            else:
                hold_signals.append((name, signal))
        
        # 2. Calculate weighted votes
        buy_vote = 0
        sell_vote = 0
        
        for name, signal in buy_signals:
            weight = self.strategy_weights.get(name, 0.1) * self.strategies[name].weight
            buy_vote += signal.confidence * signal.strength * weight
        
        for name, signal in sell_signals:
            weight = self.strategy_weights.get(name, 0.1) * self.strategies[name].weight
            sell_vote += signal.confidence * signal.strength * weight
        
        # 3. Determine ensemble direction
        if buy_vote > sell_vote and buy_vote > 0.3:
            ensemble_direction = "BUY"
            primary_signals = buy_signals
            vote_strength = buy_vote
        elif sell_vote > buy_vote and sell_vote > 0.3:
            ensemble_direction = "SELL"
            primary_signals = sell_signals
            vote_strength = sell_vote
        else:
            ensemble_direction = "HOLD"
            primary_signals = []
            vote_strength = 0
        
        # 4. Calculate consensus level
        total_signals = len(buy_signals) + len(sell_signals)
        if total_signals > 0:
            consensus_level = len(primary_signals) / total_signals
        else:
            consensus_level = 0
        
        # 5. Calculate ensemble confidence
        if ensemble_direction == "HOLD":
            ensemble_confidence = 0.0
        else:
            # Base confidence from vote strength
            base_confidence = min(0.9, vote_strength)
            
            # Consensus adjustment
            consensus_boost = consensus_level * 0.3
            
            # Strategy agreement bonus
            if len(primary_signals) >= self.min_strategies_agree:
                agreement_bonus = 0.1
            else:
                agreement_bonus = 0
            
            # Market regime alignment
            regime_alignment = await self._calculate_regime_alignment(
                ensemble_direction, market_intelligence
            )
            
            ensemble_confidence = min(0.95, base_confidence + consensus_boost + 
                                   agreement_bonus + regime_alignment)
        
        # 6. Calculate ensemble stop loss and take profit
        if ensemble_direction != "HOLD" and primary_signals:
            # Weighted average of strategy stops and targets
            total_weight = sum(self.strategy_weights.get(name, 0.1) for name, _ in primary_signals)
            
            stop_loss = sum(signal.stop_loss * self.strategy_weights.get(name, 0.1) 
                          for name, signal in primary_signals if signal.stop_loss) / total_weight
            
            take_profit = sum(signal.take_profit * self.strategy_weights.get(name, 0.1) 
                            for name, signal in primary_signals if signal.take_profit) / total_weight
        else:
            stop_loss = None
            take_profit = None
        
        # 7. Risk adjustment
        risk_scores = [signal.risk_score for _, signal in primary_signals]
        avg_risk = np.mean(risk_scores) if risk_scores else 0.5
        risk_adjusted_confidence = ensemble_confidence * (1.0 - (avg_risk * 0.3))
        
        # 8. Create ensemble signal
        ensemble_signal = EnsembleSignal(
            direction=ensemble_direction,
            confidence=ensemble_confidence,
            strength=vote_strength,
            contributing_strategies=[name for name, _ in primary_signals],
            strategy_weights={name: self.strategy_weights.get(name, 0.1) for name, _ in primary_signals},
            consensus_level=consensus_level,
            risk_adjusted_confidence=risk_adjusted_confidence,
            market_regime_alignment=regime_alignment if ensemble_direction != "HOLD" else 0,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=[signal.reasoning for _, signal in primary_signals],
            timestamp=datetime.now()
        )
        
        return ensemble_signal
    
    async def _calculate_regime_alignment(self, direction: str, market_intelligence: Dict) -> float:
        """Calculate how well signal aligns with market regime"""
        
        regime = market_intelligence.get('regime', {})
        regime_type = regime.get('regime_type', 'neutral')
        regime_confidence = regime.get('confidence', 0.5)
        
        alignment = 0
        
        if direction == "BUY":
            if regime_type in ['strong_bull', 'bull_trend']:
                alignment = 0.2 * regime_confidence
            elif regime_type in ['strong_bear', 'bear_trend']:
                alignment = -0.2 * regime_confidence
        elif direction == "SELL":
            if regime_type in ['strong_bear', 'bear_trend']:
                alignment = 0.2 * regime_confidence
            elif regime_type in ['strong_bull', 'bull_trend']:
                alignment = -0.2 * regime_confidence
        
        return alignment
    
    def _no_ensemble_signal(self, symbol: str, current_price: float) -> EnsembleSignal:
        """Return neutral ensemble signal"""
        return EnsembleSignal(
            direction="HOLD",
            confidence=0.0,
            strength=0.0,
            contributing_strategies=[],
            strategy_weights={},
            consensus_level=0.0,
            risk_adjusted_confidence=0.0,
            market_regime_alignment=0.0,
            entry_price=current_price,
            stop_loss=None,
            take_profit=None,
            reasoning=["No sufficient strategy signals"],
            timestamp=datetime.now()
        )
    
    async def update_strategy_weights(self, performance_data: Dict[str, StrategyPerformance]):
        """Update strategy weights based on performance"""
        
        print("🔄 UPDATING STRATEGY WEIGHTS...")
        
        # Calculate performance scores
        performance_scores = {}
        
        for name, perf in performance_data.items():
            if name in self.strategies:
                # Combine multiple performance metrics
                score = (perf.win_rate * 0.4 + 
                        min(2.0, perf.profit_factor) / 2.0 * 0.3 + 
                        perf.recent_performance * 0.2 +
                        perf.confidence_calibration * 0.1)
                
                performance_scores[name] = max(0.1, score)  # Minimum score
        
        if not performance_scores:
            return
        
        # Normalize to create new weights
        total_score = sum(performance_scores.values())
        
        for name in self.strategy_weights:
            if name in performance_scores:
                new_weight = performance_scores[name] / total_score
                
                # Smooth weight changes (don't change too drastically)
                old_weight = self.strategy_weights[name]
                self.strategy_weights[name] = old_weight * 0.7 + new_weight * 0.3
                
                # Update strategy weight
                if name in self.strategies:
                    self.strategies[name].weight = self.strategy_weights[name] * 5.0  # Scale for individual use
                
                print(f"   📊 {name}: {old_weight:.3f} -> {self.strategy_weights[name]:.3f}")
    
    async def optimize_ensemble_parameters(self, recent_performance: List[Dict]):
        """Optimize ensemble parameters based on recent performance"""
        
        if len(recent_performance) < 10:
            return
        
        # Analyze consensus vs performance
        consensus_performance = defaultdict(list)
        
        for trade in recent_performance:
            consensus = trade.get('consensus_level', 0.5)
            pnl = trade.get('pnl', 0)
            
            if consensus < 0.4:
                consensus_performance['low'].append(pnl)
            elif consensus < 0.7:
                consensus_performance['medium'].append(pnl)
            else:
                consensus_performance['high'].append(pnl)
        
        # Adjust consensus threshold based on performance
        for level, pnls in consensus_performance.items():
            if pnls:
                avg_pnl = np.mean(pnls)
                win_rate = sum(1 for pnl in pnls if pnl > 0) / len(pnls)
                
                if level == 'high' and win_rate > 0.8:
                    self.consensus_threshold = max(0.5, self.consensus_threshold - 0.05)
                elif level == 'low' and win_rate < 0.3:
                    self.consensus_threshold = min(0.8, self.consensus_threshold + 0.05)
    
    def get_strategy_performance_summary(self) -> Dict[str, StrategyPerformance]:
        """Get performance summary for all strategies"""
        
        performance_summary = {}
        
        for name, strategy in self.strategies.items():
            performance_summary[name] = strategy.calculate_performance_metrics()
        
        return performance_summary
    
    def update_trade_result(self, symbol: str, trade_result: Dict):
        """Update strategies with trade results"""
        
        contributing_strategies = trade_result.get('contributing_strategies', [])
        
        for strategy_name in contributing_strategies:
            if strategy_name in self.strategies:
                self.strategies[strategy_name].performance_history.append(trade_result)
    
    async def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics"""
        
        if not self.ensemble_history:
            return {}
        
        recent_signals = list(self.ensemble_history)[-100:]  # Last 100 signals
        
        # Direction distribution
        directions = [signal['signal'].direction for signal in recent_signals]
        direction_counts = {
            'BUY': directions.count('BUY'),
            'SELL': directions.count('SELL'),
            'HOLD': directions.count('HOLD')
        }
        
        # Average confidence by direction
        buy_confidences = [s['signal'].confidence for s in recent_signals if s['signal'].direction == 'BUY']
        sell_confidences = [s['signal'].confidence for s in recent_signals if s['signal'].direction == 'SELL']
        
        # Strategy contribution
        all_strategies = []
        for signal in recent_signals:
            all_strategies.extend(signal['signal'].contributing_strategies)
        
        strategy_contribution = {name: all_strategies.count(name) for name in self.strategies.keys()}
        
        return {
            'total_signals': len(recent_signals),
            'direction_distribution': direction_counts,
            'avg_buy_confidence': np.mean(buy_confidences) if buy_confidences else 0,
            'avg_sell_confidence': np.mean(sell_confidences) if sell_confidences else 0,
            'strategy_contribution': strategy_contribution,
            'current_weights': self.strategy_weights,
            'consensus_threshold': self.consensus_threshold
        }

class MultiStrategyEnsembleSystem:
    """
    🎯 MULTI-STRATEGY ENSEMBLE SYSTEM - MAIN CONTROLLER
    
    Wrapper class that provides the expected interface for the enhanced trading bot.
    Consolidates all strategy ensemble functionality with enhanced AI integration.
    """
    
    def __init__(self):
        self.strategy_ensemble = StrategyEnsemble()
        print("🎯 Multi-Strategy Ensemble System initialized")
        print("   ✅ Strategy ensemble controller ready")
        print("   📊 Dynamic weighting system active")
        print("   🧠 Performance-based optimization enabled")
    
    async def initialize(self, symbols: List[str], trading_mode: str = 'PRECISION', 
                       risk_tolerance: float = 0.02):
        """Initialize the multi-strategy ensemble system"""
        
        try:
            print(f"🚀 Initializing Multi-Strategy Ensemble for {trading_mode} mode...")
            
            # Configure ensemble based on trading mode
            if trading_mode == 'PRECISION':
                self.strategy_ensemble.min_confidence_threshold = 0.85
                self.strategy_ensemble.consensus_threshold = 0.8
            elif trading_mode == 'AGGRESSIVE':
                self.strategy_ensemble.min_confidence_threshold = 0.65
                self.strategy_ensemble.consensus_threshold = 0.6
            
            # Configure risk tolerance
            for strategy in self.strategy_ensemble.strategies.values():
                if hasattr(strategy, 'risk_tolerance'):
                    strategy.risk_tolerance = risk_tolerance
            
            print(f"   ✅ Configured for {trading_mode} mode")
            print(f"   🎯 Confidence threshold: {self.strategy_ensemble.min_confidence_threshold:.1%}")
            print(f"   🤝 Consensus requirement: {self.strategy_ensemble.consensus_threshold:.1%}")
            print(f"   🛡️ Risk tolerance: {risk_tolerance:.1%}")
            
        except Exception as e:
            print(f"   ⚠️ Error initializing ensemble: {e}")
    
    async def generate_ensemble_signals(self, market_data: Dict[str, Any], 
                                      trading_mode: str = 'PRECISION',
                                      target_accuracy: float = 0.90) -> List[Any]:
        """Generate ensemble signals for multiple symbols"""
        
        try:
            ensemble_signals = []
            
            for symbol, data in market_data.items():
                if 'prices' not in data or len(data['prices']) < 20:
                    continue
                
                # Prepare market intelligence for strategy ensemble
                market_intelligence = {
                    'regime': {
                        'regime_type': data.get('regime', 'neutral'),
                        'confidence': 0.7
                    },
                    'volatility': {
                        'forecast': data.get('volatility', 0.02)
                    },
                    'sentiment': {
                        'overall_sentiment': 0.5,
                        'confidence': 0.6
                    }
                }
                
                current_price = data['prices'][-1]
                
                # Generate ensemble signal
                ensemble_signal = await self.strategy_ensemble.generate_ensemble_signal(
                    symbol=symbol,
                    price_data=data['prices'],
                    market_intelligence=market_intelligence,
                    current_price=current_price
                )
                
                if ensemble_signal and ensemble_signal.confidence >= (target_accuracy * 0.8):
                    # Convert to AITradingSignal format expected by the bot
                    from ai_trading_engine import AITradingSignal
                    
                    ai_signal = AITradingSignal(
                        symbol=symbol,
                        action=ensemble_signal.direction,
                        confidence=ensemble_signal.confidence,
                        expected_return=2.0,  # Default expected return
                        risk_score=ensemble_signal.risk_adjusted_confidence,
                        time_horizon=120,
                        entry_price=ensemble_signal.entry_price,
                        stop_loss=ensemble_signal.stop_loss,
                        take_profit=ensemble_signal.take_profit,
                        position_size=50.0,  # Default position size
                        strategy_name=f"ENSEMBLE_{ensemble_signal.direction}",
                        ai_reasoning=f"Multi-strategy ensemble: {', '.join(ensemble_signal.reasoning)}",
                        technical_score=ensemble_signal.strength,
                        sentiment_score=0.7,
                        momentum_score=data.get('momentum', 0.0),
                        volatility_score=data.get('volatility', 0.02),
                        timestamp=datetime.now()
                    )
                    
                    ensemble_signals.append(ai_signal)
                    print(f"   ✅ {symbol}: Ensemble signal generated (Confidence: {ensemble_signal.confidence:.1%})")
            
            return ensemble_signals
            
        except Exception as e:
            print(f"   ⚠️ Error generating ensemble signals: {e}")
            return []
    
    def update_strategy_performance(self, strategy_name: str, pnl: float, trade_data: Dict):
        """Update strategy performance (delegation to ensemble)"""
        
        try:
            # Find matching strategy and update performance
            for name, strategy in self.strategy_ensemble.strategies.items():
                if strategy_name.find(name) != -1 or name.find(strategy_name.replace('LEGENDARY_TITAN_', '')) != -1:
                    strategy.performance_history.append({
                        'pnl': pnl,
                        'timestamp': datetime.now(),
                        'trade_data': trade_data
                    })
                    print(f"   📊 Updated {name} performance: ${pnl:+.2f}")
                    break
        except Exception as e:
            print(f"   ⚠️ Error updating strategy performance: {e}")
    
    def get_best_strategies(self, top_n: int = 3) -> List[Any]:
        """Get best performing strategies"""
        
        try:
            # Get performance summary
            performance_summary = self.strategy_ensemble.get_strategy_performance_summary()
            
            # Sort by win rate and recent performance
            strategy_scores = []
            for name, perf in performance_summary.items():
                score = perf.win_rate * 0.6 + perf.recent_performance * 0.4
                strategy_scores.append((name, score, perf))
            
            # Sort and return top strategies
            sorted_strategies = sorted(strategy_scores, key=lambda x: x[1], reverse=True)[:top_n]
            
            # Convert to expected format (return strategy names as enum-like objects)
            from enum import Enum
            class StrategyType(Enum):
                MOMENTUM = "momentum"
                MEAN_REVERSION = "mean_reversion"
                BREAKOUT = "breakout"
                PATTERN = "pattern"
                VOLATILITY = "volatility"
            
            best_strategies = []
            for strategy_name, score, perf in sorted_strategies:
                if "Momentum" in strategy_name:
                    best_strategies.append(StrategyType.MOMENTUM)
                elif "MeanReversion" in strategy_name:
                    best_strategies.append(StrategyType.MEAN_REVERSION)
                elif "Breakout" in strategy_name:
                    best_strategies.append(StrategyType.BREAKOUT)
                elif "Pattern" in strategy_name:
                    best_strategies.append(StrategyType.PATTERN)
                elif "Volatility" in strategy_name:
                    best_strategies.append(StrategyType.VOLATILITY)
                else:
                    best_strategies.append(StrategyType.MOMENTUM)  # Default
            
            return best_strategies if best_strategies else [StrategyType.MOMENTUM, StrategyType.MEAN_REVERSION, StrategyType.BREAKOUT]
            
        except Exception as e:
            print(f"   ⚠️ Error getting best strategies: {e}")
            # Return default strategies
            from enum import Enum
            class StrategyType(Enum):
                MOMENTUM = "momentum"
                MEAN_REVERSION = "mean_reversion"
                BREAKOUT = "breakout"
            
            return [StrategyType.MOMENTUM, StrategyType.MEAN_REVERSION, StrategyType.BREAKOUT]

# Create global instance
multi_strategy_ensemble_system = MultiStrategyEnsembleSystem()

# Global ensemble instance
strategy_ensemble = StrategyEnsemble()

# Export components
__all__ = [
    'MultiStrategyEnsembleSystem',
    'multi_strategy_ensemble_system',
    'strategy_ensemble',
    'StrategyEnsemble',
    'EnsembleSignal',
    'StrategySignal',
    'StrategyPerformance',
    'BaseStrategy',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'PatternRecognitionStrategy',
    'VolatilityStrategy'
]
