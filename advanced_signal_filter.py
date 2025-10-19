#!/usr/bin/env python3
"""
Advanced Signal Filter - 90% Win Rate Quality Control System
Ultra-selective signal filtering for the LegendaryCryptoTitanBot

Features:
- Multi-layer signal quality assessment
- Historical pattern matching
- Market regime compatibility checks
- Risk/reward ratio optimization
- Confidence score calibration
- Real-time signal validation
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalQuality(Enum):
    """Signal quality classifications"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    REJECTED = "rejected"

class FilterType(Enum):
    """Types of signal filters"""
    CONFIDENCE_FILTER = "confidence_filter"
    RISK_REWARD_FILTER = "risk_reward_filter"
    MARKET_REGIME_FILTER = "market_regime_filter"
    PATTERN_FILTER = "pattern_filter"
    MOMENTUM_FILTER = "momentum_filter"
    VOLUME_FILTER = "volume_filter"
    CORRELATION_FILTER = "correlation_filter"
    HISTORICAL_FILTER = "historical_filter"

class AdvancedSignalFilter:
    """Advanced signal filter for achieving 90% win rate"""
    
    def __init__(self,
                 target_win_rate: float = 0.90,
                 min_risk_reward_ratio: float = 2.0,
                 max_correlation_threshold: float = 0.7,
                 confidence_threshold: float = 0.85):
        """
        Initialize the advanced signal filter
        
        Args:
            target_win_rate: Target win rate (default: 90%)
            min_risk_reward_ratio: Minimum acceptable risk/reward ratio (default: 2.0)
            max_correlation_threshold: Maximum correlation between signals (default: 0.7)
            confidence_threshold: Minimum confidence threshold (default: 85%)
        """
        self.target_win_rate = target_win_rate
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.max_correlation_threshold = max_correlation_threshold
        self.confidence_threshold = confidence_threshold
        
        # Filter performance tracking
        self.filter_performance = {
            FilterType.CONFIDENCE_FILTER: {'passed': 0, 'rejected': 0},
            FilterType.RISK_REWARD_FILTER: {'passed': 0, 'rejected': 0},
            FilterType.MARKET_REGIME_FILTER: {'passed': 0, 'rejected': 0},
            FilterType.PATTERN_FILTER: {'passed': 0, 'rejected': 0},
            FilterType.MOMENTUM_FILTER: {'passed': 0, 'rejected': 0},
            FilterType.VOLUME_FILTER: {'passed': 0, 'rejected': 0},
            FilterType.CORRELATION_FILTER: {'passed': 0, 'rejected': 0},
            FilterType.HISTORICAL_FILTER: {'passed': 0, 'rejected': 0}
        }
        
        # Signal history for pattern analysis
        self.signal_history = {}
        self.execution_results = {}
        
        # Filter weights (importance of each filter)
        self.filter_weights = {
            FilterType.CONFIDENCE_FILTER: 0.25,
            FilterType.RISK_REWARD_FILTER: 0.20,
            FilterType.MARKET_REGIME_FILTER: 0.15,
            FilterType.PATTERN_FILTER: 0.12,
            FilterType.MOMENTUM_FILTER: 0.10,
            FilterType.VOLUME_FILTER: 0.08,
            FilterType.CORRELATION_FILTER: 0.05,
            FilterType.HISTORICAL_FILTER: 0.05
        }
        
        # Dynamic thresholds (adjusted based on performance)
        self.dynamic_thresholds = {
            'confidence': confidence_threshold,
            'risk_reward': min_risk_reward_ratio,
            'momentum_strength': 0.02,
            'volatility_max': 0.05,
            'pattern_score_min': 0.6
        }
        
        logger.info(f"Advanced Signal Filter initialized with {target_win_rate:.0%} win rate target")
    
    async def filter_signals(self, 
                           signals: List,
                           market_data: Dict,
                           portfolio_data: Dict,
                           price_history: Dict[str, List[float]]) -> List:
        """
        Filter signals through multiple quality control layers
        
        Args:
            signals: List of trading signals to filter
            market_data: Current market conditions
            portfolio_data: Current portfolio state
            price_history: Historical price data for all symbols
            
        Returns:
            List of filtered, high-quality signals
        """
        if not signals:
            return []
        
        filtered_signals = []
        filter_results = []
        
        print(f"\n🔬 ADVANCED SIGNAL FILTERING (Target: {self.target_win_rate:.0%} Win Rate)")
        print(f"📊 Processing {len(signals)} signals through quality control...")
        
        for i, signal in enumerate(signals):
            if not hasattr(signal, 'symbol'):
                continue
                
            symbol = signal.symbol
            
            # Initialize filter result tracking
            filter_result = {
                'signal_id': i,
                'symbol': symbol,
                'original_confidence': getattr(signal, 'confidence', 0),
                'filter_scores': {},
                'passed_filters': [],
                'failed_filters': [],
                'overall_quality': SignalQuality.REJECTED,
                'final_score': 0.0
            }
            
            # Layer 1: Confidence Filter
            confidence_score = self._apply_confidence_filter(signal)
            filter_result['filter_scores']['confidence'] = confidence_score
            if confidence_score > 0.5:
                filter_result['passed_filters'].append(FilterType.CONFIDENCE_FILTER)
                self.filter_performance[FilterType.CONFIDENCE_FILTER]['passed'] += 1
            else:
                filter_result['failed_filters'].append(FilterType.CONFIDENCE_FILTER)
                self.filter_performance[FilterType.CONFIDENCE_FILTER]['rejected'] += 1
            
            # Layer 2: Risk/Reward Filter
            rr_score = self._apply_risk_reward_filter(signal)
            filter_result['filter_scores']['risk_reward'] = rr_score
            if rr_score > 0.5:
                filter_result['passed_filters'].append(FilterType.RISK_REWARD_FILTER)
                self.filter_performance[FilterType.RISK_REWARD_FILTER]['passed'] += 1
            else:
                filter_result['failed_filters'].append(FilterType.RISK_REWARD_FILTER)
                self.filter_performance[FilterType.RISK_REWARD_FILTER]['rejected'] += 1
            
            # Layer 3: Market Regime Compatibility
            regime_score = self._apply_market_regime_filter(signal, market_data)
            filter_result['filter_scores']['market_regime'] = regime_score
            if regime_score > 0.5:
                filter_result['passed_filters'].append(FilterType.MARKET_REGIME_FILTER)
                self.filter_performance[FilterType.MARKET_REGIME_FILTER]['passed'] += 1
            else:
                filter_result['failed_filters'].append(FilterType.MARKET_REGIME_FILTER)
                self.filter_performance[FilterType.MARKET_REGIME_FILTER]['rejected'] += 1
            
            # Layer 4: Pattern Quality Filter
            pattern_score = self._apply_pattern_filter(signal, price_history.get(symbol, []))
            filter_result['filter_scores']['pattern'] = pattern_score
            if pattern_score > 0.5:
                filter_result['passed_filters'].append(FilterType.PATTERN_FILTER)
                self.filter_performance[FilterType.PATTERN_FILTER]['passed'] += 1
            else:
                filter_result['failed_filters'].append(FilterType.PATTERN_FILTER)
                self.filter_performance[FilterType.PATTERN_FILTER]['rejected'] += 1
            
            # Layer 5: Momentum Strength Filter
            momentum_score = self._apply_momentum_filter(signal, price_history.get(symbol, []))
            filter_result['filter_scores']['momentum'] = momentum_score
            if momentum_score > 0.5:
                filter_result['passed_filters'].append(FilterType.MOMENTUM_FILTER)
                self.filter_performance[FilterType.MOMENTUM_FILTER]['passed'] += 1
            else:
                filter_result['failed_filters'].append(FilterType.MOMENTUM_FILTER)
                self.filter_performance[FilterType.MOMENTUM_FILTER]['rejected'] += 1
            
            # Layer 6: Volume Confirmation Filter
            volume_score = self._apply_volume_filter(signal, market_data.get(symbol, {}))
            filter_result['filter_scores']['volume'] = volume_score
            if volume_score > 0.5:
                filter_result['passed_filters'].append(FilterType.VOLUME_FILTER)
                self.filter_performance[FilterType.VOLUME_FILTER]['passed'] += 1
            else:
                filter_result['failed_filters'].append(FilterType.VOLUME_FILTER)
                self.filter_performance[FilterType.VOLUME_FILTER]['rejected'] += 1
            
            # Layer 7: Correlation Filter (avoid too many similar signals)
            correlation_score = self._apply_correlation_filter(signal, filtered_signals)
            filter_result['filter_scores']['correlation'] = correlation_score
            if correlation_score > 0.5:
                filter_result['passed_filters'].append(FilterType.CORRELATION_FILTER)
                self.filter_performance[FilterType.CORRELATION_FILTER]['passed'] += 1
            else:
                filter_result['failed_filters'].append(FilterType.CORRELATION_FILTER)
                self.filter_performance[FilterType.CORRELATION_FILTER]['rejected'] += 1
            
            # Layer 8: Historical Performance Filter
            historical_score = self._apply_historical_filter(signal)
            filter_result['filter_scores']['historical'] = historical_score
            if historical_score > 0.5:
                filter_result['passed_filters'].append(FilterType.HISTORICAL_FILTER)
                self.filter_performance[FilterType.HISTORICAL_FILTER]['passed'] += 1
            else:
                filter_result['failed_filters'].append(FilterType.HISTORICAL_FILTER)
                self.filter_performance[FilterType.HISTORICAL_FILTER]['rejected'] += 1
            
            # Calculate final composite score
            final_score = self._calculate_composite_score(filter_result['filter_scores'])
            filter_result['final_score'] = final_score
            
            # Determine overall quality
            if final_score >= 0.9:
                filter_result['overall_quality'] = SignalQuality.EXCELLENT
            elif final_score >= 0.8:
                filter_result['overall_quality'] = SignalQuality.GOOD
            elif final_score >= 0.7:
                filter_result['overall_quality'] = SignalQuality.FAIR
            elif final_score >= 0.6:
                filter_result['overall_quality'] = SignalQuality.POOR
            else:
                filter_result['overall_quality'] = SignalQuality.REJECTED
            
            # Only accept excellent and good signals for 90% target
            if filter_result['overall_quality'] in [SignalQuality.EXCELLENT, SignalQuality.GOOD]:
                # Enhance signal with filter information
                enhanced_signal = self._enhance_signal_with_filter_data(signal, filter_result)
                filtered_signals.append(enhanced_signal)
                
                print(f"   ✅ {symbol}: {filter_result['overall_quality'].value.upper()} "
                      f"(Score: {final_score:.2f}, Filters: {len(filter_result['passed_filters'])}/8)")
            else:
                print(f"   ❌ {symbol}: {filter_result['overall_quality'].value.upper()} "
                      f"(Score: {final_score:.2f}, Failed: {len(filter_result['failed_filters'])} filters)")
            
            filter_results.append(filter_result)
        
        # Update dynamic thresholds based on filter performance
        self._update_dynamic_thresholds()
        
        acceptance_rate = len(filtered_signals) / len(signals) * 100 if signals else 0
        print(f"🎯 Filter Results: {len(filtered_signals)}/{len(signals)} signals passed "
              f"({acceptance_rate:.1f}% acceptance rate)")
        
        return filtered_signals
    
    def _apply_confidence_filter(self, signal) -> float:
        """Apply confidence-based filtering with dynamic calibration"""
        confidence = getattr(signal, 'confidence', 0)
        
        # Base score from confidence
        if confidence >= self.dynamic_thresholds['confidence']:
            base_score = confidence
        else:
            base_score = confidence / 2  # Penalize low confidence
        
        # Boost score if confidence is exceptionally high
        if confidence >= 0.95:
            base_score *= 1.1
        elif confidence >= 0.90:
            base_score *= 1.05
        
        return min(1.0, base_score)
    
    def _apply_risk_reward_filter(self, signal) -> float:
        """Apply risk/reward ratio filtering"""
        try:
            entry_price = getattr(signal, 'entry_price', 0)
            stop_loss = getattr(signal, 'stop_loss', 0)
            take_profit = getattr(signal, 'take_profit', 0)
            
            if entry_price == 0 or stop_loss == 0 or take_profit == 0:
                return 0.3  # Low score for incomplete signal data
            
            # Calculate risk and reward
            if hasattr(signal, 'action') and signal.action == 'BUY':
                risk = abs(entry_price - stop_loss) / entry_price
                reward = abs(take_profit - entry_price) / entry_price
            else:  # SELL
                risk = abs(stop_loss - entry_price) / entry_price
                reward = abs(entry_price - take_profit) / entry_price
            
            if risk == 0:
                return 0.2  # Very low score for zero risk (unrealistic)
            
            risk_reward_ratio = reward / risk
            
            # Score based on risk/reward ratio
            if risk_reward_ratio >= self.dynamic_thresholds['risk_reward']:
                # Excellent risk/reward
                score = 0.8 + min(0.2, (risk_reward_ratio - self.min_risk_reward_ratio) / 5)
            elif risk_reward_ratio >= 1.5:
                # Acceptable risk/reward
                score = 0.6 + (risk_reward_ratio - 1.5) / 2 * 0.2
            elif risk_reward_ratio >= 1.0:
                # Marginal risk/reward
                score = 0.4 + (risk_reward_ratio - 1.0) / 0.5 * 0.2
            else:
                # Poor risk/reward
                score = risk_reward_ratio / 1.0 * 0.4
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error in risk/reward filter: {e}")
            return 0.3
    
    def _apply_market_regime_filter(self, signal, market_data: Dict) -> float:
        """Apply market regime compatibility filtering"""
        try:
            strategy = getattr(signal, 'strategy_name', 'unknown')
            action = getattr(signal, 'action', 'unknown')
            
            # Extract market regime information
            regime = market_data.get('regime', 'sideways')
            volatility = market_data.get('volatility', 0.02)
            trend_strength = market_data.get('trend_strength', 0.5)
            
            score = 0.5  # Base score
            
            # Strategy-regime compatibility matrix
            if 'MOMENTUM' in strategy.upper():
                if regime in ['bull_trend', 'bear_trend'] and trend_strength > 0.6:
                    score += 0.3  # Momentum works well in trending markets
                elif regime in ['volatile', 'sideways']:
                    score -= 0.2  # Momentum struggles in choppy markets
            
            elif 'MEAN_REVERSION' in strategy.upper():
                if regime in ['sideways', 'consolidation'] and volatility < 0.03:
                    score += 0.3  # Mean reversion works in range-bound markets
                elif regime in ['bull_trend', 'bear_trend']:
                    score -= 0.2  # Mean reversion fights trends
            
            elif 'TREND_FOLLOWING' in strategy.upper():
                if regime in ['bull_trend', 'bear_trend'] and trend_strength > 0.7:
                    score += 0.4  # Trend following excels in strong trends
                elif regime in ['sideways', 'volatile']:
                    score -= 0.3  # Trend following fails in choppy markets
            
            elif 'BREAKOUT' in strategy.upper():
                if regime == 'consolidation' or volatility > 0.04:
                    score += 0.3  # Breakouts work after consolidation or in volatility
            
            elif 'SCALPING' in strategy.upper():
                if volatility > 0.02 and volatility < 0.05:
                    score += 0.2  # Scalping needs moderate volatility
                elif volatility > 0.05:
                    score -= 0.2  # Too much volatility for scalping
            
            # Action-regime compatibility
            if action == 'BUY':
                if regime in ['bull_trend', 'accumulation']:
                    score += 0.1
                elif regime in ['bear_trend', 'distribution', 'crash']:
                    score -= 0.2
            elif action == 'SELL':
                if regime in ['bear_trend', 'distribution', 'crash']:
                    score += 0.1
                elif regime in ['bull_trend', 'accumulation']:
                    score -= 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error in market regime filter: {e}")
            return 0.5
    
    def _apply_pattern_filter(self, signal, price_history: List[float]) -> float:
        """Apply pattern quality filtering"""
        try:
            if len(price_history) < 30:
                return 0.4  # Lower score for insufficient data
            
            symbol = getattr(signal, 'symbol', 'unknown')
            action = getattr(signal, 'action', 'unknown')
            
            pattern_score = 0.5  # Base score
            
            # Check for classic patterns that support the signal
            if self._detect_bull_flag(price_history) and action == 'BUY':
                pattern_score += 0.3
            elif self._detect_bear_flag(price_history) and action == 'SELL':
                pattern_score += 0.3
            
            # Check for reversal patterns
            if self._detect_hammer_pattern(price_history) and action == 'BUY':
                pattern_score += 0.2
            elif self._detect_shooting_star_pattern(price_history) and action == 'SELL':
                pattern_score += 0.2
            
            # Check for continuation patterns
            if self._detect_continuation_pattern(price_history, action):
                pattern_score += 0.25
            
            # Check for volume confirmation (simulated)
            if self._has_volume_confirmation(price_history, action):
                pattern_score += 0.15
            
            # Penalize for conflicting patterns
            if self._has_conflicting_patterns(price_history, action):
                pattern_score -= 0.3
            
            return max(0.0, min(1.0, pattern_score))
            
        except Exception as e:
            logger.error(f"Error in pattern filter: {e}")
            return 0.5
    
    def _apply_momentum_filter(self, signal, price_history: List[float]) -> float:
        """Apply momentum strength filtering"""
        try:
            if len(price_history) < 10:
                return 0.4
            
            action = getattr(signal, 'action', 'unknown')
            
            # Calculate multi-period momentum
            short_momentum = self._calculate_momentum(price_history, 3)
            medium_momentum = self._calculate_momentum(price_history, 7)
            long_momentum = self._calculate_momentum(price_history, 15)
            
            # Weighted momentum score
            weighted_momentum = (short_momentum * 0.5 + medium_momentum * 0.3 + long_momentum * 0.2)
            
            score = 0.5  # Base score
            
            # Check momentum alignment with signal direction
            if action == 'BUY' and weighted_momentum > self.dynamic_thresholds['momentum_strength']:
                score += 0.4
            elif action == 'SELL' and weighted_momentum < -self.dynamic_thresholds['momentum_strength']:
                score += 0.4
            
            # Bonus for strong momentum alignment across timeframes
            momentum_alignment = self._calculate_momentum_alignment(short_momentum, medium_momentum, long_momentum, action)
            score += momentum_alignment * 0.3
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error in momentum filter: {e}")
            return 0.5
    
    def _apply_volume_filter(self, signal, market_data: Dict) -> float:
        """Apply volume confirmation filtering"""
        try:
            # Simulate volume analysis (would use real volume data)
            volume_trend = market_data.get('volume_trend', 'neutral')
            relative_volume = market_data.get('relative_volume', 1.0)
            action = getattr(signal, 'action', 'unknown')
            
            score = 0.5  # Base score
            
            # Volume confirmation for the signal direction
            if action == 'BUY':
                if volume_trend == 'increasing' and relative_volume > 1.2:
                    score += 0.3  # High volume supports buying
                elif volume_trend == 'decreasing':
                    score -= 0.2  # Low volume doesn't support buying
            elif action == 'SELL':
                if volume_trend == 'increasing' and relative_volume > 1.2:
                    score += 0.3  # High volume supports selling (distribution)
                elif volume_trend == 'decreasing':
                    score -= 0.1  # Low volume is neutral for selling
            
            # Bonus for exceptionally high volume (breakout confirmation)
            if relative_volume > 2.0:
                score += 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error in volume filter: {e}")
            return 0.6  # Neutral score if volume data unavailable
    
    def _apply_correlation_filter(self, signal, existing_signals: List) -> float:
        """Apply correlation filtering to avoid redundant signals"""
        try:
            if not existing_signals:
                return 1.0  # Perfect score if no existing signals
            
            symbol = getattr(signal, 'symbol', 'unknown')
            action = getattr(signal, 'action', 'unknown')
            strategy = getattr(signal, 'strategy_name', 'unknown')
            
            correlation_penalties = 0
            
            for existing_signal in existing_signals:
                existing_symbol = getattr(existing_signal, 'symbol', 'unknown')
                existing_action = getattr(existing_signal, 'action', 'unknown')
                existing_strategy = getattr(existing_signal, 'strategy_name', 'unknown')
                
                # Penalize for same symbol
                if existing_symbol == symbol:
                    correlation_penalties += 0.5
                
                # Penalize for same action across highly correlated pairs
                if self._are_correlated_pairs(symbol, existing_symbol) and action == existing_action:
                    correlation_penalties += 0.3
                
                # Penalize for same strategy type
                if self._are_similar_strategies(strategy, existing_strategy):
                    correlation_penalties += 0.2
            
            # Calculate final correlation score
            max_acceptable_correlation = len(existing_signals) * 0.2  # Allow some correlation
            
            if correlation_penalties <= max_acceptable_correlation:
                return 1.0
            else:
                return max(0.0, 1.0 - (correlation_penalties - max_acceptable_correlation))
                
        except Exception as e:
            logger.error(f"Error in correlation filter: {e}")
            return 0.7
    
    def _apply_historical_filter(self, signal) -> float:
        """Apply historical performance filtering based on similar past signals"""
        try:
            symbol = getattr(signal, 'symbol', 'unknown')
            strategy = getattr(signal, 'strategy_name', 'unknown')
            confidence = getattr(signal, 'confidence', 0)
            
            # Look for similar historical signals
            similar_signals = self._find_similar_historical_signals(symbol, strategy, confidence)
            
            if not similar_signals:
                return 0.7  # Neutral score if no historical data
            
            # Calculate historical success rate
            successful_signals = sum(1 for s in similar_signals if s.get('outcome', 'unknown') == 'profit')
            success_rate = successful_signals / len(similar_signals)
            
            # Score based on historical performance
            if success_rate >= 0.9:
                return 1.0
            elif success_rate >= 0.8:
                return 0.9
            elif success_rate >= 0.7:
                return 0.8
            elif success_rate >= 0.6:
                return 0.6
            else:
                return 0.3  # Poor historical performance
                
        except Exception as e:
            logger.error(f"Error in historical filter: {e}")
            return 0.6
    
    def _calculate_composite_score(self, filter_scores: Dict[str, float]) -> float:
        """Calculate weighted composite score from all filters"""
        total_score = 0.0
        total_weight = 0.0
        
        for filter_name, score in filter_scores.items():
            # Map filter name to FilterType enum
            filter_type = None
            for ft in FilterType:
                if ft.value.replace('_filter', '') == filter_name:
                    filter_type = ft
                    break
            
            if filter_type and filter_type in self.filter_weights:
                weight = self.filter_weights[filter_type]
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _enhance_signal_with_filter_data(self, signal, filter_result: Dict):
        """Enhance signal with filtering information"""
        # Add filter metadata to signal
        if hasattr(signal, '__dict__'):
            signal.filter_quality = filter_result['overall_quality'].value
            signal.filter_score = filter_result['final_score']
            signal.passed_filters = len(filter_result['passed_filters'])
            signal.total_filters = len(filter_result['filter_scores'])
            signal.filter_confidence_boost = filter_result['final_score'] * 0.1
            
            # Boost the signal's confidence based on filter score
            original_confidence = getattr(signal, 'confidence', 0)
            enhanced_confidence = min(0.99, original_confidence + signal.filter_confidence_boost)
            signal.confidence = enhanced_confidence
        
        return signal
    
    def _calculate_momentum(self, price_history: List[float], period: int) -> float:
        """Calculate momentum over a specific period"""
        if len(price_history) <= period:
            return 0.0
        
        return (price_history[-1] - price_history[-period]) / price_history[-period]
    
    def _calculate_momentum_alignment(self, 
                                    short_momentum: float, 
                                    medium_momentum: float, 
                                    long_momentum: float, 
                                    action: str) -> float:
        """Calculate how well momentum aligns across timeframes"""
        momentums = [short_momentum, medium_momentum, long_momentum]
        
        if action == 'BUY':
            # For buy signals, we want positive momentum
            aligned_count = sum(1 for m in momentums if m > 0)
        else:  # SELL
            # For sell signals, we want negative momentum
            aligned_count = sum(1 for m in momentums if m < 0)
        
        return aligned_count / 3  # Perfect alignment = 1.0
    
    def _detect_bull_flag(self, price_history: List[float]) -> bool:
        """Detect bull flag pattern"""
        if len(price_history) < 20:
            return False
        
        # Look for initial strong move up, then sideways consolidation
        early_prices = price_history[-20:-10]
        recent_prices = price_history[-10:]
        
        # Strong initial move
        initial_move = (early_prices[-1] - early_prices[0]) / early_prices[0]
        
        # Sideways consolidation (low volatility)
        consolidation_volatility = np.std(recent_prices) / np.mean(recent_prices)
        
        return initial_move > 0.03 and consolidation_volatility < 0.02
    
    def _detect_bear_flag(self, price_history: List[float]) -> bool:
        """Detect bear flag pattern"""
        if len(price_history) < 20:
            return False
        
        # Look for initial strong move down, then sideways consolidation
        early_prices = price_history[-20:-10]
        recent_prices = price_history[-10:]
        
        # Strong initial move down
        initial_move = (early_prices[-1] - early_prices[0]) / early_prices[0]
        
        # Sideways consolidation
        consolidation_volatility = np.std(recent_prices) / np.mean(recent_prices)
        
        return initial_move < -0.03 and consolidation_volatility < 0.02
    
    def _detect_hammer_pattern(self, price_history: List[float]) -> bool:
        """Detect hammer candlestick pattern (bullish reversal)"""
        if len(price_history) < 5:
            return False
        
        # Simplified hammer detection using price action
        recent_prices = price_history[-5:]
        
        # Look for price decline followed by recovery
        low_point = min(recent_prices[:-1])
        current_price = recent_prices[-1]
        previous_high = max(price_history[-10:-5]) if len(price_history) >= 10 else recent_prices[0]
        
        # Hammer: significant decline from high, then recovery
        decline = (previous_high - low_point) / previous_high
        recovery = (current_price - low_point) / low_point
        
        return decline > 0.02 and recovery > 0.01
    
    def _detect_shooting_star_pattern(self, price_history: List[float]) -> bool:
        """Detect shooting star pattern (bearish reversal)"""
        if len(price_history) < 5:
            return False
        
        recent_prices = price_history[-5:]
        
        # Look for price spike followed by decline
        high_point = max(recent_prices[:-1])
        current_price = recent_prices[-1]
        previous_low = min(price_history[-10:-5]) if len(price_history) >= 10 else recent_prices[0]
        
        # Shooting star: significant rise from low, then decline
        rise = (high_point - previous_low) / previous_low
        decline = (high_point - current_price) / high_point
        
        return rise > 0.02 and decline > 0.01
    
    def _detect_continuation_pattern(self, price_history: List[float], action: str) -> bool:
        """Detect continuation patterns that support the signal direction"""
        if len(price_history) < 15:
            return False
        
        # Calculate overall trend direction
        trend = (price_history[-1] - price_history[-15]) / price_history[-15]
        
        # Check for pullback/continuation setup
        recent_pullback = (price_history[-5] - price_history[-1]) / price_history[-5]
        
        if action == 'BUY':
            # Bullish continuation: uptrend with small pullback
            return trend > 0.02 and recent_pullback > 0.005 and recent_pullback < 0.03
        else:  # SELL
            # Bearish continuation: downtrend with small bounce
            return trend < -0.02 and recent_pullback < -0.005 and recent_pullback > -0.03
    
    def _has_volume_confirmation(self, price_history: List[float], action: str) -> bool:
        """Check for volume confirmation (simulated)"""
        # In real implementation, this would analyze actual volume data
        # For simulation, we'll use price volatility as a proxy
        if len(price_history) < 5:
            return False
        
        recent_volatility = np.std(price_history[-5:]) / np.mean(price_history[-5:])
        
        # Higher volatility suggests higher volume activity
        return recent_volatility > 0.015  # Threshold for "high volume"
    
    def _has_conflicting_patterns(self, price_history: List[float], action: str) -> bool:
        """Check for patterns that conflict with the signal"""
        if len(price_history) < 10:
            return False
        
        # Check for opposite patterns
        if action == 'BUY':
            # Conflicting patterns for buy signals
            return (self._detect_shooting_star_pattern(price_history) or 
                   self._detect_bear_flag(price_history))
        else:  # SELL
            # Conflicting patterns for sell signals
            return (self._detect_hammer_pattern(price_history) or 
                   self._detect_bull_flag(price_history))
    
    def _are_correlated_pairs(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are highly correlated"""
        # Common crypto correlations
        btc_pairs = ['BTC/USDT', 'BTC/USD', 'BTCUSDT']
        eth_pairs = ['ETH/USDT', 'ETH/USD', 'ETHUSDT']
        
        # BTC and ETH are highly correlated
        if (symbol1 in btc_pairs and symbol2 in eth_pairs) or \
           (symbol1 in eth_pairs and symbol2 in btc_pairs):
            return True
        
        # Same base currency pairs are correlated
        base1 = symbol1.split('/')[0] if '/' in symbol1 else symbol1[:3]
        base2 = symbol2.split('/')[0] if '/' in symbol2 else symbol2[:3]
        
        return base1 == base2
    
    def _are_similar_strategies(self, strategy1: str, strategy2: str) -> bool:
        """Check if two strategies are similar"""
        # Group similar strategies
        trend_strategies = ['MOMENTUM', 'TREND_FOLLOWING', 'TREND_FOLLOW']
        reversal_strategies = ['MEAN_REVERSION', 'REVERSAL']
        breakout_strategies = ['BREAKOUT', 'VOLATILITY_BREAKOUT']
        scalping_strategies = ['SCALPING', 'SCALP', 'MICRO_SCALP']
        
        for group in [trend_strategies, reversal_strategies, breakout_strategies, scalping_strategies]:
            if any(s in strategy1.upper() for s in group) and any(s in strategy2.upper() for s in group):
                return True
        
        return False
    
    def _find_similar_historical_signals(self, symbol: str, strategy: str, confidence: float) -> List[Dict]:
        """Find similar historical signals for comparison"""
        similar_signals = []
        
        if symbol in self.signal_history:
            for historical_signal in self.signal_history[symbol]:
                # Check similarity criteria
                confidence_diff = abs(historical_signal.get('confidence', 0) - confidence)
                strategy_match = strategy in historical_signal.get('strategy', '')
                
                if confidence_diff < 0.1 and strategy_match:
                    similar_signals.append(historical_signal)
        
        return similar_signals[-20:]  # Return last 20 similar signals
    
    def _update_dynamic_thresholds(self):
        """Update dynamic thresholds based on filter performance"""
        try:
            # Calculate overall filter effectiveness
            total_passed = sum(perf['passed'] for perf in self.filter_performance.values())
            total_rejected = sum(perf['rejected'] for perf in self.filter_performance.values())
            
            if total_passed + total_rejected < 20:  # Need minimum data
                return
            
            acceptance_rate = total_passed / (total_passed + total_rejected)
            
            # If acceptance rate is too high (>30%), increase selectivity
            if acceptance_rate > 0.3:
                self.dynamic_thresholds['confidence'] = min(0.95, self.dynamic_thresholds['confidence'] + 0.01)
                self.dynamic_thresholds['risk_reward'] = min(4.0, self.dynamic_thresholds['risk_reward'] + 0.1)
                self.dynamic_thresholds['pattern_score_min'] += 0.02
            
            # If acceptance rate is too low (<5%), decrease selectivity
            elif acceptance_rate < 0.05:
                self.dynamic_thresholds['confidence'] = max(0.7, self.dynamic_thresholds['confidence'] - 0.01)
                self.dynamic_thresholds['risk_reward'] = max(1.5, self.dynamic_thresholds['risk_reward'] - 0.1)
                self.dynamic_thresholds['pattern_score_min'] -= 0.02
            
            logger.info(f"Dynamic thresholds updated - acceptance rate: {acceptance_rate:.1%}")
            
        except Exception as e:
            logger.error(f"Error updating dynamic thresholds: {e}")
    
    def record_signal_outcome(self, signal, outcome: str, actual_pnl: float):
        """Record the outcome of a signal for learning"""
        try:
            symbol = getattr(signal, 'symbol', 'unknown')
            
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            
            # Record signal and its outcome
            signal_record = {
                'timestamp': datetime.now().isoformat(),
                'strategy': getattr(signal, 'strategy_name', 'unknown'),
                'action': getattr(signal, 'action', 'unknown'),
                'confidence': getattr(signal, 'confidence', 0),
                'expected_return': getattr(signal, 'expected_return', 0),
                'filter_score': getattr(signal, 'filter_score', 0),
                'outcome': outcome,
                'actual_pnl': actual_pnl,
                'success': outcome == 'profit'
            }
            
            self.signal_history[symbol].append(signal_record)
            
            # Keep history manageable
            if len(self.signal_history[symbol]) > 100:
                self.signal_history[symbol] = self.signal_history[symbol][-100:]
            
            logger.info(f"Recorded outcome for {symbol}: {outcome} (P&L: ${actual_pnl:+.2f})")
            
        except Exception as e:
            logger.error(f"Error recording signal outcome: {e}")
    
    def get_filter_performance_report(self) -> Dict:
        """Get detailed performance report of all filters"""
        report = {
            'overall_target': self.target_win_rate,
            'current_thresholds': self.dynamic_thresholds,
            'filter_weights': {ft.value: weight for ft, weight in self.filter_weights.items()},
            'filter_performance': {}
        }
        
        # Calculate performance for each filter
        for filter_type, performance in self.filter_performance.items():
            total = performance['passed'] + performance['rejected']
            acceptance_rate = performance['passed'] / total if total > 0 else 0
            
            report['filter_performance'][filter_type.value] = {
                'total_signals': total,
                'passed': performance['passed'],
                'rejected': performance['rejected'],
                'acceptance_rate': acceptance_rate
            }
        
        return report
    
    def adjust_filter_weights(self, performance_feedback: Dict[str, float]):
        """Adjust filter weights based on performance feedback"""
        try:
            for filter_name, effectiveness in performance_feedback.items():
                # Find corresponding FilterType
                for filter_type in FilterType:
                    if filter_type.value == filter_name:
                        current_weight = self.filter_weights.get(filter_type, 0.1)
                        
                        # Adjust weight based on effectiveness
                        if effectiveness > 0.8:  # High effectiveness
                            new_weight = min(0.4, current_weight * 1.1)
                        elif effectiveness < 0.5:  # Low effectiveness
                            new_weight = max(0.02, current_weight * 0.9)
                        else:
                            new_weight = current_weight
                        
                        self.filter_weights[filter_type] = new_weight
                        break
            
            # Normalize weights to sum to 1
            total_weight = sum(self.filter_weights.values())
            if total_weight != 1.0:
                for filter_type in self.filter_weights:
                    self.filter_weights[filter_type] /= total_weight
            
            logger.info("Filter weights adjusted based on performance feedback")
            
        except Exception as e:
            logger.error(f"Error adjusting filter weights: {e}")

# Example integration with the main trading bot:
#
# # In the main bot's signal generation:
# signal_filter = AdvancedSignalFilter(target_win_rate=0.90)
# 
# # Filter signals before execution
# high_quality_signals = await signal_filter.filter_signals(
#     signals=raw_signals,
#     market_data=market_data,
#     portfolio_data=portfolio,
#     price_history=self.price_history
# )
#
# # Record outcomes for learning
# # After trade completion:
# signal_filter.record_signal_outcome(signal, 'profit', actual_pnl)
