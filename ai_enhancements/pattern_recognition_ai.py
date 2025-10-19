"""
🎯 PATTERN RECOGNITION AI
Detects profitable chart patterns: Head & Shoulders, Double Bottom, etc.
Trades ONLY high-probability setups!
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.signal import argrelextrema
from collections import deque


class ChartPattern:
    """Chart pattern types"""
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    CUP_HANDLE = "cup_handle"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"


class PatternRecognitionAI:
    """
    🧠 Advanced Chart Pattern Recognition
    
    Identifies classic patterns with high win rates:
    - Head & Shoulders (80% bearish)
    - Double Bottom (85% bullish)
    - Bull/Bear Flags (75% continuation)
    - Triangles, Wedges, etc.
    """
    
    def __init__(self):
        self.pattern_history = deque(maxlen=100)
        self.pattern_win_rates = self._initialize_win_rates()
        
    def _initialize_win_rates(self) -> Dict:
        """Historical win rates for each pattern"""
        return {
            ChartPattern.DOUBLE_BOTTOM: 0.85,
            ChartPattern.INVERSE_HEAD_SHOULDERS: 0.80,
            ChartPattern.BULL_FLAG: 0.75,
            ChartPattern.CUP_HANDLE: 0.82,
            ChartPattern.ASCENDING_TRIANGLE: 0.78,
            ChartPattern.BREAKOUT: 0.72,
            ChartPattern.HEAD_SHOULDERS: 0.80,
            ChartPattern.DOUBLE_TOP: 0.83,
            ChartPattern.BEAR_FLAG: 0.74,
            ChartPattern.DESCENDING_TRIANGLE: 0.76,
            ChartPattern.BREAKDOWN: 0.70,
            ChartPattern.SYMMETRICAL_TRIANGLE: 0.68,
            ChartPattern.WEDGE_RISING: 0.70,
            ChartPattern.WEDGE_FALLING: 0.72,
            ChartPattern.TRIPLE_TOP: 0.77,
            ChartPattern.TRIPLE_BOTTOM: 0.79
        }
    
    def detect_patterns(self, prices: List[float], volumes: List[float] = None) -> List[Dict]:
        """
        🔍 Detect all chart patterns
        
        Args:
            prices: Price history (50-200 bars)
            volumes: Volume history (optional)
            
        Returns:
            List of detected patterns with:
            {
                'pattern': ChartPattern,
                'confidence': float (0-1),
                'expected_win_rate': float,
                'action': 'BUY' or 'SELL',
                'entry_price': float,
                'target': float,
                'stop_loss': float,
                'pattern_name': str
            }
        """
        if len(prices) < 30:
            return []
        
        prices = np.array(prices)
        detected_patterns = []
        
        # Find peaks and troughs
        peaks = self._find_peaks(prices)
        troughs = self._find_troughs(prices)
        
        # Check for each pattern type
        patterns_to_check = [
            ('double_bottom', self._detect_double_bottom),
            ('double_top', self._detect_double_top),
            ('head_shoulders', self._detect_head_shoulders),
            ('inverse_hs', self._detect_inverse_head_shoulders),
            ('bull_flag', self._detect_bull_flag),
            ('bear_flag', self._detect_bear_flag),
            ('ascending_triangle', self._detect_ascending_triangle),
            ('descending_triangle', self._detect_descending_triangle),
            ('breakout', self._detect_breakout),
            ('cup_handle', self._detect_cup_handle)
        ]
        
        for pattern_name, detector_func in patterns_to_check:
            pattern = detector_func(prices, peaks, troughs)
            if pattern:
                detected_patterns.append(pattern)
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detected_patterns
    
    def _find_peaks(self, prices: np.ndarray, order: int = 5) -> np.ndarray:
        """Find local peaks"""
        peaks = argrelextrema(prices, np.greater, order=order)[0]
        return peaks
    
    def _find_troughs(self, prices: np.ndarray, order: int = 5) -> np.ndarray:
        """Find local troughs"""
        troughs = argrelextrema(prices, np.less, order=order)[0]
        return troughs
    
    def _detect_double_bottom(self, prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> Optional[Dict]:
        """🟢 Detect Double Bottom (BULLISH - 85% win rate)"""
        if len(troughs) < 2:
            return None
        
        # Look for two similar lows
        for i in range(len(troughs) - 1):
            t1_idx = troughs[i]
            t2_idx = troughs[i + 1]
            
            t1_price = prices[t1_idx]
            t2_price = prices[t2_idx]
            
            # Bottoms should be similar (within 2%)
            if abs(t1_price - t2_price) / t1_price < 0.02:
                # Find peak between them
                peak_between = prices[t1_idx:t2_idx].max()
                current_price = prices[-1]
                
                # Confirm breakout above peak
                if current_price > peak_between * 1.01:
                    target = current_price + (peak_between - min(t1_price, t2_price))
                    stop = min(t1_price, t2_price) * 0.98
                    
                    return {
                        'pattern': ChartPattern.DOUBLE_BOTTOM,
                        'confidence': 0.85,
                        'expected_win_rate': self.pattern_win_rates[ChartPattern.DOUBLE_BOTTOM],
                        'action': 'BUY',
                        'entry_price': current_price,
                        'target': target,
                        'stop_loss': stop,
                        'pattern_name': '🟢 DOUBLE BOTTOM (Bullish)',
                        'description': 'Strong bullish reversal pattern'
                    }
        
        return None
    
    def _detect_double_top(self, prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> Optional[Dict]:
        """🔴 Detect Double Top (BEARISH - 83% win rate)"""
        if len(peaks) < 2:
            return None
        
        for i in range(len(peaks) - 1):
            p1_idx = peaks[i]
            p2_idx = peaks[i + 1]
            
            p1_price = prices[p1_idx]
            p2_price = prices[p2_idx]
            
            # Tops should be similar (within 2%)
            if abs(p1_price - p2_price) / p1_price < 0.02:
                # Find trough between them
                trough_between = prices[p1_idx:p2_idx].min()
                current_price = prices[-1]
                
                # Confirm breakdown below trough
                if current_price < trough_between * 0.99:
                    target = current_price - (max(p1_price, p2_price) - trough_between)
                    stop = max(p1_price, p2_price) * 1.02
                    
                    return {
                        'pattern': ChartPattern.DOUBLE_TOP,
                        'confidence': 0.83,
                        'expected_win_rate': self.pattern_win_rates[ChartPattern.DOUBLE_TOP],
                        'action': 'SELL',
                        'entry_price': current_price,
                        'target': target,
                        'stop_loss': stop,
                        'pattern_name': '🔴 DOUBLE TOP (Bearish)',
                        'description': 'Strong bearish reversal pattern'
                    }
        
        return None
    
    def _detect_head_shoulders(self, prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> Optional[Dict]:
        """🔴 Detect Head & Shoulders (BEARISH - 80% win rate)"""
        if len(peaks) < 3:
            return None
        
        # Look for 3 peaks: left shoulder, head, right shoulder
        for i in range(len(peaks) - 2):
            left_shoulder_idx = peaks[i]
            head_idx = peaks[i + 1]
            right_shoulder_idx = peaks[i + 2]
            
            ls_price = prices[left_shoulder_idx]
            head_price = prices[head_idx]
            rs_price = prices[right_shoulder_idx]
            
            # Head should be higher than shoulders
            if head_price > ls_price * 1.02 and head_price > rs_price * 1.02:
                # Shoulders should be similar
                if abs(ls_price - rs_price) / ls_price < 0.03:
                    # Find neckline (support level)
                    neckline = min(prices[left_shoulder_idx:head_idx].min(), 
                                  prices[head_idx:right_shoulder_idx].min())
                    current_price = prices[-1]
                    
                    # Confirm breakdown
                    if current_price < neckline * 0.99:
                        target = current_price - (head_price - neckline)
                        stop = rs_price * 1.02
                        
                        return {
                            'pattern': ChartPattern.HEAD_SHOULDERS,
                            'confidence': 0.80,
                            'expected_win_rate': self.pattern_win_rates[ChartPattern.HEAD_SHOULDERS],
                            'action': 'SELL',
                            'entry_price': current_price,
                            'target': target,
                            'stop_loss': stop,
                            'pattern_name': '🔴 HEAD & SHOULDERS (Bearish)',
                            'description': 'Classic bearish reversal'
                        }
        
        return None
    
    def _detect_inverse_head_shoulders(self, prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> Optional[Dict]:
        """🟢 Detect Inverse Head & Shoulders (BULLISH - 80% win rate)"""
        if len(troughs) < 3:
            return None
        
        for i in range(len(troughs) - 2):
            ls_idx = troughs[i]
            head_idx = troughs[i + 1]
            rs_idx = troughs[i + 2]
            
            ls_price = prices[ls_idx]
            head_price = prices[head_idx]
            rs_price = prices[rs_idx]
            
            # Head should be lower than shoulders
            if head_price < ls_price * 0.98 and head_price < rs_price * 0.98:
                # Shoulders similar
                if abs(ls_price - rs_price) / ls_price < 0.03:
                    # Neckline (resistance)
                    neckline = max(prices[ls_idx:head_idx].max(), 
                                  prices[head_idx:rs_idx].max())
                    current_price = prices[-1]
                    
                    # Confirm breakout
                    if current_price > neckline * 1.01:
                        target = current_price + (neckline - head_price)
                        stop = rs_price * 0.98
                        
                        return {
                            'pattern': ChartPattern.INVERSE_HEAD_SHOULDERS,
                            'confidence': 0.80,
                            'expected_win_rate': self.pattern_win_rates[ChartPattern.INVERSE_HEAD_SHOULDERS],
                            'action': 'BUY',
                            'entry_price': current_price,
                            'target': target,
                            'stop_loss': stop,
                            'pattern_name': '🟢 INVERSE HEAD & SHOULDERS (Bullish)',
                            'description': 'Classic bullish reversal'
                        }
        
        return None
    
    def _detect_bull_flag(self, prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> Optional[Dict]:
        """🟢 Detect Bull Flag (BULLISH CONTINUATION - 75% win rate)"""
        if len(prices) < 40:
            return None
        
        # Look for strong uptrend followed by consolidation
        recent_30 = prices[-30:]
        prior_30 = prices[-60:-30]
        
        # Prior trend should be up
        prior_trend = (prior_30[-1] - prior_30[0]) / prior_30[0]
        if prior_trend < 0.03:  # At least 3% up move
            return None
        
        # Recent should consolidate or drift slightly down
        recent_trend = (recent_30[-1] - recent_30[0]) / recent_30[0]
        if recent_trend > 0.01:  # Should be flat or down
            return None
        
        # Breakout above consolidation
        consolidation_high = recent_30.max()
        current_price = prices[-1]
        
        if current_price > consolidation_high * 1.005:
            target = current_price + prior_trend * current_price
            stop = recent_30.min() * 0.98
            
            return {
                'pattern': ChartPattern.BULL_FLAG,
                'confidence': 0.75,
                'expected_win_rate': self.pattern_win_rates[ChartPattern.BULL_FLAG],
                'action': 'BUY',
                'entry_price': current_price,
                'target': target,
                'stop_loss': stop,
                'pattern_name': '🟢 BULL FLAG (Bullish Continuation)',
                'description': 'Continuation of uptrend'
            }
        
        return None
    
    def _detect_bear_flag(self, prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> Optional[Dict]:
        """🔴 Detect Bear Flag (BEARISH CONTINUATION - 74% win rate)"""
        if len(prices) < 40:
            return None
        
        recent_30 = prices[-30:]
        prior_30 = prices[-60:-30]
        
        # Prior trend should be down
        prior_trend = (prior_30[-1] - prior_30[0]) / prior_30[0]
        if prior_trend > -0.03:  # At least 3% down move
            return None
        
        # Recent consolidation
        recent_trend = (recent_30[-1] - recent_30[0]) / recent_30[0]
        if recent_trend < -0.01:  # Should be flat or up
            return None
        
        # Breakdown below consolidation
        consolidation_low = recent_30.min()
        current_price = prices[-1]
        
        if current_price < consolidation_low * 0.995:
            target = current_price + prior_trend * current_price
            stop = recent_30.max() * 1.02
            
            return {
                'pattern': ChartPattern.BEAR_FLAG,
                'confidence': 0.74,
                'expected_win_rate': self.pattern_win_rates[ChartPattern.BEAR_FLAG],
                'action': 'SELL',
                'entry_price': current_price,
                'target': target,
                'stop_loss': stop,
                'pattern_name': '🔴 BEAR FLAG (Bearish Continuation)',
                'description': 'Continuation of downtrend'
            }
        
        return None
    
    def _detect_ascending_triangle(self, prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> Optional[Dict]:
        """🟢 Ascending Triangle (BULLISH - 78% win rate)"""
        # Simplified detection
        if len(peaks) < 2 or len(troughs) < 2:
            return None
        
        # Peaks at similar level (resistance)
        recent_peaks = peaks[-2:]
        if abs(prices[recent_peaks[0]] - prices[recent_peaks[1]]) / prices[recent_peaks[0]] < 0.02:
            # Troughs rising
            recent_troughs = troughs[-2:]
            if prices[recent_troughs[1]] > prices[recent_troughs[0]] * 1.01:
                current_price = prices[-1]
                resistance = prices[recent_peaks].mean()
                
                if current_price > resistance * 1.01:
                    target = current_price + (current_price - prices[recent_troughs].min())
                    stop = prices[recent_troughs[-1]] * 0.98
                    
                    return {
                        'pattern': ChartPattern.ASCENDING_TRIANGLE,
                        'confidence': 0.78,
                        'expected_win_rate': self.pattern_win_rates[ChartPattern.ASCENDING_TRIANGLE],
                        'action': 'BUY',
                        'entry_price': current_price,
                        'target': target,
                        'stop_loss': stop,
                        'pattern_name': '🟢 ASCENDING TRIANGLE (Bullish)',
                        'description': 'Bullish breakout pattern'
                    }
        
        return None
    
    def _detect_descending_triangle(self, prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> Optional[Dict]:
        """🔴 Descending Triangle (BEARISH - 76% win rate)"""
        if len(peaks) < 2 or len(troughs) < 2:
            return None
        
        # Troughs at similar level (support)
        recent_troughs = troughs[-2:]
        if abs(prices[recent_troughs[0]] - prices[recent_troughs[1]]) / prices[recent_troughs[0]] < 0.02:
            # Peaks declining
            recent_peaks = peaks[-2:]
            if prices[recent_peaks[1]] < prices[recent_peaks[0]] * 0.99:
                current_price = prices[-1]
                support = prices[recent_troughs].mean()
                
                if current_price < support * 0.99:
                    target = current_price - (prices[recent_peaks].max() - current_price)
                    stop = prices[recent_peaks[-1]] * 1.02
                    
                    return {
                        'pattern': ChartPattern.DESCENDING_TRIANGLE,
                        'confidence': 0.76,
                        'expected_win_rate': self.pattern_win_rates[ChartPattern.DESCENDING_TRIANGLE],
                        'action': 'SELL',
                        'entry_price': current_price,
                        'target': target,
                        'stop_loss': stop,
                        'pattern_name': '🔴 DESCENDING TRIANGLE (Bearish)',
                        'description': 'Bearish breakdown pattern'
                    }
        
        return None
    
    def _detect_breakout(self, prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> Optional[Dict]:
        """🟢 Breakout Detection (72% win rate)"""
        if len(prices) < 30:
            return None
        
        # Find recent range
        recent_20 = prices[-20:]
        range_high = recent_20.max()
        range_low = recent_20.min()
        current_price = prices[-1]
        
        # Breakout above range
        if current_price > range_high * 1.01:
            target = current_price + (range_high - range_low)
            stop = range_high * 0.99
            
            return {
                'pattern': ChartPattern.BREAKOUT,
                'confidence': 0.72,
                'expected_win_rate': self.pattern_win_rates[ChartPattern.BREAKOUT],
                'action': 'BUY',
                'entry_price': current_price,
                'target': target,
                'stop_loss': stop,
                'pattern_name': '🟢 BREAKOUT (Bullish)',
                'description': 'Price breaking above resistance'
            }
        
        return None
    
    def _detect_cup_handle(self, prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> Optional[Dict]:
        """🟢 Cup & Handle (BULLISH - 82% win rate)"""
        # Simplified detection
        if len(prices) < 50:
            return None
        
        # Look for U-shape followed by small pullback
        mid_point = len(prices) // 2
        left_half = prices[:mid_point]
        right_half = prices[mid_point:]
        
        # Cup formation
        if left_half[0] > left_half[-1] and right_half[-1] > right_half[0]:
            # Handle (small pullback)
            if right_half[-10] > right_half[-5]:
                current_price = prices[-1]
                cup_high = max(left_half[0], right_half[-10])
                
                if current_price > cup_high * 1.01:
                    cup_depth = cup_high - min(left_half.min(), right_half.min())
                    target = current_price + cup_depth
                    stop = right_half[-10] * 0.98
                    
                    return {
                        'pattern': ChartPattern.CUP_HANDLE,
                        'confidence': 0.82,
                        'expected_win_rate': self.pattern_win_rates[ChartPattern.CUP_HANDLE],
                        'action': 'BUY',
                        'entry_price': current_price,
                        'target': target,
                        'stop_loss': stop,
                        'pattern_name': '🟢 CUP & HANDLE (Bullish)',
                        'description': 'Strong bullish continuation'
                    }
        
        return None
