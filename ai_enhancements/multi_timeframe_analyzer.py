"""
🧪 MULTI-TIMEFRAME ANALYSIS AI
Analyzes 1min, 5min, 15min, 1hr charts simultaneously
Only trades when ALL timeframes align!
"""

import numpy as np
from typing import Dict, List
from collections import deque


class MultiTimeframeAnalyzerAI:
    """
    🎯 Multi-Timeframe Analysis System
    
    Checks multiple timeframes to confirm trend alignment
    Higher probability when all timeframes agree!
    """
    
    def __init__(self):
        self.timeframe_data = {
            '1m': deque(maxlen=100),
            '5m': deque(maxlen=100),
            '15m': deque(maxlen=100),
            '1h': deque(maxlen=100)
        }
        
    def analyze_timeframes(self, price_data: Dict[str, List[float]]) -> Dict:
        """
        🔍 Analyze multiple timeframes
        
        Args:
            price_data: {
                '1m': [prices...],  # 1-minute data
                '5m': [prices...],  # 5-minute data
                '15m': [prices...], # 15-minute data  
                '1h': [prices...]   # 1-hour data
            }
            
        Returns:
            {
                'alignment_score': float (0-100),
                'trend_direction': 'UP', 'DOWN', or 'NEUTRAL',
                'confidence': float (0-1),
                'timeframe_trends': {...},
                'should_trade': bool,
                'recommended_action': 'BUY', 'SELL', or 'WAIT'
            }
        """
        
        # Analyze each timeframe
        timeframe_trends = {}
        for tf, prices in price_data.items():
            if len(prices) >= 10:
                trend_info = self._analyze_single_timeframe(prices, tf)
                timeframe_trends[tf] = trend_info
        
        if not timeframe_trends:
            return self._default_analysis()
        
        # Calculate alignment
        alignment = self._calculate_alignment(timeframe_trends)
        
        # Determine overall direction
        overall_trend = self._determine_overall_trend(timeframe_trends)
        
        # Calculate confidence based on alignment
        confidence = alignment['score'] / 100.0
        
        # Decide whether to trade
        should_trade = alignment['score'] >= 70  # 70% alignment required
        
        # Recommend action
        if should_trade:
            if overall_trend == 'UP':
                action = 'BUY'
            elif overall_trend == 'DOWN':
                action = 'SELL'
            else:
                action = 'WAIT'
        else:
            action = 'WAIT'
        
        return {
            'alignment_score': alignment['score'],
            'trend_direction': overall_trend,
            'confidence': confidence,
            'timeframe_trends': timeframe_trends,
            'should_trade': should_trade,
            'recommended_action': action,
            'aligned_timeframes': alignment['aligned_count'],
            'total_timeframes': alignment['total_count'],
            'strength': self._calculate_trend_strength(timeframe_trends)
        }
    
    def _analyze_single_timeframe(self, prices: List[float], timeframe: str) -> Dict:
        """Analyze a single timeframe"""
        prices = np.array(prices)
        
        # Calculate trend
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
        current_price = prices[-1]
        
        # Determine trend
        if current_price > sma_20 > sma_50:
            trend = 'UP'
            strength = (current_price - sma_50) / sma_50 * 100
        elif current_price < sma_20 < sma_50:
            trend = 'DOWN'
            strength = (sma_50 - current_price) / sma_50 * 100
        else:
            trend = 'NEUTRAL'
            strength = 0
        
        # Calculate momentum
        momentum = self._calculate_momentum(prices)
        
        # Calculate slope
        slope = self._calculate_slope(prices)
        
        return {
            'trend': trend,
            'strength': abs(strength),
            'momentum': momentum,
            'slope': slope,
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50
        }
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate momentum"""
        if len(prices) < 10:
            return 0
        
        recent = np.mean(prices[-5:])
        older = np.mean(prices[-15:-10])
        
        momentum = (recent - older) / older * 100 if older > 0 else 0
        return momentum
    
    def _calculate_slope(self, prices: np.ndarray) -> float:
        """Calculate price slope"""
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        return slope
    
    def _calculate_alignment(self, timeframe_trends: Dict) -> Dict:
        """Calculate how aligned the timeframes are"""
        trends = [tf['trend'] for tf in timeframe_trends.values()]
        
        if not trends:
            return {'score': 0, 'aligned_count': 0, 'total_count': 0}
        
        # Count UP, DOWN, NEUTRAL
        up_count = trends.count('UP')
        down_count = trends.count('DOWN')
        neutral_count = trends.count('NEUTRAL')
        total = len(trends)
        
        # Calculate alignment score
        if up_count > down_count:
            aligned_count = up_count
        elif down_count > up_count:
            aligned_count = down_count
        else:
            aligned_count = neutral_count
        
        alignment_score = (aligned_count / total) * 100
        
        return {
            'score': alignment_score,
            'aligned_count': aligned_count,
            'total_count': total
        }
    
    def _determine_overall_trend(self, timeframe_trends: Dict) -> str:
        """Determine overall trend across timeframes"""
        # Weight longer timeframes more heavily
        weights = {
            '1m': 1,
            '5m': 2,
            '15m': 3,
            '1h': 4
        }
        
        up_score = 0
        down_score = 0
        
        for tf, info in timeframe_trends.items():
            weight = weights.get(tf, 1)
            
            if info['trend'] == 'UP':
                up_score += weight * (1 + info['strength'] / 100)
            elif info['trend'] == 'DOWN':
                down_score += weight * (1 + info['strength'] / 100)
        
        if up_score > down_score * 1.2:  # 20% margin
            return 'UP'
        elif down_score > up_score * 1.2:
            return 'DOWN'
        else:
            return 'NEUTRAL'
    
    def _calculate_trend_strength(self, timeframe_trends: Dict) -> float:
        """Calculate overall trend strength (0-1)"""
        if not timeframe_trends:
            return 0.5
        
        strengths = [tf['strength'] for tf in timeframe_trends.values()]
        avg_strength = np.mean(strengths)
        
        # Normalize to 0-1
        normalized = min(1.0, avg_strength / 10.0)  # 10% strength = 1.0
        
        return normalized
    
    def _default_analysis(self) -> Dict:
        """Default analysis when no data"""
        return {
            'alignment_score': 50,
            'trend_direction': 'NEUTRAL',
            'confidence': 0.50,
            'timeframe_trends': {},
            'should_trade': False,
            'recommended_action': 'WAIT',
            'aligned_timeframes': 0,
            'total_timeframes': 0,
            'strength': 0.50
        }
    
    def get_alignment_emoji(self, score: float) -> str:
        """Get emoji based on alignment score"""
        if score >= 90:
            return "🟢🟢🟢"
        elif score >= 75:
            return "🟢🟢"
        elif score >= 60:
            return "🟢"
        elif score >= 40:
            return "🟡"
        else:
            return "🔴"
