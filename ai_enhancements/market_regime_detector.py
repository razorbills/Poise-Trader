"""
📊 MARKET REGIME DETECTION AI
Detects bull, bear, sideways, high/low volatility markets
Adapts trading strategy automatically!
"""

import numpy as np
from typing import Dict, List, Tuple
from enum import Enum
from collections import deque


class MarketRegime(Enum):
    """Market condition types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BULLISH_BREAKOUT = "bullish_breakout"
    BEARISH_BREAKDOWN = "bearish_breakdown"


class MarketRegimeDetectorAI:
    """
    🧠 Intelligent Market Regime Detection
    
    Analyzes market conditions and returns:
    - Regime type (bull/bear/sideways)
    - Volatility level (high/low)
    - Recommended strategy adjustments
    """
    
    def __init__(self):
        self.regime_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=50)
        
    def detect_regime(self, prices: List[float], volumes: List[float] = None) -> Dict:
        """
        🎯 Detect current market regime
        
        Args:
            prices: List of recent prices (50-100 bars)
            volumes: Optional volume data
            
        Returns:
            {
                'regime': MarketRegime,
                'confidence': float (0-1),
                'volatility': float,
                'trend_strength': float,
                'recommended_allocation': float,
                'recommended_stop_loss': float,
                'recommended_take_profit': float
            }
        """
        if len(prices) < 20:
            return self._default_regime()
        
        prices = np.array(prices)
        
        # Calculate key metrics
        trend_direction = self._calculate_trend(prices)
        trend_strength = self._calculate_trend_strength(prices)
        volatility = self._calculate_volatility(prices)
        momentum = self._calculate_momentum(prices)
        
        # Detect regime
        regime = self._classify_regime(trend_direction, trend_strength, volatility, momentum)
        
        # Get strategy recommendations
        recommendations = self._get_strategy_recommendations(regime, volatility, trend_strength)
        
        result = {
            'regime': regime,
            'confidence': recommendations['confidence'],
            'volatility': volatility,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'momentum': momentum,
            'recommended_allocation': recommendations['allocation'],
            'recommended_stop_loss': recommendations['stop_loss'],
            'recommended_take_profit': recommendations['take_profit'],
            'recommended_confidence_threshold': recommendations['confidence_threshold']
        }
        
        # Store in history
        self.regime_history.append(regime)
        self.volatility_history.append(volatility)
        
        return result
    
    def _calculate_trend(self, prices: np.ndarray) -> float:
        """Calculate trend direction (-1 to 1)"""
        # Linear regression slope
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize slope relative to price
        normalized_slope = slope / np.mean(prices) * 100
        
        return np.clip(normalized_slope, -1, 1)
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate how strong the trend is (0-1)"""
        # Use R-squared from linear regression
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        predicted = slope * x + intercept
        
        ss_res = np.sum((prices - predicted) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return max(0, min(1, r_squared))
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate volatility (standard deviation of returns)"""
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(len(prices))
        
        return volatility
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate momentum (rate of change)"""
        if len(prices) < 10:
            return 0
        
        # Compare recent vs older prices
        recent_avg = np.mean(prices[-10:])
        older_avg = np.mean(prices[-20:-10])
        
        momentum = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        
        return momentum
    
    def _classify_regime(self, trend_dir: float, trend_strength: float, 
                         volatility: float, momentum: float) -> MarketRegime:
        """Classify the market regime based on metrics"""
        
        # High volatility threshold
        if volatility > 0.05:  # 5% volatility
            if abs(trend_dir) > 0.5 and trend_strength > 0.6:
                if trend_dir > 0:
                    return MarketRegime.BULLISH_BREAKOUT
                else:
                    return MarketRegime.BEARISH_BREAKDOWN
            else:
                return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility
        if volatility < 0.015:  # 1.5% volatility
            return MarketRegime.LOW_VOLATILITY
        
        # Trending markets
        if trend_strength > 0.6:
            if trend_dir > 0.3:
                return MarketRegime.BULL
            elif trend_dir < -0.3:
                return MarketRegime.BEAR
        
        # Default to sideways
        return MarketRegime.SIDEWAYS
    
    def _get_strategy_recommendations(self, regime: MarketRegime, 
                                     volatility: float, 
                                     trend_strength: float) -> Dict:
        """Get recommended strategy parameters for the regime"""
        
        recommendations = {
            MarketRegime.BULL: {
                'allocation': 0.80,  # 80% capital allocation
                'stop_loss': 2.0,    # 2% stop loss
                'take_profit': 4.0,  # 4% take profit
                'confidence_threshold': 0.50,  # Lower threshold, more trades
                'confidence': 0.85
            },
            MarketRegime.BEAR: {
                'allocation': 0.30,  # 30% capital (defensive)
                'stop_loss': 1.5,    # Tighter stop
                'take_profit': 2.5,  # Lower target
                'confidence_threshold': 0.70,  # Higher threshold, fewer trades
                'confidence': 0.80
            },
            MarketRegime.SIDEWAYS: {
                'allocation': 0.50,  # 50% capital
                'stop_loss': 1.5,    # Tight stop for scalping
                'take_profit': 2.0,  # Quick profits
                'confidence_threshold': 0.60,  # Medium threshold
                'confidence': 0.75
            },
            MarketRegime.HIGH_VOLATILITY: {
                'allocation': 0.40,  # Smaller positions
                'stop_loss': 3.0,    # Wider stops (avoid whipsaws)
                'take_profit': 5.0,  # Bigger targets
                'confidence_threshold': 0.65,  # More selective
                'confidence': 0.70
            },
            MarketRegime.LOW_VOLATILITY: {
                'allocation': 0.60,  # Larger positions (safer)
                'stop_loss': 1.0,    # Tight stops
                'take_profit': 1.5,  # Small targets
                'confidence_threshold': 0.55,  # More trades
                'confidence': 0.85
            },
            MarketRegime.BULLISH_BREAKOUT: {
                'allocation': 0.90,  # Maximum allocation!
                'stop_loss': 2.5,    # Medium stop
                'take_profit': 6.0,  # Big target!
                'confidence_threshold': 0.45,  # Aggressive
                'confidence': 0.90
            },
            MarketRegime.BEARISH_BREAKDOWN: {
                'allocation': 0.20,  # Minimal allocation
                'stop_loss': 1.0,    # Very tight stop
                'take_profit': 2.0,  # Small target
                'confidence_threshold': 0.80,  # Very selective
                'confidence': 0.75
            }
        }
        
        return recommendations.get(regime, recommendations[MarketRegime.SIDEWAYS])
    
    def _default_regime(self) -> Dict:
        """Return default regime when not enough data"""
        return {
            'regime': MarketRegime.SIDEWAYS,
            'confidence': 0.50,
            'volatility': 0.02,
            'trend_strength': 0.50,
            'trend_direction': 0.0,
            'momentum': 0.0,
            'recommended_allocation': 0.50,
            'recommended_stop_loss': 2.0,
            'recommended_take_profit': 3.5,
            'recommended_confidence_threshold': 0.60
        }
    
    def get_regime_name(self, regime: MarketRegime) -> str:
        """Get human-readable regime name with emoji"""
        names = {
            MarketRegime.BULL: "🐂 BULL MARKET",
            MarketRegime.BEAR: "🐻 BEAR MARKET",
            MarketRegime.SIDEWAYS: "↔️ SIDEWAYS",
            MarketRegime.HIGH_VOLATILITY: "⚡ HIGH VOLATILITY",
            MarketRegime.LOW_VOLATILITY: "😴 LOW VOLATILITY",
            MarketRegime.BULLISH_BREAKOUT: "🚀 BULLISH BREAKOUT",
            MarketRegime.BEARISH_BREAKDOWN: "💥 BEARISH BREAKDOWN"
        }
        return names.get(regime, "UNKNOWN")
