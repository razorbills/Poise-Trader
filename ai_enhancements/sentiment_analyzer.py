"""
📰 SENTIMENT ANALYSIS AI
Analyzes crypto news and social media sentiment
(Placeholder - can be enhanced with real API integration)
"""

import random
from typing import Dict, List
from collections import deque
from datetime import datetime


class SentimentAnalyzer:
    """
    🧠 Crypto Sentiment Analysis
    
    Analyzes market sentiment from:
    - News headlines
    - Social media (Twitter, Reddit)
    - Market fear/greed indicators
    """
    
    def __init__(self):
        self.sentiment_history = deque(maxlen=100)
        
        # Simulated sentiment keywords (can be replaced with real API)
        self.bullish_keywords = [
            'moon', 'bullish', 'rally', 'breakout', 'surge', 'pump',
            'adoption', 'partnership', 'upgrade', 'buy', 'accumulate'
        ]
        self.bearish_keywords = [
            'crash', 'dump', 'bearish', 'sell', 'fear', 'regulation',
            'hack', 'scam', 'ban', 'decline', 'drop'
        ]
    
    def analyze_sentiment(self, symbol: str = None) -> Dict:
        """
        📊 Analyze current market sentiment
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC/USDT')
            
        Returns:
            {
                'sentiment_score': float (0-100),
                'sentiment_label': 'BULLISH', 'NEUTRAL', or 'BEARISH',
                'confidence': float (0-1),
                'sources': List[Dict],
                'recommendation': str
            }
        """
        # Simplified sentiment (can be enhanced with real API calls)
        # For now, use randomized sentiment with slight bias
        
        # Simulate sentiment score (0-100)
        # In production, this would aggregate real news/social data
        base_sentiment = random.uniform(30, 70)
        
        # Add some randomness
        noise = random.gauss(0, 10)
        sentiment_score = max(0, min(100, base_sentiment + noise))
        
        # Classify sentiment
        if sentiment_score >= 65:
            sentiment_label = 'BULLISH'
            recommendation = 'Positive sentiment supports buying'
        elif sentiment_score <= 35:
            sentiment_label = 'BEARISH'
            recommendation = 'Negative sentiment suggests caution'
        else:
            sentiment_label = 'NEUTRAL'
            recommendation = 'Mixed sentiment, no clear signal'
        
        # Confidence based on how extreme the sentiment is
        confidence = abs(sentiment_score - 50) / 50  # 0 at 50, 1 at 0 or 100
        
        result = {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'confidence': confidence,
            'sources': self._generate_sample_sources(sentiment_label),
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.sentiment_history.append(result)
        
        return result
    
    def _generate_sample_sources(self, sentiment_label: str) -> List[Dict]:
        """Generate sample sentiment sources (placeholder)"""
        sources = []
        
        if sentiment_label == 'BULLISH':
            sources = [
                {'source': 'Twitter', 'sentiment': 'Positive', 'volume': 'High'},
                {'source': 'Reddit', 'sentiment': 'Bullish', 'volume': 'Medium'},
                {'source': 'News', 'sentiment': 'Optimistic', 'volume': 'Medium'}
            ]
        elif sentiment_label == 'BEARISH':
            sources = [
                {'source': 'Twitter', 'sentiment': 'Negative', 'volume': 'High'},
                {'source': 'Reddit', 'sentiment': 'Bearish', 'volume': 'Medium'},
                {'source': 'News', 'sentiment': 'Pessimistic', 'volume': 'Low'}
            ]
        else:
            sources = [
                {'source': 'Twitter', 'sentiment': 'Mixed', 'volume': 'Medium'},
                {'source': 'Reddit', 'sentiment': 'Neutral', 'volume': 'Low'},
                {'source': 'News', 'sentiment': 'Unclear', 'volume': 'Low'}
            ]
        
        return sources
    
    def get_sentiment_trend(self, lookback: int = 10) -> str:
        """Get sentiment trend over last N readings"""
        if len(self.sentiment_history) < lookback:
            return 'INSUFFICIENT_DATA'
        
        recent_scores = [s['sentiment_score'] for s in list(self.sentiment_history)[-lookback:]]
        
        # Calculate trend
        first_half = sum(recent_scores[:lookback//2]) / (lookback//2)
        second_half = sum(recent_scores[lookback//2:]) / (lookback - lookback//2)
        
        if second_half > first_half + 10:
            return 'IMPROVING'
        elif second_half < first_half - 10:
            return 'DETERIORATING'
        else:
            return 'STABLE'
    
    def get_sentiment_stats(self) -> Dict:
        """Get sentiment analysis statistics"""
        if not self.sentiment_history:
            return {
                'avg_sentiment': 50,
                'bullish_pct': 33,
                'bearish_pct': 33,
                'neutral_pct': 33,
                'sample_size': 0
            }
        
        scores = [s['sentiment_score'] for s in self.sentiment_history]
        labels = [s['sentiment_label'] for s in self.sentiment_history]
        
        return {
            'avg_sentiment': sum(scores) / len(scores),
            'bullish_pct': (labels.count('BULLISH') / len(labels)) * 100,
            'bearish_pct': (labels.count('BEARISH') / len(labels)) * 100,
            'neutral_pct': (labels.count('NEUTRAL') / len(labels)) * 100,
            'sample_size': len(self.sentiment_history),
            'current_trend': self.get_sentiment_trend()
        }


class FeatureEngineer:
    """
    💡 FEATURE ENGINEERING AI
    Creates custom indicators from raw data
    """
    
    def __init__(self):
        pass
    
    def engineer_features(self, prices: List[float], volumes: List[float] = None) -> Dict:
        """
        🔧 Engineer advanced features
        
        Args:
            prices: Price history
            volumes: Volume history (optional)
            
        Returns:
            Dictionary of engineered features
        """
        import numpy as np
        
        if len(prices) < 20:
            return {}
        
        prices = np.array(prices)
        
        features = {}
        
        # 1. Volume-Weighted Momentum
        if volumes is not None and len(volumes) > 0:
            volumes = np.array(volumes)
            returns = np.diff(prices) / prices[:-1]
            vwm = np.sum(returns[-20:] * volumes[-20:]) / np.sum(volumes[-20:])
            features['volume_weighted_momentum'] = vwm
        
        # 2. Multi-Timeframe Trend Alignment
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
        
        alignment_score = 0
        if sma_10 > sma_20 > sma_50:
            alignment_score = 1.0  # Perfect bullish alignment
        elif sma_10 < sma_20 < sma_50:
            alignment_score = -1.0  # Perfect bearish alignment
        
        features['trend_alignment'] = alignment_score
        
        # 3. Volatility-Adjusted Support/Resistance
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        volatility = np.std(np.diff(prices) / prices[:-1])
        
        # Adjust support/resistance levels by volatility
        support = recent_low * (1 - volatility)
        resistance = recent_high * (1 + volatility)
        
        features['dynamic_support'] = support
        features['dynamic_resistance'] = resistance
        
        # 4. Price-Volume Correlation
        if volumes is not None and len(volumes) >= 20:
            price_returns = np.diff(prices[-20:]) / prices[-21:-1]
            volume_changes = np.diff(volumes[-20:]) / volumes[-21:-1]
            
            correlation = np.corrcoef(price_returns, volume_changes)[0, 1]
            features['price_volume_correlation'] = correlation if not np.isnan(correlation) else 0
        
        # 5. Order Flow Imbalance (simplified)
        if volumes is not None and len(volumes) >= 10:
            recent_volume = np.mean(volumes[-5:])
            avg_volume = np.mean(volumes[-20:])
            
            imbalance = (recent_volume - avg_volume) / avg_volume
            features['order_flow_imbalance'] = imbalance
        
        # 6. Momentum Oscillator
        if len(prices) >= 30:
            roc = (prices[-1] - prices[-20]) / prices[-20]  # 20-period ROC
            features['momentum_oscillator'] = roc
        
        # 7. Volatility Regime
        if len(prices) >= 50:
            short_vol = np.std(np.diff(prices[-20:]) / prices[-21:-1])
            long_vol = np.std(np.diff(prices[-50:]) / prices[-51:-1])
            
            vol_regime = short_vol / long_vol if long_vol > 0 else 1
            features['volatility_regime'] = vol_regime
        
        return features
    
    def calculate_feature_importance(self, features: Dict, target_correlation: float = None) -> Dict:
        """Calculate importance score for each feature"""
        # Simplified importance based on absolute values
        importance = {}
        
        for feature, value in features.items():
            # Normalize importance (higher absolute value = more important)
            if isinstance(value, (int, float)):
                importance[feature] = min(1.0, abs(value) * 10)
        
        return importance
