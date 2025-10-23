#!/usr/bin/env python3
"""
🧠 PROFESSIONAL MARKET PSYCHOLOGY & SENTIMENT SYSTEM
Advanced psychological analysis and sentiment tracking

FEATURES:
✅ Fear & Greed Index Calculation
✅ Market Sentiment Analysis
✅ Crowd Psychology Detection
✅ Smart Money vs Retail Detection
✅ Emotional State Management
✅ FOMO/Panic Detection
✅ Market Manipulation Detection
✅ Whale Activity Tracking
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum

class MarketSentiment(Enum):
    """Market sentiment states"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"

class CrowdBehavior(Enum):
    """Crowd psychology patterns"""
    PANIC_SELLING = "panic_selling"
    CAPITULATION = "capitulation"
    ACCUMULATION = "accumulation"
    FOMO_BUYING = "fomo_buying"
    EUPHORIA = "euphoria"
    DISTRIBUTION = "distribution"

class TraderEmotion(Enum):
    """Individual trader emotional states"""
    CONFIDENT = "confident"
    CAUTIOUS = "cautious"
    FEARFUL = "fearful"
    GREEDY = "greedy"
    TILTED = "tilted"
    FOCUSED = "focused"
    EXHAUSTED = "exhausted"

@dataclass
class PsychologyProfile:
    """Complete psychology profile"""
    market_sentiment: MarketSentiment
    fear_greed_index: float  # 0-100
    crowd_behavior: CrowdBehavior
    smart_money_sentiment: str
    retail_sentiment: str
    manipulation_detected: bool
    whale_activity: str
    contrarian_signal: Optional[str]
    confidence_score: float

class MarketPsychologyAnalyzer:
    """Advanced market psychology analysis"""
    
    def __init__(self):
        self.sentiment_history = deque(maxlen=100)
        self.fear_greed_components = {
            'volatility': 0.25,
            'momentum': 0.25,
            'volume': 0.15,
            'put_call_ratio': 0.15,
            'safe_haven_demand': 0.10,
            'social_sentiment': 0.10
        }
        
    async def analyze_market_psychology(self, market_data: Dict) -> PsychologyProfile:
        """Complete psychological analysis of market"""
        
        # Calculate Fear & Greed Index
        fear_greed = await self._calculate_fear_greed_index(market_data)
        
        # Detect crowd behavior
        crowd = await self._detect_crowd_behavior(market_data, fear_greed)
        
        # Analyze smart money vs retail
        smart_money, retail = await self._analyze_participant_sentiment(market_data)
        
        # Check for manipulation
        manipulation = await self._detect_manipulation(market_data)
        
        # Track whale activity
        whale_activity = await self._track_whale_activity(market_data)
        
        # Generate contrarian signals
        contrarian = self._generate_contrarian_signal(fear_greed, crowd)
        
        # Determine overall sentiment
        sentiment = self._classify_sentiment(fear_greed)
        
        # Calculate confidence
        confidence = self._calculate_confidence(market_data, manipulation)
        
        profile = PsychologyProfile(
            market_sentiment=sentiment,
            fear_greed_index=fear_greed,
            crowd_behavior=crowd,
            smart_money_sentiment=smart_money,
            retail_sentiment=retail,
            manipulation_detected=manipulation,
            whale_activity=whale_activity,
            contrarian_signal=contrarian,
            confidence_score=confidence
        )
        
        # Store in history
        self.sentiment_history.append({
            'timestamp': datetime.now(),
            'profile': profile
        })
        
        return profile
    
    async def _calculate_fear_greed_index(self, market_data: Dict) -> float:
        """Calculate Fear & Greed Index (0-100)"""
        
        components = {}
        
        # Volatility (inverse - high vol = fear)
        volatility = market_data.get('volatility', 0.02)
        vol_score = max(0, min(100, (1 - volatility * 50) * 100))
        components['volatility'] = vol_score
        
        # Momentum
        price_change = market_data.get('price_change_7d', 0)
        momentum_score = 50 + (price_change * 500)  # Scale to 0-100
        momentum_score = max(0, min(100, momentum_score))
        components['momentum'] = momentum_score
        
        # Volume
        volume_ratio = market_data.get('volume_vs_average', 1.0)
        volume_score = min(100, volume_ratio * 50)
        components['volume'] = volume_score
        
        # Put/Call Ratio (inverse)
        put_call = market_data.get('put_call_ratio', 1.0)
        put_call_score = max(0, min(100, (2 - put_call) * 50))
        components['put_call_ratio'] = put_call_score
        
        # Safe haven demand (inverse)
        safe_haven = market_data.get('safe_haven_flow', 0)
        safe_haven_score = max(0, min(100, 100 - safe_haven))
        components['safe_haven_demand'] = safe_haven_score
        
        # Social sentiment
        social = market_data.get('social_sentiment', 0.5)
        social_score = social * 100
        components['social_sentiment'] = social_score
        
        # Weighted average
        total_score = 0
        for component, weight in self.fear_greed_components.items():
            total_score += components.get(component, 50) * weight
        
        return round(total_score, 1)
    
    async def _detect_crowd_behavior(self, market_data: Dict, fear_greed: float) -> CrowdBehavior:
        """Detect current crowd psychology"""
        
        price_change = market_data.get('price_change_24h', 0)
        volume = market_data.get('volume_vs_average', 1.0)
        
        # Extreme fear + high volume + falling = panic selling
        if fear_greed < 20 and volume > 2.0 and price_change < -0.05:
            return CrowdBehavior.PANIC_SELLING
        
        # Extreme fear + low volume = capitulation
        elif fear_greed < 15 and volume < 0.5:
            return CrowdBehavior.CAPITULATION
        
        # Low fear + steady accumulation
        elif 25 < fear_greed < 45 and 0.8 < volume < 1.2:
            return CrowdBehavior.ACCUMULATION
        
        # High greed + high volume + rising = FOMO
        elif fear_greed > 75 and volume > 1.5 and price_change > 0.03:
            return CrowdBehavior.FOMO_BUYING
        
        # Extreme greed = euphoria
        elif fear_greed > 85:
            return CrowdBehavior.EUPHORIA
        
        # High prices + declining volume = distribution
        elif fear_greed > 70 and volume < 0.8:
            return CrowdBehavior.DISTRIBUTION
        
        else:
            return CrowdBehavior.ACCUMULATION
    
    async def _analyze_participant_sentiment(self, market_data: Dict) -> Tuple[str, str]:
        """Analyze smart money vs retail sentiment"""
        
        # Smart money indicators
        large_transactions = market_data.get('large_tx_count', 0)
        exchange_flows = market_data.get('exchange_netflow', 0)
        
        # Retail indicators  
        small_transactions = market_data.get('small_tx_count', 0)
        social_mentions = market_data.get('social_volume', 0)
        
        # Smart money sentiment
        if large_transactions > 100 and exchange_flows < 0:
            smart_money = "ACCUMULATING"
        elif large_transactions > 100 and exchange_flows > 0:
            smart_money = "DISTRIBUTING"
        else:
            smart_money = "NEUTRAL"
        
        # Retail sentiment
        if social_mentions > 1000 and small_transactions > 10000:
            retail = "EUPHORIC"
        elif social_mentions < 100:
            retail = "FEARFUL"
        else:
            retail = "NEUTRAL"
        
        return smart_money, retail
    
    async def _detect_manipulation(self, market_data: Dict) -> bool:
        """Detect potential market manipulation"""
        
        # Spoofing detection
        bid_ask_imbalance = market_data.get('bid_ask_imbalance', 0)
        order_cancellation_rate = market_data.get('order_cancel_rate', 0)
        
        # Wash trading detection
        volume_spike = market_data.get('volume_vs_average', 1.0)
        price_stability = market_data.get('price_volatility', 0.02)
        
        # Pump and dump detection
        social_spike = market_data.get('social_sentiment_spike', False)
        price_spike = abs(market_data.get('price_change_1h', 0)) > 0.05
        
        manipulation_signals = 0
        
        # Check for spoofing
        if abs(bid_ask_imbalance) > 0.3 and order_cancellation_rate > 0.7:
            manipulation_signals += 1
        
        # Check for wash trading
        if volume_spike > 5.0 and price_stability < 0.01:
            manipulation_signals += 1
        
        # Check for pump and dump
        if social_spike and price_spike:
            manipulation_signals += 1
        
        return manipulation_signals >= 2
    
    async def _track_whale_activity(self, market_data: Dict) -> str:
        """Track whale movements"""
        
        whale_transactions = market_data.get('whale_tx_count', 0)
        whale_accumulation = market_data.get('whale_accumulation_trend', 0)
        
        if whale_transactions > 10:
            if whale_accumulation > 0:
                return "WHALES_BUYING"
            else:
                return "WHALES_SELLING"
        else:
            return "WHALES_INACTIVE"
    
    def _generate_contrarian_signal(self, fear_greed: float, crowd: CrowdBehavior) -> Optional[str]:
        """Generate contrarian trading signals"""
        
        # Extreme fear = potential buying opportunity
        if fear_greed < 20 and crowd == CrowdBehavior.PANIC_SELLING:
            return "CONTRARIAN_BUY"
        
        # Extreme greed = potential selling opportunity
        elif fear_greed > 80 and crowd == CrowdBehavior.EUPHORIA:
            return "CONTRARIAN_SELL"
        
        # Capitulation = major buying opportunity
        elif crowd == CrowdBehavior.CAPITULATION:
            return "STRONG_CONTRARIAN_BUY"
        
        return None
    
    def _classify_sentiment(self, fear_greed: float) -> MarketSentiment:
        """Classify market sentiment"""
        
        if fear_greed < 20:
            return MarketSentiment.EXTREME_FEAR
        elif fear_greed < 40:
            return MarketSentiment.FEAR
        elif fear_greed < 60:
            return MarketSentiment.NEUTRAL
        elif fear_greed < 80:
            return MarketSentiment.GREED
        else:
            return MarketSentiment.EXTREME_GREED
    
    def _calculate_confidence(self, market_data: Dict, manipulation: bool) -> float:
        """Calculate confidence in analysis"""
        
        base_confidence = 0.7
        
        # Reduce confidence if manipulation detected
        if manipulation:
            base_confidence *= 0.5
        
        # Increase confidence with more data
        data_quality = market_data.get('data_quality', 0.8)
        base_confidence *= data_quality
        
        return min(1.0, base_confidence)


class PersonalPsychologyManager:
    """Manage trader's personal psychology"""
    
    def __init__(self):
        self.emotional_state = TraderEmotion.FOCUSED
        self.stress_level = 0.3  # 0-1
        self.fatigue_level = 0.0  # 0-1
        self.tilt_indicators = []
        self.performance_impact = 1.0  # Multiplier
        
    async def assess_personal_state(self, trading_stats: Dict) -> Dict:
        """Assess trader's psychological state"""
        
        # Check for tilt indicators
        consecutive_losses = trading_stats.get('consecutive_losses', 0)
        daily_trades = trading_stats.get('daily_trades', 0)
        daily_pnl_pct = trading_stats.get('daily_pnl_pct', 0)
        
        # Update stress level
        if consecutive_losses > 3:
            self.stress_level = min(1.0, self.stress_level + 0.2)
            self.tilt_indicators.append("consecutive_losses")
        
        if daily_pnl_pct < -0.03:
            self.stress_level = min(1.0, self.stress_level + 0.3)
            self.tilt_indicators.append("large_daily_loss")
        
        # Update fatigue
        if daily_trades > 10:
            self.fatigue_level = min(1.0, daily_trades / 20)
            self.tilt_indicators.append("overtrading")
        
        # Determine emotional state
        if self.stress_level > 0.7:
            self.emotional_state = TraderEmotion.TILTED
            self.performance_impact = 0.5  # 50% performance
        elif self.stress_level > 0.5:
            self.emotional_state = TraderEmotion.FEARFUL
            self.performance_impact = 0.7
        elif self.fatigue_level > 0.7:
            self.emotional_state = TraderEmotion.EXHAUSTED
            self.performance_impact = 0.6
        elif daily_pnl_pct > 0.05:
            self.emotional_state = TraderEmotion.GREEDY
            self.performance_impact = 0.8  # Overconfidence penalty
        else:
            self.emotional_state = TraderEmotion.FOCUSED
            self.performance_impact = 1.0
        
        # Generate recommendations
        recommendations = []
        
        if self.emotional_state == TraderEmotion.TILTED:
            recommendations.append("STOP TRADING - Take a break")
            recommendations.append("Review and journal recent trades")
        elif self.emotional_state == TraderEmotion.EXHAUSTED:
            recommendations.append("Reduce position sizes")
            recommendations.append("Focus on A+ setups only")
        elif self.emotional_state == TraderEmotion.GREEDY:
            recommendations.append("Stick to trading plan")
            recommendations.append("Don't increase position sizes")
        
        return {
            'emotional_state': self.emotional_state.value,
            'stress_level': self.stress_level,
            'fatigue_level': self.fatigue_level,
            'performance_impact': self.performance_impact,
            'tilt_indicators': self.tilt_indicators,
            'recommendations': recommendations,
            'should_trade': self.performance_impact > 0.6
        }
    
    def reset_daily(self):
        """Reset psychological state for new day"""
        
        self.stress_level = max(0, self.stress_level - 0.3)
        self.fatigue_level = 0
        self.tilt_indicators = []
        
        if self.stress_level < 0.3:
            self.emotional_state = TraderEmotion.FOCUSED
            self.performance_impact = 1.0


class SentimentAggregator:
    """Aggregate sentiment from multiple sources"""
    
    def __init__(self):
        self.sources = {
            'twitter': 0.2,
            'reddit': 0.15,
            'news': 0.25,
            'on_chain': 0.3,
            'technical': 0.1
        }
        
    async def get_aggregated_sentiment(self, symbol: str) -> Dict:
        """Get aggregated sentiment score"""
        
        sentiments = {}
        
        # Simulate gathering from different sources
        # In production, would call actual APIs
        
        # Twitter sentiment
        sentiments['twitter'] = np.random.uniform(-1, 1)
        
        # Reddit sentiment
        sentiments['reddit'] = np.random.uniform(-1, 1)
        
        # News sentiment
        sentiments['news'] = np.random.uniform(-1, 1)
        
        # On-chain sentiment (netflows, active addresses, etc)
        sentiments['on_chain'] = np.random.uniform(-1, 1)
        
        # Technical sentiment (RSI, moving averages)
        sentiments['technical'] = np.random.uniform(-1, 1)
        
        # Calculate weighted average
        total_sentiment = 0
        for source, weight in self.sources.items():
            total_sentiment += sentiments[source] * weight
        
        # Classify
        if total_sentiment > 0.5:
            classification = "VERY_BULLISH"
        elif total_sentiment > 0.2:
            classification = "BULLISH"
        elif total_sentiment > -0.2:
            classification = "NEUTRAL"
        elif total_sentiment > -0.5:
            classification = "BEARISH"
        else:
            classification = "VERY_BEARISH"
        
        return {
            'symbol': symbol,
            'aggregate_score': round(total_sentiment, 3),
            'classification': classification,
            'sources': sentiments,
            'confidence': self._calculate_sentiment_confidence(sentiments),
            'timestamp': datetime.now()
        }
    
    def _calculate_sentiment_confidence(self, sentiments: Dict) -> float:
        """Calculate confidence in sentiment reading"""
        
        # Check agreement between sources
        values = list(sentiments.values())
        
        # If all sources agree (low standard deviation)
        std_dev = np.std(values)
        
        # Lower std = higher confidence
        confidence = max(0.3, 1.0 - std_dev)
        
        return round(confidence, 2)
