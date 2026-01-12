#!/usr/bin/env python3
"""
📊 ALTERNATIVE DATA INTEGRATION SYSTEM
Social Sentiment, On-Chain Analytics, Options Flow & Macro Economic Data
"""

import asyncio
import aiohttp
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import re
from collections import defaultdict, deque
import hashlib
import os

_IS_RENDER = bool(os.getenv('RENDER_EXTERNAL_URL') or os.getenv('RENDER_SERVICE_NAME'))

TextBlob = None
Web3 = None
tweepy = None
praw = None
requests = None

SOCIAL_AVAILABLE = False
ONCHAIN_AVAILABLE = False


def _ensure_social_imports() -> bool:
    global tweepy, praw, TextBlob, requests, SOCIAL_AVAILABLE
    try:
        if SOCIAL_AVAILABLE and TextBlob is not None:
            return True
        if _IS_RENDER:
            # Defer heavy imports on Render unless explicitly needed.
            pass
        import requests as _requests
        import tweepy as _tweepy
        import praw as _praw
        from textblob import TextBlob as _TextBlob
        requests = _requests
        tweepy = _tweepy
        praw = _praw
        TextBlob = _TextBlob
        SOCIAL_AVAILABLE = True
        return True
    except Exception:
        SOCIAL_AVAILABLE = False
        return False


def _ensure_onchain_imports() -> bool:
    global Web3, requests, ONCHAIN_AVAILABLE
    try:
        if ONCHAIN_AVAILABLE and Web3 is not None:
            return True
        if _IS_RENDER:
            pass
        import requests as _requests
        from web3 import Web3 as _Web3
        requests = _requests
        Web3 = _Web3
        ONCHAIN_AVAILABLE = True
        return True
    except Exception:
        ONCHAIN_AVAILABLE = False
        return False

@dataclass
class SentimentData:
    """Social sentiment metrics"""
    source: str
    symbol: str
    sentiment_score: float  # -1 to 1
    volume: int  # Number of mentions
    trending_score: float
    fear_greed_index: float
    timestamp: datetime

@dataclass
class SentimentScore:
    symbol: str
    score: float
    confidence: float
    sources: Dict[str, float]
    timestamp: datetime

@dataclass
class OnChainMetrics:
    """Blockchain analytics data"""
    symbol: str
    whale_movements: List[Dict]
    exchange_flows: Dict
    holder_concentration: float
    network_activity: Dict
    defi_metrics: Dict
    timestamp: datetime

@dataclass
class OptionsFlowData:
    """Options market data and flow analysis"""
    symbol: str
    put_call_ratio: float
    implied_volatility: float
    options_volume: int
    unusual_activity: List[Dict]
    max_pain: float
    gamma_exposure: float
    timestamp: datetime

@dataclass
class MacroIndicator:
    """Macroeconomic indicator"""
    indicator_name: str
    value: float
    previous_value: float
    impact_level: str  # 'high', 'medium', 'low'
    market_expectation: float
    surprise_index: float
    timestamp: datetime

class SocialSentimentAnalyzer:
    """Advanced social sentiment analysis from multiple sources"""
    
    def __init__(self):
        self.sentiment_history = defaultdict(lambda: deque(maxlen=1000))
        self.mention_tracker = defaultdict(int)
        self.influence_weights = {
            'twitter': 0.4,
            'reddit': 0.3,
            'news': 0.2,
            'telegram': 0.1
        }
        
        # Crypto-specific keywords and weights
        self.bullish_keywords = {
            'moon', 'bullish', 'pump', 'hodl', 'diamond hands', 'to the moon',
            'buy the dip', 'accumulating', 'breakout', 'rally', 'surge'
        }
        
        self.bearish_keywords = {
            'dump', 'bearish', 'crash', 'sell', 'paper hands', 'panic',
            'liquidation', 'bear market', 'correction', 'downtrend'
        }
        
    async def analyze_twitter_sentiment(self, symbols: List[str]) -> Dict[str, SentimentData]:
        """Analyze Twitter sentiment for crypto symbols"""
        sentiments = {}
        
        if not _ensure_social_imports():
            return sentiments
        
        try:
            # Mock Twitter API data (replace with actual API calls)
            for symbol in symbols:
                tweets = await self._fetch_twitter_data(symbol)
                
                total_sentiment = 0
                tweet_count = 0
                trending_score = 0
                
                for tweet in tweets:
                    # Sentiment analysis
                    blob = TextBlob(tweet['text'])
                    sentiment = blob.sentiment.polarity
                    
                    # Crypto-specific sentiment adjustment
                    crypto_sentiment = self._analyze_crypto_sentiment(tweet['text'])
                    adjusted_sentiment = (sentiment * 0.6) + (crypto_sentiment * 0.4)
                    
                    # Weight by retweets and likes
                    engagement = tweet.get('retweet_count', 0) + tweet.get('like_count', 0)
                    weight = min(10, 1 + np.log(1 + engagement))
                    
                    total_sentiment += adjusted_sentiment * weight
                    tweet_count += weight
                    trending_score += engagement
                
                if tweet_count > 0:
                    avg_sentiment = total_sentiment / tweet_count
                    sentiments[symbol] = SentimentData(
                        source='twitter',
                        symbol=symbol,
                        sentiment_score=avg_sentiment,
                        volume=len(tweets),
                        trending_score=trending_score / 1000,  # Normalized
                        fear_greed_index=self._calculate_fear_greed(avg_sentiment),
                        timestamp=datetime.now()
                    )
                    
        except Exception as e:
            logging.error(f"Twitter sentiment analysis error: {e}")
        
        return sentiments
    
    async def _fetch_twitter_data(self, symbol: str) -> List[Dict]:
        """Fetch Twitter data (mock implementation)"""
        # In production, use actual Twitter API v2
        mock_tweets = [
            {
                'text': f'{symbol} looking bullish! Great fundamentals and strong community support. #crypto #bullish',
                'retweet_count': 25,
                'like_count': 150
            },
            {
                'text': f'Just bought more {symbol}. This dip is a gift! Diamond hands 💎🙌',
                'retweet_count': 12,
                'like_count': 89
            },
            {
                'text': f'{symbol} technical analysis shows potential breakout. Volume increasing.',
                'retweet_count': 8,
                'like_count': 45
            }
        ]
        
        return mock_tweets
    
    def _analyze_crypto_sentiment(self, text: str) -> float:
        """Analyze crypto-specific sentiment indicators"""
        text_lower = text.lower()
        
        bullish_score = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
        bearish_score = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
        
        # Emoji analysis
        moon_emojis = text.count('🚀') + text.count('🌙') + text.count('📈')
        crash_emojis = text.count('📉') + text.count('💀') + text.count('😰')
        
        bullish_score += moon_emojis * 0.5
        bearish_score += crash_emojis * 0.5
        
        # Normalize to -1 to 1 range
        total_signals = bullish_score + bearish_score
        if total_signals == 0:
            return 0.0
        
        return (bullish_score - bearish_score) / total_signals
    
    async def analyze_reddit_sentiment(self, symbols: List[str]) -> Dict[str, SentimentData]:
        """Analyze Reddit sentiment from crypto subreddits"""
        sentiments = {}
        
        if not _ensure_social_imports():
            return sentiments
        
        try:
            for symbol in symbols:
                posts = await self._fetch_reddit_data(symbol)
                
                total_sentiment = 0
                post_count = 0
                
                for post in posts:
                    # Combine title and body for analysis
                    text = f"{post['title']} {post['body']}"
                    
                    blob = TextBlob(text)
                    sentiment = blob.sentiment.polarity
                    
                    crypto_sentiment = self._analyze_crypto_sentiment(text)
                    adjusted_sentiment = (sentiment * 0.5) + (crypto_sentiment * 0.5)
                    
                    # Weight by upvotes and comments
                    engagement = post.get('upvotes', 0) + post.get('num_comments', 0)
                    weight = min(5, 1 + np.log(1 + engagement))
                    
                    total_sentiment += adjusted_sentiment * weight
                    post_count += weight
                
                if post_count > 0:
                    avg_sentiment = total_sentiment / post_count
                    sentiments[symbol] = SentimentData(
                        source='reddit',
                        symbol=symbol,
                        sentiment_score=avg_sentiment,
                        volume=len(posts),
                        trending_score=sum(p.get('upvotes', 0) for p in posts) / 100,
                        fear_greed_index=self._calculate_fear_greed(avg_sentiment),
                        timestamp=datetime.now()
                    )
                    
        except Exception as e:
            logging.error(f"Reddit sentiment analysis error: {e}")
        
        return sentiments
    
    async def _fetch_reddit_data(self, symbol: str) -> List[Dict]:
        """Fetch Reddit data (mock implementation)"""
        # Mock Reddit posts
        mock_posts = [
            {
                'title': f'{symbol} Price Analysis and Future Predictions',
                'body': f'Technical analysis suggests {symbol} is forming a bullish pattern. Strong support levels.',
                'upvotes': 45,
                'num_comments': 23
            },
            {
                'title': f'Why I\'m accumulating {symbol}',
                'body': 'Great project fundamentals and undervalued compared to competitors.',
                'upvotes': 78,
                'num_comments': 34
            }
        ]
        
        return mock_posts
    
    def _calculate_fear_greed(self, sentiment: float) -> float:
        """Calculate fear and greed index from sentiment"""
        # Convert sentiment (-1 to 1) to fear/greed (0 to 100)
        # 0 = Extreme Fear, 50 = Neutral, 100 = Extreme Greed
        return max(0, min(100, (sentiment + 1) * 50))

class OnChainAnalyzer:
    """Advanced blockchain and on-chain analytics"""
    
    def __init__(self):
        self.whale_threshold = 1000000  # $1M+ movements considered whale activity
        self.exchange_addresses = {
            'binance': ['0x...', '0x...'],  # Known exchange addresses
            'coinbase': ['0x...', '0x...'],
            'kraken': ['0x...', '0x...']
        }
        
    async def analyze_whale_movements(self, symbol: str) -> List[Dict]:
        """Analyze large wallet movements and whale activity"""
        whale_movements = []
        
        if not _ensure_onchain_imports():
            return whale_movements
        
        try:
            # Mock whale movement data
            mock_movements = [
                {
                    'from_address': '0x742d35Cc6634C0532925a3b8D99bc7e1b8f',
                    'to_address': '0x28C6c06298d514Db089934071355E5743bf21d60',
                    'amount_usd': 2500000,
                    'amount_tokens': 1250,
                    'transaction_hash': '0xabcdef...',
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'movement_type': 'exchange_to_wallet',  # or 'wallet_to_exchange'
                    'exchange': 'binance'
                },
                {
                    'from_address': '0x8ba1f109551bD432803012645Hac136c22C',
                    'to_address': '0x742d35Cc6634C0532925a3b8D99bc7e1b8f',
                    'amount_usd': 1800000,
                    'amount_tokens': 900,
                    'transaction_hash': '0xfedcba...',
                    'timestamp': datetime.now() - timedelta(hours=5),
                    'movement_type': 'wallet_to_wallet',
                    'exchange': None
                }
            ]
            
            whale_movements = mock_movements
            
        except Exception as e:
            logging.error(f"Whale movement analysis error: {e}")
        
        return whale_movements
    
    async def analyze_exchange_flows(self, symbol: str) -> Dict:
        """Analyze inflows and outflows from exchanges"""
        try:
            if not _ensure_onchain_imports():
                return {}
            # Mock exchange flow data
            flows = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'net_flow': -3500000,  # Negative = net outflow (bullish)
                'inflow_addresses': 450,
                'outflow_addresses': 620,
                'largest_single_inflow': 2100000,
                'largest_single_outflow': 3200000,
                'exchange_breakdown': {
                    'binance': {'inflow': 6000000, 'outflow': 7200000},
                    'coinbase': {'inflow': 4500000, 'outflow': 5800000},
                    'kraken': {'inflow': 2200000, 'outflow': 2900000},
                    'others': {'inflow': 2300000, 'outflow': 2600000}
                }
            }
            
            return flows
            
        except Exception as e:
            logging.error(f"Exchange flow analysis error: {e}")
            return {}
    
    async def analyze_holder_metrics(self, symbol: str) -> Dict:
        """Analyze holder distribution and concentration"""
        try:
            # Mock holder metrics
            metrics = {
                'total_holders': 1250000,
                'holder_concentration': {
                    'top_10_holders_pct': 45.2,
                    'top_100_holders_pct': 72.8,
                    'top_1000_holders_pct': 89.5
                },
                'distribution_change_24h': {
                    'small_holders': +0.15,  # Percentage point change
                    'medium_holders': -0.08,
                    'large_holders': -0.07
                },
                'new_holders_24h': 2340,
                'active_addresses_24h': 45600,
                'gini_coefficient': 0.85  # Wealth distribution inequality (0=equal, 1=unequal)
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Holder metrics analysis error: {e}")
            return {}

class OptionsFlowAnalyzer:
    """Options market analysis and unusual activity detection"""
    
    def __init__(self):
        self.volume_threshold = 1000  # Contracts for unusual activity
        self.iv_history = defaultdict(lambda: deque(maxlen=100))
        
    async def analyze_options_flow(self, symbol: str) -> OptionsFlowData:
        """Analyze options market data and flow"""
        try:
            # Mock options data (in production, use real options data providers)
            options_data = {
                'total_volume': 125000,
                'put_volume': 45000,
                'call_volume': 80000,
                'put_call_ratio': 45000 / 80000,
                'implied_volatility': 0.85,
                'iv_rank': 75.2,  # Percentile rank
                'gamma_exposure': -2.5e9,  # Negative = put bias
                'max_pain': 42500,  # Price with max option pain
                'unusual_activity': [
                    {
                        'strike': 45000,
                        'expiry': '2024-01-26',
                        'type': 'call',
                        'volume': 5000,
                        'open_interest': 1200,
                        'volume_oi_ratio': 4.17,
                        'premium': 1250000,
                        'flow_type': 'sweep'  # Large coordinated buy
                    },
                    {
                        'strike': 40000,
                        'expiry': '2024-01-19',
                        'type': 'put',
                        'volume': 3500,
                        'open_interest': 800,
                        'volume_oi_ratio': 4.38,
                        'premium': 890000,
                        'flow_type': 'block'  # Large single trade
                    }
                ]
            }
            
            return OptionsFlowData(
                symbol=symbol,
                put_call_ratio=options_data['put_call_ratio'],
                implied_volatility=options_data['implied_volatility'],
                options_volume=options_data['total_volume'],
                unusual_activity=options_data['unusual_activity'],
                max_pain=options_data['max_pain'],
                gamma_exposure=options_data['gamma_exposure'],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"Options flow analysis error: {e}")
            return OptionsFlowData(
                symbol=symbol, put_call_ratio=1.0, implied_volatility=0.5,
                options_volume=0, unusual_activity=[], max_pain=0,
                gamma_exposure=0, timestamp=datetime.now()
            )

class MacroDataProvider:
    """Macroeconomic indicators and real-time integration"""
    
    def __init__(self):
        self.indicators = {
            'CPI': {'impact': 'high', 'frequency': 'monthly'},
            'NFP': {'impact': 'high', 'frequency': 'monthly'},
            'FOMC_Rate': {'impact': 'high', 'frequency': 'quarterly'},
            'GDP': {'impact': 'medium', 'frequency': 'quarterly'},
            'Retail_Sales': {'impact': 'medium', 'frequency': 'monthly'},
            'Unemployment': {'impact': 'medium', 'frequency': 'monthly'},
            'DXY': {'impact': 'high', 'frequency': 'real-time'},
            'VIX': {'impact': 'high', 'frequency': 'real-time'},
            'Gold_Price': {'impact': 'medium', 'frequency': 'real-time'},
            'Oil_Price': {'impact': 'medium', 'frequency': 'real-time'}
        }
        
    async def fetch_economic_calendar(self) -> List[MacroIndicator]:
        """Fetch upcoming economic events and data releases"""
        indicators = []
        
        try:
            # Mock economic calendar data
            mock_events = [
                {
                    'name': 'CPI_YoY',
                    'value': 3.2,
                    'previous': 3.7,
                    'forecast': 3.1,
                    'release_time': datetime.now() + timedelta(hours=2),
                    'impact': 'high'
                },
                {
                    'name': 'NFP',
                    'value': None,  # Not yet released
                    'previous': 199000,
                    'forecast': 180000,
                    'release_time': datetime.now() + timedelta(days=2),
                    'impact': 'high'
                },
                {
                    'name': 'FOMC_Rate_Decision',
                    'value': None,
                    'previous': 5.25,
                    'forecast': 5.50,
                    'release_time': datetime.now() + timedelta(days=5),
                    'impact': 'high'
                }
            ]
            
            for event in mock_events:
                if event['value'] is not None:
                    surprise = (event['value'] - event['forecast']) / event['forecast'] if event['forecast'] != 0 else 0
                    
                    indicators.append(MacroIndicator(
                        indicator_name=event['name'],
                        value=event['value'],
                        previous_value=event['previous'],
                        impact_level=event['impact'],
                        market_expectation=event['forecast'],
                        surprise_index=surprise,
                        timestamp=datetime.now()
                    ))
                    
        except Exception as e:
            logging.error(f"Economic calendar fetch error: {e}")
        
        return indicators
    
    async def get_real_time_indicators(self) -> Dict[str, float]:
        """Get real-time market indicators"""
        try:
            # Mock real-time data
            indicators = {
                'DXY': 103.45,  # Dollar Index
                'VIX': 18.72,   # Volatility Index
                'TNX': 4.85,    # 10-Year Treasury Yield
                'Gold': 1985.50,
                'Oil_WTI': 82.15,
                'BTCDOM': 51.2,  # Bitcoin Dominance
                'TOTAL_MCAP': 1750000000000,  # Total crypto market cap
                'Fear_Greed_Index': 65
            }
            
            return indicators
            
        except Exception as e:
            logging.error(f"Real-time indicators error: {e}")
            return {}

class AlternativeDataAggregator:
    """Master aggregator for all alternative data sources"""
    
    def __init__(self):
        self.sentiment_analyzer = SocialSentimentAnalyzer()
        self.onchain_analyzer = OnChainAnalyzer()
        self.options_analyzer = OptionsFlowAnalyzer()
        self.macro_provider = MacroDataProvider()
        
        self.data_cache = {}
        self.last_update = defaultdict(datetime)
        
    async def get_comprehensive_analysis(self, symbols: List[str]) -> Dict:
        """Get comprehensive alternative data analysis for symbols"""
        analysis = {
            'timestamp': datetime.now(),
            'symbols': {},
            'macro_environment': {},
            'market_sentiment': {},
            'risk_factors': []
        }
        
        try:
            # Social sentiment analysis
            twitter_sentiment = await self.sentiment_analyzer.analyze_twitter_sentiment(symbols)
            reddit_sentiment = await self.sentiment_analyzer.analyze_reddit_sentiment(symbols)
            
            # On-chain analysis
            onchain_data = {}
            for symbol in symbols:
                whale_movements = await self.onchain_analyzer.analyze_whale_movements(symbol)
                exchange_flows = await self.onchain_analyzer.analyze_exchange_flows(symbol)
                holder_metrics = await self.onchain_analyzer.analyze_holder_metrics(symbol)
                
                onchain_data[symbol] = OnChainMetrics(
                    symbol=symbol,
                    whale_movements=whale_movements,
                    exchange_flows=exchange_flows,
                    holder_concentration=holder_metrics.get('gini_coefficient', 0.5),
                    network_activity={'active_addresses_24h': holder_metrics.get('active_addresses_24h', 0)},
                    defi_metrics={},
                    timestamp=datetime.now()
                )
            
            # Options flow analysis
            options_data = {}
            for symbol in symbols:
                options_data[symbol] = await self.options_analyzer.analyze_options_flow(symbol)
            
            # Macro economic data
            economic_indicators = await self.macro_provider.fetch_economic_calendar()
            real_time_indicators = await self.macro_provider.get_real_time_indicators()
            
            # Compile comprehensive analysis
            for symbol in symbols:
                analysis['symbols'][symbol] = {
                    'sentiment': {
                        'twitter': twitter_sentiment.get(symbol),
                        'reddit': reddit_sentiment.get(symbol),
                        'composite_score': self._calculate_composite_sentiment(
                            twitter_sentiment.get(symbol),
                            reddit_sentiment.get(symbol)
                        )
                    },
                    'onchain': onchain_data.get(symbol),
                    'options': options_data.get(symbol),
                    'risk_score': self._calculate_risk_score(symbol, onchain_data.get(symbol), options_data.get(symbol))
                }
            
            analysis['macro_environment'] = {
                'economic_calendar': economic_indicators,
                'real_time_indicators': real_time_indicators,
                'market_regime': self._determine_market_regime(real_time_indicators)
            }
            
            analysis['risk_factors'] = self._identify_risk_factors(analysis)
            
        except Exception as e:
            logging.error(f"Comprehensive analysis error: {e}")
        
        return analysis
    
    def _calculate_composite_sentiment(self, twitter_data: SentimentData, reddit_data: SentimentData) -> Dict:
        """Calculate composite sentiment score from multiple sources"""
        if not twitter_data and not reddit_data:
            return {'score': 0.0, 'confidence': 0.0}
        
        scores = []
        weights = []
        
        if twitter_data:
            scores.append(twitter_data.sentiment_score)
            weights.append(0.6)  # Twitter weighted higher
        
        if reddit_data:
            scores.append(reddit_data.sentiment_score)
            weights.append(0.4)
        
        # Weighted average
        composite_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # Confidence based on agreement between sources
        if len(scores) > 1:
            agreement = 1 - abs(scores[0] - scores[1])  # Higher agreement = higher confidence
            confidence = min(0.95, 0.5 + agreement * 0.45)
        else:
            confidence = 0.7  # Single source confidence
        
        return {
            'score': composite_score,
            'confidence': confidence,
            'sources_count': len(scores),
            'agreement': agreement if len(scores) > 1 else 1.0
        }
    
    def _calculate_risk_score(self, symbol: str, onchain_data: OnChainMetrics, options_data: OptionsFlowData) -> float:
        """Calculate comprehensive risk score for a symbol"""
        risk_factors = []
        
        # On-chain risk factors
        if onchain_data:
            # Whale concentration risk
            if onchain_data.holder_concentration > 0.8:
                risk_factors.append(0.3)  # High concentration risk
            
            # Exchange flow risk
            net_flow = onchain_data.exchange_flows.get('net_flow', 0)
            if net_flow > 0:  # Net inflow to exchanges (bearish)
                risk_factors.append(0.2)
        
        # Options market risk
        if options_data:
            # High put/call ratio indicates bearish sentiment
            if options_data.put_call_ratio > 1.2:
                risk_factors.append(0.25)
            
            # High implied volatility indicates uncertainty
            if options_data.implied_volatility > 0.8:
                risk_factors.append(0.15)
        
        # Calculate composite risk (0 = low risk, 1 = high risk)
        if not risk_factors:
            return 0.3  # Base risk level
        
        return min(1.0, sum(risk_factors))
    
    def _determine_market_regime(self, indicators: Dict[str, float]) -> str:
        """Determine current market regime based on macro indicators"""
        vix = indicators.get('VIX', 20)
        dxy = indicators.get('DXY', 100)
        fear_greed = indicators.get('Fear_Greed_Index', 50)
        
        if vix > 30:
            return 'high_volatility'
        elif vix < 15 and fear_greed > 70:
            return 'risk_on'
        elif fear_greed < 30:
            return 'risk_off'
        elif dxy > 105:
            return 'dollar_strength'
        else:
            return 'neutral'
    
    def _identify_risk_factors(self, analysis: Dict) -> List[str]:
        """Identify current market risk factors"""
        risk_factors = []
        
        macro = analysis.get('macro_environment', {})
        real_time = macro.get('real_time_indicators', {})
        
        # High volatility risk
        if real_time.get('VIX', 0) > 25:
            risk_factors.append('High market volatility detected')
        
        # Dollar strength risk for crypto
        if real_time.get('DXY', 0) > 105:
            risk_factors.append('Strong USD headwind for crypto')
        
        # Extreme fear/greed
        fear_greed = real_time.get('Fear_Greed_Index', 50)
        if fear_greed < 25:
            risk_factors.append('Extreme fear in markets')
        elif fear_greed > 80:
            risk_factors.append('Extreme greed - potential reversal risk')
        
        # Check for whale activity across symbols
        for symbol, data in analysis.get('symbols', {}).items():
            onchain = data.get('onchain')
            if onchain and len(onchain.whale_movements) > 2:  # Multiple whale movements
                risk_factors.append(f'High whale activity in {symbol}')
        
        return risk_factors

# Global instance
alternative_data = AlternativeDataAggregator()
