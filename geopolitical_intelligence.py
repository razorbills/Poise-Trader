"""
🌍 GEOPOLITICAL INTELLIGENCE SYSTEM 🌍
Reads central bank speeches, oil supply news, Elon tweets
Auto-fetches economic events and switches to defense mode
"""

import asyncio
import json
import re
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import aiohttp
import logging

logger = logging.getLogger(__name__)

class NewsImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DefenseMode(Enum):
    NORMAL = "normal"
    CAUTIOUS = "cautious"
    DEFENSIVE = "defensive"
    HALT = "halt"

@dataclass
class GeopoliticalEvent:
    """Geopolitical news event"""
    title: str
    content: str
    source: str
    timestamp: datetime
    impact_level: NewsImpact
    affected_assets: List[str]
    sentiment_score: float
    confidence: float
    keywords: List[str]

@dataclass
class EconomicEvent:
    """Economic calendar event"""
    name: str
    date: datetime
    impact: NewsImpact
    currency: str
    previous: Optional[float]
    forecast: Optional[float]
    actual: Optional[float]
    affected_assets: List[str]
    time_until: int  # seconds

@dataclass
class DefenseSignal:
    """Defense mode activation signal"""
    trigger_event: str
    defense_level: DefenseMode
    duration_minutes: int
    affected_assets: List[str]
    reasoning: str
    confidence: float

class GeopoliticalIntelligence:
    """Advanced geopolitical news scraper and analyzer"""
    
    def __init__(self):
        self.news_sources = {
            'fed': 'https://www.federalreserve.gov/newsevents.htm',
            'ecb': 'https://www.ecb.europa.eu/press/html/index.en.html',
            'reuters': 'https://www.reuters.com/business/finance/',
            'bloomberg': 'https://www.bloomberg.com/economics',
            'twitter': 'https://api.twitter.com/2/tweets/search/recent',
            'reddit': 'https://www.reddit.com/r/CryptoCurrency/hot.json'
        }
        
        # High-impact keywords and their impact levels
        self.keyword_impact_map = {
            # Critical impact
            'interest rate': NewsImpact.CRITICAL,
            'fed meeting': NewsImpact.CRITICAL,
            'rate hike': NewsImpact.CRITICAL,
            'rate cut': NewsImpact.CRITICAL,
            'recession': NewsImpact.CRITICAL,
            'bitcoin etf': NewsImpact.CRITICAL,
            'crypto ban': NewsImpact.CRITICAL,
            
            # High impact
            'inflation': NewsImpact.HIGH,
            'cpi data': NewsImpact.HIGH,
            'employment': NewsImpact.HIGH,
            'gdp': NewsImpact.HIGH,
            'geopolitical': NewsImpact.HIGH,
            'sanctions': NewsImpact.HIGH,
            'trade war': NewsImpact.HIGH,
            'oil supply': NewsImpact.HIGH,
            'regulatory': NewsImpact.HIGH,
            'sec': NewsImpact.HIGH,
            
            # Medium impact
            'elon musk': NewsImpact.MEDIUM,
            'tesla': NewsImpact.MEDIUM,
            'earnings': NewsImpact.MEDIUM,
            'china': NewsImpact.MEDIUM,
            'russia': NewsImpact.MEDIUM,
            'ukraine': NewsImpact.MEDIUM,
            
            # Low impact
            'upgrade': NewsImpact.LOW,
            'partnership': NewsImpact.LOW,
            'adoption': NewsImpact.LOW
        }
        
        # Economic events calendar
        self.economic_calendar = []
        self.events_cache = []
        self.defense_signals = []
        self.current_defense_mode = DefenseMode.NORMAL
        
        # Elon Musk tweet patterns
        self.elon_patterns = [
            r'bitcoin|btc|crypto|doge|dogecoin',
            r'tesla.*bitcoin|bitcoin.*tesla',
            r'to the moon|diamond hands|hodl',
            r'sec|regulation|government'
        ]
        
    async def scrape_all_sources(self) -> List[GeopoliticalEvent]:
        """Scrape all news sources simultaneously"""
        events = []
        
        # Run all scrapers in parallel
        scraping_tasks = [
            self._scrape_fed_news(),
            self._scrape_financial_news(),
            self._scrape_elon_tweets(),
            self._scrape_crypto_reddit(),
            self._scrape_oil_news()
        ]
        
        results = await asyncio.gather(*scraping_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                events.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"News scraping error: {result}")
        
        # Sort by impact level and timestamp
        events.sort(key=lambda x: (x.impact_level.value, x.timestamp), reverse=True)
        
        self.events_cache = events
        return events
    
    async def _scrape_fed_news(self) -> List[GeopoliticalEvent]:
        """Scrape Federal Reserve news and speeches"""
        events = []
        
        # Simulated Fed news (replace with real scraping)
        fed_events = [
            {
                'title': 'Fed Chair Powell: Inflation Progress Encouraging',
                'content': 'Federal Reserve Chair Jerome Powell stated that recent inflation data shows encouraging progress toward the 2% target. The Fed remains data-dependent on future rate decisions.',
                'keywords': ['fed meeting', 'inflation', 'jerome powell'],
                'sentiment': 0.6  # Slightly positive
            },
            {
                'title': 'FOMC Minutes Reveal Split on Rate Path',
                'content': 'Federal Open Market Committee minutes show disagreement among members about the pace of future rate adjustments amid economic uncertainty.',
                'keywords': ['fed meeting', 'interest rate', 'fomc'],
                'sentiment': 0.4  # Slightly negative
            }
        ]
        
        for event_data in fed_events:
            impact_level = self._determine_impact_level(event_data['keywords'])
            
            event = GeopoliticalEvent(
                title=event_data['title'],
                content=event_data['content'],
                source='fed',
                timestamp=datetime.now(),
                impact_level=impact_level,
                affected_assets=['USD', 'BTC', 'GOLD', 'SPY'],
                sentiment_score=event_data['sentiment'],
                confidence=0.9,
                keywords=event_data['keywords']
            )
            events.append(event)
        
        return events
    
    async def _scrape_financial_news(self) -> List[GeopoliticalEvent]:
        """Scrape Reuters/Bloomberg financial news"""
        events = []
        
        financial_news = [
            {
                'title': 'Oil Prices Surge on Middle East Supply Concerns',
                'content': 'Crude oil prices jumped 3% as geopolitical tensions in the Middle East raise concerns about supply disruptions.',
                'keywords': ['oil supply', 'geopolitical'],
                'sentiment': 0.3,  # Negative for markets
                'assets': ['OIL', 'USD', 'GOLD']
            },
            {
                'title': 'China GDP Growth Beats Expectations',
                'content': 'Chinese GDP growth exceeded forecasts, boosting optimism about global economic recovery and commodity demand.',
                'keywords': ['china', 'gdp', 'growth'],
                'sentiment': 0.7,  # Positive
                'assets': ['CNY', 'COPPER', 'BTC']
            }
        ]
        
        for news in financial_news:
            impact_level = self._determine_impact_level(news['keywords'])
            
            event = GeopoliticalEvent(
                title=news['title'],
                content=news['content'],
                source='reuters',
                timestamp=datetime.now(),
                impact_level=impact_level,
                affected_assets=news['assets'],
                sentiment_score=news['sentiment'],
                confidence=0.8,
                keywords=news['keywords']
            )
            events.append(event)
        
        return events
    
    async def _scrape_elon_tweets(self) -> List[GeopoliticalEvent]:
        """Scrape Elon Musk tweets for crypto impact"""
        events = []
        
        # Simulated Elon tweets
        elon_tweets = [
            {
                'text': 'Bitcoin is the future of money. Tesla will never sell our Bitcoin holdings. 🚀',
                'timestamp': datetime.now() - timedelta(hours=2),
                'engagement': 50000  # likes + retweets
            },
            {
                'text': 'Dogecoin to the moon! Much wow, very currency 🐕',
                'timestamp': datetime.now() - timedelta(hours=6),
                'engagement': 75000
            }
        ]
        
        for tweet in elon_tweets:
            # Check if tweet matches crypto patterns
            crypto_relevance = 0
            matched_patterns = []
            
            for pattern in self.elon_patterns:
                if re.search(pattern, tweet['text'].lower()):
                    crypto_relevance += 1
                    matched_patterns.append(pattern)
            
            if crypto_relevance > 0:
                # Impact based on engagement and crypto relevance
                if tweet['engagement'] > 100000 and crypto_relevance >= 2:
                    impact = NewsImpact.HIGH
                elif tweet['engagement'] > 50000:
                    impact = NewsImpact.MEDIUM
                else:
                    impact = NewsImpact.LOW
                
                sentiment = 0.8 if 'moon' in tweet['text'].lower() or '🚀' in tweet['text'] else 0.6
                
                event = GeopoliticalEvent(
                    title=f"Elon Musk Tweet: Crypto Impact",
                    content=tweet['text'],
                    source='twitter',
                    timestamp=tweet['timestamp'],
                    impact_level=impact,
                    affected_assets=['BTC', 'DOGE', 'ETH'],
                    sentiment_score=sentiment,
                    confidence=0.7,
                    keywords=['elon musk'] + matched_patterns
                )
                events.append(event)
        
        return events
    
    async def _scrape_crypto_reddit(self) -> List[GeopoliticalEvent]:
        """Scrape crypto Reddit for sentiment"""
        events = []
        
        # Simulated Reddit posts
        reddit_posts = [
            {
                'title': 'SEC Chair Hints at Clearer Crypto Regulations',
                'content': 'Gary Gensler suggests the SEC is working on comprehensive crypto regulations...',
                'upvotes': 2500,
                'comments': 450,
                'keywords': ['sec', 'regulatory']
            },
            {
                'title': 'Whale Alert: 10,000 BTC Moved to Exchange',
                'content': 'Large Bitcoin transfer detected, possible sell pressure incoming...',
                'upvotes': 1800,
                'comments': 320,
                'keywords': ['whale', 'bitcoin', 'exchange']
            }
        ]
        
        for post in reddit_posts:
            if post['upvotes'] > 1000:  # Only high-engagement posts
                impact = self._determine_impact_level(post['keywords'])
                sentiment = 0.3 if 'sell' in post['content'].lower() else 0.6
                
                event = GeopoliticalEvent(
                    title=post['title'],
                    content=post['content'],
                    source='reddit',
                    timestamp=datetime.now(),
                    impact_level=impact,
                    affected_assets=['BTC', 'ETH', 'CRYPTO'],
                    sentiment_score=sentiment,
                    confidence=0.6,
                    keywords=post['keywords']
                )
                events.append(event)
        
        return events
    
    async def _scrape_oil_news(self) -> List[GeopoliticalEvent]:
        """Scrape oil supply and geopolitical news"""
        events = []
        
        oil_news = [
            {
                'title': 'OPEC+ Considers Production Cut Extension',
                'content': 'OPEC+ members are discussing extending production cuts through Q2 2024 to support oil prices.',
                'keywords': ['oil supply', 'opec'],
                'sentiment': 0.7,  # Bullish for oil
                'assets': ['OIL', 'CAD', 'RUB']
            }
        ]
        
        for news in oil_news:
            impact = self._determine_impact_level(news['keywords'])
            
            event = GeopoliticalEvent(
                title=news['title'],
                content=news['content'],
                source='oil_news',
                timestamp=datetime.now(),
                impact_level=impact,
                affected_assets=news['assets'],
                sentiment_score=news['sentiment'],
                confidence=0.8,
                keywords=news['keywords']
            )
            events.append(event)
        
        return events
    
    def _determine_impact_level(self, keywords: List[str]) -> NewsImpact:
        """Determine impact level based on keywords"""
        max_impact = NewsImpact.LOW
        
        for keyword in keywords:
            if keyword in self.keyword_impact_map:
                current_impact = self.keyword_impact_map[keyword]
                if current_impact.value == 'critical':
                    return NewsImpact.CRITICAL
                elif current_impact.value == 'high' and max_impact.value != 'critical':
                    max_impact = NewsImpact.HIGH
                elif current_impact.value == 'medium' and max_impact.value == 'low':
                    max_impact = NewsImpact.MEDIUM
        
        return max_impact
    
    async def fetch_economic_calendar(self) -> List[EconomicEvent]:
        """Fetch upcoming economic events"""
        events = []
        
        # Simulated economic calendar
        upcoming_events = [
            {
                'name': 'US CPI (Consumer Price Index)',
                'date': datetime.now() + timedelta(days=2),
                'impact': NewsImpact.CRITICAL,
                'currency': 'USD',
                'previous': 3.2,
                'forecast': 3.1,
                'assets': ['USD', 'GOLD', 'BTC', 'SPY']
            },
            {
                'name': 'Federal Reserve Interest Rate Decision',
                'date': datetime.now() + timedelta(days=7),
                'impact': NewsImpact.CRITICAL,
                'currency': 'USD',
                'previous': 5.25,
                'forecast': 5.50,
                'assets': ['USD', 'ALL_MARKETS']
            },
            {
                'name': 'Non-Farm Payrolls',
                'date': datetime.now() + timedelta(days=5),
                'impact': NewsImpact.HIGH,
                'currency': 'USD',
                'previous': 150000,
                'forecast': 180000,
                'assets': ['USD', 'GOLD', 'SPY']
            },
            {
                'name': 'Bitcoin ETF Approval Decision',
                'date': datetime.now() + timedelta(days=14),
                'impact': NewsImpact.CRITICAL,
                'currency': 'USD',
                'previous': None,
                'forecast': None,
                'assets': ['BTC', 'ETH', 'CRYPTO']
            },
            {
                'name': 'ECB Interest Rate Decision',
                'date': datetime.now() + timedelta(days=10),
                'impact': NewsImpact.HIGH,
                'currency': 'EUR',
                'previous': 4.50,
                'forecast': 4.50,
                'assets': ['EUR', 'GOLD', 'BTC']
            }
        ]
        
        for event_data in upcoming_events:
            time_until = int((event_data['date'] - datetime.now()).total_seconds())
            
            event = EconomicEvent(
                name=event_data['name'],
                date=event_data['date'],
                impact=event_data['impact'],
                currency=event_data['currency'],
                previous=event_data['previous'],
                forecast=event_data['forecast'],
                actual=None,
                affected_assets=event_data['assets'],
                time_until=time_until
            )
            events.append(event)
        
        # Sort by date
        events.sort(key=lambda x: x.date)
        self.economic_calendar = events
        
        return events
    
    async def analyze_defense_signals(self) -> List[DefenseSignal]:
        """Analyze events and generate defense mode signals"""
        signals = []
        
        # Check recent high-impact events
        critical_events = [e for e in self.events_cache if e.impact_level == NewsImpact.CRITICAL]
        high_impact_events = [e for e in self.events_cache if e.impact_level == NewsImpact.HIGH]
        
        # Check upcoming economic events
        upcoming_critical = [e for e in self.economic_calendar if e.impact == NewsImpact.CRITICAL and e.time_until < 86400]  # 24 hours
        
        # Generate defense signals
        if critical_events:
            for event in critical_events:
                defense_level = DefenseMode.HALT if 'rate' in event.title.lower() else DefenseMode.DEFENSIVE
                duration = 240 if defense_level == DefenseMode.HALT else 120  # minutes
                
                signal = DefenseSignal(
                    trigger_event=event.title,
                    defense_level=defense_level,
                    duration_minutes=duration,
                    affected_assets=event.affected_assets,
                    reasoning=f"Critical event detected: {event.title}",
                    confidence=event.confidence
                )
                signals.append(signal)
        
        # Pre-emptive defense for upcoming events
        if upcoming_critical:
            for event in upcoming_critical:
                hours_until = event.time_until / 3600
                
                if hours_until <= 2:  # 2 hours before
                    defense_level = DefenseMode.HALT
                    duration = 180
                elif hours_until <= 6:  # 6 hours before
                    defense_level = DefenseMode.DEFENSIVE
                    duration = 120
                else:
                    defense_level = DefenseMode.CAUTIOUS
                    duration = 60
                
                signal = DefenseSignal(
                    trigger_event=f"Upcoming: {event.name}",
                    defense_level=defense_level,
                    duration_minutes=duration,
                    affected_assets=event.affected_assets,
                    reasoning=f"Pre-emptive defense for {event.name} in {hours_until:.1f} hours",
                    confidence=0.8
                )
                signals.append(signal)
        
        # Market stress defense
        negative_sentiment_events = [e for e in self.events_cache if e.sentiment_score < 0.3]
        if len(negative_sentiment_events) >= 3:  # Multiple negative events
            signal = DefenseSignal(
                trigger_event="Multiple negative sentiment events",
                defense_level=DefenseMode.CAUTIOUS,
                duration_minutes=90,
                affected_assets=['ALL'],
                reasoning="High negative sentiment detected across multiple sources",
                confidence=0.7
            )
            signals.append(signal)
        
        self.defense_signals = signals
        return signals
    
    def get_current_defense_mode(self) -> DefenseMode:
        """Get current recommended defense mode"""
        if not self.defense_signals:
            return DefenseMode.NORMAL
        
        # Return highest defense level from active signals
        active_signals = [s for s in self.defense_signals if s.confidence > 0.6]
        if not active_signals:
            return DefenseMode.NORMAL
        
        defense_levels = {'normal': 0, 'cautious': 1, 'defensive': 2, 'halt': 3}
        max_level = max(defense_levels[s.defense_level.value] for s in active_signals)
        
        for level, value in defense_levels.items():
            if value == max_level:
                return DefenseMode(level)
        
        return DefenseMode.NORMAL
    
    def get_trading_adjustments(self, defense_mode: DefenseMode) -> Dict[str, Any]:
        """Get trading parameter adjustments for defense mode"""
        adjustments = {
            DefenseMode.NORMAL: {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'confidence_threshold': 0.6,
                'max_positions': 5,
                'trading_enabled': True
            },
            DefenseMode.CAUTIOUS: {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 0.8,  # Tighter stops
                'confidence_threshold': 0.7,
                'max_positions': 3,
                'trading_enabled': True
            },
            DefenseMode.DEFENSIVE: {
                'position_size_multiplier': 0.4,
                'stop_loss_multiplier': 0.6,  # Much tighter stops
                'confidence_threshold': 0.8,
                'max_positions': 2,
                'trading_enabled': True
            },
            DefenseMode.HALT: {
                'position_size_multiplier': 0.0,
                'stop_loss_multiplier': 0.5,
                'confidence_threshold': 0.9,
                'max_positions': 0,
                'trading_enabled': False
            }
        }
        
        return adjustments[defense_mode]
    
    async def scan_regulatory_threats(self) -> List[Dict[str, Any]]:
        """Scan for regulatory threats and compliance issues"""
        try:
            regulatory_threats = []
            
            # Check recent events for regulatory keywords
            regulatory_keywords = ['sec', 'regulation', 'ban', 'regulatory', 'legal', 'compliance', 'cftc', 'finra']
            
            for event in self.events_cache:
                # Check if event contains regulatory keywords
                content_lower = (event.title + ' ' + event.content).lower()
                matches = [keyword for keyword in regulatory_keywords if keyword in content_lower]
                
                if matches:
                    threat_level = 'high' if event.impact_level == NewsImpact.CRITICAL else 'medium'
                    
                    threat = {
                        'title': event.title,
                        'description': event.content,
                        'threat_level': threat_level,
                        'affected_assets': event.affected_assets,
                        'keywords': matches,
                        'confidence': event.confidence,
                        'timestamp': event.timestamp,
                        'source': event.source
                    }
                    regulatory_threats.append(threat)
            
            # Add simulated regulatory threats if no events found
            if not regulatory_threats:
                regulatory_threats = [
                    {
                        'title': 'SEC Crypto Regulation Framework Review',
                        'description': 'The SEC is reviewing comprehensive crypto regulation framework',
                        'threat_level': 'medium',
                        'affected_assets': ['BTC', 'ETH', 'CRYPTO'],
                        'keywords': ['sec', 'regulation'],
                        'confidence': 0.7,
                        'timestamp': datetime.now(),
                        'source': 'regulatory_monitor'
                    }
                ]
            
            # Sort by threat level and confidence
            regulatory_threats.sort(key=lambda x: (x['threat_level'] == 'high', x['confidence']), reverse=True)
            
            return regulatory_threats[:10]  # Return top 10 threats
            
        except Exception as e:
            logger.error(f"Regulatory threat scanning error: {e}")
            return []
    
    async def assess_global_risk(self, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess global geopolitical and economic risk levels"""
        try:
            print(f"🌍 Assessing global risk levels...")
            
            # Risk factors assessment
            risk_factors = {
                'geopolitical_risk': 0.5,
                'economic_risk': 0.5,
                'regulatory_risk': 0.5,
                'market_stress_risk': 0.5
            }
            
            # Analyze recent events for risk levels
            if self.events_cache:
                critical_count = len([e for e in self.events_cache if e.impact_level == NewsImpact.CRITICAL])
                high_count = len([e for e in self.events_cache if e.impact_level == NewsImpact.HIGH])
                
                # Geopolitical risk from news events
                geopolitical_keywords = ['war', 'sanctions', 'conflict', 'geopolitical', 'china', 'russia']
                geopolitical_events = [e for e in self.events_cache if any(keyword in e.title.lower() for keyword in geopolitical_keywords)]
                risk_factors['geopolitical_risk'] = min(0.9, 0.3 + len(geopolitical_events) * 0.15)
                
                # Economic risk from economic events
                economic_keywords = ['recession', 'inflation', 'unemployment', 'gdp', 'fed', 'interest rate']
                economic_events = [e for e in self.events_cache if any(keyword in e.title.lower() for keyword in economic_keywords)]
                risk_factors['economic_risk'] = min(0.9, 0.3 + len(economic_events) * 0.1)
                
                # Regulatory risk
                regulatory_keywords = ['regulation', 'ban', 'sec', 'regulatory', 'legal', 'compliance']
                regulatory_events = [e for e in self.events_cache if any(keyword in e.title.lower() for keyword in regulatory_keywords)]
                risk_factors['regulatory_risk'] = min(0.9, 0.2 + len(regulatory_events) * 0.2)
            
            # Market stress risk from upcoming events
            if self.economic_calendar:
                critical_upcoming = [e for e in self.economic_calendar if e.impact == NewsImpact.CRITICAL and e.time_until < 86400]
                risk_factors['market_stress_risk'] = min(0.9, 0.3 + len(critical_upcoming) * 0.2)
            
            # Overall risk score
            overall_risk = np.mean(list(risk_factors.values()))
            
            # Risk level classification
            if overall_risk > 0.7:
                risk_level = 'HIGH'
                recommended_action = 'Reduce positions, increase cash allocation'
            elif overall_risk > 0.5:
                risk_level = 'MODERATE'
                recommended_action = 'Monitor closely, use tighter stops'
            else:
                risk_level = 'LOW'
                recommended_action = 'Normal trading operations'
            
            # Current defense mode assessment
            current_defense = self.get_current_defense_mode()
            defense_adjustments = self.get_trading_adjustments(current_defense)
            
            # Risk assessment summary
            risk_assessment = {
                'overall_risk_score': overall_risk,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'current_defense_mode': current_defense.value,
                'recommended_action': recommended_action,
                'defense_adjustments': defense_adjustments,
                'high_impact_events': len([e for e in self.events_cache if e.impact_level in [NewsImpact.CRITICAL, NewsImpact.HIGH]]),
                'upcoming_critical_events': len([e for e in self.economic_calendar if e.impact == NewsImpact.CRITICAL and e.time_until < 172800]),  # 48 hours
                'confidence': 0.8 if len(self.events_cache) > 5 else 0.6
            }
            
            print(f"   🌍 Global risk level: {risk_level} (score: {overall_risk:.2f})")
            print(f"   🛡️ Defense mode: {current_defense.value}")
            print(f"   📊 Recommendation: {recommended_action}")
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Global risk assessment error: {e}")
            return {
                'overall_risk_score': 0.5,
                'risk_level': 'MODERATE',
                'current_defense_mode': 'normal',
                'confidence': 0.5
            }
