#!/usr/bin/env python3
"""
🧠 ADVANCED AI TRADING SYSTEM
Multi-strategy brain with regime detection, sentiment analysis, and more
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
import logging
from collections import deque
import random
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_REAL_TRADING_ENABLED = str(os.getenv('REAL_TRADING', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
_STRICT_REAL_DATA = str(os.getenv('STRICT_REAL_DATA', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
ALLOW_SIMULATED_FEATURES = (
    str(os.getenv('ALLOW_SIMULATED_FEATURES', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
    and not _REAL_TRADING_ENABLED
    and not _STRICT_REAL_DATA
)

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "bull"
    BEAR = "bear" 
    CRAB = "crab"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class TradingStrategy(Enum):
    """Available trading strategies"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"

@dataclass
class StrategyPerformance:
    """Performance metrics for each strategy"""
    strategy: TradingStrategy
    total_trades: int = 0
    profitable_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_allocation: float = 0.0

@dataclass
class SentimentData:
    """Sentiment analysis results"""
    twitter_sentiment: float = 0.0
    reddit_sentiment: float = 0.0
    news_sentiment: float = 0.0
    fear_greed_index: float = 0.5
    composite_sentiment: float = 0.0
    confidence: float = 0.0

@dataclass
class OnChainData:
    """On-chain intelligence data"""
    btc_dominance: float = 0.0
    stablecoin_inflows: float = 0.0
    whale_activity: float = 0.0
    exchange_flows: float = 0.0
    network_activity: float = 0.0

@dataclass
class TradingExplanation:
    """Explainable AI trading decision"""
    confidence: float
    strategy: TradingStrategy
    regime: MarketRegime
    signal_strength: float
    expected_return: float
    risk_level: str
    reasoning: str
    contributing_factors: Dict[str, float]

class MultiStrategyBrain:
    """Multi-strategy trading brain with meta-controller"""
    
    def __init__(self, initial_capital: float = 1000.0):
        self.strategies = {strategy: StrategyPerformance(strategy) for strategy in TradingStrategy}
        self.total_capital = initial_capital
        self.performance_window = 100  # Trades to consider for performance
        self.trade_history = deque(maxlen=self.performance_window)
        
        # Initialize equal allocation
        for strategy in self.strategies.values():
            strategy.current_allocation = 1.0 / len(TradingStrategy)
    
    def update_strategy_performance(self, strategy: TradingStrategy, pnl: float, trade_data: Dict):
        """Update performance metrics for a strategy"""
        perf = self.strategies[strategy]
        perf.total_trades += 1
        perf.total_pnl += pnl
        
        if pnl > 0:
            perf.profitable_trades += 1
        
        perf.win_rate = perf.profitable_trades / perf.total_trades if perf.total_trades > 0 else 0
        perf.avg_return = perf.total_pnl / perf.total_trades if perf.total_trades > 0 else 0
        
        # Store trade for meta-controller
        self.trade_history.append({
            'strategy': strategy,
            'pnl': pnl,
            'timestamp': time.time(),
            'trade_data': trade_data
        })
        
        # Rebalance allocations
        self._rebalance_allocations()
    
    def _rebalance_allocations(self):
        """Dynamically rebalance capital allocation based on performance"""
        if len(self.trade_history) < 10:  # Need minimum data
            return
        
        # Calculate recent performance (last 20 trades)
        recent_trades = list(self.trade_history)[-20:]
        strategy_scores = {}
        
        for strategy in TradingStrategy:
            strategy_trades = [t for t in recent_trades if t['strategy'] == strategy]
            
            if len(strategy_trades) == 0:
                strategy_scores[strategy] = 0.0
                continue
            
            # Score based on win rate, avg return, and consistency
            recent_pnl = [t['pnl'] for t in strategy_trades]
            win_rate = len([p for p in recent_pnl if p > 0]) / len(recent_pnl)
            avg_return = np.mean(recent_pnl)
            consistency = 1.0 - (np.std(recent_pnl) / (abs(avg_return) + 0.01))  # Avoid div by 0
            
            strategy_scores[strategy] = (win_rate * 0.4 + avg_return * 0.4 + consistency * 0.2)
        
        # Convert scores to allocations (with minimum 5% per strategy)
        total_score = sum(max(0, score) for score in strategy_scores.values())
        
        if total_score > 0:
            for strategy, score in strategy_scores.items():
                base_allocation = 0.05  # 5% minimum
                performance_allocation = max(0, score) / total_score * 0.75  # 75% based on performance
                self.strategies[strategy].current_allocation = base_allocation + performance_allocation
        
        logger.info(f"🧠 Strategy Allocations Updated: {[(s.name, f'{perf.current_allocation:.1%}') for s, perf in self.strategies.items()]}")
    
    def get_strategy_allocation(self, strategy: TradingStrategy) -> float:
        """Get current capital allocation for strategy"""
        return self.strategies[strategy].current_allocation

class RegimeDetector:
    """Market regime detection system"""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.price_history = deque(maxlen=lookback_period)
        self.volume_history = deque(maxlen=lookback_period)
        self.current_regime = MarketRegime.UNKNOWN
        
    def add_data(self, price: float, volume: float = 1.0):
        """Add new market data point"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        if len(self.price_history) >= 20:  # Minimum data needed
            self.current_regime = self._detect_regime()
    
    def _detect_regime(self) -> MarketRegime:
        """Detect current market regime"""
        if len(self.price_history) < 20:
            return MarketRegime.UNKNOWN
        
        prices = np.array(list(self.price_history))
        
        # Calculate metrics
        volatility = self._calculate_volatility(prices)
        trend_strength = self._calculate_trend_strength(prices)
        hurst_exponent = self._calculate_hurst_exponent(prices)
        
        # Regime classification logic
        if volatility > 0.8:  # High volatility
            return MarketRegime.VOLATILE
        elif trend_strength > 0.6:  # Strong trend
            if prices[-1] > prices[-10]:  # Recent uptrend
                return MarketRegime.BULL
            else:
                return MarketRegime.BEAR
        elif hurst_exponent < 0.4:  # Mean-reverting
            return MarketRegime.CRAB
        else:
            return MarketRegime.UNKNOWN
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate normalized volatility"""
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(len(returns))
        return min(1.0, volatility * 10)  # Normalize to 0-1
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression"""
        x = np.arange(len(prices))
        correlation = np.corrcoef(x, prices)[0, 1]
        return abs(correlation)  # Absolute correlation as trend strength
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent for mean reversion detection"""
        if len(prices) < 10:
            return 0.5
        
        try:
            # Simple Hurst calculation
            log_prices = np.log(prices)
            returns = np.diff(log_prices)
            
            # Calculate R/S statistic
            mean_return = np.mean(returns)
            cumulative_deviations = np.cumsum(returns - mean_return)
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            S = np.std(returns)
            
            if S == 0:
                return 0.5
            
            # Rough Hurst estimate
            hurst = np.log(R/S) / np.log(len(returns))
            return max(0.0, min(1.0, hurst))
        except:
            return 0.5

class SentimentAnalyzer:
    """Sentiment analysis from multiple sources"""
    
    def __init__(self):
        self.sentiment_history = deque(maxlen=100)
        self.last_update = 0
        self.update_interval = 300  # 5 minutes
        
    async def get_sentiment_data(self) -> SentimentData:
        """Aggregate sentiment from multiple sources"""
        current_time = time.time()
        
        # Check if we need to update
        if current_time - self.last_update < self.update_interval:
            if self.sentiment_history:
                return self.sentiment_history[-1]
        
        if not ALLOW_SIMULATED_FEATURES:
            sentiment_data = SentimentData(
                twitter_sentiment=0.0,
                reddit_sentiment=0.0,
                news_sentiment=0.0,
                fear_greed_index=0.5,
                composite_sentiment=0.0,
                confidence=0.0
            )
            self.sentiment_history.append(sentiment_data)
            self.last_update = current_time
            return sentiment_data

        try:
            twitter_sentiment = await self._get_twitter_sentiment()
            reddit_sentiment = await self._get_reddit_sentiment()
            news_sentiment = await self._get_news_sentiment()
            fear_greed = await self._get_fear_greed_index()
            
            # Composite sentiment calculation
            composite = (twitter_sentiment * 0.3 + reddit_sentiment * 0.2 + 
                        news_sentiment * 0.3 + fear_greed * 0.2)
            
            sentiment_data = SentimentData(
                twitter_sentiment=twitter_sentiment,
                reddit_sentiment=reddit_sentiment,
                news_sentiment=news_sentiment,
                fear_greed_index=fear_greed,
                composite_sentiment=composite,
                confidence=0.75  # Confidence in sentiment data
            )
            
            self.sentiment_history.append(sentiment_data)
            self.last_update = current_time
            
            return sentiment_data
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            # Return neutral sentiment on error
            return SentimentData(composite_sentiment=0.5, confidence=0.1)
    
    async def _get_twitter_sentiment(self) -> float:
        """Get Twitter/X sentiment (simulated)"""
        # In production: integrate with Twitter API and finBERT
        return random.uniform(0.3, 0.7)  # Simulated sentiment
    
    async def _get_reddit_sentiment(self) -> float:
        """Get Reddit sentiment (simulated)"""
        # In production: integrate with Reddit API
        return random.uniform(0.4, 0.6)
    
    async def _get_news_sentiment(self) -> float:
        """Get news sentiment (simulated)"""
        # In production: integrate with news APIs
        return random.uniform(0.35, 0.65)
    
    async def _get_fear_greed_index(self) -> float:
        """Get Fear & Greed Index (simulated)"""
        # In production: integrate with real F&G index
        return random.uniform(0.2, 0.8)

class OnChainAnalyzer:
    """On-chain intelligence analyzer"""
    
    def __init__(self):
        self.onchain_history = deque(maxlen=50)
        self.last_update = 0
        self.update_interval = 600  # 10 minutes
    
    async def get_onchain_data(self) -> OnChainData:
        """Get on-chain intelligence data"""
        current_time = time.time()
        
        if current_time - self.last_update < self.update_interval:
            if self.onchain_history:
                return self.onchain_history[-1]
        
        if not ALLOW_SIMULATED_FEATURES:
            onchain_data = OnChainData(
                btc_dominance=0.0,
                stablecoin_inflows=0.0,
                whale_activity=0.0,
                exchange_flows=0.0,
                network_activity=0.0
            )
            self.onchain_history.append(onchain_data)
            self.last_update = current_time
            return onchain_data

        try:
            btc_dominance = await self._get_btc_dominance()
            stablecoin_flows = await self._get_stablecoin_flows()
            whale_activity = await self._get_whale_activity()
            exchange_flows = await self._get_exchange_flows()
            network_activity = await self._get_network_activity()
            
            onchain_data = OnChainData(
                btc_dominance=btc_dominance,
                stablecoin_inflows=stablecoin_flows,
                whale_activity=whale_activity,
                exchange_flows=exchange_flows,
                network_activity=network_activity
            )
            
            self.onchain_history.append(onchain_data)
            self.last_update = current_time
            
            return onchain_data
            
        except Exception as e:
            logger.warning(f"On-chain analysis failed: {e}")
            return OnChainData()  # Return neutral data
    
    async def _get_btc_dominance(self) -> float:
        """Get BTC dominance percentage"""
        return random.uniform(0.4, 0.6)  # Simulated 40-60%
    
    async def _get_stablecoin_flows(self) -> float:
        """Get stablecoin flow indicator"""
        return random.uniform(-0.2, 0.3)  # Simulated flow
    
    async def _get_whale_activity(self) -> float:
        """Get whale wallet activity"""
        return random.uniform(0.3, 0.7)
    
    async def _get_exchange_flows(self) -> float:
        """Get exchange inflow/outflow data"""
        return random.uniform(-0.1, 0.2)
    
    async def _get_network_activity(self) -> float:
        """Get network activity metrics"""
        return random.uniform(0.4, 0.8)

class SelfHealingWatchdog:
    """Self-healing system watchdog"""
    
    def __init__(self):
        self.is_healthy = True
        self.error_count = 0
        self.last_heartbeat = time.time()
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        self.error_threshold = 5
        
    async def monitor_system_health(self):
        """Monitor system health continuously"""
        while True:
            try:
                await self._check_api_connectivity()
                await self._check_market_data()
                await self._check_trading_engine()
                
                if self.is_healthy:
                    self.last_heartbeat = time.time()
                    self.error_count = max(0, self.error_count - 1)  # Slowly recover
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                await self._handle_system_error(e)
    
    async def _check_api_connectivity(self):
        """Check API connectivity"""
        try:
            # Simulate API check (in production: ping exchange APIs)
            await asyncio.sleep(0.1)
            if ALLOW_SIMULATED_FEATURES and random.random() < 0.02:  # 2% chance of simulated failure
                raise Exception("API connectivity lost")
        except Exception as e:
            self.error_count += 1
            logger.error(f"API connectivity check failed: {e}")
            raise
    
    async def _check_market_data(self):
        """Check market data feed"""
        try:
            # Simulate data feed check
            await asyncio.sleep(0.1)
            if ALLOW_SIMULATED_FEATURES and random.random() < 0.01:  # 1% chance of failure
                raise Exception("Market data feed interrupted")
        except Exception as e:
            self.error_count += 1
            logger.error(f"Market data check failed: {e}")
            raise
    
    async def _check_trading_engine(self):
        """Check trading engine health"""
        try:
            # Simulate trading engine check
            await asyncio.sleep(0.1)
            if ALLOW_SIMULATED_FEATURES and random.random() < 0.005:  # 0.5% chance of failure
                raise Exception("Trading engine malfunction")
        except Exception as e:
            self.error_count += 1
            logger.error(f"Trading engine check failed: {e}")
            raise
    
    async def _handle_system_error(self, error: Exception):
        """Handle system errors with recovery"""
        logger.error(f"System error detected: {error}")
        
        if self.error_count >= self.error_threshold:
            self.is_healthy = False
            logger.critical("System health compromised, initiating recovery")
            
            if self.recovery_attempts < self.max_recovery_attempts:
                await self._attempt_recovery()
            else:
                logger.critical("Max recovery attempts reached, system shutdown required")
                # In production: send alert, pause trading, etc.
    
    async def _attempt_recovery(self):
        """Attempt system recovery"""
        self.recovery_attempts += 1
        logger.info(f"Recovery attempt {self.recovery_attempts}/{self.max_recovery_attempts}")
        
        try:
            # Recovery steps
            await asyncio.sleep(5)  # Wait before retry
            await self._reset_connections()
            await asyncio.sleep(2)
            await self._verify_recovery()
            
            # Reset if successful
            self.error_count = 0
            self.is_healthy = True
            self.recovery_attempts = 0
            logger.info("System recovery successful")
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
    
    async def _reset_connections(self):
        """Reset API connections"""
        logger.info("Resetting API connections...")
        await asyncio.sleep(1)  # Simulate reset
    
    async def _verify_recovery(self):
        """Verify system recovery"""
        logger.info("Verifying system recovery...")
        await self._check_api_connectivity()
        await self._check_market_data()

class ExplainableAI:
    """Explainable AI system for trading decisions"""
    
    def __init__(self):
        self.decision_history = deque(maxlen=100)
    
    def generate_explanation(self, 
                           confidence: float,
                           strategy: TradingStrategy,
                           regime: MarketRegime,
                           sentiment: SentimentData,
                           onchain: OnChainData,
                           technical_signals: Dict[str, float]) -> TradingExplanation:
        """Generate explainable trading decision"""
        
        # Calculate contributing factors
        factors = {}
        
        # Technical factors
        if 'sma_signal' in technical_signals:
            factors['SMA Crossover'] = technical_signals['sma_signal']
        if 'rsi' in technical_signals:
            factors['RSI Momentum'] = technical_signals['rsi']
        if 'macd' in technical_signals:
            factors['MACD Signal'] = technical_signals['macd']
        
        # Sentiment factors
        factors['Market Sentiment'] = sentiment.composite_sentiment - 0.5  # Center around 0
        factors['Fear/Greed'] = sentiment.fear_greed_index - 0.5
        
        # On-chain factors
        factors['BTC Dominance'] = (onchain.btc_dominance - 0.5) * 2  # Normalize
        factors['Whale Activity'] = onchain.whale_activity
        
        # Regime factor
        regime_scores = {
            MarketRegime.BULL: 0.3,
            MarketRegime.BEAR: -0.3,
            MarketRegime.CRAB: 0.0,
            MarketRegime.VOLATILE: 0.1,
            MarketRegime.UNKNOWN: 0.0
        }
        factors['Market Regime'] = regime_scores.get(regime, 0.0)
        
        # Calculate signal strength
        signal_strength = sum(abs(v) for v in factors.values()) / len(factors) if factors else 0.0
        
        # Expected return estimation
        expected_return = confidence * signal_strength * 0.02  # Base 2% max return
        
        # Risk level
        risk_level = "LOW" if confidence < 0.5 else "MEDIUM" if confidence < 0.8 else "HIGH"
        
        # Generate reasoning
        top_factors = sorted(factors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        reasoning = f"Strategy {strategy.value} selected for {regime.value} market. "
        reasoning += f"Key signals: {', '.join([f'{name}({value:+.2f})' for name, value in top_factors])}. "
        reasoning += f"Sentiment: {sentiment.composite_sentiment:.2f}, Confidence: {confidence:.2f}"
        
        explanation = TradingExplanation(
            confidence=confidence,
            strategy=strategy,
            regime=regime,
            signal_strength=signal_strength,
            expected_return=expected_return,
            risk_level=risk_level,
            reasoning=reasoning,
            contributing_factors=factors
        )
        
        self.decision_history.append(explanation)
        return explanation
    
    def log_decision(self, explanation: TradingExplanation, symbol: str, action: str):
        """Log trading decision with explanation"""
        logger.info(f"""
🤖 TRADING DECISION EXPLANATION:
{'='*60}
Symbol: {symbol} | Action: {action}
Confidence: {explanation.confidence:.1%}
Strategy: {explanation.strategy.value}
Market Regime: {explanation.regime.value}
Signal Strength: {explanation.signal_strength:.2f}
Expected Return: {explanation.expected_return:+.2%}
Risk Level: {explanation.risk_level}

💡 Reasoning: {explanation.reasoning}

📊 Contributing Factors:
{chr(10).join([f'   {name}: {value:+.3f}' for name, value in explanation.contributing_factors.items()])}
{'='*60}
        """)

# Global instances
multi_strategy_brain = MultiStrategyBrain()
regime_detector = RegimeDetector()
sentiment_analyzer = SentimentAnalyzer()
onchain_analyzer = OnChainAnalyzer()
watchdog = SelfHealingWatchdog()
explainable_ai = ExplainableAI()

async def initialize_advanced_ai():
    """Initialize all advanced AI systems"""
    logger.info("🚀 Initializing Advanced AI Systems...")
    
    # Start watchdog monitoring
    asyncio.create_task(watchdog.monitor_system_health())
    
    logger.info("✅ Advanced AI Systems Online!")

if __name__ == "__main__":
    print("🧠 Advanced AI Trading System")
    print("Features: Multi-strategy brain, regime detection, sentiment analysis, watchdog, on-chain intelligence, explainable AI")
    asyncio.run(initialize_advanced_ai())
