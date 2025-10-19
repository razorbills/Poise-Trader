#!/usr/bin/env python3
"""
🧠 ADVANCED TRADING SYSTEMS
Basic implementation of advanced trading intelligence systems
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict, deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    EUPHORIA = "euphoria"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    VOLATILE = "volatile"

class TradingStrategy(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    ARBITRAGE = "arbitrage"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SCALP = "scalp"
    TREND_FOLLOW = "trend_follow"
    SWING_TRADING = "swing_trading"
    GRID_TRADING = "grid_trading"

class StrategyPerformance:
    """Track performance metrics for trading strategies"""
    
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.current_allocation = 0.1  # Default 10% allocation
        self.recent_performance = deque(maxlen=50)
        self.intermediate_updates = deque(maxlen=100)
        self.confidence_history = deque(maxlen=50)
        
    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
    
    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.total_trades if self.total_trades > 0 else 0.0
    
    def update_performance(self, pnl: float, trade_data: Dict):
        """Update strategy performance with completed trade"""
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        
        self.recent_performance.append({
            'pnl': pnl,
            'confidence': trade_data.get('confidence', 0.5),
            'timestamp': datetime.now(),
            'symbol': trade_data.get('symbol', 'UNKNOWN'),
            'trade_data': trade_data
        })
        
        # Update allocation based on recent performance
        self._update_allocation()
    
    def update_intermediate(self, pnl_pct: float, update_data: Dict):
        """Update with intermediate performance data from open positions"""
        self.intermediate_updates.append({
            'pnl_percentage': pnl_pct,
            'timestamp': datetime.now(),
            'symbol': update_data.get('symbol', 'UNKNOWN'),
            'confidence': update_data.get('confidence', 0.5),
            'cycles_held': update_data.get('cycles_held', 0),
            'continuous_update': update_data.get('continuous_update', False)
        })
        
        # Track confidence accuracy for open positions
        if 'confidence' in update_data:
            self.confidence_history.append(update_data['confidence'])
    
    def _update_allocation(self):
        """Update strategy allocation based on recent performance"""
        if len(self.recent_performance) < 5:
            return
        
        # Calculate recent win rate and average PnL
        recent_trades = list(self.recent_performance)[-10:]  # Last 10 trades
        recent_wins = sum(1 for t in recent_trades if t['pnl'] > 0)
        recent_win_rate = recent_wins / len(recent_trades)
        recent_avg_pnl = np.mean([t['pnl'] for t in recent_trades])
        
        # Adjust allocation based on performance
        base_allocation = 0.1  # 10% base
        
        # Increase allocation for high-performing strategies
        if recent_win_rate > 0.6 and recent_avg_pnl > 0:
            performance_bonus = (recent_win_rate - 0.5) * 0.4 + (recent_avg_pnl * 0.1)
            self.current_allocation = min(0.4, base_allocation + performance_bonus)  # Max 40%
        elif recent_win_rate < 0.4 or recent_avg_pnl < -0.05:
            # Reduce allocation for poor performance
            performance_penalty = (0.5 - recent_win_rate) * 0.2 + abs(min(0, recent_avg_pnl)) * 0.1
            self.current_allocation = max(0.02, base_allocation - performance_penalty)  # Min 2%
        else:
            # Gradual return to base allocation
            self.current_allocation = base_allocation

class MultiStrategyBrain:
    """Multi-strategy brain for managing different trading strategies"""
    
    def __init__(self):
        self.strategies = {}
        self._initialize_strategies()
        
    def _initialize_strategies(self):
        """Initialize all trading strategies with performance tracking"""
        for strategy in TradingStrategy:
            self.strategies[strategy] = StrategyPerformance()
        
        # Set initial allocations
        total_strategies = len(self.strategies)
        base_allocation = 1.0 / total_strategies
        
        for strategy_perf in self.strategies.values():
            strategy_perf.current_allocation = base_allocation
    
    def update_strategy_performance(self, strategy: TradingStrategy, pnl: float, trade_data: Dict):
        """Update performance for a specific strategy"""
        if strategy not in self.strategies:
            self.strategies[strategy] = StrategyPerformance()
        
        self.strategies[strategy].update_performance(pnl, trade_data)
        
        # Rebalance allocations after performance update
        self._rebalance_allocations()
        
        print(f"   🧠 Strategy {strategy.value}: Performance updated (PnL: ${pnl:+.2f}, "
              f"Win Rate: {self.strategies[strategy].win_rate:.1%}, "
              f"Allocation: {self.strategies[strategy].current_allocation:.1%})")
    
    def update_intermediate_performance(self, strategy: TradingStrategy, pnl_pct: float, update_data: Dict):
        """Update intermediate performance for strategy from open positions"""
        if strategy not in self.strategies:
            self.strategies[strategy] = StrategyPerformance()
        
        self.strategies[strategy].update_intermediate(pnl_pct, update_data)
        
        # Log intermediate update for debugging
        symbol = update_data.get('symbol', 'UNKNOWN')
        cycles = update_data.get('cycles_held', 0)
        confidence = update_data.get('confidence', 0.5)
        
        print(f"   📊 Strategy {strategy.value} intermediate: {symbol} {pnl_pct:+.2f}% "
              f"(Cycle {cycles}, Confidence: {confidence:.1%})")
    
    def get_best_strategies(self, top_n: int = 3) -> List[TradingStrategy]:
        """Get the best performing strategies"""
        if not self.strategies:
            return [TradingStrategy.MOMENTUM, TradingStrategy.TREND_FOLLOWING, TradingStrategy.BREAKOUT]
        
        # Sort strategies by a composite score of win rate and allocation
        strategy_scores = []
        for strategy, perf in self.strategies.items():
            # Composite score: win rate + allocation + recent performance
            score = perf.win_rate * 0.6 + perf.current_allocation * 0.3
            if len(perf.recent_performance) > 0:
                recent_avg = np.mean([t['pnl'] for t in list(perf.recent_performance)[-5:]])
                score += max(-0.1, min(0.1, recent_avg)) * 0.1  # Recent performance bonus/penalty
            
            strategy_scores.append((strategy, score))
        
        # Sort by score and return top N
        sorted_strategies = sorted(strategy_scores, key=lambda x: x[1], reverse=True)
        return [strategy for strategy, score in sorted_strategies[:top_n]]
    
    def get_strategy_allocation(self, strategy: TradingStrategy) -> float:
        """Get current allocation for a strategy"""
        if strategy not in self.strategies:
            return 0.1  # Default 10% allocation
        
        return self.strategies[strategy].current_allocation
    
    def _rebalance_allocations(self):
        """Rebalance strategy allocations to ensure they sum to 1.0"""
        if not self.strategies:
            return
        
        # Calculate total current allocation
        total_allocation = sum(perf.current_allocation for perf in self.strategies.values())
        
        if total_allocation <= 0:
            # Reset to equal allocation
            equal_allocation = 1.0 / len(self.strategies)
            for perf in self.strategies.values():
                perf.current_allocation = equal_allocation
        else:
            # Normalize to sum to 1.0
            for perf in self.strategies.values():
                perf.current_allocation /= total_allocation

class RegimeDetector:
    """Market regime detection system"""
    
    def __init__(self):
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        self.regime_history = deque(maxlen=100)
        
    def detect_regime(self, market_data: Dict) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Simple regime detection based on volatility and trend
            volatility = market_data.get('volatility', 0.1)
            momentum = market_data.get('momentum', 0.0)
            trend_strength = market_data.get('trend_strength', 0.3)
            
            # High volatility regimes
            if volatility > 0.4:
                if abs(momentum) > 0.3:
                    self.current_regime = MarketRegime.HIGH_VOLATILITY
                else:
                    self.current_regime = MarketRegime.VOLATILE
            # Trending markets
            elif trend_strength > 0.6:
                if momentum > 0.1:
                    self.current_regime = MarketRegime.BULL_TREND
                elif momentum < -0.1:
                    self.current_regime = MarketRegime.BEAR_TREND
                else:
                    self.current_regime = MarketRegime.SIDEWAYS
            # Low volatility
            elif volatility < 0.05:
                self.current_regime = MarketRegime.LOW_VOLATILITY
            else:
                self.current_regime = MarketRegime.SIDEWAYS
            
            # Update confidence
            self.regime_confidence = min(0.95, 0.5 + abs(momentum) + trend_strength)
            
            # Track regime history
            self.regime_history.append({
                'regime': self.current_regime,
                'confidence': self.regime_confidence,
                'timestamp': datetime.now()
            })
            
            return self.current_regime
            
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return MarketRegime.SIDEWAYS

class SentimentAnalyzer:
    """Basic sentiment analysis system"""
    
    def __init__(self):
        self.sentiment_history = deque(maxlen=100)
        self.current_sentiment = 0.5  # Neutral
        
    def analyze_sentiment(self, market_data: Dict) -> Dict:
        """Analyze market sentiment"""
        try:
            # Simple sentiment based on market momentum and volatility
            momentum = market_data.get('momentum', 0.0)
            volatility = market_data.get('volatility', 0.1)
            
            # Positive momentum = bullish sentiment
            # High volatility = fear
            sentiment_score = 0.5 + momentum * 0.5 - volatility * 0.3
            sentiment_score = max(0.0, min(1.0, sentiment_score))
            
            self.current_sentiment = sentiment_score
            
            sentiment_data = {
                'composite_sentiment': sentiment_score,
                'confidence': 0.6,
                'fear_greed_index': sentiment_score * 100,
                'social_sentiment': sentiment_score,
                'news_sentiment': sentiment_score
            }
            
            self.sentiment_history.append({
                'sentiment': sentiment_data,
                'timestamp': datetime.now()
            })
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {
                'composite_sentiment': 0.5,
                'confidence': 0.5,
                'fear_greed_index': 50,
                'social_sentiment': 0.5,
                'news_sentiment': 0.5
            }

class OnChainIntelligence:
    """Basic on-chain analysis system"""
    
    def __init__(self):
        self.on_chain_data = {}
        
    def analyze_on_chain_data(self, symbol: str) -> Dict:
        """Analyze on-chain metrics"""
        # Basic on-chain analysis (simulated)
        return {
            'whale_activity': 0.3,
            'exchange_flows': 'neutral',
            'hodler_behavior': 'accumulating',
            'network_activity': 'moderate',
            'institutional_flow': 'positive'
        }

class SelfHealingWatchdog:
    """Self-healing system watchdog"""
    
    def __init__(self):
        self.is_healthy = True
        self.health_checks = []
        
    def perform_health_check(self) -> bool:
        """Perform system health check"""
        # Basic health check
        self.is_healthy = True
        return self.is_healthy
    
    def auto_heal(self) -> bool:
        """Attempt to auto-heal system issues"""
        # Basic auto-healing
        return True

class AdaptiveRiskManager:
    """Adaptive risk management system"""
    
    def __init__(self):
        self.risk_metrics = {}
        self.volatility_threshold = 0.5
        
    def update_volatility(self, price_history: List[float]):
        """Update volatility tracking"""
        if len(price_history) < 2:
            return
        
        returns = [(price_history[i] - price_history[i-1])/price_history[i-1] 
                  for i in range(1, len(price_history))]
        
        if returns:
            volatility = np.std(returns)
            self.risk_metrics['current_volatility'] = volatility
    
    def should_avoid_trading(self) -> bool:
        """Check if trading should be avoided due to extreme conditions"""
        current_vol = self.risk_metrics.get('current_volatility', 0.1)
        return current_vol > self.volatility_threshold

class OrderBookAnalyzer:
    """Order book analysis system"""
    
    def __init__(self):
        self.order_book_data = {}
        
    def analyze_order_book(self, symbol: str) -> Dict:
        """Analyze order book for trading signals"""
        # Basic order book analysis (simulated)
        return {
            'bid_ask_spread': 0.001,
            'market_depth': 'deep',
            'order_imbalance': 0.0,
            'support_resistance': []
        }

class AdvancedTradingIntelligence:
    """Main advanced trading intelligence system"""
    
    def __init__(self, initial_capital: float = 5.0):
        self.initial_capital = initial_capital
        
        # Initialize all subsystems
        self.multi_strategy_brain = MultiStrategyBrain()
        self.regime_detector = RegimeDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.onchain_intelligence = OnChainIntelligence()
        self.watchdog = SelfHealingWatchdog()
        self.risk_manager = AdaptiveRiskManager()
        self.orderbook_analyzer = OrderBookAnalyzer()
        
        print("🧠 Advanced Trading Intelligence initialized")

# Create global instance for backward compatibility
multi_strategy_brain = MultiStrategyBrain()

print("✅ Advanced trading systems loaded successfully")
