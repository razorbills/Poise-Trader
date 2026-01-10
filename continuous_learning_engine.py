#!/usr/bin/env python3
"""
🧠 CONTINUOUS LEARNING ENGINE
Self-learning AI that feeds on millions of data points and constantly improves

Features:
- Multi-source data aggregation (APIs, historical data, live feeds)
- Real-time pattern discovery
- Reinforcement learning from every trade
- Strategy evolution and optimization
- Cross-market intelligence gathering
- Historical backtesting on millions of data points
"""

import asyncio
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
from datetime import datetime, timedelta
import random

_REAL_TRADING_ENABLED = str(os.getenv('REAL_TRADING', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
_STRICT_REAL_DATA = str(os.getenv('STRICT_REAL_DATA', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
ALLOW_SIMULATED_FEATURES = (
    str(os.getenv('ALLOW_SIMULATED_FEATURES', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
    and not _REAL_TRADING_ENABLED
    and not _STRICT_REAL_DATA
)


class DataAggregator:
    """
 MASSIVE DATA COLLECTION SYSTEM
    Gathers trading data from every available source
    """
    
    def __init__(self):
        self.data_sources = {
            'live_mexc': True,
            'historical_cache': True,
            'synthetic_generation': True,
            'cross_market': True,
            'pattern_library': True
        }

        if not ALLOW_SIMULATED_FEATURES:
            self.data_sources['synthetic_generation'] = False
        
        self.collected_data_points = 0
        self.pattern_database = {}
        self.historical_cache_file = os.path.join("data", "historical_training_data.json")
        
        # Load existing historical data
        self._load_historical_cache()
    
    def _load_historical_cache(self):
        """Load previously collected data"""
        try:
            if os.path.exists(self.historical_cache_file):
                with open(self.historical_cache_file, 'r') as f:
                    cache = json.load(f)
                    self.collected_data_points = cache.get('total_points', 0)
                    self.pattern_database = cache.get('patterns', {})
                    print(f"📚 Loaded {self.collected_data_points:,} historical data points")
        except Exception as e:
            print(f"⚠️ Could not load historical cache: {e}")
    
    def _save_historical_cache(self):
        """Save collected data for future sessions"""
        try:
            os.makedirs("data", exist_ok=True)
            cache = {
                'total_points': self.collected_data_points,
                'patterns': self.pattern_database,
                'last_update': datetime.now().isoformat()
            }
            with open(self.historical_cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
    
    async def gather_live_data(self, data_feed, symbols: List[str]) -> Dict:
        """Collect real-time data from live feeds"""
        try:
            prices = await data_feed.get_multiple_prices(symbols)
            
            data_batch = {
                'timestamp': datetime.now().isoformat(),
                'source': 'live_mexc',
                'data': {}
            }
            
            for symbol, price in prices.items():
                data_batch['data'][symbol] = {
                    'price': price,
                    'timestamp': datetime.now().isoformat()
                }
                self.collected_data_points += 1
            
            return data_batch
            
        except Exception as e:
            print(f"⚠️ Live data collection error: {e}")
            return {}
    
    def generate_synthetic_data(self, base_prices: Dict[str, float], num_scenarios: int = 1000) -> List[Dict]:
        """
        Generate thousands of synthetic scenarios based on real market behavior
        Uses Monte Carlo simulation to create realistic price movements
        """
        synthetic_scenarios = []
        
        try:
            for symbol, base_price in base_prices.items():
                # Generate 1000 scenarios for each symbol
                for i in range(num_scenarios):
                    # Simulate realistic price movements using random walk
                    volatility = random.uniform(0.01, 0.05)  # 1-5% volatility
                    num_steps = 50  # 50 time steps per scenario
                    
                    prices = [base_price]
                    for step in range(num_steps):
                        # Geometric Brownian Motion simulation
                        drift = random.uniform(-0.001, 0.001)
                        shock = np.random.normal(0, volatility)
                        
                        new_price = prices[-1] * (1 + drift + shock)
                        prices.append(new_price)
                    
                    # Calculate features from this scenario
                    scenario = {
                        'symbol': symbol,
                        'prices': prices,
                        'volatility': volatility,
                        'returns': [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))],
                        'max_drawdown': self._calculate_drawdown(prices),
                        'sharpe_ratio': self._calculate_sharpe(prices),
                        'trend': 'UP' if prices[-1] > prices[0] else 'DOWN'
                    }
                    
                    synthetic_scenarios.append(scenario)
                    self.collected_data_points += len(prices)
            
            print(f"🔬 Generated {len(synthetic_scenarios):,} synthetic scenarios ({len(synthetic_scenarios) * 50:,} data points)")
            
        except Exception as e:
            print(f"⚠️ Synthetic data generation error: {e}")
        
        return synthetic_scenarios
    
    def _calculate_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = prices[0]
        max_dd = 0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe(self, prices: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns)
    
    def extract_patterns_from_data(self, price_data: List[float]) -> Dict:
        """Extract and catalog trading patterns from price data"""
        patterns = {
            'momentum_patterns': [],
            'reversal_patterns': [],
            'continuation_patterns': [],
            'volatility_patterns': []
        }
        
        try:
            if len(price_data) < 20:
                return patterns
            
            # Momentum patterns
            for i in range(10, len(price_data) - 10):
                momentum = (price_data[i] - price_data[i-10]) / price_data[i-10]
                future_move = (price_data[i+10] - price_data[i]) / price_data[i]
                
                if abs(momentum) > 0.02:  # Strong momentum
                    patterns['momentum_patterns'].append({
                        'strength': momentum,
                        'outcome': future_move,
                        'success': (momentum * future_move) > 0  # Same direction
                    })
            
            # Reversal patterns
            for i in range(5, len(price_data) - 5):
                if price_data[i] > price_data[i-5] and price_data[i] > price_data[i+5]:
                    # Local peak - potential reversal
                    future_move = (price_data[i+5] - price_data[i]) / price_data[i]
                    patterns['reversal_patterns'].append({
                        'type': 'peak',
                        'outcome': future_move
                    })
            
            # Store in database
            pattern_id = f"pattern_{len(self.pattern_database)}"
            self.pattern_database[pattern_id] = {
                'discovered': datetime.now().isoformat(),
                'patterns': patterns,
                'data_points': len(price_data)
            }
            
        except Exception as e:
            print(f"⚠️ Pattern extraction error: {e}")
        
        return patterns
    
    def get_statistics(self) -> Dict:
        """Get data collection statistics"""
        return {
            'total_data_points': self.collected_data_points,
            'patterns_discovered': len(self.pattern_database),
            'data_sources_active': sum(self.data_sources.values()),
            'cache_size_mb': os.path.getsize(self.historical_cache_file) / 1024 / 1024 if os.path.exists(self.historical_cache_file) else 0
        }


class ContinuousLearningEngine:
    """
    🧠 SELF-LEARNING AI ENGINE
    Learns from every trade and continuously improves strategies
    """
    
    def __init__(self):
        self.data_aggregator = DataAggregator()
        
        # Learning components
        self.strategy_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0})
        self.pattern_success_rates = {}
        self.learned_rules = []
        
        # Experience replay buffer (stores last 10,000 trades)
        self.experience_buffer = deque(maxlen=10000)
        
        # Model parameters (simple Q-learning style)
        self.q_table = defaultdict(lambda: {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0})
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        
        # Performance tracking
        self.learning_iterations = 0
        self.total_training_samples = 0
        
        # Load previous learning
        self._load_learned_knowledge()
    
    def _load_learned_knowledge(self):
        """Load previously learned patterns and strategies"""
        try:
            knowledge_path = os.path.join("data", "learned_knowledge.json")
            if os.path.exists(knowledge_path):
                with open(knowledge_path, 'r') as f:
                    knowledge = json.load(f)
                    self.learning_iterations = knowledge.get('iterations', 0)
                    self.learned_rules = knowledge.get('rules', [])
                    self.pattern_success_rates = knowledge.get('patterns', {})
                    print(f"🧠 Loaded {len(self.learned_rules)} learned rules from previous sessions")
        except Exception as e:
            print(f"⚠️ Could not load knowledge: {e}")
    
    def _save_learned_knowledge(self):
        """Save learned knowledge for future sessions"""
        try:
            os.makedirs("data", exist_ok=True)
            knowledge = {
                'iterations': self.learning_iterations,
                'q_table': dict(self.q_table),
                'rules': self.learned_rules,
                'patterns': self.pattern_success_rates,
                'last_update': datetime.now().isoformat()
            }
            knowledge_path = os.path.join("data", "learned_knowledge.json")
            with open(knowledge_path, 'w') as f:
                json.dump(knowledge, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save knowledge: {e}")
    
    async def feed_on_data(self, data_feed, symbols: List[str], price_history: Dict):
        """
        🍽️ FEED THE AI - Collect and learn from massive data
        This runs continuously in the background
        """
        try:
            print("🧠 FEEDING AI ENGINE - Gathering massive data...")
            
            # 1. Collect live data
            live_data = await self.data_aggregator.gather_live_data(data_feed, symbols)
            
            # 2. Generate synthetic scenarios for training
            if live_data and 'data' in live_data and self.data_aggregator.data_sources.get('synthetic_generation', False):
                current_prices = {s: d['price'] for s, d in live_data['data'].items()}
                synthetic_data = self.data_aggregator.generate_synthetic_data(current_prices, num_scenarios=500)
                
                # 3. Learn from synthetic scenarios
                for scenario in synthetic_data[:100]:  # Process 100 at a time
                    self._learn_from_scenario(scenario)
            
            # 4. Extract patterns from price history
            for symbol, prices in price_history.items():
                if len(prices) >= 50:
                    patterns = self.data_aggregator.extract_patterns_from_data(list(prices))
                    self._update_pattern_knowledge(patterns)
            
            # 5. Save learned knowledge
            self.learning_iterations += 1
            if self.learning_iterations % 10 == 0:  # Save every 10 iterations
                self.data_aggregator._save_historical_cache()
                self._save_learned_knowledge()
            
            stats = self.data_aggregator.get_statistics()
            print(f"✅ Data feeding complete: {stats['total_data_points']:,} total points, {stats['patterns_discovered']} patterns")
            
        except Exception as e:
            print(f"⚠️ Data feeding error: {e}")
    
    def _learn_from_scenario(self, scenario: Dict):
        """Learn optimal actions from a scenario"""
        try:
            prices = scenario['prices']
            
            # Analyze different entry points
            for i in range(5, len(prices) - 10):
                state = self._create_state(prices, i)
                
                # Determine best action in hindsight
                future_price = prices[i + 10]
                current_price = prices[i]
                
                if future_price > current_price * 1.01:  # 1%+ gain
                    best_action = 'BUY'
                    reward = (future_price - current_price) / current_price
                elif future_price < current_price * 0.99:  # 1%+ loss
                    best_action = 'SELL'
                    reward = (current_price - future_price) / current_price
                else:
                    best_action = 'HOLD'
                    reward = 0.0
                
                # Update Q-table
                self._update_q_value(state, best_action, reward)
                
                self.total_training_samples += 1
            
        except Exception as e:
            pass
    
    def _create_state(self, prices: List[float], index: int) -> str:
        """Create a state representation from price data"""
        if index < 5:
            return "unknown"
        
        # Simple state: momentum and volatility bucket
        momentum = (prices[index] - prices[index-5]) / prices[index-5]
        recent_volatility = np.std([prices[i] / prices[i-1] - 1 for i in range(index-5, index)])
        
        momentum_bucket = "up" if momentum > 0.01 else "down" if momentum < -0.01 else "flat"
        vol_bucket = "high" if recent_volatility > 0.02 else "low"
        
        return f"{momentum_bucket}_{vol_bucket}"
    
    def _update_q_value(self, state: str, action: str, reward: float):
        """Update Q-learning table"""
        current_q = self.q_table[state][action]
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.q_table[state][action] = new_q
    
    def _update_pattern_knowledge(self, patterns: Dict):
        """Update knowledge about pattern success rates"""
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if 'success' in pattern:
                    pattern_key = f"{pattern_type}_{pattern.get('type', 'generic')}"
                    
                    if pattern_key not in self.pattern_success_rates:
                        self.pattern_success_rates[pattern_key] = {'successes': 0, 'failures': 0}
                    
                    if pattern['success']:
                        self.pattern_success_rates[pattern_key]['successes'] += 1
                    else:
                        self.pattern_success_rates[pattern_key]['failures'] += 1
    
    def learn_from_trade(self, trade_result: Dict):
        """Learn from actual executed trade"""
        try:
            # Store in experience buffer
            self.experience_buffer.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': trade_result.get('symbol'),
                'action': trade_result.get('action'),
                'entry_price': trade_result.get('entry_price'),
                'exit_price': trade_result.get('exit_price'),
                'pnl': trade_result.get('pnl', 0),
                'win': trade_result.get('pnl', 0) > 0,
                'confidence': trade_result.get('confidence', 0),
                'strategy': trade_result.get('strategy', 'unknown')
            })
            
            # Update strategy performance
            strategy = trade_result.get('strategy', 'unknown')
            if trade_result.get('pnl', 0) > 0:
                self.strategy_performance[strategy]['wins'] += 1
            else:
                self.strategy_performance[strategy]['losses'] += 1
            
            self.strategy_performance[strategy]['total_pnl'] += trade_result.get('pnl', 0)
            
            # Discover new rules from successful trades
            if len(self.experience_buffer) >= 100:
                self._discover_winning_patterns()
            
        except Exception as e:
            print(f"⚠️ Trade learning error: {e}")
    
    def _discover_winning_patterns(self):
        """Analyze experience buffer to discover winning patterns"""
        try:
            # Analyze last 100 trades
            recent_trades = list(self.experience_buffer)[-100:]
            
            winning_trades = [t for t in recent_trades if t['win']]
            
            if len(winning_trades) >= 20:  # Need sufficient winning examples
                # Find common characteristics
                avg_confidence = np.mean([t['confidence'] for t in winning_trades])
                
                if avg_confidence > 0.6:
                    rule = {
                        'type': 'confidence_threshold',
                        'min_confidence': avg_confidence * 0.9,
                        'discovered': datetime.now().isoformat(),
                        'sample_size': len(winning_trades)
                    }
                    
                    # Add rule if not already present
                    if rule not in self.learned_rules:
                        self.learned_rules.append(rule)
                        print(f"🎓 LEARNED NEW RULE: Require {avg_confidence:.1%} confidence (from {len(winning_trades)} winning trades)")
        
        except Exception as e:
            pass
    
    def get_learned_insights(self) -> Dict:
        """Get insights from learned knowledge"""
        try:
            # Analyze best strategies
            best_strategy = None
            best_win_rate = 0.0
            
            for strategy, perf in self.strategy_performance.items():
                total = perf['wins'] + perf['losses']
                if total >= 10:  # Need sufficient sample size
                    win_rate = perf['wins'] / total
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_strategy = strategy
            
            # Analyze best patterns
            best_patterns = []
            for pattern_name, stats in self.pattern_success_rates.items():
                total = stats['successes'] + stats['failures']
                if total >= 10:
                    success_rate = stats['successes'] / total
                    if success_rate > 0.7:  # 70%+ success
                        best_patterns.append({
                            'pattern': pattern_name,
                            'success_rate': success_rate,
                            'sample_size': total
                        })
            
            return {
                'total_learning_iterations': self.learning_iterations,
                'total_training_samples': self.total_training_samples,
                'experience_buffer_size': len(self.experience_buffer),
                'learned_rules_count': len(self.learned_rules),
                'best_strategy': best_strategy,
                'best_strategy_win_rate': best_win_rate,
                'best_patterns': sorted(best_patterns, key=lambda x: x['success_rate'], reverse=True)[:5],
                'data_statistics': self.data_aggregator.get_statistics()
            }
            
        except Exception as e:
            print(f"⚠️ Insights error: {e}")
            return {}
    
    def get_recommendation(self, symbol: str, current_state: str) -> Dict:
        """Get AI recommendation based on learned knowledge"""
        try:
            # Get Q-values for current state
            q_values = self.q_table.get(current_state, {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0})
            
            # Find best action
            best_action = max(q_values, key=q_values.get)
            confidence = abs(q_values[best_action])
            
            return {
                'action': best_action,
                'confidence': min(confidence, 1.0),
                'q_values': q_values,
                'learned_from_samples': self.total_training_samples
            }
            
        except Exception as e:
            return {'action': 'HOLD', 'confidence': 0.0}


class MassiveDataTrainer:
    """
    🎓 TRAINS ON MILLIONS OF DATA POINTS
    Background training system that never stops learning
    """
    
    def __init__(self, learning_engine: ContinuousLearningEngine):
        self.learning_engine = learning_engine
        self.training_active = False
    
    async def start_continuous_training(self, data_feed, symbols: List[str], price_history: Dict):
        """Start continuous background training"""
        self.training_active = True
        
        print("🎓 STARTING CONTINUOUS TRAINING MODE")
        print("   Bot will learn from every market tick...")
        
        iteration = 0
        while self.training_active:
            try:
                # Feed the AI engine
                await self.learning_engine.feed_on_data(data_feed, symbols, price_history)
                
                iteration += 1
                
                # Every 5 iterations, show progress
                if iteration % 5 == 0:
                    insights = self.learning_engine.get_learned_insights()
                    print(f"📚 Training Progress: {insights['total_training_samples']:,} samples, {insights['learned_rules_count']} rules")
                
                # Wait a bit before next iteration
                await asyncio.sleep(30)  # Train every 30 seconds
                
            except Exception as e:
                print(f"⚠️ Training error: {e}")
                await asyncio.sleep(60)
    
    def stop_training(self):
        """Stop continuous training"""
        self.training_active = False
        print("⏸️ Continuous training paused")
