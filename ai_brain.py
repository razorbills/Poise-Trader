#!/usr/bin/env python3
"""
 PERSISTENT AI BRAIN SYSTEM
Saves and loads AI trading knowledge across sessions
"""

import json
import os
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict, deque
from ml_components import neural_predictor, rl_optimizer, pattern_engine, TradingSignalML
from ultra_ai_optimizer import UltraAIOptimizer

_IS_RENDER = bool(os.getenv('RENDER_EXTERNAL_URL') or os.getenv('RENDER_SERVICE_NAME'))


def _file_too_large_for_render(path: str, max_mb: float = 5.0) -> bool:
    try:
        if not _IS_RENDER:
            return False
        if not path or not os.path.exists(path):
            return False
        size = os.path.getsize(path)
        return size > float(max_mb) * 1024 * 1024
    except Exception:
        return False

class AIBrain:
    """Persistent AI brain that learns and remembers across sessions"""
    
    def __init__(self, brain_file: str = "ai_brain.json"):
        self.brain_file = brain_file
        self.backup_file = "ai_brain_backup.json"

        try:
            self.learning_min_seconds_between_updates = float(os.getenv('AI_LEARNING_MIN_SECONDS_BETWEEN_UPDATES', '0') or 0)
        except Exception:
            self.learning_min_seconds_between_updates = 0.0

        try:
            self.learning_min_trades_between_saves = int(float(os.getenv('AI_LEARNING_MIN_TRADES_BETWEEN_SAVES', '1') or 1))
        except Exception:
            self.learning_min_trades_between_saves = 1

        self._last_learning_update_ts = 0.0
        self._trades_since_save = 0
        
        # ML components
        self.neural_predictor = neural_predictor
        self.rl_optimizer = rl_optimizer
        self.pattern_engine = pattern_engine
        self.performance_history = deque(maxlen=1000)
        self.position_performance = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'win_rate': 0})
        
        # Initialize Ultra AI Optimizer for advanced learning
        self.ultra_optimizer = UltraAIOptimizer()
        
        self.brain = {
            'version': '3.0',
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_trades': 0,
            'total_profit_loss': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'learning_sessions': 0,
            
            # Strategy performance tracking
            'strategy_performance': {
                'technical': {'wins': 0, 'losses': 0, 'total_return': 0.0},
                'sentiment': {'wins': 0, 'losses': 0, 'total_return': 0.0},
                'momentum': {'wins': 0, 'losses': 0, 'total_return': 0.0},
                'mean_reversion': {'wins': 0, 'losses': 0, 'total_return': 0.0},
                'breakout': {'wins': 0, 'losses': 0, 'total_return': 0.0}
            },
            
            # Market condition patterns
            'market_patterns': {
                'bull_market': {'trades': 0, 'success_rate': 0.5},
                'bear_market': {'trades': 0, 'success_rate': 0.5},
                'sideways': {'trades': 0, 'success_rate': 0.5},
                'high_volatility': {'trades': 0, 'success_rate': 0.5},
                'low_volatility': {'trades': 0, 'success_rate': 0.5}
            },
            
            # Symbol-specific learning
            'symbol_knowledge': {},
            
            # Time-based patterns
            'time_patterns': {
                'hour_performance': {str(i): {'trades': 0, 'profit': 0.0} for i in range(24)},
                'day_performance': {str(i): {'trades': 0, 'profit': 0.0} for i in range(7)}
            },
            
            # Confidence adjustments based on past performance
            'confidence_adjustments': {
                'high_confidence_accuracy': 0.5,  # Accuracy when confidence > 0.7
                'medium_confidence_accuracy': 0.5, # Accuracy when confidence 0.3-0.7
                'low_confidence_accuracy': 0.5     # Accuracy when confidence < 0.3
            },
            
            # Risk management learnings
            'risk_learnings': {
                'optimal_position_sizes': {},
                'stop_loss_effectiveness': {},
                'take_profit_levels': {}
            },
            
            # Recent trades for pattern analysis
            'recent_trades': [],
            
            # ML performance tracking
            'ml_performance': {
                'neural_accuracy': 0.0,
                'pattern_success': 0.0,
                'rl_reward': 0.0
            },
            
            # Position tracking
            'position_tracking': {},
            
            # Execution quality
            'execution_metrics': {
                'slippage': [],
                'fill_rate': 0.0,
                'avg_execution_time': 0.0
            },
            
            # Learning sessions tracking
            'knowledge_sessions': []
        }
        self.load_brain()
        
    def load_brain(self):
        """Load AI brain from file"""
        try:
            if os.path.exists(self.brain_file):
                if _file_too_large_for_render(self.brain_file, max_mb=5.0):
                    print("🧠 AI brain file is large - skipping load on Render")
                    self.save_brain()
                    return
                with open(self.brain_file, 'r') as f:
                    loaded_brain = json.load(f)
                    
                # Merge with default structure (handles version updates)
                self._merge_brain_data(loaded_brain)
                
                # Load ultra optimizer state
                try:
                    if not _file_too_large_for_render(getattr(self.ultra_optimizer, 'learning_file', ''), max_mb=5.0) and not _file_too_large_for_render(getattr(self.ultra_optimizer, 'winning_patterns_file', ''), max_mb=5.0):
                        self.ultra_optimizer.load_state()
                except Exception:
                    pass

                try:
                    if _IS_RENDER:
                        rt = self.brain.get('recent_trades')
                        if isinstance(rt, list) and len(rt) > 200:
                            self.brain['recent_trades'] = rt[-200:]

                        wrh = self.brain.get('win_rate_history')
                        if isinstance(wrh, list) and len(wrh) > 50:
                            self.brain['win_rate_history'] = wrh[-50:]

                        ks = self.brain.get('knowledge_sessions')
                        if isinstance(ks, list) and len(ks) > 20:
                            self.brain['knowledge_sessions'] = ks[-20:]

                        em = self.brain.get('execution_metrics')
                        if isinstance(em, dict) and isinstance(em.get('slippage'), list) and len(em.get('slippage') or []) > 50:
                            em['slippage'] = list(em.get('slippage') or [])[-50:]
                            self.brain['execution_metrics'] = em

                        sk = self.brain.get('symbol_knowledge')
                        if isinstance(sk, dict) and len(sk) > 120:
                            trimmed = {}
                            keys = list(sk.keys())[-120:]
                            for k in keys:
                                trimmed[k] = sk.get(k)
                            self.brain['symbol_knowledge'] = trimmed
                except Exception:
                    pass
                
                print(f"🧠 AI BRAIN LOADED: {self.brain['total_trades']} trades, {self.brain['learning_sessions']} sessions")
                print(f"   📈 Win Rate: {self.brain.get('win_rate', 0):.1%}")
                print(f"   🤖 ML Accuracy: {self.brain.get('ml_performance', {}).get('neural_accuracy', 0):.1%}")
                
                # Check if we have winning patterns
                if hasattr(self.ultra_optimizer, 'winning_patterns') and self.ultra_optimizer.winning_patterns:
                    print(f"   🎯 Winning Patterns: {len(self.ultra_optimizer.winning_patterns)}")
                
                self._print_brain_summary()
            else:
                print("🧠 NEW AI BRAIN CREATED")
                self.save_brain()
        except Exception as e:
            print(f"⚠️ Error loading AI brain: {e}")
            print("🧠 Starting with fresh brain...")
            if os.path.exists(self.backup_file):
                try:
                    if _file_too_large_for_render(self.backup_file, max_mb=5.0):
                        return
                    with open(self.backup_file, 'r') as f:
                        self.brain = json.load(f)
                    print("✅ Restored from backup")
                except:
                    pass
    
    def _merge_brain_data(self, loaded_brain: Dict):
        """Merge loaded brain data with current structure"""
        for key, value in loaded_brain.items():
            if key in self.brain:
                if isinstance(value, dict) and isinstance(self.brain[key], dict):
                    self.brain[key].update(value)
                else:
                    self.brain[key] = value
    
    def save_brain(self):
        """Save AI brain to file"""
        try:
            self.brain['last_updated'] = datetime.now().isoformat()
            
            # Create backup
            if os.path.exists(self.brain_file):
                with open(self.brain_file, 'r') as f:
                    backup_data = f.read()
                with open(self.backup_file, 'w') as f:
                    f.write(backup_data)
            
            # Save current brain
            with open(self.brain_file, 'w') as f:
                json.dump(self.brain, f, indent=2)
            
            # Save ultra optimizer state
            self.ultra_optimizer.save_state()
                
            # Only print save message if we have trades (reduce noise)
            if self.brain['total_trades'] > 0:
                recent_win_rate = self.brain.get('win_rate', 0)
                print(f"💾 AI BRAIN SAVED: {self.brain['total_trades']} trades | {recent_win_rate:.1%} win rate")
        except Exception as e:
            print(f"⚠️ Error saving AI brain: {e}")
    
    def learn_from_trade(self, trade_data: Dict):
        """🔥 Enhanced learning from trade with comprehensive market data"""
        try:
            try:
                now_ts = time.time()
                if self.learning_min_seconds_between_updates and self.learning_min_seconds_between_updates > 0:
                    if self._last_learning_update_ts and (now_ts - float(self._last_learning_update_ts)) < float(self.learning_min_seconds_between_updates):
                        return
                self._last_learning_update_ts = now_ts
            except Exception:
                pass

            # Extract trade information
            symbol = trade_data.get('symbol', 'UNKNOWN')
            action = trade_data.get('action', 'UNKNOWN')
            profit_loss = trade_data.get('profit_loss', 0.0)
            confidence = trade_data.get('confidence', 0.5)
            strategy_scores = trade_data.get('strategy_scores', {})
            market_conditions = trade_data.get('market_conditions', {})
            
            # 🔥 NEW: Extract comprehensive market data if available
            comprehensive_data = trade_data.get('comprehensive_market_data', {})
            orderbook_data = comprehensive_data.get('orderbook', {})
            trade_flow_data = comprehensive_data.get('recent_trades', {})
            ticker_data = comprehensive_data.get('ticker_24h', {})
            klines_data = comprehensive_data.get('klines', {})
            
            # Update total statistics
            self.brain['total_trades'] += 1
            self.brain['total_profit_loss'] += profit_loss
            
            # Learn from strategy performance
            was_profitable = profit_loss > 0
            for strategy, score in strategy_scores.items():
                if strategy in self.brain['strategy_performance']:
                    if was_profitable:
                        self.brain['strategy_performance'][strategy]['wins'] += 1
                    else:
                        self.brain['strategy_performance'][strategy]['losses'] += 1
                    
                    self.brain['strategy_performance'][strategy]['total_return'] += profit_loss
            
            # Learn from market conditions
            volatility = market_conditions.get('volatility', 2.0)
            trend_strength = market_conditions.get('trend_strength', 0.0)
            
            if volatility > 3.0:
                condition = 'high_volatility'
            elif volatility < 1.0:
                condition = 'low_volatility'
            elif trend_strength > 0.3:
                condition = 'bull_market'
            elif trend_strength < -0.3:
                condition = 'bear_market'
            else:
                condition = 'sideways'
            
            self.brain['market_patterns'][condition]['trades'] += 1
            if was_profitable:
                current_rate = self.brain['market_patterns'][condition]['success_rate']
                total_trades = self.brain['market_patterns'][condition]['trades']
                new_rate = (current_rate * (total_trades - 1) + 1.0) / total_trades
                self.brain['market_patterns'][condition]['success_rate'] = new_rate
            
            # Learn symbol-specific patterns
            if symbol not in self.brain['symbol_knowledge']:
                self.brain['symbol_knowledge'][symbol] = {
                    'trades': 0,
                    'profit_loss': 0.0,
                    'best_strategies': {},
                    'volatility_preference': 0.0
                }
            
            self.brain['symbol_knowledge'][symbol]['trades'] += 1
            self.brain['symbol_knowledge'][symbol]['profit_loss'] += profit_loss
            
            # Learn confidence accuracy
            if confidence > 0.7:
                conf_key = 'high_confidence_accuracy'
            elif confidence > 0.3:
                conf_key = 'medium_confidence_accuracy'
            else:
                conf_key = 'low_confidence_accuracy'
            
            current_acc = self.brain['confidence_adjustments'][conf_key]
            # Update accuracy with exponential moving average
            new_acc = current_acc * 0.9 + (1.0 if was_profitable else 0.0) * 0.1
            self.brain['confidence_adjustments'][conf_key] = new_acc
            
            # Learn time patterns
            current_hour = datetime.now().hour
            current_day = datetime.now().weekday()
            
            # Ensure time_patterns structure exists (backward compatibility)
            if 'time_patterns' not in self.brain:
                self.brain['time_patterns'] = {
                    'hour_performance': {str(i): {'trades': 0, 'profit': 0.0} for i in range(24)},
                    'day_performance': {str(i): {'trades': 0, 'profit': 0.0} for i in range(7)}
                }
            
            # Ensure time pattern keys exist (in case loaded from old brain file)
            hour_key = str(current_hour)
            day_key = str(current_day)
            
            if hour_key not in self.brain['time_patterns']['hour_performance']:
                self.brain['time_patterns']['hour_performance'][hour_key] = {'trades': 0, 'profit': 0.0}
            if day_key not in self.brain['time_patterns']['day_performance']:
                self.brain['time_patterns']['day_performance'][day_key] = {'trades': 0, 'profit': 0.0}
            
            self.brain['time_patterns']['hour_performance'][hour_key]['trades'] += 1
            self.brain['time_patterns']['hour_performance'][hour_key]['profit'] += profit_loss
            
            self.brain['time_patterns']['day_performance'][day_key]['trades'] += 1
            self.brain['time_patterns']['day_performance'][day_key]['profit'] += profit_loss
            
            # Store recent trade for pattern recognition
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': action,
                'profit_loss': profit_loss,
                'confidence': confidence,
                'strategy_scores': strategy_scores,
                'market_conditions': market_conditions,
                # 🔥 NEW: Include comprehensive market data for deeper learning
                'orderbook_spread': orderbook_data.get('spread_pct', 0),
                'orderbook_imbalance': orderbook_data.get('volume_imbalance', 0),
                'buy_sell_ratio': trade_flow_data.get('buy_sell_ratio', 1.0),
                'volume_pressure': trade_flow_data.get('volume_pressure', 0),
                'price_change_24h': ticker_data.get('price_change_pct', 0),
                'volatility_24h': ticker_data.get('volatility_24h', 0),
                'market_sentiment': comprehensive_data.get('market_sentiment', 0),
                'liquidity_score': comprehensive_data.get('liquidity_score', 0),
                'momentum_score': comprehensive_data.get('momentum_score', 0)
            }
            
            # 🔥 NEW: Learn which market conditions lead to wins/losses
            if 'market_condition_performance' not in self.brain:
                self.brain['market_condition_performance'] = {}
            
            # Track performance by spread
            spread_category = 'tight' if orderbook_data.get('spread_pct', 999) < 0.05 else 'wide'
            if spread_category not in self.brain['market_condition_performance']:
                self.brain['market_condition_performance'][spread_category] = {'wins': 0, 'losses': 0}
            if was_profitable:
                self.brain['market_condition_performance'][spread_category]['wins'] += 1
            else:
                self.brain['market_condition_performance'][spread_category]['losses'] += 1
            
            # Track performance by buy/sell pressure
            pressure = trade_flow_data.get('volume_pressure', 0)
            pressure_category = 'buying' if pressure > 0.2 else 'selling' if pressure < -0.2 else 'neutral'
            if pressure_category not in self.brain['market_condition_performance']:
                self.brain['market_condition_performance'][pressure_category] = {'wins': 0, 'losses': 0}
            if was_profitable:
                self.brain['market_condition_performance'][pressure_category]['wins'] += 1
            else:
                self.brain['market_condition_performance'][pressure_category]['losses'] += 1
            
            # Track performance by liquidity
            liquidity = comprehensive_data.get('liquidity_score', 0)
            liquidity_category = 'high_liquidity' if liquidity > 0.7 else 'low_liquidity'
            if liquidity_category not in self.brain['market_condition_performance']:
                self.brain['market_condition_performance'][liquidity_category] = {'wins': 0, 'losses': 0}
            if was_profitable:
                self.brain['market_condition_performance'][liquidity_category]['wins'] += 1
            else:
                self.brain['market_condition_performance'][liquidity_category]['losses'] += 1
            
            self.brain['recent_trades'].append(trade_record)
            
            # Keep only last 1000 trades
            if len(self.brain['recent_trades']) > 1000:
                self.brain['recent_trades'] = self.brain['recent_trades'][-1000:]
            
            # Update win rate tracking
            total_recent = len(self.brain['recent_trades'])
            if total_recent > 0:
                recent_wins = sum(1 for t in self.brain['recent_trades'][-100:] if t['profit_loss'] > 0)
                recent_win_rate = recent_wins / min(100, total_recent)
                
                # Track win rate improvement
                if 'win_rate_history' not in self.brain:
                    self.brain['win_rate_history'] = []
                
                self.brain['win_rate_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'win_rate': recent_win_rate,
                    'total_trades': total_recent
                })
                
                # Keep only last 100 win rate records
                if len(self.brain['win_rate_history']) > 100:
                    self.brain['win_rate_history'] = self.brain['win_rate_history'][-100:]
                
                # Check if we're improving towards 90% win rate
                if recent_win_rate > 0.85:
                    print(f"   🎯 APPROACHING TARGET: {recent_win_rate:.1%} win rate!")
                elif recent_win_rate > 0.7:
                    print(f"   📈 Good progress: {recent_win_rate:.1%} win rate")
                elif recent_win_rate < 0.4:
                    print(f"   ⚠️ Need improvement: {recent_win_rate:.1%} win rate")
            
            # Update strategy weights based on performance
            self._update_strategy_weights()
            
            # Update performance metrics
            self._update_performance_metrics(trade_data)
        
            # ML learning
            self._ml_learn(trade_data)
            
            # Ultra AI Optimizer learning for advanced pattern recognition
            self.ultra_optimizer.learn_from_trade(trade_data)
            
            # Get recommendations from ultra optimizer
            recommendations = self.ultra_optimizer.get_recommendations()
            if recommendations:
                print(f"   🎯 AI Recommendations:")
                for rec in recommendations[:3]:  # Show top 3
                    print(f"      • {rec}")
        
            # Save immediately after every trade to ensure persistence
            try:
                self._trades_since_save = int(getattr(self, '_trades_since_save', 0) or 0) + 1
            except Exception:
                self._trades_since_save = 1

            should_save = True
            try:
                should_save = self._trades_since_save >= int(self.learning_min_trades_between_saves)
            except Exception:
                should_save = True

            if should_save:
                try:
                    self.save_brain()
                    self._trades_since_save = 0
                except Exception:
                    pass
            
            print(f"🧠 LEARNED FROM TRADE: {symbol} {action} → ${profit_loss:+.2f}")
        
        except Exception as e:
            print(f"⚠️ Error learning from trade: {e}")
    
    def update_position_performance(self, symbol: str, position_data: Dict):
        """Update performance metrics for a position"""
        try:
            pnl = position_data.get('pnl', 0)
            entry_price = position_data.get('entry_price', 0)
            exit_price = position_data.get('exit_price', 0)
            hold_time = position_data.get('hold_time', 0)
            
            if symbol not in self.position_performance:
                self.position_performance[symbol] = {
                    'total_pnl': 0,
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'avg_hold_time': 0,
                    'best_trade': 0,
                    'worst_trade': 0
                }
            
            perf = self.position_performance[symbol]
            perf['total_pnl'] += pnl
            perf['trades'] += 1
            
            if pnl > 0:
                perf['wins'] += 1
                perf['best_trade'] = max(perf['best_trade'], pnl)
            else:
                perf['losses'] += 1
                perf['worst_trade'] = min(perf['worst_trade'], pnl)
            
            # Update average hold time
            perf['avg_hold_time'] = ((perf['avg_hold_time'] * (perf['trades'] - 1)) + hold_time) / perf['trades']
            
            # Calculate win rate
            perf['win_rate'] = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
            
            # Store in brain
            if 'position_performance' not in self.brain:
                self.brain['position_performance'] = {}
            self.brain['position_performance'][symbol] = perf
            
            print(f"   📊 Position Performance Updated: {symbol} - Win Rate: {perf['win_rate']:.1%}")
            
        except Exception as e:
            print(f"⚠️ Error updating position performance: {e}")

    def update_position_tracking(self, symbol: str, position_data: Dict):
        try:
            if 'position_tracking' not in self.brain:
                self.brain['position_tracking'] = {}

            tracking = dict(position_data) if isinstance(position_data, dict) else {'data': position_data}
            tracking['timestamp'] = datetime.now().isoformat()
            self.brain['position_tracking'][symbol] = tracking
        except Exception as e:
            print(f"⚠️ Error updating position tracking: {e}")
    
    def learn_from_execution(self, execution_data: Dict):
        """Learn from trade execution quality"""
        try:
            symbol = execution_data.get('symbol', '')
            expected_price = execution_data.get('expected_price', 0)
            actual_price = execution_data.get('actual_price', 0)
            execution_time = execution_data.get('execution_time', 0)
            order_type = execution_data.get('order_type', 'MARKET')
            
            # Calculate slippage
            if expected_price > 0:
                slippage = abs(actual_price - expected_price) / expected_price
                self.brain['execution_metrics']['slippage'].append(slippage)
                
                # Keep only recent slippage data
                if len(self.brain['execution_metrics']['slippage']) > 100:
                    self.brain['execution_metrics']['slippage'] = self.brain['execution_metrics']['slippage'][-100:]
            
            # Update execution time
            if execution_time > 0:
                current_avg = self.brain['execution_metrics'].get('avg_execution_time', 0)
                total_execs = len(self.brain['execution_metrics'].get('slippage', []))
                new_avg = ((current_avg * (total_execs - 1)) + execution_time) / total_execs if total_execs > 0 else execution_time
                self.brain['execution_metrics']['avg_execution_time'] = new_avg
            
            # Learn optimal execution strategies
            if 'execution_learnings' not in self.brain:
                self.brain['execution_learnings'] = {}
            
            if symbol not in self.brain['execution_learnings']:
                self.brain['execution_learnings'][symbol] = {
                    'best_order_type': order_type,
                    'avg_slippage': 0,
                    'optimal_time': 0
                }
            
            # Update execution learnings
            exec_learn = self.brain['execution_learnings'][symbol]
            if slippage < exec_learn.get('avg_slippage', 1):
                exec_learn['best_order_type'] = order_type
            
            print(f"   ⚡ Execution Learning: {symbol} - Slippage: {slippage:.2%}")
            
        except Exception as e:
            print(f"⚠️ Error learning from execution: {e}")
    
    def _update_performance_metrics(self, trade_data: Dict):
        """Update overall performance metrics"""
        try:
            # Add to performance history
            self.performance_history.append(trade_data)
            
            # Calculate win rate
            wins = sum(1 for t in self.brain['recent_trades'] if t.get('profit_loss', 0) > 0)
            total = len(self.brain['recent_trades'])
            self.brain['win_rate'] = wins / total if total > 0 else 0
            
            # Calculate Sharpe ratio
            if len(self.performance_history) > 10:
                returns = [t.get('profit_loss', 0) for t in self.performance_history]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                self.brain['sharpe_ratio'] = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            
        except Exception as e:
            print(f"⚠️ Error updating performance metrics: {e}")
    
    def _ml_learn(self, trade_data: Dict):
        """Apply machine learning to learn from trade"""
        try:
            symbol = trade_data.get('symbol', '')
            profit_loss = trade_data.get('profit_loss', 0)
            market_conditions = trade_data.get('market_conditions', {})
            
            # Reinforcement learning update
            state = self.rl_optimizer.get_state(market_conditions)
            action = 'BUY' if trade_data.get('action') == 'BUY' else 'SELL'
            reward = self.rl_optimizer.calculate_reward({'pnl': profit_loss, 'risk': 1})
            next_state = self.rl_optimizer.get_state(market_conditions)  # Simplified for now
            
            self.rl_optimizer.update_q_value(state, action, reward, next_state)
            
            # Update ML performance
            self.brain['ml_performance']['rl_reward'] = reward
            
            # Pattern recognition learning
            if 'price_history' in trade_data:
                patterns = self.pattern_engine.detect_patterns(trade_data['price_history'])
                if patterns:
                    # Track pattern success
                    for pattern in patterns:
                        if profit_loss > 0:
                            self.brain['ml_performance']['pattern_success'] += 0.1
                        else:
                            self.brain['ml_performance']['pattern_success'] -= 0.05
            
            print(f"   🤖 ML Learning: RL Reward: {reward:.2f}")
            
        except Exception as e:
            print(f"⚠️ Error in ML learning: {e}")
    
    def get_enhanced_prediction(self, symbol: str, price_history: List[float], market_data: Dict = None) -> Dict:
        """Get enhanced prediction using Ultra AI Optimizer"""
        try:
            # Get base ML prediction
            ml_signal = self.get_ml_prediction(symbol, price_history)
            
            # Get ultra optimizer prediction
            if market_data:
                ultra_prediction = self.ultra_optimizer.predict_trade_outcome(
                    symbol=symbol,
                    confidence=ml_signal.confidence,
                    strategy_scores=market_data.get('strategy_scores', {}),
                    market_conditions=market_data.get('market_conditions', {})
                )
                
                # Combine predictions
                combined_confidence = (ml_signal.confidence * 0.4 + ultra_prediction.get('confidence', 0.5) * 0.6)
                
                # Apply win rate boost if we have good patterns
                win_prob = ultra_prediction.get('win_probability', 0.5)
                if win_prob > 0.7:
                    combined_confidence = min(0.95, combined_confidence * 1.2)
                elif win_prob < 0.3:
                    combined_confidence = max(0.1, combined_confidence * 0.7)
                
                # Derive fields not provided by the optimizer for compatibility
                risk_score = 1.0 - win_prob  # Lower risk when win probability is high
                pattern_strength = win_prob if ultra_prediction.get('pattern_matches') else max(0.3, win_prob * 0.8)
                
                return {
                    'action': ml_signal.action,
                    'confidence': combined_confidence,
                    'ml_confidence': ml_signal.confidence,
                    'ultra_confidence': ultra_prediction.get('confidence', 0.5),
                    'win_probability': win_prob,
                    'risk_score': risk_score,
                    'pattern_strength': pattern_strength,
                    'stop_loss': ml_signal.stop_loss,
                    'take_profit': ml_signal.take_profit,
                    'optimal_size': min(ml_signal.optimal_size, 0.05)
                }
            else:
                # Fallback to ML signal only
                return {
                    'action': ml_signal.action,
                    'confidence': ml_signal.confidence,
                    'ml_confidence': ml_signal.confidence,
                    'ultra_confidence': 0.5,
                    'win_probability': 0.5,
                    'risk_score': 0.5,
                    'pattern_strength': 0.5,
                    'stop_loss': ml_signal.stop_loss,
                    'take_profit': ml_signal.take_profit,
                    'optimal_size': ml_signal.optimal_size
                }
                
        except Exception as e:
            print(f"⚠️ Enhanced prediction error: {e}")
            # Return safe default
            return {
                'action': 'HOLD',
                'confidence': 0.3,
                'ml_confidence': 0.3,
                'ultra_confidence': 0.3,
                'win_probability': 0.3,
                'risk_score': 0.7,
                'pattern_strength': 0.3,
                'stop_loss': 0,
                'take_profit': 0,
                'optimal_size': 0.01
            }
    
    def get_ml_prediction(self, symbol: str, price_history: List[float]) -> TradingSignalML:
        """Get ML-enhanced trading prediction"""
        try:
            # Neural network prediction
            nn_price, nn_confidence = self.neural_predictor.predict(price_history)
            
            # Pattern detection
            patterns = self.pattern_engine.detect_patterns(price_history)
            pattern_score = max([p.confidence for p in patterns]) if patterns else 0.5
            expected_move = np.mean([p.expected_move for p in patterns]) if patterns else 0
            
            # RL action selection
            current_price = price_history[-1] if price_history else 0
            market_state = {
                'rsi': 50,  # Simplified
                'trend': 'up' if nn_price > current_price else 'down',
                'volatility': 'medium'
            }
            rl_action = self.rl_optimizer.choose_action(self.rl_optimizer.get_state(market_state))
            
            # Combine predictions
            if nn_price > current_price * 1.002 and rl_action != 'SELL':
                action = 'BUY'
                confidence = (nn_confidence + pattern_score) / 2
            elif nn_price < current_price * 0.998 and rl_action != 'BUY':
                action = 'SELL'
                confidence = (nn_confidence + pattern_score) / 2
            else:
                action = 'HOLD'
                confidence = 0.4
            
            # Calculate optimal position size based on confidence
            optimal_size = min(0.05, confidence * 0.03)  # Max 5% of capital
            
            # Risk management
            stop_loss = current_price * (0.98 if action == 'BUY' else 1.02)
            take_profit = current_price * (1.03 if action == 'BUY' else 0.97)
            
            return TradingSignalML(
                symbol=symbol,
                action=action,
                confidence=confidence,
                ml_score=(nn_confidence + pattern_score) / 2,
                nn_prediction=nn_price,
                pattern_score=pattern_score,
                risk_reward=abs(take_profit - current_price) / abs(stop_loss - current_price),
                optimal_size=optimal_size,
                expected_return=expected_move,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        except Exception as e:
            print(f"⚠️ Error getting ML prediction: {e}")
            # Return default signal
            return TradingSignalML(
                symbol=symbol,
                action='HOLD',
                confidence=0.5,
                ml_score=0.5,
                nn_prediction=price_history[-1] if price_history else 0,
                pattern_score=0.5,
                risk_reward=1.0,
                optimal_size=0.01,
                expected_return=0,
                stop_loss=0,
                take_profit=0
            )
            
        except Exception as e:
            print(f"⚠️ Learning error: {e}")
    
    def _update_strategy_weights(self):
        """Update strategy weights based on performance"""
        try:
            total_performance = 0
            strategy_performance = {}
            
            for strategy, perf in self.brain['strategy_performance'].items():
                wins = perf['wins']
                losses = perf['losses']
                total_trades = wins + losses
                
                if total_trades > 0:
                    win_rate = wins / total_trades
                    avg_return = perf['total_return'] / total_trades if total_trades > 0 else 0
                    
                    # Combined performance score
                    performance_score = win_rate * 0.6 + (avg_return / 100) * 0.4
                    strategy_performance[strategy] = max(0.01, performance_score)
                    total_performance += strategy_performance[strategy]
            
            # Initialize optimal_strategy_weights if not exists
            if 'optimal_strategy_weights' not in self.brain:
                self.brain['optimal_strategy_weights'] = {
                    'technical': 0.2,
                    'sentiment': 0.2,
                    'momentum': 0.2,
                    'mean_reversion': 0.2,
                    'breakout': 0.2
                }
            
            # Normalize weights
            if total_performance > 0:
                for strategy in strategy_performance:
                    new_weight = strategy_performance[strategy] / total_performance
                    current_weight = self.brain['optimal_strategy_weights'].get(strategy, 0.2)
                    
                    # Smooth weight updates with faster learning for winning strategies
                    if new_weight > current_weight:
                        # Faster adoption of winning strategies
                        self.brain['optimal_strategy_weights'][strategy] = (
                            current_weight * 0.6 + new_weight * 0.4
                        )
                    else:
                        # Slower reduction for losing strategies
                        self.brain['optimal_strategy_weights'][strategy] = (
                            current_weight * 0.9 + new_weight * 0.1
                        )
        except Exception as e:
            print(f"⚠️ Weight update error: {e}")
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get learned optimal strategy weights"""
        return self.brain['optimal_strategy_weights'].copy()
    
    def get_confidence_adjustment(self, base_confidence: float) -> float:
        """Adjust confidence based on past accuracy"""
        try:
            if base_confidence > 0.7:
                accuracy = self.brain['confidence_adjustments']['high_confidence_accuracy']
            elif base_confidence > 0.3:
                accuracy = self.brain['confidence_adjustments']['medium_confidence_accuracy']
            else:
                accuracy = self.brain['confidence_adjustments']['low_confidence_accuracy']
            
            # Adjust confidence based on historical accuracy
            if accuracy > 0.6:
                return min(1.0, base_confidence * 1.2)  # Boost if historically accurate
            elif accuracy < 0.4:
                return max(0.1, base_confidence * 0.8)  # Reduce if historically inaccurate
            else:
                return base_confidence
        except:
            return base_confidence
    
    def get_market_condition_multiplier(self, market_conditions: Dict) -> float:
        """Get multiplier based on current market conditions"""
        try:
            volatility = market_conditions.get('volatility', 2.0)
            trend_strength = market_conditions.get('trend_strength', 0.0)
            
            if volatility > 3.0:
                condition = 'high_volatility'
            elif volatility < 1.0:
                condition = 'low_volatility'
            elif trend_strength > 0.3:
                condition = 'bull_market'
            elif trend_strength < -0.3:
                condition = 'bear_market'
            else:
                condition = 'sideways'
            
            success_rate = self.brain['market_patterns'][condition]['success_rate']
            
            # Return multiplier based on success rate
            return 0.5 + success_rate  # Range: 0.5 to 1.5
            
        except:
            return 1.0
    
    def get_symbol_preference(self, symbol: str) -> float:
        """Get preference score for a symbol based on past performance"""
        try:
            if symbol in self.brain['symbol_knowledge']:
                symbol_data = self.brain['symbol_knowledge'][symbol]
                if symbol_data['trades'] > 0:
                    avg_profit = symbol_data['profit_loss'] / symbol_data['trades']
                    # Convert to preference score (0.5 to 1.5)
                    return max(0.5, min(1.5, 1.0 + avg_profit / 10))
            return 1.0
        except:
            return 1.0
    
    def update_position_performance(self, symbol: str, position_data: Dict):
        """Update AI brain with intermediate position performance (continuous learning)"""
        try:
            # Track position performance over time
            if symbol not in self.brain['symbol_knowledge']:
                self.brain['symbol_knowledge'][symbol] = {
                    'trades': 0,
                    'profit_loss': 0.0,
                    'best_strategies': {},
                    'volatility_preference': 0.0,
                    'position_tracking': []
                }
            
            # Add position tracking data
            tracking_data = {
                'timestamp': datetime.now().isoformat(),
                'unrealized_pnl': position_data.get('unrealized_pnl', 0),
                'pnl_percentage': position_data.get('pnl_percentage', 0),
                'cycles_held': position_data.get('cycles_held', 0),
                'confidence': position_data.get('confidence', 0.5)
            }
            
            # Keep only last 50 tracking points per symbol
            if 'position_tracking' not in self.brain['symbol_knowledge'][symbol]:
                self.brain['symbol_knowledge'][symbol]['position_tracking'] = []
            
            self.brain['symbol_knowledge'][symbol]['position_tracking'].append(tracking_data)
            if len(self.brain['symbol_knowledge'][symbol]['position_tracking']) > 50:
                self.brain['symbol_knowledge'][symbol]['position_tracking'] = \
                    self.brain['symbol_knowledge'][symbol]['position_tracking'][-50:]
            
            # Learn from position trends
            self._learn_from_position_trends(symbol, position_data)
            
        except Exception as e:
            print(f"⚠️ Position update error: {e}")
    
    def learn_from_execution(self, execution_data: Dict):
        """Learn from trade execution quality and timing"""
        try:
            symbol = execution_data.get('symbol', 'UNKNOWN')
            success = execution_data.get('execution_success', False)
            confidence = execution_data.get('confidence', 0.5)
            
            # Track execution success rates
            if 'execution_stats' not in self.brain:
                self.brain['execution_stats'] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'failed_executions': 0,
                    'confidence_vs_success': {}
                }
            
            self.brain['execution_stats']['total_executions'] += 1
            if success:
                self.brain['execution_stats']['successful_executions'] += 1
            else:
                self.brain['execution_stats']['failed_executions'] += 1
            
            # Track confidence vs execution success
            conf_bucket = f"{int(confidence * 10) / 10:.1f}"  # Round to nearest 0.1
            if conf_bucket not in self.brain['execution_stats']['confidence_vs_success']:
                self.brain['execution_stats']['confidence_vs_success'][conf_bucket] = {
                    'successes': 0,
                    'failures': 0
                }
            
            if success:
                self.brain['execution_stats']['confidence_vs_success'][conf_bucket]['successes'] += 1
            else:
                self.brain['execution_stats']['confidence_vs_success'][conf_bucket]['failures'] += 1
            
            # Update symbol-specific execution stats
            if symbol in self.brain['symbol_knowledge']:
                if 'execution_quality' not in self.brain['symbol_knowledge'][symbol]:
                    self.brain['symbol_knowledge'][symbol]['execution_quality'] = 0.5
                
                # Update execution quality score
                current_quality = self.brain['symbol_knowledge'][symbol]['execution_quality']
                new_quality = current_quality * 0.9 + (1.0 if success else 0.0) * 0.1
                self.brain['symbol_knowledge'][symbol]['execution_quality'] = new_quality
            
        except Exception as e:
            print(f"⚠️ Execution learning error: {e}")
    
    def _learn_from_position_trends(self, symbol: str, position_data: Dict):
        """Learn from position performance trends"""
        try:
            tracking = self.brain['symbol_knowledge'][symbol].get('position_tracking', [])
            if len(tracking) < 3:
                return
            
            # Analyze recent trend
            recent_pnls = [t['pnl_percentage'] for t in tracking[-5:]]
            trend = 'improving' if recent_pnls[-1] > recent_pnls[0] else 'declining'
            
            # Update learning based on trend
            if trend == 'improving' and position_data.get('confidence', 0) > 0.5:
                # Boost confidence for this type of setup
                strategy_scores = position_data.get('strategy_scores', {})
                for strategy, score in strategy_scores.items():
                    if strategy in self.brain['strategy_performance'] and score > 0.5:
                        # Slightly increase the perceived effectiveness
                        self.brain['strategy_performance'][strategy]['wins'] += 0.1
            
        except Exception as e:
            print(f"⚠️ Trend learning error: {e}")
    
    def start_learning_session(self):
        """Mark the start of a learning session"""
        self.brain['learning_sessions'] += 1
        session_data = {
            'session_id': self.brain['learning_sessions'],
            'start_time': datetime.now().isoformat(),
            'trades_at_start': self.brain['total_trades'],
            'profit_at_start': self.brain['total_profit_loss']
        }
        self.brain['knowledge_sessions'].append(session_data)
        print(f"🚀 LEARNING SESSION #{self.brain['learning_sessions']} STARTED")
    
    def end_learning_session(self):
        """Mark the end of a learning session"""
        if self.brain['knowledge_sessions']:
            last_session = self.brain['knowledge_sessions'][-1]
            last_session['end_time'] = datetime.now().isoformat()
            last_session['trades_learned'] = self.brain['total_trades'] - last_session['trades_at_start']
            last_session['profit_learned'] = self.brain['total_profit_loss'] - last_session['profit_at_start']
            
            print(f"📚 LEARNING SESSION COMPLETE:")
            print(f"   🔢 Trades: {last_session['trades_learned']}")
            print(f"   💰 P&L: ${last_session['profit_learned']:+.2f}")
        
        self.save_brain()
    
    def _print_brain_summary(self):
        """Print a summary of what the AI has learned"""
        try:
            total_trades = self.brain['total_trades']
            if total_trades == 0:
                print("🧠 Fresh brain - no experience yet")
                return
            
            total_pnl = self.brain['total_profit_loss']
            avg_trade = total_pnl / total_trades
            
            print(f"📊 AI BRAIN SUMMARY:")
            print(f"   🎯 Total Trades: {total_trades}")
            print(f"   💰 Total P&L: ${total_pnl:+.2f}")
            print(f"   📈 Avg per Trade: ${avg_trade:+.2f}")
            print(f"   🎓 Learning Sessions: {self.brain['learning_sessions']}")
            
            # Best performing strategy
            best_strategy = "unknown"
            best_return = -999999
            for strategy, perf in self.brain['strategy_performance'].items():
                if perf['total_return'] > best_return:
                    best_return = perf['total_return']
                    best_strategy = strategy
            
            if best_return > -999999:
                print(f"   🏆 Best Strategy: {best_strategy} (${best_return:+.2f})")
            
            # Best performing symbol
            if self.brain['symbol_knowledge']:
                best_symbol = max(
                    self.brain['symbol_knowledge'].items(),
                    key=lambda x: x[1]['profit_loss']
                )[0]
                best_profit = self.brain['symbol_knowledge'][best_symbol]['profit_loss']
                print(f"   💎 Best Symbol: {best_symbol} (${best_profit:+.2f})")
                
        except Exception as e:
            print(f"⚠️ Brain summary error: {e}")

# Global AI brain instance
ai_brain = AIBrain()
