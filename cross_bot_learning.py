#!/usr/bin/env python3
"""
🤝 CROSS-BOT LEARNING SYSTEM
Shared AI knowledge between multiple trading bots
"""

import json
import os
from datetime import datetime
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossBotLearningSystem:
    """Cross-bot learning system for shared AI knowledge between trading bots"""
    
    def __init__(self, shared_brain_file: str = "shared_ai_knowledge.json"):
        self.shared_brain_file = shared_brain_file
        self.load_shared_knowledge()
        
    def load_shared_knowledge(self):
        """Load shared knowledge from all bots"""
        try:
            if os.path.exists(self.shared_brain_file):
                with open(self.shared_brain_file, 'r') as f:
                    self.shared_knowledge = json.load(f)
                    logger.info(f"Loaded shared knowledge: {self.shared_knowledge.get('cross_bot_trades', 0)} cross-bot trades")
            else:
                self.shared_knowledge = self._initialize_shared_knowledge()
                self.save_shared_knowledge()
                logger.info("Initialized new shared knowledge database")
        except Exception as e:
            logger.error(f"Error loading shared knowledge: {e}")
            self.shared_knowledge = self._initialize_shared_knowledge()
    
    def _initialize_shared_knowledge(self) -> Dict:
        """Initialize empty shared knowledge structure"""
        return {
            'version': '2.0',
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'cross_bot_trades': 0,
            'micro_bot_lessons': [],
            'profit_bot_lessons': [],
            'shared_patterns': {},
            'symbol_performance': {},
            'regime_lessons': {},
            'sentiment_accuracy': {},
            'best_practices': [],
            'strategy_success_rates': {
                'TREND_FOLLOWING': {'wins': 0, 'losses': 0},
                'MEAN_REVERSION': {'wins': 0, 'losses': 0},
                'MOMENTUM': {'wins': 0, 'losses': 0},
                'BREAKOUT': {'wins': 0, 'losses': 0}
            },
            'market_regime_performance': {
                'BULL': {'trades': 0, 'profit': 0.0},
                'BEAR': {'trades': 0, 'profit': 0.0},
                'CRAB': {'trades': 0, 'profit': 0.0},
                'VOLATILE': {'trades': 0, 'profit': 0.0}
            }
        }
    
    def share_trade_lesson(self, bot_type: str, lesson: Dict):
        """Share a trade lesson with other bots"""
        try:
            # Add metadata to lesson
            lesson['timestamp'] = datetime.now().isoformat()
            lesson['bot_source'] = bot_type
            
            # Store lesson by bot type
            if bot_type == 'micro':
                self.shared_knowledge['micro_bot_lessons'].append(lesson)
                # Keep only recent lessons (last 100)
                if len(self.shared_knowledge['micro_bot_lessons']) > 100:
                    self.shared_knowledge['micro_bot_lessons'] = self.shared_knowledge['micro_bot_lessons'][-100:]
            elif bot_type == 'profit':
                self.shared_knowledge['profit_bot_lessons'].append(lesson)
                if len(self.shared_knowledge['profit_bot_lessons']) > 100:
                    self.shared_knowledge['profit_bot_lessons'] = self.shared_knowledge['profit_bot_lessons'][-100:]
            elif bot_type == 'legendary_micro':
                # Special handling for legendary trades
                if 'legendary_lessons' not in self.shared_knowledge:
                    self.shared_knowledge['legendary_lessons'] = []
                self.shared_knowledge['legendary_lessons'].append(lesson)
                if len(self.shared_knowledge['legendary_lessons']) > 50:
                    self.shared_knowledge['legendary_lessons'] = self.shared_knowledge['legendary_lessons'][-50:]
            
            # Update shared patterns
            self._update_shared_patterns(lesson)
            
            # Update strategy success rates
            self._update_strategy_success_rates(lesson)
            
            # Update market regime performance
            self._update_regime_performance(lesson)
            
            self.shared_knowledge['cross_bot_trades'] += 1
            self.shared_knowledge['last_updated'] = datetime.now().isoformat()
            
            # Save every 5 trades
            if self.shared_knowledge['cross_bot_trades'] % 5 == 0:
                self.save_shared_knowledge()
            
            logger.info(f"Shared lesson from {bot_type} bot: {lesson.get('type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error sharing lesson: {e}")
    
    def _update_shared_patterns(self, lesson: Dict):
        """Update shared trading patterns from lessons"""
        try:
            pattern_key = f"{lesson.get('symbol', 'UNKNOWN')}_{lesson.get('strategy', 'UNKNOWN')}"
            
            if pattern_key not in self.shared_knowledge['shared_patterns']:
                self.shared_knowledge['shared_patterns'][pattern_key] = {
                    'occurrences': 0,
                    'total_pnl': 0.0,
                    'avg_confidence': 0.0,
                    'success_rate': 0.0
                }
            
            pattern = self.shared_knowledge['shared_patterns'][pattern_key]
            pattern['occurrences'] += 1
            
            pnl = lesson.get('pnl', 0.0)
            pattern['total_pnl'] += pnl
            
            # Update average confidence
            confidence = lesson.get('confidence', 0.5)
            pattern['avg_confidence'] = (
                (pattern['avg_confidence'] * (pattern['occurrences'] - 1) + confidence) / 
                pattern['occurrences']
            )
            
            # Update success rate
            if pnl > 0:
                pattern['success_rate'] = (
                    (pattern['success_rate'] * (pattern['occurrences'] - 1) + 1) / 
                    pattern['occurrences']
                )
            else:
                pattern['success_rate'] = (
                    (pattern['success_rate'] * (pattern['occurrences'] - 1)) / 
                    pattern['occurrences']
                )
                
        except Exception as e:
            logger.error(f"Error updating shared patterns: {e}")
    
    def _update_strategy_success_rates(self, lesson: Dict):
        """Update strategy success rates across all bots"""
        try:
            strategy = lesson.get('strategy', '').upper()
            if strategy in self.shared_knowledge['strategy_success_rates']:
                pnl = lesson.get('pnl', 0.0)
                if pnl > 0:
                    self.shared_knowledge['strategy_success_rates'][strategy]['wins'] += 1
                else:
                    self.shared_knowledge['strategy_success_rates'][strategy]['losses'] += 1
                    
        except Exception as e:
            logger.error(f"Error updating strategy success rates: {e}")
    
    def _update_regime_performance(self, lesson: Dict):
        """Update market regime performance tracking"""
        try:
            regime = lesson.get('regime', '').upper()
            if regime in self.shared_knowledge['market_regime_performance']:
                self.shared_knowledge['market_regime_performance'][regime]['trades'] += 1
                self.shared_knowledge['market_regime_performance'][regime]['profit'] += lesson.get('pnl', 0.0)
                
        except Exception as e:
            logger.error(f"Error updating regime performance: {e}")
    
    def get_lessons_for_bot(self, bot_type: str) -> List[Dict]:
        """Get relevant lessons for specific bot type"""
        try:
            if bot_type == 'micro':
                # Micro bot learns from profit bot's successful patterns
                profit_lessons = self.shared_knowledge.get('profit_bot_lessons', [])
                legendary_lessons = self.shared_knowledge.get('legendary_lessons', [])
                
                # Combine and return most recent successful lessons
                all_lessons = profit_lessons + legendary_lessons
                successful_lessons = [l for l in all_lessons if l.get('pnl', 0) > 0]
                return successful_lessons[-20:] if successful_lessons else []
                
            elif bot_type == 'profit':
                # Profit bot learns from micro bot's efficient patterns
                micro_lessons = self.shared_knowledge.get('micro_bot_lessons', [])
                legendary_lessons = self.shared_knowledge.get('legendary_lessons', [])
                
                # Focus on lessons with high confidence
                all_lessons = micro_lessons + legendary_lessons
                high_confidence_lessons = [l for l in all_lessons if l.get('confidence', 0) > 0.6]
                return high_confidence_lessons[-20:] if high_confidence_lessons else []
                
            else:
                # Return all recent lessons for other bot types
                all_lessons = (
                    self.shared_knowledge.get('micro_bot_lessons', [])[-10:] +
                    self.shared_knowledge.get('profit_bot_lessons', [])[-10:]
                )
                return all_lessons
                
        except Exception as e:
            logger.error(f"Error getting lessons: {e}")
            return []
    
    def get_best_patterns(self, min_occurrences: int = 5) -> Dict:
        """Get the best performing patterns across all bots"""
        try:
            best_patterns = {}
            
            for pattern_key, pattern_data in self.shared_knowledge['shared_patterns'].items():
                if pattern_data['occurrences'] >= min_occurrences:
                    if pattern_data['success_rate'] > 0.6:  # 60% success rate threshold
                        best_patterns[pattern_key] = {
                            'success_rate': pattern_data['success_rate'],
                            'avg_confidence': pattern_data['avg_confidence'],
                            'total_pnl': pattern_data['total_pnl'],
                            'occurrences': pattern_data['occurrences']
                        }
            
            # Sort by success rate
            sorted_patterns = dict(sorted(
                best_patterns.items(), 
                key=lambda x: x[1]['success_rate'], 
                reverse=True
            ))
            
            return sorted_patterns
            
        except Exception as e:
            logger.error(f"Error getting best patterns: {e}")
            return {}
    
    def get_strategy_recommendations(self) -> Dict:
        """Get strategy recommendations based on cross-bot learning"""
        try:
            recommendations = {}
            
            for strategy, performance in self.shared_knowledge['strategy_success_rates'].items():
                total_trades = performance['wins'] + performance['losses']
                if total_trades > 0:
                    win_rate = performance['wins'] / total_trades
                    recommendations[strategy] = {
                        'win_rate': win_rate,
                        'total_trades': total_trades,
                        'recommendation': 'USE' if win_rate > 0.55 else 'CAUTION' if win_rate > 0.45 else 'AVOID'
                    }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting strategy recommendations: {e}")
            return {}
    
    def save_shared_knowledge(self):
        """Save shared knowledge to file"""
        try:
            with open(self.shared_brain_file, 'w') as f:
                json.dump(self.shared_knowledge, f, indent=2)
            logger.info(f"Saved shared knowledge: {self.shared_knowledge['cross_bot_trades']} trades")
        except Exception as e:
            logger.error(f"Error saving shared knowledge: {e}")
    
    def get_summary(self) -> Dict:
        """Get a summary of cross-bot learning insights"""
        try:
            summary = {
                'total_cross_bot_trades': self.shared_knowledge['cross_bot_trades'],
                'micro_bot_lessons': len(self.shared_knowledge.get('micro_bot_lessons', [])),
                'profit_bot_lessons': len(self.shared_knowledge.get('profit_bot_lessons', [])),
                'legendary_lessons': len(self.shared_knowledge.get('legendary_lessons', [])),
                'unique_patterns': len(self.shared_knowledge['shared_patterns']),
                'best_strategy': None,
                'best_regime': None
            }
            
            # Find best strategy
            best_strategy_rate = 0
            for strategy, perf in self.shared_knowledge['strategy_success_rates'].items():
                total = perf['wins'] + perf['losses']
                if total > 0:
                    win_rate = perf['wins'] / total
                    if win_rate > best_strategy_rate:
                        best_strategy_rate = win_rate
                        summary['best_strategy'] = (strategy, win_rate)
            
            # Find best regime
            best_regime_profit = -999999
            for regime, perf in self.shared_knowledge['market_regime_performance'].items():
                if perf['trades'] > 0:
                    avg_profit = perf['profit'] / perf['trades']
                    if avg_profit > best_regime_profit:
                        best_regime_profit = avg_profit
                        summary['best_regime'] = (regime, avg_profit)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {}

# Create global instance
cross_bot_learning = CrossBotLearningSystem()
