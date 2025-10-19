"""
🧠 META-LEARNING BRAIN 🧠
Auto-builds new strategies when old ones stop working
Learns how to learn and evolves trading strategies using genetic algorithms
"""

import asyncio
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetaStrategy:
    """Auto-generated strategy by meta-learning brain"""
    name: str
    code: str
    performance_score: float
    creation_time: datetime
    parent_strategies: List[str]
    market_conditions: Dict[str, Any]
    success_rate: float = 0.0
    total_trades: int = 0
    avg_return: float = 0.0

class MetaLearningBrain:
    """Meta-learning brain that learns how to learn and auto-builds strategies"""
    
    def __init__(self):
        self.meta_strategies = {}
        self.strategy_performance = defaultdict(list)
        self.learning_patterns = {}
        self.strategy_dna = {}  # Genetic algorithm for strategy evolution
        self.meta_knowledge_file = "meta_learning_brain.json"
        self.generation_counter = 0
        self.load_meta_knowledge()
        
    def load_meta_knowledge(self):
        """Load meta-learning knowledge from disk"""
        try:
            with open(self.meta_knowledge_file, 'r') as f:
                data = json.load(f)
                self.learning_patterns = data.get('learning_patterns', {})
                self.strategy_dna = data.get('strategy_dna', {})
                self.generation_counter = data.get('generation_counter', 0)
        except FileNotFoundError:
            self.learning_patterns = {}
            self.strategy_dna = {}
            self.generation_counter = 0
    
    def save_meta_knowledge(self):
        """Save meta-learning knowledge to disk"""
        data = {
            'learning_patterns': self.learning_patterns,
            'strategy_dna': self.strategy_dna,
            'generation_counter': self.generation_counter,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.meta_knowledge_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def analyze_strategy_decay(self, strategy_name: str) -> bool:
        """Detect when a strategy stops working"""
        if strategy_name not in self.strategy_performance:
            return False
        
        recent_performance = self.strategy_performance[strategy_name][-20:]  # Last 20 trades
        if len(recent_performance) < 10:
            return False
        
        # Calculate performance trend
        recent_returns = [p['return'] for p in recent_performance]
        trend = np.polyfit(range(len(recent_returns)), recent_returns, 1)[0]
        
        # Strategy is decaying if trend is negative and recent performance is poor
        avg_recent_return = np.mean(recent_returns[-10:])
        
        if trend < -0.001 and avg_recent_return < 0:
            print(f"🚨 META-BRAIN: Strategy decay detected: {strategy_name}")
            print(f"   📉 Trend: {trend:.4f}, Avg Return: {avg_recent_return:.4f}")
            return True
        
        return False
    
    async def auto_build_strategy(self, market_conditions: Dict[str, Any]) -> Optional[MetaStrategy]:
        """Auto-build new strategy using genetic algorithm"""
        print("🧬 META-BRAIN: Auto-building new strategy...")
        
        # Analyze successful patterns from past strategies
        successful_patterns = self._extract_successful_patterns()
        
        if not successful_patterns:
            print("⚠️ No successful patterns found, using base template")
            successful_patterns = self._get_base_patterns()
        
        # Generate strategy DNA by combining successful elements
        new_dna = self._generate_strategy_dna(successful_patterns, market_conditions)
        
        # Build strategy code from DNA
        strategy_code = self._dna_to_strategy_code(new_dna)
        
        # Create new meta-strategy
        self.generation_counter += 1
        strategy_name = f"META_GEN{self.generation_counter}_{datetime.now().strftime('%H%M%S')}"
        
        meta_strategy = MetaStrategy(
            name=strategy_name,
            code=strategy_code,
            performance_score=0.0,
            creation_time=datetime.now(),
            parent_strategies=list(successful_patterns.keys()),
            market_conditions=market_conditions.copy()
        )
        
        self.meta_strategies[strategy_name] = meta_strategy
        self.save_meta_knowledge()
        
        print(f"🎯 New meta-strategy created: {strategy_name}")
        print(f"   🧬 Generation: {self.generation_counter}")
        print(f"   👨‍👩‍👧‍👦 Parents: {len(meta_strategy.parent_strategies)}")
        
        return meta_strategy
    
    def _extract_successful_patterns(self) -> Dict[str, Any]:
        """Extract patterns from successful strategies"""
        patterns = {}
        
        for strategy_name, performance_data in self.strategy_performance.items():
            if len(performance_data) < 5:
                continue
            
            # Calculate success metrics
            returns = [p['return'] for p in performance_data]
            win_rate = len([r for r in returns if r > 0]) / len(returns)
            avg_return = np.mean(returns)
            
            if win_rate > 0.6 and avg_return > 0.01:  # Successful strategy
                patterns[strategy_name] = {
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'conditions': performance_data[-1].get('market_conditions', {}),
                    'indicators': performance_data[-1].get('indicators', ['rsi', 'sma', 'momentum']),
                    'risk_profile': performance_data[-1].get('risk_profile', 'medium')
                }
        
        return patterns
    
    def _get_base_patterns(self) -> Dict[str, Any]:
        """Get base patterns when no successful strategies exist"""
        return {
            'base_momentum': {
                'win_rate': 0.65,
                'avg_return': 0.02,
                'conditions': {'volatility': 0.02, 'trend_strength': 0.7},
                'indicators': ['rsi', 'sma_20', 'momentum'],
                'risk_profile': 'medium'
            },
            'base_mean_reversion': {
                'win_rate': 0.60,
                'avg_return': 0.015,
                'conditions': {'volatility': 0.015, 'trend_strength': 0.3},
                'indicators': ['rsi', 'bollinger_bands', 'stdev'],
                'risk_profile': 'low'
            }
        }
    
    async def analyze_strategy_evolution(self, current_performance: Dict, market_conditions: Dict) -> Dict:
        """Analyze strategy evolution and adaptation"""
        try:
            evolution_analysis = {
                'improvement_score': 1.0,
                'adaptation_needed': False,
                'recommended_changes': [],
                'evolution_confidence': 0.5
            }
            
            # Analyze current performance vs historical
            win_rate = current_performance.get('win_rate', 0.5)
            total_trades = current_performance.get('total_trades', 0)
            
            if total_trades >= 10:
                # Check if strategies are evolving positively
                if win_rate > 0.65:
                    evolution_analysis['improvement_score'] = 1.2
                    evolution_analysis['evolution_confidence'] = 0.8
                elif win_rate < 0.45:
                    evolution_analysis['adaptation_needed'] = True
                    evolution_analysis['recommended_changes'].append('increase_selectivity')
                    evolution_analysis['evolution_confidence'] = 0.6
            
            return evolution_analysis
        except Exception as e:
            logger.error(f"Strategy evolution analysis error: {e}")
            return {'improvement_score': 1.0, 'adaptation_needed': False}
    
    async def generate_new_strategies(self, successful_patterns: List[Dict], market_regime: str) -> List[Dict]:
        """Generate new strategy combinations based on successful patterns"""
        try:
            new_strategies = []
            
            # Generate strategies based on market regime
            if market_regime in ['bull', 'bullish_trend']:
                new_strategies.append({
                    'name': 'META_BULL_MOMENTUM',
                    'description': 'Enhanced momentum strategy for bull markets',
                    'confidence_boost': 0.05,
                    'applies_to_regime': 'bull',
                    'indicators': ['momentum', 'volume', 'rsi']
                })
            elif market_regime in ['bear', 'bearish_trend']:
                new_strategies.append({
                    'name': 'META_BEAR_REVERSION',
                    'description': 'Mean reversion strategy for bear markets',
                    'confidence_boost': 0.03,
                    'applies_to_regime': 'bear',
                    'indicators': ['rsi', 'bollinger', 'support_resistance']
                })
            else:
                new_strategies.append({
                    'name': 'META_SIDEWAYS_SCALP',
                    'description': 'Scalping strategy for sideways markets',
                    'confidence_boost': 0.02,
                    'applies_to_regime': 'sideways',
                    'indicators': ['micro_momentum', 'volatility', 'bid_ask_spread']
                })
            
            return new_strategies
        except Exception as e:
            logger.error(f"New strategy generation error: {e}")
            return []
    
    async def get_meta_insights(self) -> Dict:
        """Get meta-learning insights and recommendations"""
        try:
            insights = {
                'learning_velocity': 'moderate',
                'adaptation_success_rate': 0.7,
                'strategy_diversity': 0.6,
                'meta_recommendations': [
                    'Continue current learning path',
                    'Monitor strategy performance closely'
                ]
            }
            
            # Calculate learning velocity based on recent adaptations
            if self.generation_counter > 5:
                insights['learning_velocity'] = 'high'
                insights['meta_recommendations'].append('Consider reducing strategy generation frequency')
            elif self.generation_counter < 2:
                insights['learning_velocity'] = 'slow'
                insights['meta_recommendations'].append('Increase exploration of new strategies')
            
            return insights
        except Exception as e:
            logger.error(f"Meta insights error: {e}")
            return {'learning_velocity': 'moderate'}
    
    def _generate_strategy_dna(self, successful_patterns: Dict[str, Any], market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new strategy DNA by combining successful elements"""
        dna = {
            'indicators': [],
            'entry_conditions': [],
            'exit_conditions': [],
            'risk_management': {},
            'market_filters': [],
            'confidence_factors': []
        }
        
        # Combine indicators from successful strategies
        all_indicators = []
        for pattern in successful_patterns.values():
            all_indicators.extend(pattern.get('indicators', []))
        
        # Select most common indicators (genetic selection)
        indicator_counts = defaultdict(int)
        for indicator in all_indicators:
            indicator_counts[indicator] += 1
        
        # Take top 5 indicators
        dna['indicators'] = [ind for ind, count in sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        # Generate conditions based on market conditions and successful patterns
        volatility = market_conditions.get('volatility', 0.02)
        trend_strength = market_conditions.get('trend_strength', 0.5)
        
        # Entry conditions (evolved from successful patterns)
        if volatility > 0.025:
            dna['entry_conditions'].append('volatility_breakout')
            dna['confidence_factors'].append('high_volatility_boost')
        
        if trend_strength > 0.7:
            dna['entry_conditions'].append('trend_following')
            dna['confidence_factors'].append('strong_trend_boost')
        elif trend_strength < 0.3:
            dna['entry_conditions'].append('mean_reversion')
            dna['confidence_factors'].append('weak_trend_reversion')
        
        # Exit conditions (learned from successful exits)
        dna['exit_conditions'].extend(['profit_target', 'stop_loss', 'time_decay'])
        
        # Risk management DNA (evolved from best performers)
        avg_win_rate = np.mean([p['win_rate'] for p in successful_patterns.values()])
        avg_return = np.mean([p['avg_return'] for p in successful_patterns.values()])
        
        dna['risk_management'] = {
            'stop_loss': max(0.015, volatility * 1.5),  # Adaptive to volatility
            'take_profit': max(0.03, volatility * 3.0),  # Risk-reward ratio
            'position_size': min(0.15, 0.05 / volatility),  # Inverse volatility sizing
            'max_drawdown': 0.10,
            'confidence_threshold': max(0.6, avg_win_rate * 0.9)
        }
        
        # Market filters (learned environmental conditions)
        dna['market_filters'] = ['volume_filter', 'spread_filter', 'time_filter']
        
        return dna
    
    def _dna_to_strategy_code(self, dna: Dict[str, Any]) -> str:
        """Convert strategy DNA to executable strategy code"""
        
        # Generate indicator calculations
        indicator_code = []
        for indicator in dna['indicators']:
            if indicator == 'rsi':
                indicator_code.append("    rsi = indicators.get('rsi', 50)")
            elif indicator == 'sma_20':
                indicator_code.append("    sma_20 = indicators.get('sma_20', current_price)")
            elif indicator == 'momentum':
                indicator_code.append("    momentum = indicators.get('momentum', 0)")
            elif indicator == 'bollinger_bands':
                indicator_code.append("    bb_upper = indicators.get('bb_upper', current_price * 1.02)")
                indicator_code.append("    bb_lower = indicators.get('bb_lower', current_price * 0.98)")
            elif indicator == 'stdev':
                indicator_code.append("    volatility = indicators.get('volatility', 0.02)")
        
        # Generate entry conditions
        entry_code = []
        for condition in dna['entry_conditions']:
            if condition == 'volatility_breakout':
                entry_code.append("volatility > 0.025 and rsi < 35")
            elif condition == 'trend_following':
                entry_code.append("sma_20 > current_price * 1.001 and momentum > 0.01")
            elif condition == 'mean_reversion':
                entry_code.append("rsi > 70 or (current_price < bb_lower and rsi < 30)")
        
        # Generate exit conditions
        exit_code = []
        for condition in dna['exit_conditions']:
            if condition == 'profit_target':
                exit_code.append(f"unrealized_pnl_pct > {dna['risk_management']['take_profit']:.3f}")
            elif condition == 'stop_loss':
                exit_code.append(f"unrealized_pnl_pct < -{dna['risk_management']['stop_loss']:.3f}")
            elif condition == 'time_decay':
                exit_code.append("position_age > 3600")  # 1 hour max hold
        
        # Generate confidence factors
        confidence_code = []
        for factor in dna['confidence_factors']:
            if factor == 'high_volatility_boost':
                confidence_code.append("confidence *= 1.1 if volatility > 0.03 else 1.0")
            elif factor == 'strong_trend_boost':
                confidence_code.append("confidence *= 1.15 if abs(momentum) > 0.02 else 1.0")
            elif factor == 'weak_trend_reversion':
                confidence_code.append("confidence *= 1.05 if rsi > 65 or rsi < 35 else 1.0")
        
        # Build complete strategy code
        strategy_template = f'''
async def meta_evolved_strategy(market_data, indicators, position_data=None):
    """Auto-generated meta-strategy - Generation {self.generation_counter}"""
    
    current_price = market_data.get('price', 0)
    if current_price <= 0:
        return None
    
    # Technical indicators
{chr(10).join(indicator_code)}
    
    # Base confidence
    confidence = {dna['risk_management']['confidence_threshold']:.2f}
    
    # Market filters
    volume = market_data.get('volume', 0)
    spread = market_data.get('spread', 0.001)
    
    if volume < 1000 or spread > 0.005:  # Volume and spread filters
        return None
    
    # Entry logic
    entry_signal = False
    if {' and '.join(entry_code) if entry_code else 'False'}:
        entry_signal = True
        
        # Confidence adjustments
{chr(10).join(['        ' + code for code in confidence_code])}
    
    # Exit logic (if we have a position)
    exit_signal = False
    if position_data:
        unrealized_pnl_pct = position_data.get('unrealized_pnl_pct', 0)
        position_age = position_data.get('age_seconds', 0)
        
        if {' or '.join(exit_code) if exit_code else 'False'}:
            exit_signal = True
    
    # Risk management
    position_size = min({dna['risk_management']['position_size']:.3f}, 0.1)
    stop_loss = {dna['risk_management']['stop_loss']:.3f}
    take_profit = {dna['risk_management']['take_profit']:.3f}
    
    if entry_signal and confidence >= {dna['risk_management']['confidence_threshold']:.2f}:
        return {{
            'action': 'BUY',
            'confidence': confidence,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy_name': 'META_EVOLVED',
            'reasoning': 'Meta-learned strategy with evolved DNA'
        }}
    elif exit_signal:
        return {{
            'action': 'SELL',
            'confidence': confidence * 0.9,
            'reasoning': 'Meta-learned exit condition triggered'
        }}
    
    return None
'''
        
        return strategy_template
    
    def update_strategy_performance(self, strategy_name: str, trade_result: Dict[str, Any]):
        """Update performance tracking for meta-strategies"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
        
        performance_record = {
            'return': trade_result.get('pnl_pct', 0),
            'timestamp': datetime.now().isoformat(),
            'market_conditions': trade_result.get('market_conditions', {}),
            'indicators': trade_result.get('indicators', []),
            'confidence': trade_result.get('confidence', 0.5),
            'risk_profile': trade_result.get('risk_profile', 'medium')
        }
        
        self.strategy_performance[strategy_name].append(performance_record)
        
        # Keep only recent performance (last 100 trades)
        if len(self.strategy_performance[strategy_name]) > 100:
            self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-100:]
        
        # Update meta-strategy stats if it exists
        if strategy_name in self.meta_strategies:
            strategy = self.meta_strategies[strategy_name]
            strategy.total_trades += 1
            
            # Update success rate
            recent_returns = [p['return'] for p in self.strategy_performance[strategy_name][-20:]]
            strategy.success_rate = len([r for r in recent_returns if r > 0]) / len(recent_returns)
            strategy.avg_return = np.mean(recent_returns)
            strategy.performance_score = strategy.success_rate * strategy.avg_return * 100
        
        self.save_meta_knowledge()
    
    def get_best_meta_strategies(self, top_n: int = 3) -> List[MetaStrategy]:
        """Get best performing meta-strategies"""
        strategies = list(self.meta_strategies.values())
        strategies.sort(key=lambda s: s.performance_score, reverse=True)
        return strategies[:top_n]
    
    async def evolve_strategies(self) -> List[MetaStrategy]:
        """Evolve existing strategies through genetic algorithm"""
        print("🧬 META-BRAIN: Evolving strategies through genetic algorithm...")
        
        best_strategies = self.get_best_meta_strategies(top_n=5)
        if len(best_strategies) < 2:
            return []
        
        evolved_strategies = []
        
        # Cross-breed top strategies
        for i in range(len(best_strategies)):
            for j in range(i + 1, len(best_strategies)):
                parent1 = best_strategies[i]
                parent2 = best_strategies[j]
                
                # Create hybrid market conditions
                hybrid_conditions = {}
                for key in parent1.market_conditions:
                    if key in parent2.market_conditions:
                        hybrid_conditions[key] = (parent1.market_conditions[key] + parent2.market_conditions[key]) / 2
                
                # Create evolved strategy
                evolved = await self.auto_build_strategy(hybrid_conditions)
                if evolved:
                    evolved.parent_strategies = [parent1.name, parent2.name]
                    evolved_strategies.append(evolved)
        
        print(f"🎯 Evolved {len(evolved_strategies)} new strategies")
        return evolved_strategies
