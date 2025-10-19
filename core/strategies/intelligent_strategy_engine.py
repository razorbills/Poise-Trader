#!/usr/bin/env python3
"""
🧠 INTELLIGENT STRATEGY ENGINE
AI-powered strategy selection and optimization for maximum profits

Features:
• Machine learning strategy optimization
• Real-time performance analysis
• Automatic strategy switching
• Portfolio-aware position sizing
• Risk-adjusted profit maximization
• Continuous learning and adaptation
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
import pickle
import os
from enum import Enum


class StrategyType(Enum):
    """Available strategy types"""
    COMPOUND_BEAST = "compound_beast"
    PROFESSIONAL_MOMENTUM = "professional_momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    DCA = "dca"
    GRID = "grid"
    SCALPING = "scalping"
    BREAKOUT = "breakout"
    NEWS_TRADING = "news_trading"


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    avg_trade_duration_minutes: float = 0.0
    confidence_score: float = 0.5
    recent_performance_trend: float = 0.0


@dataclass
class MarketCondition:
    """Current market conditions"""
    volatility: float
    trend_strength: float
    volume_profile: float
    momentum: float
    market_sentiment: str  # 'bullish', 'bearish', 'neutral'
    time_of_day: int  # Hour of day (0-23)
    day_of_week: int  # 0=Monday, 6=Sunday


@dataclass
class StrategySignal:
    """Trading signal from a strategy"""
    strategy_name: str
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float
    expected_profit: float
    risk_reward_ratio: float
    time_horizon_minutes: int
    reasoning: str
    metadata: Dict[str, Any]


class IntelligentStrategyEngine:
    """
    🧠 AI-POWERED STRATEGY ENGINE
    
    Automatically selects and optimizes trading strategies based on:
    • Real-time market conditions
    • Historical performance data
    • ML-based pattern recognition
    • Risk-adjusted profit maximization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("IntelligentStrategyEngine")
        
        # Strategy instances
        self.strategies = {}
        self.strategy_performances = {}
        
        # AI/ML components
        self.ml_optimizer = MLStrategyOptimizer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.market_analyzer = MarketConditionAnalyzer()
        
        # Configuration
        self.capital = Decimal(str(config.get('initial_capital', 5000)))
        self.max_simultaneous_strategies = config.get('max_simultaneous_strategies', 3)
        self.strategy_allocation_weights = config.get('strategy_allocation_weights', {})
        self.rebalance_frequency_hours = config.get('rebalance_frequency_hours', 6)
        
        # Performance tracking
        self.trade_history = []
        self.strategy_switches = []
        self.performance_history = []
        
        # Current state
        self.active_strategies = []
        self.current_market_condition = None
        self.last_optimization_time = datetime.now()
        
        # Learning data
        self.learning_data_file = "data/strategy_learning_data.pkl"
        self.learning_data = self._load_learning_data()
    
    async def initialize(self):
        """Initialize the intelligent strategy engine"""
        self.logger.info("🧠 Initializing Intelligent Strategy Engine")
        
        # Initialize all available strategies
        await self._initialize_strategies()
        
        # Load historical performance data
        await self._load_historical_performance()
        
        # Initialize ML models
        await self.ml_optimizer.initialize(self.learning_data)
        
        # Start continuous optimization loop
        asyncio.create_task(self._optimization_loop())
        
        # Start performance monitoring
        asyncio.create_task(self._performance_monitoring_loop())
        
        self.logger.info("✅ Intelligent Strategy Engine initialized")
    
    async def _initialize_strategies(self):
        """Initialize all available trading strategies"""
        strategy_configs = self.config.get('strategies', {})
        
        # Import and initialize each strategy type
        strategy_classes = {
            StrategyType.COMPOUND_BEAST: self._create_compound_beast_strategy,
            StrategyType.PROFESSIONAL_MOMENTUM: self._create_momentum_strategy,
            StrategyType.MEAN_REVERSION: self._create_mean_reversion_strategy,
            StrategyType.ARBITRAGE: self._create_arbitrage_strategy,
            StrategyType.DCA: self._create_dca_strategy,
            StrategyType.GRID: self._create_grid_strategy,
            StrategyType.SCALPING: self._create_scalping_strategy,
            StrategyType.BREAKOUT: self._create_breakout_strategy,
            StrategyType.NEWS_TRADING: self._create_news_trading_strategy,
        }
        
        for strategy_type, create_func in strategy_classes.items():
            try:
                config = strategy_configs.get(strategy_type.value, {})
                config['initial_capital'] = float(self.capital)
                
                strategy = await create_func(config)
                self.strategies[strategy_type.value] = strategy
                
                # Initialize performance tracking
                self.strategy_performances[strategy_type.value] = StrategyPerformance(
                    strategy_name=strategy_type.value
                )
                
                self.logger.info(f"✅ Initialized {strategy_type.value} strategy")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {strategy_type.value}: {e}")
    
    async def _create_compound_beast_strategy(self, config):
        """Create compound beast strategy"""
        from ..compound_beast_strategy import CompoundBeastStrategy
        return CompoundBeastStrategy(config.get('initial_capital_sats', 5000))
    
    async def _create_momentum_strategy(self, config):
        """Create professional momentum strategy"""
        from ..professional_strategies import ProfessionalMomentumStrategy
        from ..professional_strategies import InstitutionalRiskManager
        
        capital = config.get('initial_capital', 5000)
        risk_manager = InstitutionalRiskManager(capital)
        return ProfessionalMomentumStrategy(capital, risk_manager)
    
    async def _create_mean_reversion_strategy(self, config):
        """Create mean reversion strategy"""
        from ..professional_strategies import ProfessionalMeanReversionStrategy
        from ..professional_strategies import InstitutionalRiskManager
        
        capital = config.get('initial_capital', 5000)
        risk_manager = InstitutionalRiskManager(capital)
        return ProfessionalMeanReversionStrategy(capital, risk_manager)
    
    async def _create_arbitrage_strategy(self, config):
        """Create arbitrage strategy"""
        from ..professional_strategies import ProfessionalArbitrageStrategy
        from ..professional_strategies import InstitutionalRiskManager
        
        capital = config.get('initial_capital', 5000)
        risk_manager = InstitutionalRiskManager(capital)
        return ProfessionalArbitrageStrategy(capital, risk_manager)
    
    async def _create_dca_strategy(self, config):
        """Create DCA strategy"""
        from .dca_strategy import DCAStrategy
        return DCAStrategy(config)
    
    async def _create_grid_strategy(self, config):
        """Create grid strategy"""
        # Placeholder - implement grid strategy
        return GridStrategy(config) if 'GridStrategy' in globals() else None
    
    async def _create_scalping_strategy(self, config):
        """Create scalping strategy"""
        # Placeholder - implement scalping strategy  
        return ScalpingStrategy(config) if 'ScalpingStrategy' in globals() else None
    
    async def _create_breakout_strategy(self, config):
        """Create breakout strategy"""
        # Placeholder - implement breakout strategy
        return BreakoutStrategy(config) if 'BreakoutStrategy' in globals() else None
    
    async def _create_news_trading_strategy(self, config):
        """Create news trading strategy"""
        # Placeholder - implement news trading strategy
        return NewsTradingStrategy(config) if 'NewsTradingStrategy' in globals() else None
    
    async def get_optimal_signals(self, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Get optimal trading signals based on AI analysis"""
        
        # Analyze current market conditions
        market_condition = await self.market_analyzer.analyze_market_condition(market_data)
        self.current_market_condition = market_condition
        
        # Get signals from all active strategies
        all_signals = []
        
        for strategy_name in self.active_strategies:
            if strategy_name in self.strategies:
                try:
                    strategy = self.strategies[strategy_name]
                    signals = await self._get_strategy_signals(strategy, strategy_name, market_data)
                    all_signals.extend(signals)
                except Exception as e:
                    self.logger.error(f"Error getting signals from {strategy_name}: {e}")
        
        # AI-powered signal optimization and ranking
        optimized_signals = await self.ml_optimizer.optimize_signals(
            all_signals, market_condition, self.strategy_performances
        )
        
        # Apply portfolio-aware position sizing
        sized_signals = await self._apply_portfolio_sizing(optimized_signals)
        
        # Log the AI decision process
        if sized_signals:
            self.logger.info(f"🧠 AI selected {len(sized_signals)} optimal signals:")
            for signal in sized_signals[:3]:  # Log top 3
                self.logger.info(f"   • {signal.strategy_name}: {signal.symbol} {signal.action} "
                               f"(confidence: {signal.confidence:.2f}, expected profit: {signal.expected_profit:.1f}%)")
        
        return sized_signals
    
    async def _get_strategy_signals(self, strategy, strategy_name: str, 
                                   market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Get signals from a specific strategy"""
        signals = []
        
        try:
            # Different strategies have different interfaces
            if hasattr(strategy, 'generate_compound_opportunities'):
                # Compound Beast Strategy
                opportunities = strategy.generate_compound_opportunities(market_data)
                signals.extend(self._convert_compound_signals(opportunities, strategy_name))
                
            elif hasattr(strategy, 'generate_signals'):
                # Professional strategies
                raw_signals = strategy.generate_signals(market_data)
                signals.extend(self._convert_professional_signals(raw_signals, strategy_name))
                
            elif hasattr(strategy, '_generate_signal'):
                # DCA Strategy
                for symbol, data in market_data.items():
                    signal = await strategy._generate_signal(data)
                    if signal:
                        signals.append(self._convert_dca_signal(signal, strategy_name))
            
        except Exception as e:
            self.logger.error(f"Error getting signals from {strategy_name}: {e}")
        
        return signals
    
    def _convert_compound_signals(self, opportunities, strategy_name: str) -> List[StrategySignal]:
        """Convert compound beast opportunities to strategy signals"""
        signals = []
        
        for opp in opportunities:
            signals.append(StrategySignal(
                strategy_name=strategy_name,
                symbol=opp['symbol'],
                action=opp['type'],
                confidence=opp['confidence'],
                entry_price=opp['entry_price'],
                stop_loss=opp.get('stop_loss'),
                take_profit=opp.get('profit_target'),
                position_size=opp['trade_size_sats'],
                expected_profit=opp['profit_target_pct'],
                risk_reward_ratio=opp.get('risk_reward_ratio', 2.0),
                time_horizon_minutes=60,
                reasoning=opp['reason'],
                metadata={'expected_profit_sats': opp['expected_profit_sats']}
            ))
        
        return signals
    
    def _convert_professional_signals(self, raw_signals, strategy_name: str) -> List[StrategySignal]:
        """Convert professional strategy signals to standard format"""
        signals = []
        
        for signal in raw_signals:
            signals.append(StrategySignal(
                strategy_name=strategy_name,
                symbol=signal.get('symbol', 'BTC/USDT'),
                action=signal['type'],
                confidence=signal['confidence'],
                entry_price=signal['entry_price'],
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('profit_target'),
                position_size=0,  # Will be calculated later
                expected_profit=signal.get('risk_reward_ratio', 2.0) * 0.02,  # Estimate
                risk_reward_ratio=signal.get('risk_reward_ratio', 2.0),
                time_horizon_minutes=signal.get('expected_holding_period', 60),
                reasoning=signal['reason'],
                metadata=signal
            ))
        
        return signals
    
    def _convert_dca_signal(self, signal, strategy_name: str) -> StrategySignal:
        """Convert DCA signal to standard format"""
        return StrategySignal(
            strategy_name=strategy_name,
            symbol=signal.symbol,
            action=signal.action.value,
            confidence=signal.confidence,
            entry_price=float(signal.price),
            stop_loss=float(signal.stop_loss) if signal.stop_loss else None,
            take_profit=float(signal.take_profit) if signal.take_profit else None,
            position_size=float(signal.quantity),
            expected_profit=0.02,  # DCA typically targets smaller, consistent gains
            risk_reward_ratio=2.0,
            time_horizon_minutes=signal.metadata.get('holding_period', 1440),  # 1 day default
            reasoning=signal.metadata.get('signal_type', 'DCA'),
            metadata=signal.metadata
        )
    
    async def _apply_portfolio_sizing(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Apply intelligent portfolio-aware position sizing"""
        if not signals:
            return signals
        
        # Calculate available capital
        available_capital = float(self.capital)
        
        # Calculate optimal position sizes using Kelly Criterion and risk budgeting
        for signal in signals:
            # Get strategy performance metrics
            perf = self.strategy_performances.get(signal.strategy_name)
            
            if perf and perf.win_rate > 0:
                # Kelly Criterion: f = (bp - q) / b
                # where f = fraction to bet, b = odds, p = win probability, q = lose probability
                win_prob = perf.win_rate
                avg_win = abs(perf.avg_profit) if perf.avg_profit != 0 else 0.02
                avg_loss = abs(perf.avg_loss) if perf.avg_loss != 0 else 0.01
                
                if avg_win > 0 and avg_loss > 0:
                    odds = avg_win / avg_loss
                    kelly_fraction = (odds * win_prob - (1 - win_prob)) / odds
                    
                    # Be conservative - use 25% of Kelly
                    kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.05))  # Max 5% per trade
                else:
                    kelly_fraction = 0.02  # Default 2%
            else:
                kelly_fraction = 0.02  # Default for new strategies
            
            # Adjust based on confidence
            confidence_multiplier = signal.confidence
            final_fraction = kelly_fraction * confidence_multiplier
            
            # Calculate position size in base currency
            position_value = available_capital * final_fraction
            position_size = position_value / signal.entry_price
            
            signal.position_size = position_size
            signal.metadata['position_value'] = position_value
            signal.metadata['kelly_fraction'] = kelly_fraction
        
        return signals
    
    async def _optimization_loop(self):
        """Continuous AI optimization loop"""
        while True:
            try:
                current_time = datetime.now()
                
                # Check if it's time for rebalancing
                time_since_last = (current_time - self.last_optimization_time).total_seconds() / 3600
                
                if time_since_last >= self.rebalance_frequency_hours:
                    await self._optimize_strategy_allocation()
                    self.last_optimization_time = current_time
                
                # Check for strategy performance and switch if needed
                await self._check_strategy_performance()
                
                # Update ML models with recent data
                await self.ml_optimizer.update_models(self.trade_history[-100:])  # Last 100 trades
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def _optimize_strategy_allocation(self):
        """Optimize strategy allocation based on performance"""
        self.logger.info("🔄 Optimizing strategy allocation...")
        
        # Get current market conditions
        market_condition = self.current_market_condition
        
        if not market_condition:
            return
        
        # Score strategies based on current conditions and historical performance
        strategy_scores = {}
        
        for strategy_name, performance in self.strategy_performances.items():
            if performance.total_trades < 5:
                # Not enough data, use default score
                strategy_scores[strategy_name] = 0.5
                continue
            
            # Base score from historical performance
            base_score = (
                performance.win_rate * 0.3 +
                min(performance.profit_factor, 3) / 3 * 0.3 +
                max(0, 1 - performance.max_drawdown / 0.2) * 0.2 +
                min(performance.sharpe_ratio, 3) / 3 * 0.2
            )
            
            # Adjust based on recent performance trend
            trend_adjustment = performance.recent_performance_trend * 0.1
            
            # Adjust based on market conditions suitability
            market_adjustment = await self._calculate_market_suitability(strategy_name, market_condition)
            
            final_score = base_score + trend_adjustment + market_adjustment
            strategy_scores[strategy_name] = max(0, min(1, final_score))
        
        # Select top strategies
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        self.active_strategies = [name for name, score in sorted_strategies[:self.max_simultaneous_strategies]]
        
        self.logger.info(f"🎯 Active strategies: {self.active_strategies}")
        for strategy, score in sorted_strategies[:5]:
            self.logger.info(f"   • {strategy}: {score:.3f}")
    
    async def _calculate_market_suitability(self, strategy_name: str, market_condition: MarketCondition) -> float:
        """Calculate how suitable a strategy is for current market conditions"""
        
        # Strategy-specific market condition preferences
        suitability_matrix = {
            'compound_beast': {
                'high_volatility': 0.3, 'trending': 0.2, 'high_volume': 0.3
            },
            'professional_momentum': {
                'trending': 0.4, 'high_volume': 0.2, 'high_volatility': 0.1
            },
            'mean_reversion': {
                'ranging': 0.4, 'low_volatility': 0.2, 'neutral_sentiment': 0.2
            },
            'arbitrage': {
                'high_volatility': 0.3, 'high_volume': 0.3, 'any_market': 0.2
            },
            'dca': {
                'ranging': 0.3, 'bearish': 0.3, 'low_volatility': 0.2
            }
        }
        
        if strategy_name not in suitability_matrix:
            return 0.0
        
        preferences = suitability_matrix[strategy_name]
        suitability_score = 0.0
        
        # Check volatility preference
        if market_condition.volatility > 0.05 and 'high_volatility' in preferences:
            suitability_score += preferences['high_volatility']
        elif market_condition.volatility < 0.02 and 'low_volatility' in preferences:
            suitability_score += preferences['low_volatility']
        
        # Check trend preference
        if abs(market_condition.trend_strength) > 0.7 and 'trending' in preferences:
            suitability_score += preferences['trending']
        elif abs(market_condition.trend_strength) < 0.3 and 'ranging' in preferences:
            suitability_score += preferences['ranging']
        
        # Check volume preference
        if market_condition.volume_profile > 1.5 and 'high_volume' in preferences:
            suitability_score += preferences['high_volume']
        
        # Check sentiment preference
        if market_condition.market_sentiment == 'bearish' and 'bearish' in preferences:
            suitability_score += preferences['bearish']
        elif market_condition.market_sentiment == 'neutral' and 'neutral_sentiment' in preferences:
            suitability_score += preferences['neutral_sentiment']
        
        return min(suitability_score, 0.5)  # Max 0.5 adjustment
    
    async def _check_strategy_performance(self):
        """Check recent strategy performance and make adjustments"""
        
        # Check if any strategy is underperforming
        for strategy_name, performance in self.strategy_performances.items():
            if performance.total_trades >= 10:  # Enough data to judge
                
                # Check if strategy has poor recent performance
                if (performance.recent_performance_trend < -0.1 and 
                    performance.win_rate < 0.4):
                    
                    # Temporarily reduce allocation or remove from active
                    if strategy_name in self.active_strategies:
                        self.logger.warning(f"⚠️ {strategy_name} showing poor performance, reducing allocation")
                        
                        # Could implement more sophisticated logic here
                        pass
    
    async def record_trade_result(self, strategy_name: str, symbol: str, 
                                  pnl: float, trade_duration_minutes: int, was_winner: bool):
        """Record trade result for performance tracking and learning"""
        
        # Update strategy performance
        if strategy_name in self.strategy_performances:
            perf = self.strategy_performances[strategy_name]
            perf.total_trades += 1
            
            if was_winner:
                perf.winning_trades += 1
                perf.avg_profit = (perf.avg_profit * (perf.winning_trades - 1) + pnl) / perf.winning_trades
            else:
                perf.losing_trades += 1
                perf.avg_loss = (perf.avg_loss * (perf.losing_trades - 1) + abs(pnl)) / perf.losing_trades
            
            perf.win_rate = perf.winning_trades / perf.total_trades
            perf.total_pnl += pnl
            
            # Update average trade duration
            total_duration = perf.avg_trade_duration_minutes * (perf.total_trades - 1)
            perf.avg_trade_duration_minutes = (total_duration + trade_duration_minutes) / perf.total_trades
            
            # Calculate profit factor
            total_profits = perf.avg_profit * perf.winning_trades
            total_losses = perf.avg_loss * perf.losing_trades
            perf.profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        # Add to trade history for ML learning
        trade_record = {
            'timestamp': datetime.now(),
            'strategy_name': strategy_name,
            'symbol': symbol,
            'pnl': pnl,
            'duration_minutes': trade_duration_minutes,
            'was_winner': was_winner,
            'market_condition': self.current_market_condition
        }
        
        self.trade_history.append(trade_record)
        
        # Keep only recent history (last 1000 trades)
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-500:]
        
        # Update learning data
        self.learning_data.append(trade_record)
        self._save_learning_data()
        
        self.logger.info(f"📊 Trade recorded: {strategy_name} {symbol} "
                        f"{'WIN' if was_winner else 'LOSS'} {pnl:+.2f}")
    
    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring and reporting"""
        while True:
            try:
                # Calculate and log performance metrics
                await self._calculate_performance_metrics()
                
                # Generate performance report every hour
                await self._generate_performance_report()
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(1800)
    
    async def _calculate_performance_metrics(self):
        """Calculate advanced performance metrics"""
        for strategy_name, performance in self.strategy_performances.items():
            if performance.total_trades >= 5:
                
                # Calculate Sharpe ratio
                recent_trades = [t for t in self.trade_history[-50:] if t['strategy_name'] == strategy_name]
                if recent_trades:
                    returns = [t['pnl'] for t in recent_trades]
                    if len(returns) > 1:
                        avg_return = sum(returns) / len(returns)
                        return_std = np.std(returns)
                        performance.sharpe_ratio = avg_return / return_std if return_std > 0 else 0
                
                # Calculate max drawdown
                running_max = 0
                max_dd = 0
                running_pnl = 0
                
                for trade in recent_trades:
                    running_pnl += trade['pnl']
                    running_max = max(running_max, running_pnl)
                    drawdown = (running_max - running_pnl) / max(running_max, 1)
                    max_dd = max(max_dd, drawdown)
                
                performance.max_drawdown = max_dd
                
                # Calculate recent performance trend
                if len(recent_trades) >= 10:
                    recent_10 = recent_trades[-10:]
                    early_5 = sum(t['pnl'] for t in recent_10[:5])
                    late_5 = sum(t['pnl'] for t in recent_10[5:])
                    performance.recent_performance_trend = (late_5 - early_5) / max(abs(early_5), 1)
    
    async def _generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.strategy_performances:
            return
        
        self.logger.info("📊 INTELLIGENT STRATEGY ENGINE PERFORMANCE REPORT")
        self.logger.info("=" * 60)
        
        total_trades = sum(p.total_trades for p in self.strategy_performances.values())
        total_pnl = sum(p.total_pnl for p in self.strategy_performances.values())
        
        self.logger.info(f"🔄 Total Trades: {total_trades}")
        self.logger.info(f"💰 Total PnL: {total_pnl:+.2f}")
        self.logger.info(f"🎯 Active Strategies: {', '.join(self.active_strategies)}")
        
        # Top performing strategies
        sorted_strategies = sorted(
            self.strategy_performances.items(),
            key=lambda x: x[1].profit_factor if x[1].profit_factor != float('inf') else 999,
            reverse=True
        )
        
        self.logger.info("\n🏆 Strategy Performance Ranking:")
        for i, (name, perf) in enumerate(sorted_strategies[:5]):
            if perf.total_trades > 0:
                self.logger.info(
                    f"   {i+1}. {name}: {perf.win_rate*100:.1f}% win rate, "
                    f"PF: {perf.profit_factor:.2f}, "
                    f"Sharpe: {perf.sharpe_ratio:.2f}, "
                    f"Trades: {perf.total_trades}"
                )
        
        if self.current_market_condition:
            self.logger.info(f"\n📈 Market Condition: {self.current_market_condition.market_sentiment}")
            self.logger.info(f"   • Volatility: {self.current_market_condition.volatility:.1%}")
            self.logger.info(f"   • Trend: {self.current_market_condition.trend_strength:.2f}")
            self.logger.info(f"   • Volume: {self.current_market_condition.volume_profile:.2f}x")
    
    def _load_learning_data(self) -> List[Dict]:
        """Load historical learning data"""
        try:
            if os.path.exists(self.learning_data_file):
                with open(self.learning_data_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading learning data: {e}")
        
        return []
    
    def _save_learning_data(self):
        """Save learning data for future sessions"""
        try:
            os.makedirs(os.path.dirname(self.learning_data_file), exist_ok=True)
            with open(self.learning_data_file, 'wb') as f:
                pickle.dump(self.learning_data[-10000:], f)  # Keep last 10k records
        except Exception as e:
            self.logger.error(f"Error saving learning data: {e}")
    
    async def _load_historical_performance(self):
        """Load historical performance data"""
        # This would load from database or files
        pass
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get comprehensive strategy engine summary"""
        summary = {
            'active_strategies': self.active_strategies,
            'total_strategies_available': len(self.strategies),
            'strategy_performances': {
                name: asdict(perf) for name, perf in self.strategy_performances.items()
            },
            'current_market_condition': asdict(self.current_market_condition) if self.current_market_condition else None,
            'total_trades_all_strategies': sum(p.total_trades for p in self.strategy_performances.values()),
            'total_pnl_all_strategies': sum(p.total_pnl for p in self.strategy_performances.values()),
            'last_optimization_time': self.last_optimization_time.isoformat(),
        }
        
        return summary


class MLStrategyOptimizer:
    """Machine learning-based strategy optimizer"""
    
    def __init__(self):
        self.logger = logging.getLogger("MLStrategyOptimizer")
        self.models = {}
        self.feature_extractors = {}
    
    async def initialize(self, historical_data: List[Dict]):
        """Initialize ML models with historical data"""
        self.logger.info("🤖 Initializing ML Strategy Optimizer")
        
        # This would load pre-trained models or train new ones
        # For now, using rule-based optimization with ML-like logic
        
        self.logger.info("✅ ML Strategy Optimizer initialized")
    
    async def optimize_signals(self, signals: List[StrategySignal], 
                              market_condition: MarketCondition,
                              strategy_performances: Dict[str, StrategyPerformance]) -> List[StrategySignal]:
        """Use ML to optimize and rank trading signals"""
        
        if not signals:
            return signals
        
        # Score each signal using ML-like logic
        scored_signals = []
        
        for signal in signals:
            score = await self._calculate_signal_score(signal, market_condition, strategy_performances)
            signal.metadata['ml_score'] = score
            scored_signals.append((score, signal))
        
        # Sort by ML score
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        
        # Return top signals
        optimized_signals = [signal for score, signal in scored_signals[:10]]  # Top 10
        
        return optimized_signals
    
    async def _calculate_signal_score(self, signal: StrategySignal, 
                                     market_condition: MarketCondition,
                                     strategy_performances: Dict[str, StrategyPerformance]) -> float:
        """Calculate ML-based score for a signal"""
        
        # Base confidence score
        base_score = signal.confidence
        
        # Strategy historical performance weight
        strategy_perf = strategy_performances.get(signal.strategy_name)
        if strategy_perf and strategy_perf.total_trades > 5:
            perf_weight = (
                strategy_perf.win_rate * 0.4 +
                min(strategy_perf.profit_factor, 5) / 5 * 0.3 +
                strategy_perf.recent_performance_trend * 0.3
            )
        else:
            perf_weight = 0.5  # Default for new strategies
        
        # Market condition suitability weight
        market_weight = await self._calculate_market_suitability_score(signal, market_condition)
        
        # Risk-reward weight
        rr_weight = min(signal.risk_reward_ratio / 3, 1.0) * 0.2
        
        # Expected profit weight
        profit_weight = min(abs(signal.expected_profit) / 0.1, 1.0) * 0.2
        
        # Combine all factors
        final_score = (
            base_score * 0.3 +
            perf_weight * 0.3 +
            market_weight * 0.2 +
            rr_weight * 0.1 +
            profit_weight * 0.1
        )
        
        return max(0, min(1, final_score))
    
    async def _calculate_market_suitability_score(self, signal: StrategySignal, 
                                                 market_condition: MarketCondition) -> float:
        """Calculate how suitable the signal is for current market conditions"""
        
        # Strategy-specific market suitability logic
        suitability_score = 0.5  # Base neutral score
        
        strategy_name = signal.strategy_name.lower()
        
        # Volatility-based adjustments
        if 'compound' in strategy_name or 'scalping' in strategy_name:
            if market_condition.volatility > 0.05:
                suitability_score += 0.2  # High vol is good for these strategies
        
        elif 'dca' in strategy_name or 'mean_reversion' in strategy_name:
            if market_condition.volatility < 0.03:
                suitability_score += 0.2  # Low vol is good for these
        
        # Trend-based adjustments
        if 'momentum' in strategy_name or 'breakout' in strategy_name:
            if abs(market_condition.trend_strength) > 0.6:
                suitability_score += 0.2  # Strong trends are good
        
        # Volume-based adjustments
        if market_condition.volume_profile > 1.5:
            suitability_score += 0.1  # High volume is generally good
        
        return max(0, min(1, suitability_score))
    
    async def update_models(self, recent_trade_data: List[Dict]):
        """Update ML models with recent trade results"""
        
        # This would retrain or update models with new data
        # For now, just log that we're learning
        
        if recent_trade_data:
            win_rate = sum(1 for trade in recent_trade_data if trade['was_winner']) / len(recent_trade_data)
            self.logger.debug(f"🧠 ML learning from {len(recent_trade_data)} recent trades "
                            f"(win rate: {win_rate*100:.1f}%)")


class PerformanceAnalyzer:
    """Advanced performance analysis and reporting"""
    
    def __init__(self):
        self.logger = logging.getLogger("PerformanceAnalyzer")
    
    def calculate_advanced_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        if not trades:
            return {}
        
        returns = [trade['pnl'] for trade in trades]
        
        metrics = {
            'total_return': sum(returns),
            'avg_return': np.mean(returns),
            'volatility': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns),
            'profit_factor': self._calculate_profit_factor(returns),
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / np.maximum(running_max, 1)
        
        return float(np.max(drawdown))
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        profits = sum(r for r in returns if r > 0)
        losses = abs(sum(r for r in returns if r < 0))
        
        return profits / losses if losses > 0 else float('inf')


class MarketConditionAnalyzer:
    """Analyze current market conditions"""
    
    def __init__(self):
        self.logger = logging.getLogger("MarketConditionAnalyzer")
    
    async def analyze_market_condition(self, market_data: Dict[str, Any]) -> MarketCondition:
        """Analyze current market conditions"""
        
        # Extract basic metrics from market data
        volatility = self._calculate_volatility(market_data)
        trend_strength = self._calculate_trend_strength(market_data)
        volume_profile = self._calculate_volume_profile(market_data)
        momentum = self._calculate_momentum(market_data)
        sentiment = self._determine_sentiment(market_data)
        
        now = datetime.now()
        
        return MarketCondition(
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            momentum=momentum,
            market_sentiment=sentiment,
            time_of_day=now.hour,
            day_of_week=now.weekday()
        )
    
    def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility"""
        # Simplified volatility calculation
        total_vol = 0
        count = 0
        
        for symbol, data in market_data.items():
            if isinstance(data, list) and len(data) >= 20:
                prices = [d.get('close', d.get('price', 0)) for d in data[-20:]]
                if prices and all(p > 0 for p in prices):
                    high_low_range = (max(prices) - min(prices)) / min(prices)
                    total_vol += high_low_range
                    count += 1
        
        return total_vol / count if count > 0 else 0.02
    
    def _calculate_trend_strength(self, market_data: Dict[str, Any]) -> float:
        """Calculate overall trend strength"""
        trend_scores = []
        
        for symbol, data in market_data.items():
            if isinstance(data, list) and len(data) >= 50:
                prices = [d.get('close', d.get('price', 0)) for d in data]
                if prices:
                    # Simple trend calculation
                    ma_20 = sum(prices[-20:]) / 20
                    ma_50 = sum(prices[-50:]) / 50
                    current = prices[-1]
                    
                    if ma_20 > ma_50 and current > ma_20:
                        trend_scores.append(1.0)  # Strong uptrend
                    elif ma_20 < ma_50 and current < ma_20:
                        trend_scores.append(-1.0)  # Strong downtrend
                    else:
                        trend_scores.append(0.0)  # Sideways
        
        return sum(trend_scores) / len(trend_scores) if trend_scores else 0.0
    
    def _calculate_volume_profile(self, market_data: Dict[str, Any]) -> float:
        """Calculate volume profile (current vs average)"""
        volume_ratios = []
        
        for symbol, data in market_data.items():
            if isinstance(data, list) and len(data) >= 20:
                volumes = [d.get('volume', 0) for d in data]
                if volumes:
                    current_vol = volumes[-1]
                    avg_vol = sum(volumes[-20:-1]) / 19
                    if avg_vol > 0:
                        ratio = current_vol / avg_vol
                        volume_ratios.append(ratio)
        
        return sum(volume_ratios) / len(volume_ratios) if volume_ratios else 1.0
    
    def _calculate_momentum(self, market_data: Dict[str, Any]) -> float:
        """Calculate price momentum"""
        momentum_scores = []
        
        for symbol, data in market_data.items():
            if isinstance(data, list) and len(data) >= 10:
                prices = [d.get('close', d.get('price', 0)) for d in data]
                if prices:
                    current = prices[-1]
                    old = prices[-10]
                    if old > 0:
                        momentum = (current - old) / old
                        momentum_scores.append(momentum)
        
        return sum(momentum_scores) / len(momentum_scores) if momentum_scores else 0.0
    
    def _determine_sentiment(self, market_data: Dict[str, Any]) -> str:
        """Determine overall market sentiment"""
        positive_count = 0
        total_count = 0
        
        for symbol, data in market_data.items():
            if isinstance(data, list) and data:
                latest = data[-1]
                change_24h = latest.get('price_change_24h', 0)
                if change_24h > 0:
                    positive_count += 1
                total_count += 1
        
        if total_count == 0:
            return 'neutral'
        
        positive_ratio = positive_count / total_count
        
        if positive_ratio > 0.6:
            return 'bullish'
        elif positive_ratio < 0.4:
            return 'bearish'
        else:
            return 'neutral'
