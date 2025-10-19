#!/usr/bin/env python3
"""
🚀 ENHANCED AI PROFIT BOT
Ultimate AI trading system with all advanced features:
- Multi-strategy brain
- Regime detection  
- Sentiment analysis
- Self-healing watchdog
- On-chain intelligence
- Explainable AI
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
import random

# Import our AI components
from ai_trading_engine import (
    AdvancedTechnicalAnalyzer, 
    SentimentAnalyzer, 
    AIStrategyEngine,
    AITradingSignal
)
from live_paper_trading_test import LiveMexcDataFeed, LivePaperTradingManager
from ai_brain import ai_brain
from advanced_ai_system import (
    multi_strategy_brain, regime_detector, sentiment_analyzer as adv_sentiment,
    onchain_analyzer, watchdog, explainable_ai, TradingStrategy, MarketRegime,
    initialize_advanced_ai
)

class EnhancedAIProfitBot:
    """Enhanced AI trading bot with all advanced features"""
    
    def __init__(self, initial_capital: float = 5000.0):
        # Start AI learning session
        ai_brain.start_learning_session()
        
        # Initialize multi-strategy brain with our capital
        multi_strategy_brain.total_capital = initial_capital
        
        # Core components
        self.trader = LivePaperTradingManager(initial_capital)
        self.data_feed = LiveMexcDataFeed()
        
        # AI systems
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.strategy_engine = AIStrategyEngine()
        
        # Configuration
        self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "ADA/USDT", "MATIC/USDT", "DOT/USDT", "LINK/USDT"]
        self.price_history = {symbol: deque(maxlen=200) for symbol in self.symbols}
        self.active_signals = {}
        self.trade_count = 0
        self.initial_capital = initial_capital
        
        # ENHANCED TRADING PARAMETERS
        self.confidence_threshold = 0.25  # 25% confidence minimum
        self.max_concurrent_positions = 5  # More positions for diversification
        self.min_trade_size = 50.0  # $50 minimum trade
        self.max_position_size = initial_capital * 0.15  # Max 15% per trade
        self.stop_loss_pct = 3.0  # 3% stop loss
        self.take_profit_pct = 5.0  # 5% take profit target
        
        # ADVANCED FEATURES
        self.regime_adaptive = True  # Adapt to market regimes
        self.multi_strategy = True  # Use multiple strategies
        self.sentiment_boost = True  # Boost trades with sentiment
        self.onchain_filter = True  # Filter with on-chain data
        self.explainable_mode = True  # Generate explanations
        
        print("🚀 ENHANCED AI PROFIT BOT INITIALIZED")
        print(f"💰 Starting Capital: ${initial_capital:,.2f}")
        print("🧠 Advanced Features Enabled:")
        print("   ✅ Multi-strategy brain with meta-controller")
        print("   ✅ Market regime detection & adaptation")
        print("   ✅ Multi-source sentiment analysis")
        print("   ✅ Self-healing system watchdog")
        print("   ✅ On-chain intelligence integration")
        print("   ✅ Explainable AI decision system")
        
    async def run_enhanced_trading(self, cycles: int = 10):
        """Run enhanced AI trading for specified cycles"""
        
        print(f"\n🔥 STARTING ENHANCED AI PROFIT TRADING")
        print(f"🎯 Target: Maximize returns on ${self.initial_capital:,.2f}")
        print(f"🔄 Running {cycles} trading cycles...")
        print("💡 Press Ctrl+C to stop early\n")
        
        # Initialize advanced AI systems
        print("🧠 Initializing Advanced AI Systems...")
        await initialize_advanced_ai()
        print("✅ All advanced AI systems online!\n")
        
        for cycle in range(1, cycles + 1):
            try:
                print(f"\n{'='*70}")
                print(f"🤖 ENHANCED AI TRADING CYCLE #{cycle}/{cycles}")
                print(f"{'='*70}")
                
                # Step 1: System health check
                if not watchdog.is_healthy:
                    print("⚠️ System health compromised, skipping cycle...")
                    await asyncio.sleep(60)
                    continue
                
                # Step 2: Collect enhanced market data
                await self._collect_enhanced_market_data()
                
                # Step 3: Generate multi-strategy signals
                ai_signals = await self._generate_enhanced_signals()
                
                # Step 4: Execute strategic trades
                await self._execute_strategic_trades(ai_signals)
                
                # Step 5: Manage positions with advanced logic
                await self._manage_enhanced_positions()
                
                # Step 6: Show comprehensive status
                await self._show_enhanced_status(cycle, cycles)
                
                # Wait between cycles
                if cycle < cycles:
                    print(f"⏰ Next cycle in 60 seconds...")
                    await asyncio.sleep(60)
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Enhanced trading stopped by user at cycle {cycle}")
                break
            except Exception as e:
                print(f"❌ Error in cycle {cycle}: {e}")
                
        # End session and show results
        ai_brain.end_learning_session()
        await self._show_final_enhanced_results()
    
    async def _collect_enhanced_market_data(self):
        """Collect and process enhanced market data"""
        print("📊 COLLECTING ENHANCED MARKET DATA...")
        
        # Get live prices
        live_prices = await self.data_feed.get_multiple_prices(self.symbols)
        
        if not live_prices:
            print("⚠️ No market data received")
            return
        
        # Process prices and update regime detector
        for symbol, price in live_prices.items():
            self.price_history[symbol].append(price)
            regime_detector.add_data(price, 1.0)  # Add to regime detector
            
            if len(self.price_history[symbol]) > 1:
                prev_price = self.price_history[symbol][-2]
                change_pct = (price - prev_price) / prev_price * 100
                trend = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                print(f"   {trend} {symbol:12} ${price:>12,.2f} ({change_pct:+6.2f}%)")
            else:
                print(f"   📊 {symbol:12} ${price:>12,.2f}")
    
    async def _generate_enhanced_signals(self) -> List[AITradingSignal]:
        """Generate signals using all advanced AI features"""
        print("\n🧠 ADVANCED AI SIGNAL GENERATION...")
        
        # Get all advanced data
        sentiment_data = await adv_sentiment.get_sentiment_data()
        onchain_data = await onchain_analyzer.get_onchain_data()
        current_regime = regime_detector.current_regime
        
        print(f"🎆 Market Regime: {current_regime.value.upper()}")
        print(f"💭 Composite Sentiment: {sentiment_data.composite_sentiment:.2f}")
        print(f"⛓️ BTC Dominance: {onchain_data.btc_dominance:.1%}")
        print(f"🐋 Whale Activity: {onchain_data.whale_activity:.2f}")
        
        signals = []
        
        for symbol in self.symbols:
            if len(self.price_history[symbol]) < 10:
                continue
                
            # Calculate technical indicators
            current_price = self.price_history[symbol][-1]
            prices = list(self.price_history[symbol])
            
            # Technical analysis
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
            
            rsi = self._calculate_rsi(prices)
            macd = self._calculate_macd(prices)
            volatility = self._calculate_volatility(prices)
            momentum = self._calculate_momentum(prices)
            
            # Create technical signals dictionary for explainable AI
            technical_signals = {
                'sma_signal': (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0,
                'rsi': (rsi - 50) / 50,  # Normalize RSI
                'macd': macd,
                'volatility': volatility,
                'momentum': momentum
            }
            
            # Generate signals for each strategy
            strategies_to_try = [
                TradingStrategy.TREND_FOLLOWING,
                TradingStrategy.MEAN_REVERSION,
                TradingStrategy.MOMENTUM,
                TradingStrategy.BREAKOUT
            ]
            
            for strategy in strategies_to_try:
                # Get strategy allocation from multi-strategy brain
                allocation = multi_strategy_brain.get_strategy_allocation(strategy)
                
                if allocation < 0.05:  # Skip strategies with < 5% allocation
                    continue
                
                # Generate strategy-specific signal
                signal = self._generate_strategy_signal(
                    symbol, strategy, current_price, technical_signals,
                    sentiment_data, onchain_data, current_regime
                )
                
                if signal and signal.confidence >= self.confidence_threshold:
                    # Apply regime filtering
                    if self._should_trade_in_regime(strategy, current_regime):
                        # Generate explanation
                        if self.explainable_mode:
                            explanation = explainable_ai.generate_explanation(
                                signal.confidence, strategy, current_regime,
                                sentiment_data, onchain_data, technical_signals
                            )
                            explainable_ai.log_decision(explanation, symbol, signal.action)
                        
                        signals.append(signal)
                        print(f"🎯 SIGNAL: {signal.action} {symbol} | Strategy: {strategy.value}")
                        print(f"   📊 Confidence: {signal.confidence:.1%} | Expected: {signal.expected_return:+.2f}%")
        
        return signals
    
    def _generate_strategy_signal(self, symbol: str, strategy: TradingStrategy, 
                                 current_price: float, technical_signals: Dict,
                                 sentiment_data, onchain_data, regime: MarketRegime) -> Optional[AITradingSignal]:
        """Generate signal for specific strategy"""
        
        confidence = 0.0
        expected_return = 0.0
        action = 'HOLD'
        
        if strategy == TradingStrategy.TREND_FOLLOWING:
            # Trend following logic
            sma_signal = technical_signals['sma_signal']
            momentum = technical_signals['momentum']
            
            if sma_signal > 0.02 and momentum > 0.5:  # Strong uptrend
                action = 'BUY'
                confidence = min(0.9, abs(sma_signal) * 5 + momentum * 0.3)
                expected_return = sma_signal * 2.0 + momentum * 0.5
            elif sma_signal < -0.02 and momentum < -0.5:  # Strong downtrend
                action = 'SELL'
                confidence = min(0.9, abs(sma_signal) * 5 + abs(momentum) * 0.3)
                expected_return = abs(sma_signal) * 2.0 + abs(momentum) * 0.5
                
        elif strategy == TradingStrategy.MEAN_REVERSION:
            # Mean reversion logic
            rsi = technical_signals['rsi']
            
            if rsi < -0.6:  # Oversold
                action = 'BUY'
                confidence = min(0.8, abs(rsi) * 1.2)
                expected_return = abs(rsi) * 1.5
            elif rsi > 0.6:  # Overbought
                action = 'SELL'
                confidence = min(0.8, abs(rsi) * 1.2)
                expected_return = abs(rsi) * 1.5
                
        elif strategy == TradingStrategy.MOMENTUM:
            # Momentum strategy
            momentum = technical_signals['momentum']
            macd = technical_signals['macd']
            
            if momentum > 0.3 and macd > 0.1:
                action = 'BUY'
                confidence = min(0.85, momentum * 2 + abs(macd) * 3)
                expected_return = momentum * 1.5 + macd * 2
            elif momentum < -0.3 and macd < -0.1:
                action = 'SELL'
                confidence = min(0.85, abs(momentum) * 2 + abs(macd) * 3)
                expected_return = abs(momentum) * 1.5 + abs(macd) * 2
                
        elif strategy == TradingStrategy.BREAKOUT:
            # Breakout strategy
            volatility = technical_signals['volatility']
            momentum = technical_signals['momentum']
            
            if volatility > 0.5 and abs(momentum) > 0.4:
                action = 'BUY' if momentum > 0 else 'SELL'
                confidence = min(0.8, volatility * 1.5 + abs(momentum) * 1.2)
                expected_return = volatility * 1.0 + abs(momentum) * 1.3
        
        if action == 'HOLD':
            return None
        
        # Apply sentiment and on-chain boosts
        if self.sentiment_boost:
            sentiment_factor = (sentiment_data.composite_sentiment - 0.5) * 2  # -1 to +1
            if (action == 'BUY' and sentiment_factor > 0) or (action == 'SELL' and sentiment_factor < 0):
                confidence += min(0.1, abs(sentiment_factor) * 0.2)
                expected_return += abs(sentiment_factor) * 0.5
        
        if self.onchain_filter:
            # BTC dominance factor
            btc_factor = (onchain_data.btc_dominance - 0.5) * 2
            if symbol == "BTC/USDT" and btc_factor > 0:
                confidence += 0.05
                expected_return += btc_factor * 0.3
        
        return AITradingSignal(
            symbol=symbol,
            action=action,
            confidence=min(0.95, confidence),
            expected_return=expected_return,
            risk_score=0.3,
            time_horizon=60,
            entry_price=current_price,
            stop_loss=None,
            take_profit=None,
            position_size=self.min_trade_size,
            strategy_name=strategy.value,
            ai_reasoning=f"{strategy.value} strategy in {regime.value} market",
            technical_score=technical_signals['momentum'],
            sentiment_score=sentiment_data.composite_sentiment,
            momentum_score=technical_signals['momentum'],
            volatility_score=technical_signals['volatility']
        )
    
    def _should_trade_in_regime(self, strategy: TradingStrategy, regime: MarketRegime) -> bool:
        """Determine if strategy should trade in current regime"""
        if not self.regime_adaptive:
            return True
        
        # Regime-strategy compatibility matrix
        compatibility = {
            MarketRegime.BULL: [TradingStrategy.TREND_FOLLOWING, TradingStrategy.MOMENTUM, TradingStrategy.BREAKOUT],
            MarketRegime.BEAR: [TradingStrategy.MEAN_REVERSION, TradingStrategy.MOMENTUM],
            MarketRegime.CRAB: [TradingStrategy.MEAN_REVERSION, TradingStrategy.ARBITRAGE],
            MarketRegime.VOLATILE: [TradingStrategy.BREAKOUT, TradingStrategy.MOMENTUM],
            MarketRegime.UNKNOWN: [TradingStrategy.TREND_FOLLOWING, TradingStrategy.MEAN_REVERSION]
        }
        
        return strategy in compatibility.get(regime, [])
    
    async def _execute_strategic_trades(self, signals: List[AITradingSignal]):
        """Execute trades with strategic allocation"""
        if not signals:
            print("🔍 No signals generated")
            return
        
        portfolio = await self.trader.get_portfolio_value()
        available_cash = portfolio['cash']
        current_positions = portfolio.get('positions', {})
        
        print(f"\n💰 STRATEGIC TRADE EXECUTION:")
        print(f"   💵 Available Cash: ${available_cash:,.2f}")
        
        # Sort signals by confidence and expected return
        sorted_signals = sorted(signals, key=lambda x: x.confidence * x.expected_return, reverse=True)
        
        active_positions = len([p for p in current_positions.values() if p.get('quantity', 0) > 0])
        
        for signal in sorted_signals:
            if active_positions >= self.max_concurrent_positions:
                print(f"   ⚠️ Max positions ({self.max_concurrent_positions}) reached")
                break
            
            if signal.symbol in current_positions and current_positions[signal.symbol].get('quantity', 0) > 0:
                print(f"   ⚠️ {signal.symbol}: Already have position")
                continue
            
            # Calculate position size based on strategy allocation
            strategy = TradingStrategy(signal.strategy_name) if signal.strategy_name in [s.value for s in TradingStrategy] else TradingStrategy.TREND_FOLLOWING
            allocation = multi_strategy_brain.get_strategy_allocation(strategy)
            
            base_size = available_cash * allocation * 0.8  # Use 80% of allocated capital
            position_size = min(base_size, self.max_position_size, available_cash * 0.2)
            
            if position_size < self.min_trade_size:
                print(f"   ⚠️ Insufficient funds for {signal.symbol}")
                continue
            
            # Execute trade
            result = await self.trader.execute_live_trade(
                signal.symbol, signal.action, position_size,
                f'{strategy.value}_{signal.strategy_name}'
            )
            
            if result['success']:
                self.trade_count += 1
                self.active_signals[signal.symbol] = signal
                active_positions += 1
                available_cash -= position_size
                
                print(f"   ✅ {signal.symbol}: ${position_size:,.2f} | {signal.action} | {strategy.value}")
                print(f"      📊 Confidence: {signal.confidence:.1%} | Expected: {signal.expected_return:+.2f}%")
            else:
                print(f"   ❌ {signal.symbol}: Trade execution failed")
    
    async def _manage_enhanced_positions(self):
        """Enhanced position management with regime awareness"""
        portfolio = await self.trader.get_portfolio_value()
        positions = portfolio.get('positions', {})
        
        if not positions:
            return
        
        print(f"\n🧠 ENHANCED POSITION MANAGEMENT:")
        current_regime = regime_detector.current_regime
        
        for symbol, position in positions.items():
            if position.get('quantity', 0) <= 0:
                continue
            
            cost_basis = position['cost_basis']
            current_value = position['current_value']
            unrealized_pnl = position['unrealized_pnl']
            pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            should_close = False
            reason = ""
            
            # Regime-adaptive exit logic
            if current_regime == MarketRegime.VOLATILE and abs(pnl_pct) > 2.0:
                should_close = True
                reason = "VOLATILE_REGIME_EXIT"
            elif pnl_pct >= self.take_profit_pct:
                should_close = True
                reason = f"PROFIT_TARGET_{self.take_profit_pct}%"
            elif pnl_pct <= -self.stop_loss_pct:
                should_close = True
                reason = f"STOP_LOSS_{self.stop_loss_pct}%"
            
            if should_close:
                await self._close_enhanced_position(symbol, position, reason)
            else:
                status = "💚" if unrealized_pnl > 0 else "❤️" if unrealized_pnl < 0 else "💛"
                print(f"   {status} {symbol}: ${current_value:,.2f} ({pnl_pct:+.2f}%) - HOLDING")
    
    async def _close_enhanced_position(self, symbol: str, position: Dict, reason: str):
        """Close position with enhanced logging"""
        result = await self.trader.execute_live_trade(
            symbol, 'SELL', position['current_value'], f'CLOSE_{reason}'
        )
        
        if result['success']:
            pnl = position['unrealized_pnl']
            print(f"   🎯 {symbol}: CLOSED - ${pnl:+,.2f} ({reason})")
            
            # Update multi-strategy brain performance
            if symbol in self.active_signals:
                signal = self.active_signals[symbol]
                strategy_name = signal.strategy_name
                
                if strategy_name in [s.value for s in TradingStrategy]:
                    strategy = TradingStrategy(strategy_name)
                    multi_strategy_brain.update_strategy_performance(
                        strategy, pnl, {
                            'symbol': symbol,
                            'confidence': signal.confidence,
                            'expected_return': signal.expected_return
                        }
                    )
                
                # Learn from trade
                trade_data = {
                    'symbol': symbol,
                    'action': signal.action,
                    'profit_loss': pnl,
                    'confidence': signal.confidence,
                    'strategy_scores': {
                        'technical': signal.technical_score,
                        'sentiment': signal.sentiment_score,
                        'momentum': signal.momentum_score
                    },
                    'market_conditions': {
                        'volatility': signal.volatility_score,
                        'trend_strength': 0.5
                    }
                }
                ai_brain.learn_from_trade(trade_data)
                del self.active_signals[symbol]
        else:
            print(f"   ❌ {symbol}: Failed to close position")
    
    async def _show_enhanced_status(self, cycle: int, total_cycles: int):
        """Show comprehensive enhanced status"""
        portfolio = await self.trader.get_portfolio_value()
        
        print(f"\n📊 ENHANCED AI STATUS (Cycle {cycle}/{total_cycles}):")
        print("=" * 70)
        
        # Portfolio metrics
        total_value = portfolio['total_value']
        cash = portfolio['cash']
        total_return_pct = ((total_value - self.initial_capital) / self.initial_capital) * 100
        
        print(f"💰 Portfolio Value: ${total_value:,.2f}")
        print(f"💵 Available Cash: ${cash:,.2f}")
        print(f"📈 Total Return: {total_return_pct:+.2f}%")
        print(f"💎 Unrealized P&L: ${total_value - self.initial_capital:+,.2f}")
        print(f"🔄 Total Trades: {self.trade_count}")
        
        # System health
        health_status = "🟢 HEALTHY" if watchdog.is_healthy else "🔴 COMPROMISED"
        print(f"🛡️ System Health: {health_status}")
        
        # Strategy allocations
        print("\n🧠 STRATEGY ALLOCATIONS:")
        for strategy, perf in multi_strategy_brain.strategies.items():
            print(f"   {strategy.value:15} {perf.current_allocation:6.1%} | Trades: {perf.total_trades:3} | Win Rate: {perf.win_rate:.1%}")
        
        # Active positions
        active_pos = [p for p in portfolio['positions'].values() if p.get('quantity', 0) > 0]
        if active_pos:
            print(f"\n🎯 ACTIVE POSITIONS ({len(active_pos)}):")
            for symbol, pos in portfolio['positions'].items():
                if pos.get('quantity', 0) > 0:
                    pnl_pct = (pos['unrealized_pnl'] / pos['cost_basis']) * 100 if pos['cost_basis'] > 0 else 0
                    emoji = "💚" if pos['unrealized_pnl'] > 0 else "❤️"
                    print(f"   {emoji} {symbol:12} ${pos['current_value']:>10,.2f} ({pnl_pct:+6.2f}%)")
        
        # Performance metrics
        if total_value > self.initial_capital:
            growth_rate = ((total_value / self.initial_capital) ** (1/cycle)) - 1 if cycle > 0 else 0
            print(f"\n🚀 PERFORMANCE METRICS:")
            print(f"   📈 Growth Rate: {growth_rate * 100:+.2f}% per cycle")
            if growth_rate > 0:
                cycles_to_double = 70 / (growth_rate * 100) if growth_rate > 0 else 0
                print(f"   💫 Cycles to Double: ~{cycles_to_double:.0f}")
        
        print("=" * 70)
    
    async def _show_final_enhanced_results(self):
        """Show comprehensive final results"""
        portfolio = await self.trader.get_portfolio_value()
        
        print(f"\n🏁 FINAL ENHANCED AI RESULTS:")
        print("=" * 80)
        
        final_value = portfolio['total_value']
        total_pnl = final_value - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        print(f"💰 Starting Capital: ${self.initial_capital:,.2f}")
        print(f"💰 Final Value: ${final_value:,.2f}")
        print(f"📈 Total Return: {total_return_pct:+.2f}%")
        print(f"💎 Total P&L: ${total_pnl:+,.2f}")
        print(f"🔄 Total Trades: {self.trade_count}")
        
        # Strategy performance summary
        print(f"\n🧠 STRATEGY PERFORMANCE SUMMARY:")
        for strategy, perf in multi_strategy_brain.strategies.items():
            if perf.total_trades > 0:
                print(f"   {strategy.value:15} | Trades: {perf.total_trades:3} | Win Rate: {perf.win_rate:6.1%} | Avg Return: {perf.avg_return:+7.2f}")
        
        # Success assessment
        if total_pnl > 0:
            print(f"\n🎉 SUCCESS: Enhanced AI generated ${total_pnl:,.2f} profit!")
        elif total_pnl == 0:
            print(f"\n😐 BREAK-EVEN: No gains or losses")
        else:
            print(f"\n📉 LOSS: Account down by ${abs(total_pnl):,.2f}")
        
        print(f"\n🧠 AI LEARNING SUMMARY:")
        print(f"   📚 Multi-strategy brain learned from {self.trade_count} trades")
        print(f"   🎯 All knowledge saved for future sessions")
        print(f"   🚀 System ready for live trading deployment!")
        
        print("=" * 80)
    
    # TECHNICAL ANALYSIS METHODS
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: List[float]) -> float:
        """Calculate MACD"""
        if len(prices) < 26:
            return 0.0
        
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        return (ema_12 - ema_26) / ema_26 if ema_26 != 0 else 0.0
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate volatility"""
        if len(prices) < 2:
            return 0.1
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return min(1.0, np.std(returns) * 100)
    
    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate momentum"""
        if len(prices) < 10:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-10]
        
        return (current_price - past_price) / past_price

async def main():
    """Main function for enhanced AI trading"""
    
    print("🚀 ENHANCED AI PROFIT BOT")
    print("=" * 50)
    print("💰 Ultimate AI trading with all advanced features")
    print("🧠 Multi-strategy, regime-aware, sentiment-driven")
    print("🛡️ Self-healing with explainable decisions")
    print()
    
    # Create enhanced bot
    enhanced_bot = EnhancedAIProfitBot(5000.0)
    
    try:
        # Run enhanced trading session
        await enhanced_bot.run_enhanced_trading(cycles=8)
        
    except KeyboardInterrupt:
        print("\n👋 Enhanced AI trading stopped by user")
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    print("🤖 Starting Enhanced AI Profit Bot...")
    asyncio.run(main())
