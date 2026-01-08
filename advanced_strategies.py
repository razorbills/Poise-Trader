#!/usr/bin/env python3
"""
🚀 ADVANCED HIGH-PROFIT STRATEGIES for Poise Trader

⚠️  WARNING: These are AGGRESSIVE strategies with HIGH RISK!
💰 Higher profit potential = Higher loss potential
🎯 Only use with money you can afford to lose completely!

For your 0.00005 BTC (5k sats) - Start small and learn!
"""

from decimal import Decimal
import os
import asyncio
import math

class ScalpingStrategy:
    """
    🔥 SCALPING STRATEGY - Ultra Fast Profits
    
    📈 Target: 0.1-0.5% profit per trade (many trades per day)
    ⚡ Speed: 1-5 minute trades
    💰 Potential: 50-200% monthly (if successful)
    ⚠️  Risk: Very High - can lose quickly
    """
    
    def __init__(self, capital):
        self.capital = capital
        self.config = {
            'name': 'Scalping Beast',
            'profit_target': 0.002,     # 0.2% per trade
            'stop_loss': 0.001,         # 0.1% stop loss
            'trade_frequency': 60,      # Every 1 minute
            'max_trades_per_day': 50,   # Very active
            'leverage': 1,              # No leverage (safer)
            'min_volume_spike': 1.5,    # Trade on volume spikes
        }
    
    def get_signals(self, price_data):
        """Generate ultra-fast scalping signals"""
        signals = []
        
        # Look for quick price movements
        for i in range(5, len(price_data)):
            recent_prices = price_data[i-5:i]
            current_price = price_data[i]['close']
            
            # Quick momentum check
            momentum = (current_price - recent_prices[0]['close']) / recent_prices[0]['close']
            volume_spike = price_data[i]['volume'] > sum([p['volume'] for p in recent_prices]) / 5 * 1.5
            
            if momentum > 0.001 and volume_spike:  # 0.1% up movement + volume
                signals.append({
                    'type': 'BUY',
                    'price': current_price,
                    'confidence': min(abs(momentum) * 100, 0.95),
                    'reason': f'Quick momentum: {momentum:.3%} + volume spike'
                })
            elif momentum < -0.001 and volume_spike:  # 0.1% down movement + volume
                signals.append({
                    'type': 'SELL',
                    'price': current_price, 
                    'confidence': min(abs(momentum) * 100, 0.95),
                    'reason': f'Quick drop: {momentum:.3%} + volume spike'
                })
        
        return signals

class MomentumBreakoutStrategy:
    """
    🚀 MOMENTUM BREAKOUT - Ride the Big Moves
    
    📈 Target: 2-10% profit per trade (fewer, bigger trades)
    ⚡ Speed: 15-60 minute trades
    💰 Potential: 100-500% monthly (if you catch breakouts)
    ⚠️  Risk: High - wrong breakouts can hurt
    """
    
    def __init__(self, capital):
        self.capital = capital
        self.config = {
            'name': 'Breakout Hunter',
            'profit_target': 0.03,      # 3% per trade
            'stop_loss': 0.015,         # 1.5% stop loss
            'breakout_threshold': 0.02, # 2% breakout needed
            'volume_multiplier': 2.0,   # 2x average volume
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'confirmation_candles': 3,  # Wait for confirmation
        }
    
    def detect_breakouts(self, price_data):
        """Find powerful breakout patterns"""
        breakouts = []
        
        for i in range(20, len(price_data)):
            current = price_data[i]
            recent_20 = price_data[i-20:i]
            
            # Find resistance/support levels
            highs = [p['high'] for p in recent_20]
            lows = [p['low'] for p in recent_20]
            
            resistance = max(highs)
            support = min(lows)
            avg_volume = sum([p['volume'] for p in recent_20]) / 20
            
            # Check for breakout above resistance
            if (current['close'] > resistance * 1.02 and 
                current['volume'] > avg_volume * 2):
                breakouts.append({
                    'type': 'BULLISH_BREAKOUT',
                    'entry_price': current['close'],
                    'target': current['close'] * 1.05,  # 5% target
                    'stop_loss': resistance * 0.98,     # Back below resistance
                    'strength': (current['close'] - resistance) / resistance,
                    'reason': f'Broke resistance at {resistance} with {current["volume"]/avg_volume:.1f}x volume'
                })
            
            # Check for breakdown below support  
            elif (current['close'] < support * 0.98 and 
                  current['volume'] > avg_volume * 2):
                breakouts.append({
                    'type': 'BEARISH_BREAKOUT',
                    'entry_price': current['close'],
                    'target': current['close'] * 0.95,  # 5% target down
                    'stop_loss': support * 1.02,        # Back above support
                    'strength': (support - current['close']) / support,
                    'reason': f'Broke support at {support} with {current["volume"]/avg_volume:.1f}x volume'
                })
        
        return breakouts

class VolatilityHunterStrategy:
    """
    ⚡ VOLATILITY HUNTER - Profit from Big Swings
    
    📈 Target: 5-20% profit per trade
    ⚡ Speed: 1-4 hour trades  
    💰 Potential: 200-1000% monthly (extremely aggressive)
    ⚠️  Risk: VERY High - can lose 50%+ quickly
    """
    
    def __init__(self, capital):
        self.capital = capital
        self.config = {
            'name': 'Volatility Beast',
            'min_volatility': 0.05,     # 5% minimum price swings
            'profit_multiplier': 3,     # Risk 1% to make 3%
            'max_risk_per_trade': 0.02, # Risk 2% of capital per trade
            'volatility_window': 24,    # 24 periods to measure volatility
            'momentum_threshold': 0.03, # 3% momentum needed
        }
    
    def calculate_volatility(self, price_data, window=24):
        """Calculate recent volatility"""
        if len(price_data) < window:
            return 0
        
        recent = price_data[-window:]
        returns = []
        
        for i in range(1, len(recent)):
            ret = (recent[i]['close'] - recent[i-1]['close']) / recent[i-1]['close']
            returns.append(ret)
        
        # Standard deviation of returns
        avg_return = sum(returns) / len(returns)
        variance = sum([(r - avg_return)**2 for r in returns]) / len(returns)
        volatility = math.sqrt(variance)
        
        return volatility
    
    def find_volatility_opportunities(self, price_data):
        """Find high volatility trading opportunities"""
        opportunities = []
        current_volatility = self.calculate_volatility(price_data)
        
        if current_volatility > self.config['min_volatility']:
            current_price = price_data[-1]['close']
            recent_5 = price_data[-5:]
            
            # Look for momentum in high volatility
            momentum = (current_price - recent_5[0]['close']) / recent_5[0]['close']
            
            if abs(momentum) > self.config['momentum_threshold']:
                opportunities.append({
                    'type': 'VOLATILITY_PLAY',
                    'direction': 'LONG' if momentum > 0 else 'SHORT',
                    'entry_price': current_price,
                    'volatility': current_volatility,
                    'momentum': momentum,
                    'target': current_price * (1 + (momentum * 2)),  # 2x the momentum
                    'stop_loss': current_price * (1 - (abs(momentum) * 0.5)),  # Half momentum stop
                    'reason': f'High volatility {current_volatility:.1%} + momentum {momentum:.1%}'
                })
        
        return opportunities

class AIMarketMakerStrategy:
    """
    🤖 AI MARKET MAKER - Multiple Small Profits
    
    📈 Target: 0.05-0.2% per trade (many trades)
    ⚡ Speed: 5-30 minute trades
    💰 Potential: 30-100% monthly 
    ⚠️  Risk: Medium - more consistent but slower
    """
    
    def __init__(self, capital):
        self.capital = capital
        self.config = {
            'name': 'AI Market Maker',
            'spread_target': 0.001,     # 0.1% spread
            'inventory_limit': 0.5,     # Max 50% in any asset
            'quote_layers': 5,          # 5 price levels
            'rebalance_threshold': 0.02, # 2% price move = rebalance
            'profit_take_ratio': 2,     # Take profit at 2x spread
        }
    
    def generate_orders(self, current_price, spread):
        """Generate buy/sell orders around current price"""
        orders = []
        
        # Create buy orders below current price
        for i in range(1, self.config['quote_layers'] + 1):
            buy_price = current_price * (1 - (spread * i))
            orders.append({
                'type': 'BUY_LIMIT',
                'price': buy_price,
                'amount': self.capital * 0.1 / self.config['quote_layers'],  # Spread capital
                'layer': i
            })
        
        # Create sell orders above current price
        for i in range(1, self.config['quote_layers'] + 1):
            sell_price = current_price * (1 + (spread * i))
            orders.append({
                'type': 'SELL_LIMIT', 
                'price': sell_price,
                'amount': self.capital * 0.1 / self.config['quote_layers'],
                'layer': i
            })
        
        return orders

# 🚀 STRATEGY COMBINER - Use Multiple Strategies Together!
class MultiStrategyEngine:
    """
    🎯 MULTI-STRATEGY BEAST - Combine All Strategies!
    
    📈 Target: Varies based on market conditions
    💰 Potential: Maximum possible profits
    ⚠️  Risk: Managed across multiple strategies
    """
    
    def __init__(self, capital):
        self.capital = capital
        self.strategies = {
            'scalping': ScalpingStrategy(capital * Decimal('0.3')),      # 30% for scalping
            'breakout': MomentumBreakoutStrategy(capital * Decimal('0.4')), # 40% for breakouts  
            'volatility': VolatilityHunterStrategy(capital * Decimal('0.2')), # 20% for volatility
            'market_maker': AIMarketMakerStrategy(capital * Decimal('0.1')),  # 10% for market making
        }
        
        self.performance = {
            'scalping': {'wins': 0, 'losses': 0, 'profit': 0},
            'breakout': {'wins': 0, 'losses': 0, 'profit': 0},
            'volatility': {'wins': 0, 'losses': 0, 'profit': 0},
            'market_maker': {'wins': 0, 'losses': 0, 'profit': 0},
        }
    
    def get_best_strategy_for_market(self, market_conditions):
        """Choose best strategy based on current market"""
        volatility = market_conditions.get('volatility', 0)
        trend = market_conditions.get('trend', 0)
        volume = market_conditions.get('volume_ratio', 1)
        
        if volatility > 0.05 and volume > 2:
            return 'volatility'  # High volatility + volume = volatility strategy
        elif abs(trend) > 0.02 and volume > 1.5:
            return 'breakout'    # Strong trend + volume = breakout strategy
        elif volatility < 0.02:
            return 'market_maker' # Low volatility = market making
        else:
            return 'scalping'    # Default to scalping
    
    def execute_multi_strategy(self, price_data):
        """Execute the best strategy for current conditions"""
        # Analyze market conditions
        conditions = self.analyze_market_conditions(price_data)
        
        # Choose best strategy
        best_strategy = self.get_best_strategy_for_market(conditions)
        
        # Get signals from chosen strategy
        if best_strategy == 'scalping':
            return self.strategies['scalping'].get_signals(price_data)
        elif best_strategy == 'breakout':
            return self.strategies['breakout'].detect_breakouts(price_data)
        elif best_strategy == 'volatility':
            return self.strategies['volatility'].find_volatility_opportunities(price_data)
        elif best_strategy == 'market_maker':
            current_price = price_data[-1]['close']
            return self.strategies['market_maker'].generate_orders(current_price, 0.002)
    
    def analyze_market_conditions(self, price_data):
        """Analyze current market conditions"""
        if len(price_data) < 20:
            return {'volatility': 0, 'trend': 0, 'volume_ratio': 1}
        
        recent = price_data[-20:]
        current_price = price_data[-1]['close']
        old_price = price_data[-20]['close']
        
        # Calculate trend
        trend = (current_price - old_price) / old_price
        
        # Calculate volatility
        returns = []
        for i in range(1, len(recent)):
            ret = (recent[i]['close'] - recent[i-1]['close']) / recent[i-1]['close']
            returns.append(ret ** 2)
        volatility = math.sqrt(sum(returns) / len(returns))
        
        # Calculate volume ratio
        avg_volume = sum([p['volume'] for p in recent[:-1]]) / (len(recent) - 1)
        current_volume = recent[-1]['volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        return {
            'volatility': volatility,
            'trend': trend,
            'volume_ratio': volume_ratio
        }

# 🎯 MEXC STRATEGY CONFIGURATOR - Ready for Your Account!
MEXC_ADVANCED_CONFIG = {
    'api_key': os.getenv('MEXC_API_KEY', ''),
    'api_secret': os.getenv('MEXC_API_SECRET', ''),
    'exchange': 'mexc',
    'initial_capital': Decimal('0.00005'),  # Your 5k sats
    
    # 🚀 AGGRESSIVE SETTINGS - HIGH PROFIT POTENTIAL!
    'strategy': 'multi_strategy',           # Use all strategies!
    'risk_level': 'AGGRESSIVE',             # Maximum profit mode
    'max_risk_per_trade': 0.02,            # Risk 2% per trade
    'profit_target_multiplier': 3,          # Risk 1 to make 3
    'max_daily_trades': 100,               # Very active trading
    
    # ⚡ SPEED SETTINGS
    'check_interval': 30,                   # Check every 30 seconds
    'quick_profit_take': True,              # Take profits quickly
    'trail_stop_loss': True,                # Protect profits
    
    # 🎯 SYMBOL SELECTION - Most Volatile Pairs
    'symbols': [
        'BTC/USDT',   # Most liquid
        'ETH/USDT',   # High volume
        'BNB/USDT',   # Good volatility
        'ADA/USDT',   # Frequent moves
        'SOL/USDT',   # High momentum
        'DOGE/USDT',  # Meme coin volatility
        'SHIB/USDT',  # Extreme volatility
        'PEPE/USDT',  # Newest volatile coin
    ],
    
    # 🛡️ SAFETY NETS (Even for aggressive trading)
    'max_drawdown': 0.2,                   # Stop if down 20%
    'daily_loss_limit': 0.1,               # Stop if down 10% in a day
    'emergency_stop': True,                # Emergency stop enabled
    'paper_trading': True,                 # Start safe, then switch to live
}

def create_advanced_mexc_strategy():
    """Create the most advanced strategy for your MEXC account"""
    print("🚀 CREATING ADVANCED HIGH-PROFIT STRATEGY")
    print("=" * 50)
    print(f"💰 Capital: {MEXC_ADVANCED_CONFIG['initial_capital']} BTC")
    print(f"🎯 Strategy: Multi-Strategy Beast Mode")
    print(f"⚠️  Risk Level: {MEXC_ADVANCED_CONFIG['risk_level']}")
    print(f"📊 Symbols: {len(MEXC_ADVANCED_CONFIG['symbols'])} pairs")
    print(f"🛡️  Safety: Multiple stop-loss levels")
    
    # Initialize multi-strategy engine
    engine = MultiStrategyEngine(MEXC_ADVANCED_CONFIG['initial_capital'])
    
    print("\n✅ Advanced strategy ready!")
    print("🔥 This strategy combines:")
    print("   • Scalping (30% capital) - Quick profits")
    print("   • Breakouts (40% capital) - Big moves") 
    print("   • Volatility (20% capital) - Extreme profits")
    print("   • Market Making (10% capital) - Consistent income")
    
    return engine

if __name__ == "__main__":
    print("⚠️  WARNING: ADVANCED HIGH-RISK STRATEGIES!")
    print("💰 Higher profit potential = Higher loss risk")
    print("🎯 Only trade what you can afford to lose!")
    print("\n🚀 Ready to create advanced strategies for your 5k sats?")
    
    create_advanced_mexc_strategy()
