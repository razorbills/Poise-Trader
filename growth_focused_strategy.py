#!/usr/bin/env python3
"""
🚀 GROWTH-FOCUSED STRATEGY for Small Amounts

The REALITY: With 5k sats ($5), even 100% gains = only $5 profit
The SOLUTION: Aggressive but SMART growth to build meaningful capital

🎯 Goal: Grow 5k sats to 50k+ sats (10x) THEN switch to professional strategies
💰 Target: 100-500% annual returns (higher risk, higher reward)
🛡️ Safety: Smart risk management, not reckless gambling

This is for growing SMALL amounts into MEANINGFUL amounts!
"""

from decimal import Decimal
import os
import math

class SmallCapitalGrowthStrategy:
    """
    🚀 SMALL CAPITAL GROWTH STRATEGY
    
    Designed specifically for growing tiny amounts (like your 5k sats):
    • Higher risk tolerance (you can afford to lose $5)
    • Focus on GROWTH over preservation
    • Take calculated risks for meaningful returns
    • Smart stop-losses to prevent total loss
    """
    
    def __init__(self, capital):
        self.capital = capital
        self.config = {
            'name': 'Small Capital Growth Beast',
            
            # 🎯 GROWTH-FOCUSED RISK SETTINGS
            'max_single_trade_risk': 0.10,     # Risk 10% per trade (vs 0.5% professional)
            'max_portfolio_risk': 0.25,        # Risk 25% of portfolio (vs 2% professional)
            'profit_target_multiplier': 5,     # Target 5x risk as profit (vs 3x professional)
            'compound_frequency': 'immediate', # Reinvest profits immediately
            
            # ⚡ AGGRESSIVE TARGETS
            'monthly_target': 0.15,            # 15% monthly (vs 2% professional)
            'min_profit_per_trade': 0.05,      # Min 5% profit per trade
            'max_trades_per_day': 20,          # Very active (vs 5 professional)
            'leverage_allowed': 1.5,           # Slight leverage (vs 1.0 professional)
            
            # 🛡️ SMART SAFETY (Not reckless)
            'emergency_stop_loss': 0.5,        # Stop if lose 50% (not 100%)
            'take_profit_quick': 0.08,         # Take profits at 8%+ moves
            'trail_profits': True,             # Protect gains with trailing stops
            'daily_profit_target': 0.03,       # Stop trading when up 3% for day
        }
    
    def calculate_growth_position_size(self, signal_strength, market_volatility):
        """Calculate aggressive but smart position sizes for growth"""
        base_risk = self.config['max_single_trade_risk']
        
        # Scale position size based on signal strength
        confidence_multiplier = signal_strength  # 0.5 to 1.0
        volatility_multiplier = min(market_volatility * 2, 1.5)  # Higher vol = bigger positions
        
        # Final position size
        position_risk = base_risk * confidence_multiplier * volatility_multiplier
        position_risk = min(position_risk, self.config['max_portfolio_risk'])
        
        return self.capital * Decimal(str(position_risk))

class MemeTokenMomentumStrategy:
    """
    🚀 MEME TOKEN MOMENTUM - High Growth Potential
    
    Target volatile coins with huge upside potential:
    • PEPE, SHIB, DOGE moves (10-50% swings)
    • Trend following on steroids
    • Quick profits, quick exits
    • Compound small wins into big gains
    """
    
    def __init__(self, capital):
        self.capital = capital
        self.config = {
            'name': 'Meme Momentum Beast',
            'target_coins': ['PEPE/USDT', 'SHIB/USDT', 'DOGE/USDT', 'WIF/USDT'],
            'momentum_threshold': 0.03,        # 3% moves to trigger
            'volume_surge_multiplier': 3,      # 3x volume increase needed
            'profit_target': 0.15,             # Target 15% gains
            'stop_loss': 0.08,                 # 8% stop loss
            'hold_time_max': 6,                # Max 6 hours hold time
            'social_sentiment_weight': 0.3,    # 30% weight on social buzz
        }
    
    def detect_meme_momentum(self, price_data, volume_data, social_data=None):
        """Detect explosive meme token momentum"""
        signals = []
        
        if len(price_data) < 10:
            return signals
        
        current_price = price_data[-1]['close']
        price_5min_ago = price_data[-5]['close']
        current_volume = volume_data[-1] if volume_data else price_data[-1]['volume']
        avg_volume = sum([p['volume'] for p in price_data[-20:-1]]) / 19
        
        # Calculate momentum
        momentum = (current_price - price_5min_ago) / price_5min_ago
        volume_surge = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Check for explosive move
        if (abs(momentum) > self.config['momentum_threshold'] and 
            volume_surge > self.config['volume_surge_multiplier']):
            
            confidence = min(abs(momentum) * 10 + (volume_surge / 5), 0.95)
            
            if momentum > 0:  # Bullish explosion
                signals.append({
                    'type': 'BUY',
                    'strategy': 'Meme Momentum',
                    'entry_price': current_price,
                    'profit_target': current_price * (1 + self.config['profit_target']),
                    'stop_loss': current_price * (1 - self.config['stop_loss']),
                    'confidence': confidence,
                    'momentum': momentum,
                    'volume_surge': volume_surge,
                    'max_hold_hours': self.config['hold_time_max'],
                    'reason': f'Meme explosion: {momentum:.1%} move, {volume_surge:.1f}x volume'
                })
            
            else:  # Bearish explosion (short opportunity)
                signals.append({
                    'type': 'SELL',
                    'strategy': 'Meme Momentum',
                    'entry_price': current_price,
                    'profit_target': current_price * (1 - self.config['profit_target']),
                    'stop_loss': current_price * (1 + self.config['stop_loss']),
                    'confidence': confidence,
                    'momentum': momentum,
                    'volume_surge': volume_surge,
                    'max_hold_hours': self.config['hold_time_max'],
                    'reason': f'Meme crash: {momentum:.1%} drop, {volume_surge:.1f}x volume'
                })
        
        return signals

class NewsEventTradingStrategy:
    """
    📰 NEWS EVENT TRADING - Capitalize on Market Moving Events
    
    Trade major news events and announcements:
    • Bitcoin ETF news
    • Major exchange listings
    • Regulatory announcements
    • Celebrity endorsements
    • Technical upgrades
    """
    
    def __init__(self, capital):
        self.capital = capital
        self.config = {
            'name': 'News Event Trader',
            'high_impact_events': [
                'ETF_APPROVAL', 'EXCHANGE_LISTING', 'REGULATORY_NEWS',
                'CELEBRITY_ENDORSEMENT', 'MAJOR_PARTNERSHIP', 'TECHNICAL_UPGRADE'
            ],
            'reaction_time_minutes': 5,        # React within 5 minutes
            'profit_target': 0.20,             # 20% profit target
            'stop_loss': 0.10,                 # 10% stop loss
            'position_size_multiplier': 1.5,   # Bigger positions for news
            'sentiment_threshold': 0.7,        # Min 70% positive sentiment
        }
    
    def analyze_news_impact(self, news_events, price_data):
        """Analyze news events for trading opportunities"""
        signals = []
        
        for event in news_events:
            impact_score = self.calculate_news_impact(event)
            
            if impact_score > 0.7:  # High impact news
                current_price = price_data[-1]['close']
                
                if event['sentiment'] > 0:  # Positive news
                    signals.append({
                        'type': 'BUY',
                        'strategy': 'News Event',
                        'entry_price': current_price,
                        'profit_target': current_price * (1 + self.config['profit_target']),
                        'stop_loss': current_price * (1 - self.config['stop_loss']),
                        'confidence': impact_score,
                        'news_type': event['type'],
                        'urgency': 'HIGH',
                        'reason': f'High impact {event["type"]}: {event["headline"]}'
                    })
                
                elif event['sentiment'] < -0.5:  # Negative news
                    signals.append({
                        'type': 'SELL',
                        'strategy': 'News Event',
                        'entry_price': current_price,
                        'profit_target': current_price * (1 - self.config['profit_target']),
                        'stop_loss': current_price * (1 + self.config['stop_loss']),
                        'confidence': impact_score,
                        'news_type': event['type'],
                        'urgency': 'HIGH',
                        'reason': f'Negative {event["type"]}: {event["headline"]}'
                    })
        
        return signals
    
    def calculate_news_impact(self, news_event):
        """Calculate the potential market impact of news"""
        base_impact = {
            'ETF_APPROVAL': 0.9,
            'EXCHANGE_LISTING': 0.7,
            'REGULATORY_NEWS': 0.8,
            'CELEBRITY_ENDORSEMENT': 0.6,
            'MAJOR_PARTNERSHIP': 0.7,
            'TECHNICAL_UPGRADE': 0.5,
        }
        
        impact = base_impact.get(news_event.get('type'), 0.3)
        
        # Adjust for recency (newer = higher impact)
        time_decay = max(0.5, 1 - (news_event.get('minutes_old', 0) / 60))
        
        # Adjust for source credibility
        source_multiplier = news_event.get('source_credibility', 0.7)
        
        return impact * time_decay * source_multiplier

class BreakoutScalpingStrategy:
    """
    ⚡ BREAKOUT SCALPING - Quick Profits from Price Breaks
    
    Rapid-fire trading on confirmed breakouts:
    • Support/resistance breaks
    • Volume-confirmed moves
    • Quick in, quick out
    • Many small wins compound to big gains
    """
    
    def __init__(self, capital):
        self.capital = capital
        self.config = {
            'name': 'Breakout Scalper',
            'min_breakout_size': 0.02,         # 2% minimum breakout
            'volume_confirmation': 2.0,        # 2x volume needed
            'profit_target': 0.03,             # 3% quick profit
            'stop_loss': 0.015,                # 1.5% stop loss
            'max_hold_minutes': 30,            # Max 30 minutes
            'trades_per_hour': 4,              # Max 4 trades per hour
            'compound_immediately': True,       # Reinvest profits immediately
        }
    
    def find_breakout_opportunities(self, price_data):
        """Find rapid breakout scalping opportunities"""
        signals = []
        
        if len(price_data) < 50:
            return signals
        
        # Find recent support/resistance levels
        recent_50 = price_data[-50:]
        current_price = price_data[-1]['close']
        current_volume = price_data[-1]['volume']
        
        # Calculate resistance (recent highs)
        highs = [p['high'] for p in recent_50[-20:]]
        resistance = max(highs)
        
        # Calculate support (recent lows)
        lows = [p['low'] for p in recent_50[-20:]]
        support = min(lows)
        
        # Average volume
        avg_volume = sum([p['volume'] for p in recent_50[-20:-1]]) / 19
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Check for resistance breakout
        breakout_above = (current_price - resistance) / resistance
        if (breakout_above > self.config['min_breakout_size'] and 
            volume_ratio > self.config['volume_confirmation']):
            
            confidence = min(breakout_above * 10 + (volume_ratio / 3), 0.9)
            
            signals.append({
                'type': 'BUY',
                'strategy': 'Breakout Scalp',
                'entry_price': current_price,
                'profit_target': current_price * (1 + self.config['profit_target']),
                'stop_loss': resistance * 0.99,  # Just below resistance
                'confidence': confidence,
                'breakout_size': breakout_above,
                'volume_ratio': volume_ratio,
                'max_hold_minutes': self.config['max_hold_minutes'],
                'reason': f'Resistance breakout: {breakout_above:.1%}, vol: {volume_ratio:.1f}x'
            })
        
        # Check for support breakdown
        breakdown_below = (support - current_price) / support
        if (breakdown_below > self.config['min_breakout_size'] and 
            volume_ratio > self.config['volume_confirmation']):
            
            confidence = min(breakdown_below * 10 + (volume_ratio / 3), 0.9)
            
            signals.append({
                'type': 'SELL',
                'strategy': 'Breakout Scalp',
                'entry_price': current_price,
                'profit_target': current_price * (1 - self.config['profit_target']),
                'stop_loss': support * 1.01,  # Just above support
                'confidence': confidence,
                'breakout_size': breakdown_below,
                'volume_ratio': volume_ratio,
                'max_hold_minutes': self.config['max_hold_minutes'],
                'reason': f'Support breakdown: {breakdown_below:.1%}, vol: {volume_ratio:.1f}x'
            })
        
        return signals

class GrowthFocusedPortfolio:
    """
    🚀 GROWTH-FOCUSED PORTFOLIO MANAGER
    
    Specifically designed for growing small amounts:
    • Higher risk tolerance
    • Aggressive compounding
    • Multi-strategy approach
    • Smart capital allocation
    """
    
    def __init__(self, initial_capital):
        self.capital = initial_capital
        self.growth_calculator = SmallCapitalGrowthStrategy(initial_capital)
        
        # Strategy allocation for GROWTH
        self.strategies = {
            'meme_momentum': MemeTokenMomentumStrategy(initial_capital * Decimal('0.4')),  # 40%
            'news_events': NewsEventTradingStrategy(initial_capital * Decimal('0.3')),    # 30%
            'breakout_scalping': BreakoutScalpingStrategy(initial_capital * Decimal('0.3')), # 30%
        }
        
        self.growth_config = {
            'target_multiplier': 10,           # Target 10x growth (5k → 50k sats)
            'monthly_compound_target': 0.15,   # 15% monthly compounding
            'risk_scaling': 'aggressive',       # Scale risk as capital grows
            'profit_reinvestment': 1.0,        # Reinvest 100% of profits
            'graduation_threshold': 50000,     # Switch to professional at 50k sats
        }
    
    def calculate_growth_trajectory(self):
        """Calculate realistic growth trajectory"""
        current_capital = float(self.capital)
        monthly_rate = self.growth_config['monthly_compound_target']
        target = self.growth_config['graduation_threshold'] * 0.00000001  # Convert sats to BTC
        
        print("🚀 GROWTH TRAJECTORY CALCULATOR")
        print("=" * 40)
        print(f"💰 Starting Capital: {current_capital:.8f} BTC ({int(current_capital * 100000000)} sats)")
        print(f"🎯 Target Capital: {target:.8f} BTC (50k sats)")
        print(f"📈 Monthly Growth Rate: {monthly_rate*100}%")
        
        capital_progression = []
        month = 0
        
        while current_capital < target and month < 24:  # Max 2 years
            month += 1
            current_capital *= (1 + monthly_rate)
            current_sats = int(current_capital * 100000000)
            capital_progression.append((month, current_capital, current_sats))
            
            if month % 3 == 0:  # Every 3 months
                print(f"   Month {month:2d}: {current_capital:.8f} BTC ({current_sats:,} sats)")
        
        final_sats = int(current_capital * 100000000)
        if final_sats >= 50000:
            print(f"\n🎉 TARGET REACHED in {month} months!")
            print(f"🎯 Final Amount: {final_sats:,} sats")
            print(f"📈 Total Growth: {((current_capital/float(self.capital))-1)*100:.1f}%")
        else:
            print(f"\n⏰ After 2 years: {final_sats:,} sats")
            print(f"📈 Total Growth: {((current_capital/float(self.capital))-1)*100:.1f}%")
        
        return capital_progression

# 🎯 GROWTH-FOCUSED MEXC CONFIG
MEXC_GROWTH_CONFIG = {
    'api_key': os.getenv('MEXC_API_KEY', ''),
    'api_secret': os.getenv('MEXC_API_SECRET', ''),
    'exchange': 'mexc',
    'initial_capital': Decimal('0.00005'),  # Your 5k sats
    
    # 🚀 GROWTH-FOCUSED SETTINGS
    'strategy_type': 'growth_aggressive',
    'target_multiplier': 10,               # Grow 10x (5k → 50k sats)
    'monthly_target': 0.15,                # 15% monthly growth
    'risk_tolerance': 'high',              # Higher risk for higher reward
    'compound_frequency': 'immediate',     # Reinvest profits immediately
    
    # ⚡ AGGRESSIVE TRADING PAIRS
    'symbols': [
        'BTC/USDT',   # Bitcoin moves
        'PEPE/USDT',  # Meme coin volatility
        'SHIB/USDT',  # Meme coin momentum  
        'DOGE/USDT',  # Celebrity-driven moves
        'WIF/USDT',   # New meme potential
        'SOL/USDT',   # High momentum altcoin
        'AVAX/USDT',  # Volatile altcoin
        'MATIC/USDT', # News-driven moves
    ],
    
    # 🎯 GROWTH EXECUTION
    'max_trades_per_day': 20,              # Very active trading
    'profit_target_min': 0.05,             # Min 5% profit per trade
    'stop_loss_max': 0.08,                 # Max 8% loss per trade
    'position_size_base': 0.10,            # 10% of capital per trade
    'leverage_max': 1.5,                   # Slight leverage allowed
    
    # 🛡️ GROWTH SAFETY NETS
    'daily_loss_limit': 0.15,              # Stop if down 15% in a day
    'emergency_stop_loss': 0.50,           # Emergency stop at 50% total loss
    'take_profit_quick': True,             # Take profits at 8%+ moves
    'trail_profits': True,                 # Protect profits with trailing stops
    'paper_trading': True,                 # Start with paper trading
}

def create_growth_focused_strategy():
    """Create aggressive growth strategy for small amounts"""
    print("🚀 CREATING GROWTH-FOCUSED STRATEGY")
    print("=" * 45)
    print("💡 THE REALITY: $5 capital needs GROWTH, not preservation!")
    print("🎯 GOAL: Grow 5k sats to 50k+ sats FAST")
    print("⚠️  APPROACH: Higher risk, higher reward")
    
    portfolio = GrowthFocusedPortfolio(MEXC_GROWTH_CONFIG['initial_capital'])
    
    print(f"\n💰 Strategy Allocation:")
    print(f"   • Meme Momentum: 40% (PEPE, SHIB, DOGE moves)")
    print(f"   • News Events: 30% (ETF, listings, announcements)")
    print(f"   • Breakout Scalping: 30% (Quick support/resistance breaks)")
    
    print(f"\n🎯 Growth Targets:")
    print(f"   • Monthly Growth: {MEXC_GROWTH_CONFIG['monthly_target']*100}%")
    print(f"   • Target Multiplier: {MEXC_GROWTH_CONFIG['target_multiplier']}x")
    print(f"   • Max Risk per Trade: 10% (vs 0.5% professional)")
    print(f"   • Profit Target: 5-15% per trade")
    
    print(f"\n🚀 Expected Results:")
    trajectory = portfolio.calculate_growth_trajectory()
    
    print(f"\n✅ GROWTH STRATEGY READY!")
    print(f"🎯 This strategy is designed for your situation:")
    print(f"   • Small capital needs BIG growth")
    print(f"   • Higher risk tolerance ($5 loss is acceptable)")
    print(f"   • Focus on BUILDING capital, not preserving it")
    print(f"   • Switch to professional strategies at 50k+ sats")
    
    return portfolio

if __name__ == "__main__":
    print("🚀 GROWTH-FOCUSED STRATEGY FOR SMALL AMOUNTS")
    print("💡 Perfect for growing 5k sats into meaningful capital")
    print("⚡ Aggressive but SMART approach to building wealth\n")
    
    create_growth_focused_strategy()
