#!/usr/bin/env python3
"""
🔥 AGGRESSIVE MODE TEST - FORCE IMMEDIATE TRADE
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_aggressive_mode():
    """Test aggressive mode with forced trade"""
    print("\n" + "="*60)
    print("🔥 AGGRESSIVE MODE TEST - FORCING IMMEDIATE TRADE")
    print("="*60)
    
    try:
        # Import the bot
        from micro_trading_bot import LegendaryCryptoTitanBot
        
        # Create bot with $5 initial capital
        bot = LegendaryCryptoTitanBot(5.0)
        
        # FORCE AGGRESSIVE MODE
        bot.trading_mode = 'AGGRESSIVE'
        bot.win_rate_optimizer_enabled = False
        bot.min_confidence_for_trade = 0.25
        bot.confidence_threshold = 0.25
        bot.base_confidence_threshold = 0.25
        bot.aggressive_trade_guarantee = True
        bot.aggressive_trade_interval = 10.0  # Trade every 10 seconds
        bot.min_trade_size = 1.0
        bot.cycle_sleep_override = 5.0  # Fast cycles
        
        print("\n⚡ AGGRESSIVE SETTINGS APPLIED:")
        print(f"   • Trading Mode: {bot.trading_mode}")
        print(f"   • Win Rate Optimizer: DISABLED")
        print(f"   • Min Confidence: {bot.min_confidence_for_trade:.0%}")
        print(f"   • Trade Interval: {bot.aggressive_trade_interval}s")
        print(f"   • Min Trade Size: ${bot.min_trade_size}")
        
        # Ensure we have symbols
        bot.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        # Initialize price history with fake data for testing
        print("\n📊 Initializing price data...")
        from collections import deque
        bot.price_history = {
            'BTC/USDT': deque([100000.0, 100050.0, 100100.0], maxlen=100),
            'ETH/USDT': deque([3500.0, 3505.0, 3510.0], maxlen=100),
            'SOL/USDT': deque([100.0, 100.5, 101.0], maxlen=100)
        }
        
        print("   ✅ Price history initialized")
        
        # Start bot
        bot.bot_running = True
        print("\n🚀 STARTING AGGRESSIVE TRADING...")
        print("   ⚡ Expecting trade within 10 seconds...")
        
        # Test signal generation
        print("\n📡 Generating signals...")
        signals = await bot._generate_micro_signals()
        print(f"   Generated {len(signals)} signals")
        
        if signals:
            for s in signals[:3]:
                print(f"   • {s.action} {s.symbol} @ ${s.entry_price:.2f} (Confidence: {s.confidence:.0%})")
        
        # Test trade execution
        print("\n💎 Executing trades...")
        await bot._execute_micro_trades(signals)
        
        # Check if trade was placed
        if hasattr(bot, 'last_trade_ts') and bot.last_trade_ts > 0:
            print("\n✅ SUCCESS! Trade was executed!")
        else:
            print("\n⚠️ No trade executed yet - forcing one now...")
            # Force a trade directly
            from ai_trading_engine import AITradingSignal
            forced_signal = AITradingSignal(
                symbol='BTC/USDT',
                action='BUY',
                confidence=0.35,
                expected_return=1.5,
                risk_score=0.3,
                time_horizon=60,
                entry_price=100000.0,
                stop_loss=99000.0,
                take_profit=101000.0,
                position_size=1.0,
                strategy_name='FORCED_TEST',
                ai_reasoning='Forced test trade',
                technical_score=0.5,
                sentiment_score=0.5,
                momentum_score=0.01,
                volatility_score=0.02
            )
            await bot._execute_micro_trades([forced_signal])
        
        print("\n" + "="*60)
        print("🎯 TEST COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting aggressive mode test...")
    asyncio.run(test_aggressive_mode())
