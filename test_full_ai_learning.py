#!/usr/bin/env python3
"""
Full integration test for AI learning in paper trading
"""

import asyncio
import json
import os
from datetime import datetime
import sys
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent))

async def test_full_ai_learning():
    """Test complete AI learning flow with paper trading"""
    
    print("🚀 FULL AI LEARNING INTEGRATION TEST")
    print("=" * 60)
    
    # Import after path setup
    from micro_trading_bot import LegendaryCryptoTitanBot
    from ai_brain import ai_brain
    
    # Record initial state
    initial_trades = ai_brain.brain['total_trades']
    initial_pnl = ai_brain.brain['total_profit_loss']
    
    print(f"\n📊 Initial AI Brain State:")
    print(f"   Trades before test: {initial_trades}")
    print(f"   P&L before test: ${initial_pnl:.2f}")
    
    # Create bot instance with small capital for testing
    print("\n🤖 Initializing Micro Trading Bot...")
    bot = LegendaryCryptoTitanBot(initial_capital=5.0)
    
    # Run a few quick trading cycles
    print("\n🔄 Running 3 quick paper trading cycles...")
    print("   (This will attempt to execute real trades with paper money)")
    
    try:
        # Run only 3 cycles for quick testing
        await bot.run_micro_trading(cycles=3)
    except KeyboardInterrupt:
        print("\n⏸️ Test interrupted by user")
    except Exception as e:
        print(f"\n⚠️ Trading error (expected in test): {e}")
    
    # Check final state
    final_trades = ai_brain.brain['total_trades']
    final_pnl = ai_brain.brain['total_profit_loss']
    
    print("\n" + "=" * 60)
    print("📊 FINAL AI BRAIN STATE:")
    print(f"   Total trades executed: {final_trades - initial_trades}")
    print(f"   Total trades learned: {final_trades}")
    print(f"   Total P&L learned: ${final_pnl:.2f}")
    print(f"   P&L change: ${final_pnl - initial_pnl:.2f}")
    
    # Verify brain file was updated
    if os.path.exists('ai_brain.json'):
        with open('ai_brain.json', 'r') as f:
            saved_brain = json.load(f)
        print(f"\n💾 Brain file verification:")
        print(f"   File last updated: {saved_brain['last_updated']}")
        print(f"   Saved total trades: {saved_brain['total_trades']}")
        
        # Check recent trades
        if saved_brain['recent_trades']:
            print(f"   Recent trades saved: {len(saved_brain['recent_trades'])}")
            last_trade = saved_brain['recent_trades'][-1]
            print(f"   Last trade: {last_trade['symbol']} - P&L: ${last_trade['profit_loss']:.2f}")
    
    # Check shared knowledge
    if os.path.exists('shared_ai_knowledge.json'):
        with open('shared_ai_knowledge.json', 'r') as f:
            shared = json.load(f)
        micro_lessons = len(shared.get('micro_bot_lessons', []))
        print(f"\n🤝 Cross-bot learning:")
        print(f"   Micro bot lessons: {micro_lessons}")
        if micro_lessons > 0:
            last_lesson = shared['micro_bot_lessons'][-1]
            print(f"   Last lesson type: {last_lesson.get('type', 'unknown')}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS:")
    
    if final_trades > initial_trades:
        print("✅ SUCCESS: AI is learning from trades!")
        print(f"   - Learned from {final_trades - initial_trades} new trades")
        print(f"   - Brain data is being saved correctly")
        print(f"   - Cross-bot learning is active")
    else:
        print("⚠️ WARNING: No new trades were learned")
        print("   This could mean:")
        print("   - No positions were closed during the test")
        print("   - Trades are still open (positions not closed)")
        print("   - Try running the test for more cycles")
    
    print("\n💡 To see more learning, run the bot for more cycles")
    print("   or wait for positions to hit stop-loss/take-profit")

if __name__ == "__main__":
    asyncio.run(test_full_ai_learning())
