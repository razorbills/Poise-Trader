#!/usr/bin/env python3
"""
🔥 PROOF THAT TRADES EXECUTE
This script will FORCE a trade and show you it actually happens
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_trade_execution():
    print("\n" + "="*70)
    print("🔥 TESTING IF BOT CAN ACTUALLY EXECUTE TRADES")
    print("="*70)
    
    try:
        # Import bot
        from micro_trading_bot import LegendaryCryptoTitanBot
        
        print("\n1️⃣ Creating bot with $10 capital...")
        bot = LegendaryCryptoTitanBot(10.0)
        
        print("\n2️⃣ Checking initial state...")
        portfolio = await bot.trader.get_portfolio_value()
        print(f"   💰 Initial Cash: ${portfolio['cash']:.2f}")
        print(f"   📊 Initial Total Value: ${portfolio['total_value']:.2f}")
        print(f"   📈 Positions: {len(portfolio['positions'])}")
        
        print("\n3️⃣ Executing TEST BUY trade...")
        print("   Symbol: BTC/USDT")
        print("   Action: BUY")
        print("   Amount: $2.00")
        
        result = await bot.trader.execute_live_trade(
            symbol='BTC/USDT',
            action='BUY',
            amount_usd=2.0,
            strategy='TEST_EXECUTION'
        )
        
        if result.get('success'):
            print("\n   ✅✅✅ TRADE EXECUTED SUCCESSFULLY! ✅✅✅")
        else:
            print(f"\n   ❌ TRADE FAILED: {result.get('error')}")
            return
        
        print("\n4️⃣ Checking portfolio after trade...")
        portfolio = await bot.trader.get_portfolio_value()
        print(f"   💰 Cash After Trade: ${portfolio['cash']:.2f}")
        print(f"   📊 Total Value: ${portfolio['total_value']:.2f}")
        print(f"   📈 Active Positions: {len(portfolio['positions'])}")
        
        if portfolio['positions']:
            print("\n   🎯 Position Details:")
            for symbol, pos in portfolio['positions'].items():
                print(f"      {symbol}:")
                print(f"         Quantity: {pos['quantity']:.8f}")
                print(f"         Value: ${pos['current_value']:.2f}")
                print(f"         P&L: ${pos['unrealized_pnl']:+.2f}")
        
        print("\n5️⃣ Executing TEST SELL trade...")
        result2 = await bot.trader.execute_live_trade(
            symbol='BTC/USDT',
            action='SELL',
            amount_usd=1.0,
            strategy='TEST_SELL'
        )
        
        if result2.get('success'):
            print("\n   ✅✅✅ SELL TRADE EXECUTED! ✅✅✅")
        else:
            print(f"\n   ❌ SELL FAILED: {result2.get('error')}")
        
        print("\n6️⃣ Final portfolio state...")
        portfolio = await bot.trader.get_portfolio_value()
        print(f"   💰 Final Cash: ${portfolio['cash']:.2f}")
        print(f"   📊 Final Total Value: ${portfolio['total_value']:.2f}")
        print(f"   💸 Total P&L: ${portfolio['total_pnl']:+.2f}")
        
        # Check trade history
        if hasattr(bot.trader, 'trade_history'):
            print(f"\n   📜 Trade History ({len(bot.trader.trade_history)} trades):")
            for i, trade in enumerate(bot.trader.trade_history, 1):
                print(f"      {i}. {trade['action']} {trade['symbol']} ${trade['amount_usd']:.2f} @ ${trade['execution_price']:.2f}")
        
        print("\n" + "="*70)
        print("✅ PROOF COMPLETE - BOT CAN EXECUTE TRADES!")
        print("="*70)
        print("\n🎯 The bot IS capable of placing trades!")
        print("   If it's not trading in regular mode, it's likely:")
        print("   1. No signals being generated (filters too strict)")
        print("   2. Win rate optimizer blocking trades")
        print("   3. Confidence thresholds too high")
        print("\n💡 Solution: Use AGGRESSIVE mode to force trades!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_trade_execution())
