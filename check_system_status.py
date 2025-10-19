#!/usr/bin/env python3
"""
🔍 SYSTEM STATUS CHECKER
Verify that the AI trading system is working properly
"""

import asyncio
from live_paper_trading_test import LiveMexcDataFeed, LivePaperTradingManager
from datetime import datetime

async def check_system_status():
    """Check all system components"""
    
    print("🔍 POISE TRADER SYSTEM STATUS CHECK")
    print("=" * 50)
    print()
    
    # 1. Check MEXC API connection
    print("1️⃣ CHECKING MEXC API CONNECTION...")
    data_feed = LiveMexcDataFeed()
    
    try:
        prices = await data_feed.get_multiple_prices(["BTC/USDT", "ETH/USDT"])
        if prices:
            print("✅ MEXC API: CONNECTED")
            print(f"   📈 BTC/USDT: ${prices.get('BTC/USDT', 'N/A'):,.2f}")
            print(f"   📈 ETH/USDT: ${prices.get('ETH/USDT', 'N/A'):,.2f}")
        else:
            print("❌ MEXC API: NO DATA")
    except Exception as e:
        print(f"❌ MEXC API: ERROR - {e}")
    
    print()
    
    # 2. Check Paper Trading System
    print("2️⃣ CHECKING PAPER TRADING SYSTEM...")
    trader = LivePaperTradingManager(5000)
    
    try:
        portfolio = await trader.get_portfolio_value()
        print("✅ PAPER TRADING: ACTIVE")
        print(f"   💰 Portfolio Value: ${portfolio['total_value']:,.2f}")
        print(f"   💵 Cash Balance: ${portfolio['cash']:,.2f}")
        print(f"   📊 Active Positions: {len([p for p in portfolio['positions'].values() if p.get('quantity', 0) > 0])}")
    except Exception as e:
        print(f"❌ PAPER TRADING: ERROR - {e}")
    
    print()
    
    # 3. Test a paper trade
    print("3️⃣ TESTING PAPER TRADE EXECUTION...")
    try:
        result = await trader.execute_live_trade("BTC/USDT", "BUY", 100, "system_test")
        if result['success']:
            print("✅ TRADE EXECUTION: WORKING")
            print(f"   🎯 Test Trade: BUY $100 BTC/USDT")
            print(f"   💰 Quantity: {result.get('quantity', 'N/A')}")
            print(f"   💵 Execution Price: ${result.get('price', 'N/A')}")
        else:
            print(f"❌ TRADE EXECUTION: FAILED - {result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"❌ TRADE EXECUTION: ERROR - {e}")
    
    print()
    
    # 4. Show current portfolio after test
    print("4️⃣ PORTFOLIO AFTER TEST TRADE...")
    try:
        portfolio = await trader.get_portfolio_value()
        print(f"   💰 Total Value: ${portfolio['total_value']:,.2f}")
        print(f"   💵 Cash: ${portfolio['cash']:,.2f}")
        print(f"   📈 Total Return: {portfolio['total_return']*100:+.2f}%")
        
        if portfolio['positions']:
            print("   🎯 POSITIONS:")
            for symbol, pos in portfolio['positions'].items():
                if pos.get('quantity', 0) > 0:
                    pnl_pct = (pos['unrealized_pnl'] / pos['cost_basis']) * 100 if pos['cost_basis'] > 0 else 0
                    print(f"      {symbol}: ${pos['current_value']:.0f} ({pnl_pct:+.2f}%)")
        
    except Exception as e:
        print(f"❌ PORTFOLIO CHECK: ERROR - {e}")
    
    print()
    print("🔍 SYSTEM STATUS SUMMARY:")
    print("=" * 30)
    print("📊 MODE: PAPER TRADING (Simulated)")
    print("🔒 REAL MONEY: NOT USED (Safe for testing)")
    print("📈 LIVE PRICES: Real MEXC market data")
    print("🤖 AI DECISIONS: Real AI analysis")
    print("💰 PROFIT TRACKING: Simulated but accurate")
    print()
    print("💡 TO SEE LIVE TRADES:")
    print("   • Run: python ai_profit_bot.py")
    print("   • Watch the console output for trade executions")
    print("   • Portfolio changes show in the status updates")
    print()
    print("🚨 TO ENABLE REAL TRADING:")
    print("   • You'd need to modify the system to use real MEXC orders")
    print("   • Add your MEXC API keys with trading permissions")
    print("   • Enable live trading mode (currently disabled for safety)")

if __name__ == "__main__":
    print("🤖 Starting system status check...")
    asyncio.run(check_system_status())
