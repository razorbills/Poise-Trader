#!/usr/bin/env python3
"""
📊 REAL-TIME PORTFOLIO TRACKER
Monitor your AI trading bot's performance in real-time
"""

import asyncio
from live_paper_trading_test import LivePaperTradingManager
from datetime import datetime
import os

async def track_portfolio():
    """Real-time portfolio tracking"""
    
    print("📊 POISE TRADER - REAL-TIME PORTFOLIO TRACKER")
    print("=" * 60)
    print("🤖 Monitoring AI trading bot performance...")
    print("💡 Press Ctrl+C to stop tracking")
    print()
    
    # Initialize portfolio manager
    trader = LivePaperTradingManager(5000)
    
    # Track performance over time
    cycle = 0
    
    while True:
        try:
            cycle += 1
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Clear screen (optional - comment out if you want history)
            # os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"\n📊 PORTFOLIO UPDATE #{cycle} - {current_time}")
            print("-" * 50)
            
            # Get portfolio status
            portfolio = await trader.get_portfolio_value()
            
            # Portfolio summary
            total_return_pct = portfolio['total_return'] * 100
            pnl_color = "🟢" if portfolio['total_pnl'] >= 0 else "🔴"
            return_color = "📈" if total_return_pct >= 0 else "📉"
            
            print(f"💰 Total Portfolio: ${portfolio['total_value']:,.2f}")
            print(f"💵 Cash Available:  ${portfolio['cash']:,.2f}")
            print(f"{return_color} Total Return:    {total_return_pct:+.2f}%")
            print(f"{pnl_color} Unrealized P&L:  ${portfolio['total_pnl']:+,.2f}")
            
            # Active positions
            active_positions = {k:v for k,v in portfolio['positions'].items() if v.get('quantity', 0) > 0}
            print(f"🎯 Active Positions: {len(active_positions)}")
            
            if active_positions:
                print("\n🏦 POSITION DETAILS:")
                for symbol, pos in active_positions.items():
                    quantity = pos['quantity']
                    current_value = pos['current_value']
                    cost_basis = pos['cost_basis']
                    unrealized_pnl = pos['unrealized_pnl']
                    pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
                    
                    # Position status
                    status_emoji = "💚" if unrealized_pnl > 0 else "❤️" if unrealized_pnl < 0 else "💛"
                    
                    print(f"  {status_emoji} {symbol:12}")
                    print(f"     💰 Quantity:     {quantity:.6f}")
                    print(f"     💵 Value:        ${current_value:.2f}")
                    print(f"     📊 Cost Basis:   ${cost_basis:.2f}")
                    print(f"     🎯 P&L:          ${unrealized_pnl:+.2f} ({pnl_pct:+.2f}%)")
                    print()
            
            # Performance metrics
            initial_capital = 5000
            roi = ((portfolio['total_value'] - initial_capital) / initial_capital) * 100
            
            print("📈 PERFORMANCE METRICS:")
            print(f"   🎯 ROI:              {roi:+.2f}%")
            print(f"   💎 Equity:           ${portfolio['total_value'] - portfolio['cash']:.2f}")
            print(f"   🏦 Cash Ratio:       {(portfolio['cash']/portfolio['total_value']*100):.1f}%")
            
            # Next update countdown
            print(f"\n🔄 Next update in 10 seconds... (Cycle #{cycle})")
            
            await asyncio.sleep(10)
            
        except KeyboardInterrupt:
            print("\n\n👋 Portfolio tracking stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            await asyncio.sleep(5)
    
    # Final summary
    print("\n🏁 FINAL PORTFOLIO SUMMARY:")
    print("=" * 40)
    final_portfolio = await trader.get_portfolio_value()
    final_return = ((final_portfolio['total_value'] - 5000) / 5000) * 100
    
    print(f"💰 Final Value:     ${final_portfolio['total_value']:,.2f}")
    print(f"💵 Cash:            ${final_portfolio['cash']:,.2f}")
    print(f"📈 Total Return:    {final_return:+.2f}%")
    print(f"💎 Total P&L:       ${final_portfolio['total_pnl']:+,.2f}")
    print()
    print("🤖 AI Trading System Performance Summary Complete!")

if __name__ == "__main__":
    print("📊 Starting real-time portfolio tracker...")
    asyncio.run(track_portfolio())
