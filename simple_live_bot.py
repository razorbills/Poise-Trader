#!/usr/bin/env python3
"""
🔥 SIMPLE LIVE POISE TRADER
Working paper trading bot with REAL live MEXC prices

✅ Uses current 2025 market prices
✅ Paper trading mode (safe)  
✅ Real-time decision making
✅ Actual profit/loss tracking
❌ No more fake 2022 data!
"""

import asyncio
import random
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import our working live data feed
from live_paper_trading_test import LiveMexcDataFeed, LivePaperTradingManager

class SimpleLiveTradingBot:
    """Simplified bot using live MEXC data"""
    
    def __init__(self, initial_capital=5000):
        self.trader = LivePaperTradingManager(initial_capital)
        self.data_feed = LiveMexcDataFeed()
        self.running = False
        self.trade_count = 0
        
        # Simple strategies
        self.symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "ADA/USDT"]
        self.last_prices = {}
        self.position_targets = {}
        
        print("🚀 Simple Live Trading Bot initialized")
        print("📡 Using REAL MEXC market data")
    
    async def get_trading_signal(self, symbol: str, current_price: float) -> dict:
        """Generate trading signals based on simple strategies"""
        
        # Get previous price for comparison
        prev_price = self.last_prices.get(symbol, current_price)
        price_change_pct = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0
        
        # Simple momentum strategy
        signal = {
            'symbol': symbol,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'action': None,
            'amount': 0,
            'confidence': 0,
            'reason': ''
        }
        
        # Random trading for demo (replace with real strategy)
        trade_probability = random.random()
        
        if trade_probability > 0.85:  # 15% chance to trade
            
            if price_change_pct < -2:  # Price dropped 2%+ - BUY opportunity
                signal['action'] = 'BUY'
                signal['amount'] = random.uniform(200, 800)
                signal['confidence'] = 0.7
                signal['reason'] = f'Price drop {price_change_pct:.1f}% - buying opportunity'
                
            elif price_change_pct > 3:  # Price rose 3%+ - SELL some profits
                signal['action'] = 'SELL'  
                signal['amount'] = random.uniform(150, 400)
                signal['confidence'] = 0.6
                signal['reason'] = f'Price rise {price_change_pct:.1f}% - taking profits'
                
            elif random.random() > 0.7:  # Random DCA buy
                signal['action'] = 'BUY'
                signal['amount'] = random.uniform(100, 300)
                signal['confidence'] = 0.5
                signal['reason'] = 'DCA - dollar cost averaging'
        
        return signal
    
    async def execute_signal(self, signal: dict):
        """Execute trading signal"""
        
        if signal['action'] in ['BUY', 'SELL']:
            print(f"\n🎯 SIGNAL DETECTED: {signal['action']} {signal['symbol']}")
            print(f"   💰 Amount: ${signal['amount']:.2f}")
            print(f"   🎲 Confidence: {signal['confidence']:.1%}")
            print(f"   📝 Reason: {signal['reason']}")
            
            # Execute the trade with live prices
            result = await self.trader.execute_live_trade(
                signal['symbol'], 
                signal['action'], 
                signal['amount'],
                'live_bot_v1'
            )
            
            if result['success']:
                self.trade_count += 1
                print(f"✅ Trade #{self.trade_count} executed successfully!")
            else:
                print(f"❌ Trade failed: {result.get('error', 'Unknown error')}")
    
    async def run_trading_loop(self):
        """Main trading loop"""
        
        self.running = True
        print("\n🔄 Starting live trading loop...")
        print("💡 Press Ctrl+C to stop")
        
        cycle = 0
        
        while self.running:
            try:
                cycle += 1
                print(f"\n{'='*60}")
                print(f"🔄 Trading Cycle #{cycle} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Get current live prices for all symbols
                live_prices = await self.data_feed.get_multiple_prices(self.symbols)
                
                if not live_prices:
                    print("⚠️ No price data received, skipping cycle")
                    await asyncio.sleep(30)
                    continue
                
                print("📊 LIVE MARKET PRICES:")
                for symbol, price in live_prices.items():
                    prev_price = self.last_prices.get(symbol, price)
                    change_pct = (price - prev_price) / prev_price * 100 if prev_price != price else 0
                    change_symbol = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                    print(f"   {change_symbol} {symbol:10} ${price:>10,.2f} ({change_pct:+5.2f}%)")
                
                # Generate and execute signals for each symbol
                for symbol, current_price in live_prices.items():
                    signal = await self.get_trading_signal(symbol, current_price)
                    
                    if signal['action']:
                        await self.execute_signal(signal)
                
                # Update price history
                self.last_prices.update(live_prices)
                
                # Show portfolio status every 5 cycles
                if cycle % 5 == 0:
                    await self.show_portfolio_status()
                
                # Wait before next cycle
                print(f"⏰ Waiting 60 seconds for next cycle...")
                await asyncio.sleep(60)  # 1 minute between cycles
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping bot...")
                self.running = False
                break
                
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                print("🔄 Continuing in 30 seconds...")
                await asyncio.sleep(30)
    
    async def show_portfolio_status(self):
        """Show current portfolio performance"""
        
        print("\n" + "="*60)
        print("📊 PORTFOLIO STATUS (with LIVE prices)")
        
        portfolio = await self.trader.get_portfolio_value()
        
        print(f"💰 Total Value: ${portfolio['total_value']:,.2f}")
        print(f"💵 Cash: ${portfolio['cash']:,.2f}")
        print(f"📈 Total Return: {portfolio['total_return']*100:+.2f}%")
        print(f"💎 P&L: ${portfolio['total_pnl']:+,.2f}")
        print(f"🔄 Total Trades: {self.trade_count}")
        
        if portfolio['positions']:
            print(f"\n🎯 ACTIVE POSITIONS:")
            for symbol, pos in portfolio['positions'].items():
                pnl_pct = (pos['unrealized_pnl'] / pos['cost_basis']) * 100 if pos['cost_basis'] > 0 else 0
                emoji = "💚" if pos['unrealized_pnl'] > 0 else "❤️" if pos['unrealized_pnl'] < 0 else "💛"
                print(f"   {emoji} {symbol:10} {pos['quantity']:>8.4f} @ ${pos['current_price']:>8,.2f} = ${pos['current_value']:>8,.2f} (PnL: ${pos['unrealized_pnl']:+7.2f})")
        
        print("="*60)

async def main():
    """Main function"""
    
    print("🚀 SIMPLE LIVE POISE TRADER")
    print("=" * 60)
    print("🔥 Features:")
    print("   ✅ Live MEXC market data")
    print("   ✅ Real-time trading decisions") 
    print("   ✅ Paper trading (safe mode)")
    print("   ✅ Current 2025 prices")
    print("   ❌ No fake data!")
    print()
    
    # Create and run bot
    bot = SimpleLiveTradingBot(5000)
    
    try:
        # Show initial portfolio
        await bot.show_portfolio_status()
        
        # Run trading loop
        await bot.run_trading_loop()
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
        
    except Exception as e:
        print(f"💥 Critical error: {e}")
        
    finally:
        # Show final results
        print("\n🏁 FINAL RESULTS:")
        await bot.show_portfolio_status()

if __name__ == "__main__":
    asyncio.run(main())
