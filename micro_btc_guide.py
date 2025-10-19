#!/usr/bin/env python3
"""
Micro BTC Trading Guide - Starting with 0.00005 BTC

Perfect for beginners with small amounts!
Learn trading with minimal risk while growing your stack.
"""

import asyncio
import logging
from decimal import Decimal
from core.strategies import StrategyFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MicroBTC")


async def micro_btc_demo():
    """Demo trading with very small BTC amounts"""
    print("🔥 MICRO BTC TRADER - Start Small, Dream Big!")
    print("=" * 55)
    
    # Your actual BTC amount
    your_btc = Decimal('0.00005')  # 5,000 sats (~$2.50)
    btc_usd_price = 50000  # Assume $50k BTC for calculations
    usd_value = float(your_btc) * btc_usd_price
    
    print(f"💰 Your Bitcoin: {your_btc} BTC")
    print(f"💵 USD Value: ~${usd_value:.2f}")
    print(f"⚡ Satoshis: {int(your_btc * 100_000_000):,} sats")
    print(f"")
    
    # Strategy for micro amounts
    print("🎯 MICRO TRADING STRATEGY:")
    print("Since you have a small amount, we'll focus on:")
    print("✅ Learning the system with minimal risk")
    print("✅ Growing your BTC through smart micro-trades") 
    print("✅ Building experience for when you have more BTC")
    print("")
    
    # Micro DCA Strategy - trade tiny amounts frequently
    print("🚀 Setting up Micro DCA Strategy...")
    
    micro_config = {
        'initial_capital': float(your_btc),   # All your BTC
        'base_currency': 'BTC',              # BTC base
        'symbols': ['ETH/BTC'],              # Just ETH to start
        'buy_interval': 1800,                # Every 30 minutes (fast for demo)
        'base_buy_amount': 0.00001,          # 1,000 sats per buy
        'max_allocation_per_symbol': 0.9,    # Use 90% max (keep some reserve)
        'price_drop_threshold': 0.02,        # Buy extra on 2% drops
        'profit_taking_threshold': 0.05,     # Take profits at 5% (smaller targets)
        'profit_taking_percentage': 0.5      # Take 50% profits
    }
    
    strategy = StrategyFactory.create_strategy('dca', micro_config)
    await strategy.initialize()
    
    print("✅ Micro strategy initialized!")
    print(f"   Capital: {strategy.portfolio.initial_capital} BTC ({int(strategy.portfolio.initial_capital * 100_000_000)} sats)")
    print(f"   Buy Size: {strategy.base_buy_amount} BTC ({int(strategy.base_buy_amount * 100_000_000)} sats)")
    print(f"   Target: {strategy.symbols[0]}")
    print("")
    
    # Show what each trade costs
    sat_per_trade = int(strategy.base_buy_amount * 100_000_000)
    usd_per_trade = strategy.base_buy_amount * btc_usd_price
    print(f"📊 Each trade: {sat_per_trade} sats (~${usd_per_trade:.3f})")
    
    max_trades = int(your_btc / strategy.base_buy_amount)
    print(f"📈 Max possible trades: ~{max_trades} trades with current balance")
    print("")
    
    # Simulate some trading
    print("💡 MICRO TRADING SIMULATION:")
    print("Let's see how your small BTC can grow...")
    print("")
    
    from core.framework.base_classes import MarketData
    import time
    
    # Simulate ETH/BTC price movements (realistic micro movements)
    eth_btc_prices = [
        Decimal('0.065000'),  # Start
        Decimal('0.064700'),  # Small drop - might trigger buy
        Decimal('0.065200'),  # Small recovery  
        Decimal('0.065800'),  # Small gain
        Decimal('0.068250'),  # 5% gain - should trigger profit taking
    ]
    
    for i, price in enumerate(eth_btc_prices):
        print(f"📊 Update {i+1}: ETH/BTC = {price}")
        
        market_data = MarketData(
            symbol='ETH/BTC',
            timestamp=int(time.time() * 1000),
            price=price,
            volume=Decimal('100'),
            exchange='demo'
        )
        
        signal = await strategy.process_market_data(market_data)
        
        if signal:
            signal_sats = int(signal.quantity * signal.price * 100_000_000)
            print(f"🎯 Signal: {signal.action.value.upper()}")
            print(f"   Amount: {signal.quantity:.8f} ETH")
            print(f"   Cost: {signal.quantity * signal.price:.8f} BTC ({signal_sats} sats)")
            
            # Execute the trade
            if signal.action.value == 'buy':
                from core.strategies.base_strategy import PositionType
                success = strategy.portfolio.open_position(
                    signal.symbol, PositionType.LONG, signal.quantity, signal.price
                )
                print(f"   {'✅ Buy executed' if success else '❌ Buy failed'}")
            else:
                pnl = strategy.portfolio.close_position(signal.symbol, signal.price)
                if pnl is not None:
                    pnl_sats = int(pnl * 100_000_000)
                    print(f"   ✅ Sell executed - P&L: {pnl:.8f} BTC ({pnl_sats:+d} sats)")
        else:
            print("   No signal generated")
        
        # Show portfolio
        portfolio_btc = strategy.portfolio.get_portfolio_value()
        portfolio_sats = int(portfolio_btc * 100_000_000)
        print(f"📋 Portfolio: {portfolio_btc:.8f} BTC ({portfolio_sats:,} sats)")
        
        await asyncio.sleep(0.5)
    
    print("\n" + "=" * 55)
    print("📊 MICRO TRADING RESULTS:")
    
    metrics = strategy.get_performance_metrics()
    final_btc = Decimal(str(metrics['current_value']))
    final_sats = int(final_btc * 100_000_000)
    initial_sats = int(your_btc * 100_000_000)
    
    print(f"   Initial: {your_btc} BTC ({initial_sats:,} sats)")
    print(f"   Final: {final_btc:.8f} BTC ({final_sats:,} sats)")
    print(f"   Return: {metrics['total_return_pct']:.2f}%")
    print(f"   Trades: {metrics['total_trades']}")
    
    sat_difference = final_sats - initial_sats
    if sat_difference > 0:
        print(f"   Profit: +{sat_difference:,} sats! 🚀")
    else:
        print(f"   Loss: {sat_difference:,} sats")
    
    # Growth projection
    print(f"\n💡 GROWTH PROJECTION:")
    if metrics['total_return_pct'] > 0:
        monthly_return = metrics['total_return_pct'] * 30  # Assuming daily gains
        print(f"   If this return continues daily for 30 days:")
        projected_btc = final_btc * (1 + monthly_return/100)
        projected_sats = int(projected_btc * 100_000_000)
        print(f"   Projected: {projected_btc:.8f} BTC ({projected_sats:,} sats)")
        print(f"   That's {projected_sats - initial_sats:,} sats profit potential!")


async def micro_btc_strategies():
    """Show different strategies for micro BTC amounts"""
    print("\n" + "=" * 55)
    print("🧠 MICRO BTC STRATEGIES")
    print("=" * 55)
    
    your_btc = Decimal('0.00005')
    
    print(f"With {your_btc} BTC, here are your options:\n")
    
    print("🎯 STRATEGY 1: MICRO DCA (Recommended)")
    print(f"   • Use all {your_btc} BTC")
    print("   • Buy ETH every 30 minutes") 
    print("   • 1,000 sats per buy (0.00001 BTC)")
    print("   • Take profits at 5% gains")
    print("   • Risk: LOW, Learning: HIGH\n")
    
    print("🎯 STRATEGY 2: ULTRA-MICRO SCALPING")
    print(f"   • Use {your_btc} BTC") 
    print("   • Trade every 5 minutes")
    print("   • 200 sats per trade (0.000002 BTC)")
    print("   • Take profits at 1-2% gains")
    print("   • Risk: MEDIUM, Learning: HIGH\n")
    
    print("🎯 STRATEGY 3: HODL + MICRO GROWTH")
    print(f"   • Keep 3,000 sats as HODL reserve")
    print(f"   • Trade with 2,000 sats")
    print("   • Conservative 1 trade per hour")
    print("   • Focus on learning, not profit")
    print("   • Risk: VERY LOW, Learning: MEDIUM\n")
    
    print("💡 RECOMMENDED APPROACH:")
    print("Start with Strategy 1 (Micro DCA) because:")
    print("✅ Learn the system with minimal risk")
    print("✅ See how strategies work in practice") 
    print("✅ Build confidence for larger amounts later")
    print("✅ Even small gains compound over time")
    print("")
    
    # Show the math
    print("🔢 THE MICRO MATH:")
    initial_sats = 5000
    print(f"Starting: {initial_sats:,} sats")
    
    scenarios = [
        ("Conservative", 0.5, "1% daily growth"),
        ("Moderate", 1.0, "2% daily growth"), 
        ("Optimistic", 2.0, "4% daily growth")
    ]
    
    for scenario, daily_pct, description in scenarios:
        monthly_growth = (1 + daily_pct/100) ** 30
        final_sats = int(initial_sats * monthly_growth)
        profit_sats = final_sats - initial_sats
        
        print(f"   {scenario}: {description}")
        print(f"     After 30 days: {final_sats:,} sats (+{profit_sats:,} sats)")
        print(f"     Total growth: {((monthly_growth - 1) * 100):.1f}%")


async def micro_btc_tips():
    """Tips for trading with micro BTC amounts"""
    print("\n" + "=" * 55)
    print("💎 MICRO BTC TRADING TIPS")
    print("=" * 55)
    
    print("🔥 MAKING THE MOST OF TINY AMOUNTS:\n")
    
    print("1. 🎯 THINK IN SATOSHIS")
    print("   • 0.00005 BTC = 5,000 sats")
    print("   • Trade 500-1,000 sats at a time")
    print("   • Every sat counts - they add up!\n")
    
    print("2. ⚡ TRADE FREQUENTLY") 
    print("   • Small amounts = can afford more trades")
    print("   • Learn from each trade")
    print("   • Compound small gains\n")
    
    print("3. 📈 FOCUS ON LEARNING")
    print("   • Perfect strategy with small risk")
    print("   • Understand market patterns")
    print("   • Build trading discipline\n")
    
    print("4. 🚀 SCALE UP GRADUALLY")
    print("   • Add more BTC as you learn")
    print("   • strategy.add_funds(Decimal('0.0001'))  # Add more")
    print("   • Successful small trades → bigger trades\n")
    
    print("5. 💡 REALISTIC EXPECTATIONS")
    print("   • Goal: Learn + small growth")
    print("   • Don't expect to get rich overnight")
    print("   • Perfect your skills for when you have more BTC\n")
    
    print("🎮 HOW TO START RIGHT NOW:")
    print("```python")
    print("# Your micro BTC config")
    print("config = {")
    print("    'initial_capital': 0.00005,    # Your 5,000 sats")
    print("    'base_currency': 'BTC',")
    print("    'symbols': ['ETH/BTC'],")  
    print("    'buy_interval': 1800,          # 30 minutes")
    print("    'base_buy_amount': 0.00001     # 1,000 sats per trade")
    print("}")
    print("")
    print("strategy = StrategyFactory.create_strategy('dca', config)")
    print("await strategy.initialize()")
    print("```")
    print("")
    print("🚀 Remember: Every whale started as a shrimp!")
    print("Your 5,000 sats today could be 50,000 sats tomorrow!")


if __name__ == "__main__":
    print("🔥 Starting Micro BTC Trading Guide...")
    try:
        asyncio.run(micro_btc_demo())
        asyncio.run(micro_btc_strategies())
        asyncio.run(micro_btc_tips())
        print("\n✅ Guide completed!")
        print("🎯 Ready to start growing your micro BTC stack!")
        print("🚀 Remember: Small amounts, big learning!")
    except KeyboardInterrupt:
        print("\n⏹️ Guide interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Guide failed")
