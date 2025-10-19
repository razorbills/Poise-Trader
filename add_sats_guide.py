#!/usr/bin/env python3
"""
How to Add Sats to Your Poise Trader Strategy

Complete guide for adding more Bitcoin (satoshis) to your trading strategy
at any time - whether it's 1,000 sats or 1,000,000 sats!
"""

import asyncio
import logging
from decimal import Decimal
from core.strategies import StrategyFactory

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("AddSats")


async def add_sats_demo():
    """Demo showing exactly how to add sats to your strategy"""
    print("⚡ HOW TO ADD SATS TO YOUR STRATEGY")
    print("=" * 50)
    
    # Start with your current amount
    current_sats = 5000  # Your current 5,000 sats
    current_btc = Decimal(str(current_sats / 100_000_000))
    
    print(f"💰 Current Holdings: {current_sats:,} sats ({current_btc} BTC)")
    print()
    
    # Create initial strategy
    print("🚀 Setting up initial strategy...")
    config = {
        'initial_capital': float(current_btc),
        'base_currency': 'BTC',
        'symbols': ['ETH/BTC'],
        'buy_interval': 1800,
        'base_buy_amount': 0.00001  # 1,000 sats per trade
    }
    
    strategy = StrategyFactory.create_strategy('dca', config)
    await strategy.initialize()
    
    print(f"✅ Strategy started with {current_sats:,} sats")
    print(f"   Available: {int(strategy.portfolio.get_available_balance('BTC') * 100_000_000):,} sats")
    print()
    
    # Method 1: Add small amounts (common case)
    print("📈 METHOD 1: Adding Small Amounts")
    print("-" * 30)
    
    additional_amounts = [2000, 3000, 5000, 10000]  # Different sat amounts
    
    for sats_to_add in additional_amounts:
        btc_to_add = Decimal(str(sats_to_add / 100_000_000))
        
        print(f"💰 Adding {sats_to_add:,} sats ({btc_to_add} BTC)...")
        
        # This is the key line - how you add sats!
        strategy.add_funds(btc_to_add)
        
        new_total_btc = strategy.portfolio.get_portfolio_value()
        new_total_sats = int(new_total_btc * 100_000_000)
        
        print(f"   ✅ Added successfully!")
        print(f"   📊 New Total: {new_total_sats:,} sats ({new_total_btc:.8f} BTC)")
        
        # Show what this means for trading
        available_btc = strategy.portfolio.get_available_balance('BTC')
        available_sats = int(available_btc * 100_000_000)
        trade_size_sats = int(strategy.base_buy_amount * 100_000_000)
        possible_trades = available_sats // trade_size_sats
        
        print(f"   💡 Available for trading: {available_sats:,} sats")
        print(f"   🎯 Possible trades: ~{possible_trades} trades ({trade_size_sats} sats each)")
        print()
        
        await asyncio.sleep(0.5)  # Pause for readability


async def add_sats_methods():
    """Show different ways to add sats"""
    print("🔧 DIFFERENT WAYS TO ADD SATS")
    print("=" * 50)
    
    # Create example strategy
    strategy = StrategyFactory.create_strategy('dca', {
        'initial_capital': 0.00005,
        'base_currency': 'BTC',
        'symbols': ['ETH/BTC']
    })
    await strategy.initialize()
    
    print("Here are all the ways you can add sats:\n")
    
    print("✅ METHOD 1: Add by BTC amount")
    print("```python")
    print("# Add 0.00001 BTC (1,000 sats)")
    print("strategy.add_funds(Decimal('0.00001'))")
    print("```")
    print()
    
    print("✅ METHOD 2: Convert sats to BTC first")
    print("```python")
    print("# Add 5,000 sats")
    print("sats_to_add = 5000")
    print("btc_amount = Decimal(str(sats_to_add / 100_000_000))")
    print("strategy.add_funds(btc_amount)")
    print("```")
    print()
    
    print("✅ METHOD 3: Use a helper function")
    print("```python")
    print("def add_sats_to_strategy(strategy, sats):")
    print("    btc_amount = Decimal(str(sats / 100_000_000))")
    print("    strategy.add_funds(btc_amount)")
    print("    return int(strategy.portfolio.get_portfolio_value() * 100_000_000)")
    print("")
    print("# Usage:")
    print("new_total = add_sats_to_strategy(strategy, 10000)  # Add 10k sats")
    print("print(f'New total: {new_total:,} sats')")
    print("```")
    print()
    
    # Demo the helper function
    def add_sats_to_strategy(strategy, sats):
        """Helper function to add sats easily"""
        btc_amount = Decimal(str(sats / 100_000_000))
        old_total = int(strategy.portfolio.get_portfolio_value() * 100_000_000)
        strategy.add_funds(btc_amount)
        new_total = int(strategy.portfolio.get_portfolio_value() * 100_000_000)
        added = new_total - old_total
        print(f"📈 Added {added:,} sats! Total now: {new_total:,} sats")
        return new_total
    
    print("🎮 TESTING THE HELPER FUNCTION:")
    add_sats_to_strategy(strategy, 3000)   # Add 3,000 sats
    add_sats_to_strategy(strategy, 7500)   # Add 7,500 sats  
    add_sats_to_strategy(strategy, 15000)  # Add 15,000 sats


async def practical_examples():
    """Show practical real-world examples"""
    print("\n" + "=" * 50)
    print("🌟 PRACTICAL EXAMPLES")
    print("=" * 50)
    
    print("Real scenarios when you'd add more sats:\n")
    
    scenarios = [
        {
            'name': 'Weekly DCA',
            'description': 'Add 10,000 sats every week',
            'code': 'strategy.add_funds(Decimal("0.0001"))  # 10k sats',
            'sats': 10000
        },
        {
            'name': 'Profit Reinvestment', 
            'description': 'Strategy made profit, reinvest 50%',
            'code': 'profit_sats = 5000\nstrategy.add_funds(Decimal(str(profit_sats / 100_000_000)))',
            'sats': 5000
        },
        {
            'name': 'Bonus Addition',
            'description': 'Got some extra BTC, add it all',
            'code': 'bonus_btc = 0.0002  # 20k sats\nstrategy.add_funds(Decimal(str(bonus_btc)))',
            'sats': 20000
        },
        {
            'name': 'Emergency Boost',
            'description': 'Market dip, add more to buy the dip',
            'code': 'dip_sats = 25000\nstrategy.add_funds(Decimal(str(dip_sats / 100_000_000)))',
            'sats': 25000
        }
    ]
    
    # Create demo strategy
    strategy = StrategyFactory.create_strategy('dca', {
        'initial_capital': 0.00005,  # Start with 5k sats
        'base_currency': 'BTC',
        'symbols': ['ETH/BTC']
    })
    await strategy.initialize()
    
    initial_sats = int(strategy.portfolio.get_portfolio_value() * 100_000_000)
    print(f"🏁 Starting with: {initial_sats:,} sats\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"📈 SCENARIO {i}: {scenario['name']}")
        print(f"   💡 {scenario['description']}")
        print(f"   📝 Code: {scenario['code']}")
        
        # Execute the addition
        sats_to_add = scenario['sats']
        btc_to_add = Decimal(str(sats_to_add / 100_000_000))
        strategy.add_funds(btc_to_add)
        
        new_total = int(strategy.portfolio.get_portfolio_value() * 100_000_000)
        print(f"   ✅ Result: +{sats_to_add:,} sats → Total: {new_total:,} sats")
        print()


async def scaling_examples():
    """Show how your strategy scales as you add more"""
    print("🚀 SCALING YOUR STRATEGY")
    print("=" * 50)
    
    print("See how your trading power grows as you add sats:\n")
    
    # Start small
    strategy = StrategyFactory.create_strategy('dca', {
        'initial_capital': 0.00005,  # 5k sats
        'base_currency': 'BTC',
        'symbols': ['ETH/BTC'], 
        'buy_interval': 1800,
        'base_buy_amount': 0.00001  # 1k sats per trade
    })
    await strategy.initialize()
    
    def show_trading_power(strategy, stage_name):
        total_btc = strategy.portfolio.get_portfolio_value()
        total_sats = int(total_btc * 100_000_000)
        available_btc = strategy.portfolio.get_available_balance('BTC')
        available_sats = int(available_btc * 100_000_000)
        
        trade_size_sats = int(strategy.base_buy_amount * 100_000_000)
        possible_trades = available_sats // trade_size_sats if trade_size_sats > 0 else 0
        
        print(f"📊 {stage_name}:")
        print(f"   💰 Total: {total_sats:,} sats")
        print(f"   ⚡ Available: {available_sats:,} sats") 
        print(f"   🎯 Possible trades: {possible_trades} ({trade_size_sats:,} sats each)")
        print()
    
    show_trading_power(strategy, "STAGE 1: Starting Small")
    
    # Add 10k sats
    strategy.add_funds(Decimal('0.0001'))
    show_trading_power(strategy, "STAGE 2: After adding 10k sats")
    
    # Add 50k sats  
    strategy.add_funds(Decimal('0.0005'))
    show_trading_power(strategy, "STAGE 3: After adding 50k more sats")
    
    # Add 100k sats
    strategy.add_funds(Decimal('0.001'))
    show_trading_power(strategy, "STAGE 4: After adding 100k more sats")
    
    print("💡 NOTICE HOW:")
    print("✅ More sats = More trading opportunities")
    print("✅ Same strategy, just scales up automatically")
    print("✅ You can add ANY amount at ANY time")
    print("✅ Strategy immediately uses new funds")


async def add_sats_checklist():
    """Provide a simple checklist for adding sats"""
    print("\n" + "=" * 50)
    print("📋 QUICK ADD SATS CHECKLIST")
    print("=" * 50)
    
    print("Follow these steps every time you want to add sats:\n")
    
    checklist = [
        "1. 🧮 Decide how many sats to add",
        "2. 🔢 Convert sats to BTC: sats ÷ 100,000,000",  
        "3. 💰 Add to strategy: strategy.add_funds(Decimal('amount'))",
        "4. ✅ Check new balance: strategy.portfolio.get_portfolio_value()",
        "5. 📊 Monitor how strategy uses new funds"
    ]
    
    for step in checklist:
        print(step)
    
    print("\n🎯 EXAMPLE WORKFLOW:")
    print("```python")
    print("# Step 1: Decide amount")
    print("sats_to_add = 15000  # Want to add 15k sats")
    print("")
    print("# Step 2: Convert to BTC")
    print("btc_amount = Decimal(str(sats_to_add / 100_000_000))")
    print("")
    print("# Step 3: Add to strategy")
    print("strategy.add_funds(btc_amount)")
    print("")
    print("# Step 4: Check new balance")
    print("new_total = strategy.portfolio.get_portfolio_value()")
    print("new_sats = int(new_total * 100_000_000)")
    print("print(f'New total: {new_sats:,} sats')")
    print("")
    print("# Step 5: Let the strategy work!")
    print("# Strategy automatically uses new funds for trading")
    print("```")
    
    print("\n⚡ REMEMBER:")
    print("• Add sats ANYTIME - strategy running or stopped")
    print("• No minimum amount - add 100 sats or 1,000,000 sats")  
    print("• Strategy immediately recognizes new funds")
    print("• More sats = More trading power!")
    print("• Your trading opportunities scale automatically")


if __name__ == "__main__":
    print("⚡ Starting Add Sats Guide...")
    try:
        asyncio.run(add_sats_demo())
        asyncio.run(add_sats_methods())
        asyncio.run(practical_examples())
        asyncio.run(scaling_examples())
        asyncio.run(add_sats_checklist())
        
        print("\n" + "=" * 50)
        print("🎉 ADD SATS GUIDE COMPLETE!")
        print("=" * 50)
        print("Now you know exactly how to:")
        print("✅ Add any amount of sats to your strategy")
        print("✅ Scale your trading power over time")
        print("✅ Reinvest profits back into trading")
        print("✅ Boost your strategy during opportunities")
        print("")
        print("🚀 Ready to start adding sats and growing your stack!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Guide interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Guide failed")
