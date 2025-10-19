#!/usr/bin/env python3
"""
BTC Demo for Poise Trader

Simple demonstration of using Bitcoin as trading capital.
Perfect for when you have BTC but no fiat money.
"""

import asyncio
import logging
from decimal import Decimal
from core.strategies import StrategyFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BTCDemo")


async def btc_demo():
    """Demo using BTC as trading capital"""
    print("₿ Welcome to Poise Trader - BTC Edition!")
    print("=" * 50)
    
    # Your BTC amount (adjust this to your actual holdings)
    your_btc = Decimal('0.1')  # 0.1 BTC for demo
    
    print(f"💰 Your BTC Holdings: {your_btc} BTC")
    print(f"📊 Let's put it to work!\n")
    
    # Conservative allocation: only use part of your BTC for trading
    trading_btc = your_btc * Decimal('0.5')  # Use 50% for trading, keep 50% as hodl
    
    print(f"🎯 Trading Allocation: {trading_btc} BTC (50% of holdings)")
    print(f"🏦 HODL Reserve: {your_btc - trading_btc} BTC (safe storage)\n")
    
    # Strategy 1: DCA into Ethereum using BTC
    print("🚀 Setting up ETH DCA Strategy...")
    eth_dca_config = {
        'initial_capital': float(trading_btc),  # Use your trading BTC
        'base_currency': 'BTC',                 # BTC as base currency
        'symbols': ['ETH/BTC'],                 # Buy ETH with BTC
        'buy_interval': 3600,                   # Buy every hour (demo speed)
        'base_buy_amount': 0.005,               # 0.005 BTC per buy
        'max_allocation_per_symbol': 0.8,       # Max 80% in ETH
        'price_drop_threshold': 0.03,           # Buy extra on 3% drops
        'profit_taking_threshold': 0.15,        # Take profits at 15% gain
        'profit_taking_percentage': 0.3         # Take 30% profits
    }
    
    # Create the strategy
    strategy = StrategyFactory.create_strategy('dca', eth_dca_config)
    await strategy.initialize()
    
    print("✅ Strategy initialized!")
    print(f"   Strategy Capital: {strategy.portfolio.initial_capital} BTC")
    print(f"   Base Currency: {strategy.portfolio.base_currency}")
    print(f"   Target: {strategy.symbols[0]}")
    print(f"   Buy Amount: {strategy.base_buy_amount} BTC per interval\n")
    
    # Show portfolio state
    print("📊 Initial Portfolio State:")
    summary = strategy.get_portfolio_summary()
    for currency, balance in summary['balances'].items():
        print(f"   {currency}: {balance}")
    
    print(f"   Total Value: {summary['total_value']} BTC")
    print(f"   Available: {strategy.portfolio.get_available_balance('BTC')} BTC")
    
    # Simulate adding more BTC later
    print("\n💰 Simulating adding more BTC to the strategy...")
    additional_btc = Decimal('0.05')  # Add 0.05 more BTC
    strategy.add_funds(additional_btc)
    
    print(f"✅ Added {additional_btc} BTC")
    print(f"   New Total: {strategy.portfolio.get_portfolio_value()} BTC")
    
    # Show how to withdraw BTC
    print("\n💸 Testing BTC withdrawal...")
    withdraw_amount = Decimal('0.02')
    success = strategy.withdraw_funds(withdraw_amount)
    
    if success:
        print(f"✅ Successfully withdrew {withdraw_amount} BTC")
    else:
        print(f"❌ Withdrawal failed (insufficient funds)")
    
    print(f"   Remaining: {strategy.portfolio.get_portfolio_value()} BTC")
    
    # Simulate some market activity
    print("\n📈 Simulating market activity...")
    from core.framework.base_classes import MarketData
    import time
    
    # Simulate ETH/BTC price movements
    eth_btc_prices = [
        Decimal('0.065'),  # Starting price
        Decimal('0.063'),  # 3% drop - should trigger enhanced buy
        Decimal('0.067'),  # Recovery
        Decimal('0.070'),  # Growth
        Decimal('0.075'),  # 15% gain - should trigger profit taking
    ]
    
    for i, price in enumerate(eth_btc_prices):
        print(f"\n📊 Market Update {i+1}: ETH/BTC = {price}")
        
        # Create market data
        market_data = MarketData(
            symbol='ETH/BTC',
            timestamp=int(time.time() * 1000),
            price=price,
            volume=Decimal('100'),
            exchange='demo'
        )
        
        # Process with strategy
        signal = await strategy.process_market_data(market_data)
        
        if signal:
            signal_type = signal.metadata.get('signal_type', 'unknown')
            print(f"🎯 Signal Generated: {signal.action.value.upper()}")
            print(f"   Type: {signal_type}")
            print(f"   Amount: {signal.quantity:.6f} ETH")
            print(f"   BTC Cost: {signal.quantity * signal.price:.6f} BTC")
            print(f"   Confidence: {signal.confidence:.2f}")
            
            # Simulate execution
            if signal.action.value == 'buy':
                # Simulate buy order
                from core.strategies.base_strategy import PositionType
                success = strategy.portfolio.open_position(
                    signal.symbol,
                    PositionType.LONG,
                    signal.quantity,
                    signal.price
                )
                if success:
                    print("   ✅ Buy executed")
                else:
                    print("   ❌ Buy failed")
            else:
                # Simulate sell order
                pnl = strategy.portfolio.close_position(signal.symbol, signal.price)
                if pnl is not None:
                    print(f"   ✅ Sell executed - P&L: {pnl:.6f} BTC")
                else:
                    print("   ❌ Sell failed")
        else:
            print("   No signal (conditions not met)")
        
        # Show current portfolio
        portfolio_value = strategy.portfolio.get_portfolio_value()
        position = strategy.portfolio.get_position('ETH/BTC')
        print(f"📋 Portfolio: {portfolio_value:.6f} BTC")
        
        if position:
            print(f"   ETH Position: {position.size:.6f} ETH @ {position.entry_price:.6f} BTC")
            print(f"   Unrealized P&L: {position.unrealized_pnl:.6f} BTC")
        
        await asyncio.sleep(1)  # Pause between updates
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 Final Strategy Performance:")
    
    metrics = strategy.get_performance_metrics()
    print(f"   Initial Capital: {metrics['initial_capital']} BTC")
    print(f"   Final Value: {metrics['current_value']:.6f} BTC")
    print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Win Rate: {metrics['win_rate_pct']:.1f}%")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    
    btc_profit = Decimal(str(metrics['current_value'])) - Decimal(str(metrics['initial_capital']))
    if btc_profit > 0:
        print(f"   BTC Profit: +{btc_profit:.6f} BTC 🚀")
    else:
        print(f"   BTC Loss: {btc_profit:.6f} BTC")
    
    print(f"\n💡 Your total BTC now: ~{Decimal(str(metrics['current_value'])) + (your_btc - trading_btc):.6f} BTC")
    print("   (Strategy BTC + HODL reserve)")


async def simple_btc_setup():
    """Show the simplest way to set up BTC trading"""
    print("\n" + "=" * 50)
    print("🔧 Simple Setup Guide")
    print("=" * 50)
    
    print("To use your BTC with Poise Trader, just:")
    print("")
    print("1. Set your BTC amount:")
    print("   your_btc = 0.5  # Your actual BTC")
    print("")
    print("2. Create a strategy config:")
    print("   config = {")
    print("       'initial_capital': 0.2,      # Use 0.2 BTC for trading")
    print("       'base_currency': 'BTC',      # BTC as base currency")
    print("       'symbols': ['ETH/BTC'],      # Trade ETH against BTC")  
    print("       'buy_interval': 3600,        # Buy every hour")
    print("       'base_buy_amount': 0.01      # 0.01 BTC per buy")
    print("   }")
    print("")
    print("3. Create and run:")
    print("   strategy = StrategyFactory.create_strategy('dca', config)")
    print("   await strategy.initialize()")
    print("")
    print("4. Add more BTC anytime:")
    print("   strategy.add_funds(Decimal('0.1'))  # Add 0.1 more BTC")
    print("")
    print("5. Check performance:")
    print("   metrics = strategy.get_performance_metrics()")
    print("   print(f'BTC Return: {metrics[\"total_return_pct\"]:.2f}%')")
    print("")
    print("That's it! Your BTC is now earning more BTC! 🚀")


if __name__ == "__main__":
    print("Starting BTC Trading Demo...")
    try:
        asyncio.run(btc_demo())
        asyncio.run(simple_btc_setup())
        print("\n✅ Demo completed successfully!")
        print("🎯 Ready to start trading with your BTC!")
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        logger.exception("Demo failed")
