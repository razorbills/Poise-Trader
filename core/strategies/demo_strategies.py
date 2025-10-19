"""
Demo script for Poise Trader Strategy Engine

This script demonstrates how to:
- Initialize strategies with funds
- Add and withdraw funds dynamically
- Monitor strategy performance
- Run multiple strategies simultaneously
"""

import asyncio
import logging
from decimal import Decimal
from typing import Dict, Any

from .dca_strategy import DCAStrategy
from .grid_strategy import GridStrategy
from ..feeds import FeedFactory
from ..framework.event_system import EventBus


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("StrategyDemo")


async def demo_fund_management():
    """Demo how to manage funds in strategies"""
    print("\n=== Fund Management Demo ===")
    
    # Create a DCA strategy with initial capital
    dca_config = {
        'initial_capital': 10000,  # $10,000 starting capital
        'base_currency': 'USDT',
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'buy_interval': 300,  # 5 minutes for demo
        'base_buy_amount': 100,  # $100 per buy
        'price_drop_threshold': 0.03,  # 3% drop trigger
        'max_allocation_per_symbol': 0.4  # 40% max per symbol
    }
    
    strategy = DCAStrategy(dca_config)
    await strategy.initialize()
    
    print("📊 Initial Portfolio State:")
    print(f"   Balance: {strategy.portfolio.balances}")
    print(f"   Total Value: ${strategy.portfolio.get_portfolio_value()}")
    
    # Add more funds
    print("\n💰 Adding $5,000 to the strategy...")
    strategy.add_funds(Decimal('5000'))
    
    print("📊 After Adding Funds:")
    print(f"   Balance: {strategy.portfolio.balances}")
    print(f"   Total Value: ${strategy.portfolio.get_portfolio_value()}")
    
    # Withdraw some funds
    print("\n💸 Attempting to withdraw $2,000...")
    success = strategy.withdraw_funds(Decimal('2000'))
    if success:
        print("✅ Withdrawal successful")
    else:
        print("❌ Withdrawal failed")
    
    print("📊 After Withdrawal:")
    print(f"   Balance: {strategy.portfolio.balances}")
    print(f"   Available: ${strategy.portfolio.get_available_balance('USDT')}")
    print(f"   Total Value: ${strategy.portfolio.get_portfolio_value()}")
    
    # Try to withdraw too much
    print("\n💸 Attempting to withdraw $20,000 (should fail)...")
    success = strategy.withdraw_funds(Decimal('20000'))
    if success:
        print("✅ Withdrawal successful")
    else:
        print("❌ Withdrawal failed (insufficient funds)")
    
    print("\n📈 Portfolio Summary:")
    summary = strategy.get_portfolio_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")


async def demo_dca_strategy():
    """Demo DCA strategy with live-like market data simulation"""
    print("\n=== DCA Strategy Demo ===")
    
    # Create DCA strategy
    dca_config = {
        'initial_capital': 5000,
        'base_currency': 'USDT',
        'symbols': ['BTC/USDT'],
        'buy_interval': 10,  # 10 seconds for demo
        'base_buy_amount': 200,
        'price_drop_threshold': 0.02,  # 2% drop
        'max_allocation_per_symbol': 0.8,  # 80% max allocation
        'profit_taking_threshold': 0.1,  # 10% profit
        'profit_taking_percentage': 0.3  # Take 30% profits
    }
    
    strategy = DCAStrategy(dca_config)
    await strategy.initialize()
    
    print("🚀 DCA Strategy initialized")
    print(f"   Initial Capital: ${strategy.portfolio.initial_capital}")
    print(f"   Buy Interval: {strategy.buy_interval} seconds")
    print(f"   Base Buy Amount: ${strategy.base_buy_amount}")
    
    # Simulate market data
    import time
    from ..framework.base_classes import MarketData
    
    # Start with BTC at $50,000
    base_price = Decimal('50000')
    prices = [
        base_price,
        base_price * Decimal('0.98'),  # 2% drop - should trigger buy
        base_price * Decimal('0.95'),  # 5% drop - should trigger enhanced buy
        base_price * Decimal('1.02'),  # 2% recovery
        base_price * Decimal('1.08'),  # 8% gain
        base_price * Decimal('1.12'),  # 12% gain - should trigger profit taking
    ]
    
    for i, price in enumerate(prices):
        print(f"\n📊 Market Update {i+1}: BTC/USDT = ${price}")
        
        # Create market data
        market_data = MarketData(
            symbol='BTC/USDT',
            timestamp=int(time.time() * 1000),
            price=price,
            volume=Decimal('100'),
            exchange='demo'
        )
        
        # Process with strategy
        signal = await strategy.process_market_data(market_data)
        
        if signal:
            print(f"📈 Signal Generated:")
            print(f"   Action: {signal.action.value}")
            print(f"   Quantity: {signal.quantity}")
            print(f"   Confidence: {signal.confidence}")
            print(f"   Metadata: {signal.metadata}")
            
            # Simulate trade execution
            if signal.action.value == 'buy':
                success = strategy.portfolio.open_position(
                    signal.symbol,
                    strategy.PositionType.LONG,
                    signal.quantity,
                    signal.price
                )
                if success:
                    print("✅ Position opened")
                else:
                    print("❌ Failed to open position")
            else:  # sell
                pnl = strategy.portfolio.close_position(signal.symbol, signal.price)
                if pnl is not None:
                    print(f"✅ Position closed, P&L: ${pnl}")
        else:
            print("   No signal generated")
        
        # Show current state
        portfolio_value = strategy.portfolio.get_portfolio_value()
        position = strategy.portfolio.get_position('BTC/USDT')
        print(f"💼 Portfolio Value: ${portfolio_value}")
        if position:
            print(f"   Position: {position.size} BTC at ${position.entry_price}")
            print(f"   Unrealized P&L: ${position.unrealized_pnl}")
        
        # Wait a bit between updates
        await asyncio.sleep(2)
    
    print("\n📊 Final DCA Summary:")
    dca_summary = strategy.get_dca_summary()
    for symbol, data in dca_summary.items():
        print(f"   {symbol}:")
        for key, value in data.items():
            print(f"     {key}: {value}")


async def demo_grid_strategy():
    """Demo Grid trading strategy"""
    print("\n=== Grid Strategy Demo ===")
    
    # Create Grid strategy
    grid_config = {
        'initial_capital': 3000,
        'base_currency': 'USDT',
        'symbols': ['ETH/USDT'],
        'grid_spacing_pct': 0.01,  # 1% spacing
        'num_grid_levels': 8,      # 4 levels above/below
        'max_grid_deviation': 0.03,  # 3% max deviation
        'order_size_base': 150,    # $150 per order
        'profit_target': 0.01      # 1% profit target
    }
    
    strategy = GridStrategy(grid_config)
    await strategy.initialize()
    
    print("🔲 Grid Strategy initialized")
    print(f"   Initial Capital: ${strategy.portfolio.initial_capital}")
    print(f"   Grid Spacing: {strategy.grid_spacing_pct * 100}%")
    print(f"   Grid Levels: {strategy.num_grid_levels}")
    
    # Simulate ETH price movement
    from ..framework.base_classes import MarketData
    import time
    
    base_price = Decimal('3000')  # ETH at $3000
    prices = [
        base_price,
        base_price * Decimal('0.99'),  # -1% (should trigger buy)
        base_price * Decimal('0.98'),  # -2% (should trigger another buy)
        base_price * Decimal('1.01'),  # +1% (should trigger sell)
        base_price * Decimal('1.02'),  # +2% (should trigger another sell)
        base_price * Decimal('0.995'), # Back to -0.5%
    ]
    
    for i, price in enumerate(prices):
        print(f"\n📊 Market Update {i+1}: ETH/USDT = ${price}")
        
        market_data = MarketData(
            symbol='ETH/USDT',
            timestamp=int(time.time() * 1000),
            price=price,
            volume=Decimal('50'),
            exchange='demo'
        )
        
        signal = await strategy.process_market_data(market_data)
        
        if signal:
            print(f"🔲 Grid Signal:")
            print(f"   Action: {signal.action.value}")
            print(f"   Quantity: {signal.quantity}")
            print(f"   Price: ${signal.price}")
            print(f"   Metadata: {signal.metadata}")
            
            # Simulate execution
            if signal.action.value == 'buy':
                success = strategy.portfolio.open_position(
                    signal.symbol,
                    strategy.PositionType.LONG,
                    signal.quantity,
                    signal.price
                )
                print("✅ Grid buy executed" if success else "❌ Grid buy failed")
            else:
                pnl = strategy.portfolio.close_position(signal.symbol, signal.price)
                print(f"✅ Grid sell executed, P&L: ${pnl}" if pnl else "❌ Grid sell failed")
        
        # Show grid state
        grid_summary = strategy.get_grid_summary()
        eth_grid = grid_summary.get('ETH/USDT', {})
        print(f"🔲 Grid State:")
        print(f"   Center Price: ${eth_grid.get('center_price', 0)}")
        print(f"   Active Buy Levels: {eth_grid.get('active_buy_levels', 0)}")
        print(f"   Active Sell Levels: {eth_grid.get('active_sell_levels', 0)}")
        print(f"   Filled Levels: {eth_grid.get('filled_levels', 0)}")
        
        await asyncio.sleep(1)


async def demo_multiple_strategies():
    """Demo running multiple strategies simultaneously"""
    print("\n=== Multiple Strategies Demo ===")
    
    # Create event bus for coordination
    event_bus = EventBus()
    await event_bus.start()
    
    # Create multiple strategies
    strategies = {}
    
    # DCA Strategy for BTC
    strategies['btc_dca'] = DCAStrategy({
        'initial_capital': 5000,
        'symbols': ['BTC/USDT'],
        'buy_interval': 15,
        'base_buy_amount': 150,
        'max_allocation_per_symbol': 1.0
    })
    
    # Grid Strategy for ETH
    strategies['eth_grid'] = GridStrategy({
        'initial_capital': 3000,
        'symbols': ['ETH/USDT'],
        'grid_spacing_pct': 0.015,
        'num_grid_levels': 6,
        'order_size_base': 200
    })
    
    # Initialize all strategies
    for name, strategy in strategies.items():
        strategy.event_bus = event_bus
        await strategy.initialize()
        print(f"✅ {name} initialized with ${strategy.portfolio.initial_capital}")
    
    # Event handler to log all trading events
    async def log_trading_events(event):
        print(f"🔔 Trading Event: {event.event_type}")
        print(f"   Data: {event.data}")
    
    event_bus.subscribe("trading.signal.generated", log_trading_events)
    event_bus.subscribe("trading.position.closed", log_trading_events)
    
    print("\n📊 Running multiple strategies for 30 seconds...")
    
    # Simulate market data for both symbols
    import time
    from ..framework.base_classes import MarketData
    
    btc_price = Decimal('50000')
    eth_price = Decimal('3000')
    
    for i in range(10):  # 10 updates
        # Update BTC price (more volatile for DCA)
        btc_change = ((-1) ** i) * Decimal('0.02')  # Alternate +/-2%
        btc_price = btc_price * (Decimal('1') + btc_change)
        
        # Update ETH price (range-bound for grid)
        eth_change = Decimal('0.01') if i % 3 == 0 else Decimal('-0.01')
        eth_price = eth_price * (Decimal('1') + eth_change)
        
        print(f"\n--- Update {i+1} ---")
        print(f"BTC: ${btc_price}, ETH: ${eth_price}")
        
        # Send data to strategies
        btc_data = MarketData('BTC/USDT', int(time.time()*1000), btc_price, Decimal('100'), exchange='demo')
        eth_data = MarketData('ETH/USDT', int(time.time()*1000), eth_price, Decimal('50'), exchange='demo')
        
        # Process with all relevant strategies
        for name, strategy in strategies.items():
            if 'btc' in name:
                signal = await strategy.process_market_data(btc_data)
            else:
                signal = await strategy.process_market_data(eth_data)
            
            if signal:
                print(f"📈 {name} generated signal: {signal.action.value}")
        
        await asyncio.sleep(3)
    
    # Show final performance
    print("\n📊 Final Performance Summary:")
    for name, strategy in strategies.items():
        metrics = strategy.get_performance_metrics()
        print(f"\n{name.upper()}:")
        print(f"   Initial: ${metrics['initial_capital']}")
        print(f"   Current: ${metrics['current_value']:.2f}")
        print(f"   Return: {metrics['total_return_pct']:.2f}%")
        print(f"   Trades: {metrics['total_trades']}")
        print(f"   Win Rate: {metrics['win_rate_pct']:.1f}%")
    
    await event_bus.stop()


async def main():
    """Run all demos"""
    print("🚀 Poise Trader Strategy Engine Demo\n")
    
    try:
        # Demo 1: Fund management
        await demo_fund_management()
        
        # Demo 2: DCA Strategy
        await demo_dca_strategy()
        
        # Demo 3: Grid Strategy
        await demo_grid_strategy()
        
        # Demo 4: Multiple strategies
        await demo_multiple_strategies()
        
        print("\n✅ All strategy demos completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo error")


if __name__ == "__main__":
    asyncio.run(main())
