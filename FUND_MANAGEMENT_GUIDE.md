# 💰 Fund Management Guide - Poise Trader

## How to Add and Manage Funds in Your Trading System

### 🚀 Quick Start

```python
from core.strategies import StrategyFactory, create_portfolio_manager
from decimal import Decimal

# Method 1: Create strategy with initial capital
config = {
    'initial_capital': 10000,  # $10,000 starting funds
    'base_currency': 'USDT',
    'symbols': ['BTC/USDT', 'ETH/USDT'],
    'buy_interval': 3600,      # 1 hour
    'base_buy_amount': 100     # $100 per buy
}

strategy = StrategyFactory.create_strategy('dca', config)
await strategy.initialize()
```

### 💵 Adding Funds to a Running Strategy

```python
# Add $5,000 more to the strategy
strategy.add_funds(Decimal('5000'))

# Add funds in different currencies
strategy.add_funds(Decimal('1000'), currency='BTC')  # Add 1 BTC
strategy.add_funds(Decimal('2000'), currency='ETH')  # Add 2 ETH

# Check updated balances
print(f"USDT Balance: {strategy.portfolio.balances['USDT']}")
print(f"Total Portfolio Value: ${strategy.portfolio.get_portfolio_value()}")
```

### 💸 Withdrawing Funds

```python
# Withdraw $2,000 (only from available balance)
success = strategy.withdraw_funds(Decimal('2000'))
if success:
    print("✅ Withdrawal successful")
else:
    print("❌ Insufficient funds")

# Check available vs locked funds
available = strategy.portfolio.get_available_balance('USDT')
locked = strategy.portfolio.locked_balances.get('USDT', Decimal('0'))
print(f"Available: ${available}, Locked: ${locked}")
```

### 📊 Portfolio Management Examples

#### Example 1: DCA Strategy with Fund Management
```python
import asyncio
from core.strategies import DCAStrategy
from decimal import Decimal

async def manage_dca_funds():
    # Initialize DCA strategy
    config = {
        'initial_capital': 5000,
        'base_currency': 'USDT',
        'symbols': ['BTC/USDT'],
        'buy_interval': 3600,      # 1 hour
        'base_buy_amount': 200,    # $200 per buy
        'max_allocation_per_symbol': 0.8  # 80% max allocation
    }
    
    strategy = DCAStrategy(config)
    await strategy.initialize()
    
    print(f"📊 Initial State:")
    print(f"   Capital: ${strategy.portfolio.initial_capital}")
    print(f"   Available: ${strategy.portfolio.get_available_balance('USDT')}")
    
    # Add more funds during operation
    strategy.add_funds(Decimal('3000'))
    print(f"💰 Added $3,000 - New total: ${strategy.portfolio.get_portfolio_value()}")
    
    # Strategy will automatically use new funds for future DCA buys
    # No manual intervention needed!
    
    return strategy

# Run it
strategy = asyncio.run(manage_dca_funds())
```

#### Example 2: Grid Strategy Fund Scaling
```python
async def scale_grid_strategy():
    # Start with smaller capital
    config = {
        'initial_capital': 2000,
        'symbols': ['ETH/USDT'],
        'grid_spacing_pct': 0.01,    # 1% spacing
        'num_grid_levels': 8,
        'order_size_base': 100       # $100 per grid level
    }
    
    strategy = GridStrategy(config)
    await strategy.initialize()
    
    # Scale up the strategy
    print("📈 Scaling up grid strategy...")
    strategy.add_funds(Decimal('5000'))  # Add $5,000 more
    
    # Optionally update parameters for larger capital
    strategy.update_grid_parameters(
        order_size_base=250,  # Increase to $250 per level
        num_grid_levels=12    # More levels with more capital
    )
    
    print("✅ Grid strategy scaled successfully!")
    return strategy
```

### 🔧 Advanced Fund Management

#### Multiple Currency Support
```python
# Portfolio with multiple currencies
strategy.add_funds(Decimal('10000'), 'USDT')  # Add USDT
strategy.add_funds(Decimal('5'), 'BTC')       # Add BTC  
strategy.add_funds(Decimal('50'), 'ETH')      # Add ETH

# Check all balances
for currency, amount in strategy.portfolio.balances.items():
    print(f"{currency}: {amount}")
```

#### Dynamic Fund Allocation
```python
def smart_fund_allocation(strategy, new_funds):
    """Intelligently allocate new funds based on current positions"""
    current_value = strategy.portfolio.get_portfolio_value()
    
    # Add the funds
    strategy.add_funds(Decimal(str(new_funds)))
    
    # Get current positions
    positions = strategy.get_positions()
    
    if not positions:
        print(f"💰 Added ${new_funds} - ready for new positions")
    else:
        total_position_value = sum(
            pos.size * pos.current_price for pos in positions.values()
        )
        allocation_pct = (total_position_value / current_value) * 100
        print(f"💼 Current allocation: {allocation_pct:.1f}%")
        print(f"💰 Added ${new_funds} - available for scaling positions")
```

### 📈 Real-time Fund Monitoring

```python
def monitor_portfolio(strategy):
    """Monitor portfolio state in real-time"""
    summary = strategy.get_portfolio_summary()
    
    print("📊 Portfolio Status:")
    print(f"   Total Value: ${summary['total_value']:.2f}")
    
    print("\n💰 Balances:")
    for currency, balance in summary['balances'].items():
        locked = summary['locked_balances'].get(currency, 0)
        available = balance - locked
        print(f"   {currency}: {balance:.2f} (Available: {available:.2f}, Locked: {locked:.2f})")
    
    if summary['positions']:
        print("\n📍 Active Positions:")
        for symbol, pos_data in summary['positions'].items():
            print(f"   {symbol}: {pos_data['size']:.4f} @ ${pos_data['entry_price']:.2f}")
            print(f"     Current P&L: ${pos_data['unrealized_pnl']:.2f}")

# Use it
monitor_portfolio(strategy)
```

### ⚠️ Fund Safety Features

#### Available vs Locked Funds
```python
# The system automatically prevents over-withdrawal
available = strategy.portfolio.get_available_balance('USDT')
locked = strategy.portfolio.locked_balances.get('USDT', Decimal('0'))

print(f"Available for withdrawal: ${available}")
print(f"Locked in positions: ${locked}")

# This will fail if amount > available
success = strategy.withdraw_funds(available + Decimal('100'))  # ❌ Will fail
```

#### Emergency Fund Management
```python
def emergency_liquidation(strategy):
    """Emergency function to liquidate all positions"""
    positions = strategy.get_positions()
    
    for symbol, position in positions.items():
        current_price = strategy.last_prices.get(symbol, position.entry_price)
        pnl = strategy.portfolio.close_position(symbol, current_price)
        print(f"🚨 Emergency close {symbol}: P&L ${pnl}")
    
    # Now all funds should be available
    total_available = strategy.portfolio.get_available_balance('USDT')
    print(f"💰 Total available after liquidation: ${total_available}")
```

### 🎯 Best Practices

1. **Start Conservative**: Begin with smaller amounts to test strategies
2. **Scale Gradually**: Add funds incrementally as strategies prove profitable  
3. **Monitor Closely**: Check available vs locked balances regularly
4. **Emergency Planning**: Always keep some funds available for emergencies
5. **Multiple Strategies**: Distribute funds across different strategies for diversification

### 📝 Common Fund Management Patterns

```python
# Pattern 1: Progressive Capital Scaling
initial_capital = 1000
for week in range(4):  # Scale up over 4 weeks
    if week > 0:
        performance = strategy.get_performance_metrics()
        if performance['total_return_pct'] > 0:  # Only add if profitable
            strategy.add_funds(Decimal(str(initial_capital * 0.5)))  # Add 50% more
            print(f"Week {week}: Added funds, total return: {performance['total_return_pct']:.1f}%")

# Pattern 2: Profit Reinvestment
def reinvest_profits(strategy, reinvestment_rate=0.5):
    """Reinvest a portion of profits back into the strategy"""
    metrics = strategy.get_performance_metrics()
    current_value = Decimal(str(metrics['current_value']))
    initial_capital = Decimal(str(metrics['initial_capital']))
    
    if current_value > initial_capital:
        profit = current_value - initial_capital
        reinvestment = profit * Decimal(str(reinvestment_rate))
        
        # This effectively compounds the capital
        strategy.add_funds(reinvestment)
        print(f"💰 Reinvested ${reinvestment} (50% of ${profit} profit)")

# Pattern 3: Risk-Based Fund Allocation
def risk_adjusted_funding(strategy, max_risk_pct=0.02):
    """Add funds based on risk tolerance"""
    portfolio_value = strategy.portfolio.get_portfolio_value()
    max_position_size = portfolio_value * Decimal(str(max_risk_pct))
    
    # Add funds to maintain risk levels
    if strategy.risk_per_trade * portfolio_value > max_position_size:
        additional_funds = max_position_size / strategy.risk_per_trade - portfolio_value
        strategy.add_funds(additional_funds)
        print(f"🛡️ Added ${additional_funds} to maintain risk levels")
```

---

## 🚀 Ready to Start?

Run the demo to see fund management in action:

```bash
cd "C:\Users\OM\Desktop\Poise Trader\core\strategies"
python demo_strategies.py
```

The demo will show you:
- ✅ How to add/withdraw funds  
- ✅ Real-time portfolio monitoring
- ✅ Multiple strategies with different capital
- ✅ Performance tracking with fund changes

Your Poise Trader system now has enterprise-grade fund management capabilities! 🎉
