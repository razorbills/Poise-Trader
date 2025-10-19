# ₿ Using Your BTC with Poise Trader

## Perfect! You Have BTC - Let's Put It to Work! 

### 🚀 **Quick Start with BTC as Capital**

```python
from core.strategies import StrategyFactory
from decimal import Decimal

# Use BTC as your base currency instead of USD
config = {
    'initial_capital': 0.5,        # 0.5 BTC starting capital  
    'base_currency': 'BTC',        # BTC as base instead of USDT
    'symbols': ['ETH/BTC', 'ADA/BTC', 'DOT/BTC'],  # Trade alts against BTC
    'buy_interval': 3600,          # 1 hour DCA
    'base_buy_amount': 0.01        # 0.01 BTC per buy
}

strategy = StrategyFactory.create_strategy('dca', config)
await strategy.initialize()
```

### 💰 **Fund Management with BTC**

```python
# Add more BTC to your strategy
strategy.add_funds(Decimal('0.1'))  # Add 0.1 BTC

# Check your BTC balance
btc_balance = strategy.portfolio.balances['BTC']
print(f"Total BTC: {btc_balance}")

# Available vs locked BTC
available_btc = strategy.portfolio.get_available_balance('BTC') 
locked_btc = strategy.portfolio.locked_balances.get('BTC', Decimal('0'))
print(f"Available: {available_btc} BTC")
print(f"Locked in trades: {locked_btc} BTC")
```

### 📊 **BTC-Based Trading Strategies**

#### **Strategy 1: Altcoin DCA with BTC**
```python
# DCA into altcoins using your BTC
altcoin_dca_config = {
    'initial_capital': 0.2,           # 0.2 BTC
    'base_currency': 'BTC',           
    'symbols': [
        'ETH/BTC',    # Ethereum
        'ADA/BTC',    # Cardano  
        'DOT/BTC',    # Polkadot
        'SOL/BTC'     # Solana
    ],
    'buy_interval': 7200,             # 2 hours
    'base_buy_amount': 0.005,         # 0.005 BTC per buy
    'max_allocation_per_symbol': 0.3, # Max 30% per alt
    'price_drop_threshold': 0.05      # Buy more on 5% drops
}

dca_strategy = StrategyFactory.create_strategy('dca', altcoin_dca_config)
```

#### **Strategy 2: BTC/USDT Grid Trading**
```python
# Grid trade BTC against USDT to accumulate more BTC
btc_grid_config = {
    'initial_capital': 50000,         # $50k USDT equivalent
    'base_currency': 'USDT',          
    'symbols': ['BTC/USDT'],
    'grid_spacing_pct': 0.02,         # 2% grid spacing
    'num_grid_levels': 10,            # 5 levels up/down
    'order_size_base': 2000,          # $2k per grid level
    'profit_target': 0.02             # 2% profit per trade
}

grid_strategy = StrategyFactory.create_strategy('grid', btc_grid_config)
```

### 🔄 **Converting Between BTC and USDT**

```python
# If you need USDT for certain strategies, the system can handle conversions
mixed_portfolio_config = {
    'initial_capital': 0.1,      # 0.1 BTC
    'base_currency': 'BTC',
    'symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/BTC'],  # Mixed pairs
    'auto_convert': True         # Auto-convert between BTC/USDT as needed
}
```

### 💡 **Practical Example: Your BTC Portfolio**

```python
import asyncio
from core.strategies import DCAStrategy, GridStrategy
from decimal import Decimal

async def setup_btc_portfolio():
    """Setup trading strategies using your BTC"""
    
    print("🚀 Setting up your BTC trading portfolio...")
    
    # Strategy 1: DCA into top altcoins (using 60% of BTC)
    altcoin_strategy = DCAStrategy({
        'initial_capital': 0.3,       # 0.3 BTC for altcoins
        'base_currency': 'BTC',
        'symbols': ['ETH/BTC', 'ADA/BTC', 'DOT/BTC'],
        'buy_interval': 3600,         # Every hour
        'base_buy_amount': 0.01,      # 0.01 BTC per buy
        'max_allocation_per_symbol': 0.4  # Max 40% in any one alt
    })
    
    # Strategy 2: Grid trade BTC/USDT (using remaining 40%)  
    btc_grid_strategy = GridStrategy({
        'initial_capital': 20000,     # $20k equivalent in USDT
        'base_currency': 'USDT', 
        'symbols': ['BTC/USDT'],
        'grid_spacing_pct': 0.015,    # 1.5% spacing
        'num_grid_levels': 8,
        'order_size_base': 1000       # $1k per level
    })
    
    # Initialize both strategies
    await altcoin_strategy.initialize()
    await btc_grid_strategy.initialize()
    
    print("✅ Portfolio initialized!")
    print(f"   Altcoin DCA: {altcoin_strategy.portfolio.initial_capital} BTC")
    print(f"   BTC Grid: ${btc_grid_strategy.portfolio.initial_capital} USDT")
    
    return altcoin_strategy, btc_grid_strategy

# Run it
strategies = asyncio.run(setup_btc_portfolio())
```

### 📈 **BTC Value Tracking**

```python
def monitor_btc_portfolio(strategies):
    """Monitor your BTC portfolio performance"""
    
    total_btc_value = Decimal('0')
    
    for strategy in strategies:
        if strategy.portfolio.base_currency == 'BTC':
            btc_value = strategy.portfolio.get_portfolio_value()
            total_btc_value += btc_value
            print(f"📊 {strategy.__class__.__name__}: {btc_value} BTC")
        else:
            # Convert USDT value to BTC (you'd need current BTC price)
            usdt_value = strategy.portfolio.get_portfolio_value()
            btc_price = 50000  # Current BTC price in USDT
            btc_equivalent = usdt_value / btc_price
            total_btc_value += btc_equivalent
            print(f"📊 {strategy.__class__.__name__}: {usdt_value} USDT (~{btc_equivalent:.6f} BTC)")
    
    print(f"\n💰 Total Portfolio: ~{total_btc_value:.6f} BTC")

# Use it
monitor_btc_portfolio(strategies)
```

### 🎯 **Smart BTC Allocation Strategies**

```python
def smart_btc_allocation(total_btc):
    """Intelligently allocate your BTC across strategies"""
    
    allocation = {
        'hodl_reserve': total_btc * Decimal('0.4'),      # 40% keep as BTC
        'altcoin_dca': total_btc * Decimal('0.3'),       # 30% DCA into alts  
        'btc_grid_trading': total_btc * Decimal('0.2'),  # 20% grid trade BTC
        'experimental': total_btc * Decimal('0.1')       # 10% for testing
    }
    
    print("🧠 Smart BTC Allocation:")
    for strategy, amount in allocation.items():
        print(f"   {strategy}: {amount:.6f} BTC")
    
    return allocation

# Example with 1 BTC
allocation = smart_btc_allocation(Decimal('1.0'))
```

### 🔥 **Advanced: BTC Yield Strategies**

```python
# Strategy to grow your BTC stack
btc_growth_config = {
    'initial_capital': 0.1,           # Start with 0.1 BTC
    'base_currency': 'BTC',
    'symbols': [
        'ETH/BTC',     # Trade ETH against BTC
        'LTC/BTC',     # Litecoin 
        'BCH/BTC'      # Bitcoin Cash
    ],
    'strategy_goal': 'accumulate_btc',  # Goal: end up with more BTC
    'buy_interval': 1800,               # 30 minutes
    'base_buy_amount': 0.002,           # 0.002 BTC per trade
    'profit_taking_in_btc': True        # Take profits in BTC, not USD
}
```

### 🛡️ **Risk Management with BTC**

```python
def btc_risk_management(strategy):
    """Manage risk when using BTC as capital"""
    
    # Current BTC position
    btc_balance = strategy.portfolio.get_portfolio_value()
    positions = strategy.get_positions()
    
    # Calculate exposure
    total_exposure = sum(
        pos.size * pos.current_price 
        for pos in positions.values()
    )
    
    exposure_pct = (total_exposure / btc_balance) * 100
    
    print(f"📊 BTC Risk Analysis:")
    print(f"   Total BTC: {btc_balance:.6f}")
    print(f"   Exposure: {exposure_pct:.1f}%")
    
    # Risk warnings
    if exposure_pct > 80:
        print("⚠️  HIGH RISK: Over 80% exposure!")
    elif exposure_pct > 60:
        print("⚠️  MEDIUM RISK: Over 60% exposure")
    else:
        print("✅ SAFE: Good risk level")

# Monitor risk
btc_risk_management(strategy)
```

### 🚀 **Getting Started with Your BTC**

1. **Decide Your Allocation:**
   ```python
   my_btc = 0.5  # Your BTC amount
   
   # Conservative: 70% hodl, 30% trade
   trading_btc = my_btc * 0.3
   
   # Aggressive: 50% hodl, 50% trade  
   trading_btc = my_btc * 0.5
   ```

2. **Choose Your Strategy:**
   ```python
   # Conservative DCA into blue-chip alts
   conservative_config = {
       'initial_capital': trading_btc,
       'base_currency': 'BTC',
       'symbols': ['ETH/BTC'],  # Just ETH
       'buy_interval': 86400,   # Once per day
       'base_buy_amount': trading_btc / 30  # Spread over 30 days
   }
   
   # Aggressive multi-alt DCA
   aggressive_config = {
       'initial_capital': trading_btc,
       'base_currency': 'BTC', 
       'symbols': ['ETH/BTC', 'ADA/BTC', 'SOL/BTC', 'DOT/BTC'],
       'buy_interval': 3600,    # Every hour
       'base_buy_amount': 0.001  # Small frequent buys
   }
   ```

3. **Start Small and Scale:**
   ```python
   # Start with a small test
   test_strategy = StrategyFactory.create_strategy('dca', {
       'initial_capital': 0.01,  # Just 0.01 BTC to test
       'base_currency': 'BTC',
       'symbols': ['ETH/BTC']
   })
   
   # Once comfortable, add more BTC
   if test_strategy.get_performance_metrics()['total_return_pct'] > 0:
       test_strategy.add_funds(Decimal('0.1'))  # Add more BTC
   ```

---

## 🎮 **Ready to Trade with Your BTC?**

Your BTC is perfect for:
- ✅ **Altcoin DCA** - Systematically accumulate promising alts
- ✅ **BTC Grid Trading** - Generate yield from BTC volatility  
- ✅ **Multi-strategy** - Diversify across different approaches
- ✅ **Risk Management** - Built-in position sizing and safety

Run this to get started:
```bash
cd "C:\Users\OM\Desktop\Poise Trader\core\strategies"
python demo_strategies.py
```

Just modify the configs to use `'base_currency': 'BTC'` and you're ready to roll! 🚀₿
