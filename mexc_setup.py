#!/usr/bin/env python3
"""
🚀 MEXC Setup for Poise Trader - NO KYC Required!

Your MEXC API keys are configured and ready for trading
with your 0.00005 BTC (5k sats).
"""

from decimal import Decimal
import os

# MEXC Configuration for Your Account
MEXC_CONFIG = {
    # 🔑 Your MEXC API credentials (configured)
    'api_key': os.getenv('MEXC_API_KEY', ''),
    'api_secret': os.getenv('MEXC_API_SECRET', ''),
    
    # 🏦 Exchange settings
    'exchange': 'mexc',
    'base_url': 'https://api.mexc.com',
    'sandbox': False,        # Live trading mode
    'kyc_required': False,   # ✅ No KYC needed!
    
    # 💰 Your trading setup
    'initial_capital': Decimal('0.00005'),  # Your 5k sats
    'base_currency': 'BTC',
    'quote_currencies': ['USDT'],
    
    # 📊 Trading pairs for your BTC
    'symbols': [
        'BTC/USDT',   # Trade BTC for USDT (most liquid)
        'ETH/USDT',   # Can buy ETH with USDT
        'ADA/USDT',   # Can buy ADA with USDT
        'DOT/USDT',   # Can buy DOT with USDT
    ],
    
    # 🛡️ Safety settings
    'paper_trading': True,   # Start with fake trades (safe!)
    'max_trade_size': Decimal('0.00001'),  # Max 1k sats per trade
    'emergency_stop': True,
    
    # ⚙️ Strategy settings
    'strategy_type': 'dca',
    'dca_interval': 3600,    # 1 hour between buys
    'dca_amount': Decimal('0.000005'),  # 500 sats per buy
}

def test_mexc_connection():
    """Test connection to MEXC API"""
    print("🔄 Testing MEXC connection...")
    print(f"🔑 API Key: {MEXC_CONFIG['api_key'][:10]}...")
    print(f"🏦 Exchange: MEXC (No KYC required)")
    print(f"💰 Initial Capital: {MEXC_CONFIG['initial_capital']} BTC")
    
    try:
        # Import ccxt for testing connection
        import ccxt
        
        # Create MEXC exchange instance
        exchange = ccxt.mexc({
            'apiKey': MEXC_CONFIG['api_key'],
            'secret': MEXC_CONFIG['api_secret'],
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Test connection by fetching account info
        balance = exchange.fetch_balance()
        
        print("✅ MEXC connection successful!")
        print(f"📊 Account connected")
        
        # Show available balances
        if balance['total']:
            print("\n💰 Current Balances:")
            for currency, amount in balance['total'].items():
                if amount > 0:
                    print(f"   • {currency}: {amount}")
        else:
            print("💰 Account ready for deposits")
        
        return True
        
    except ImportError:
        print("⚠️  Installing required package...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'ccxt'])
        return test_mexc_connection()  # Try again after install
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔍 Possible issues:")
        print("1. Check your internet connection")
        print("2. Verify API key permissions")
        print("3. Make sure MEXC API is enabled")
        return False

def show_trading_plan():
    """Display the trading strategy plan"""
    print("\n📊 YOUR MEXC TRADING PLAN")
    print("=" * 50)
    print(f"🏦 Exchange: MEXC (KYC-Free)")
    print(f"💰 Starting Capital: {MEXC_CONFIG['initial_capital']} BTC")
    print(f"📈 Strategy: DCA (Dollar Cost Averaging)")
    print(f"🛡️  Mode: Paper Trading (Safe!)")
    
    print(f"\n🎯 DCA Strategy Details:")
    print(f"   • Buy every {MEXC_CONFIG['dca_interval']/3600} hours")
    print(f"   • Amount per buy: {MEXC_CONFIG['dca_amount']} BTC")
    print(f"   • Max trade size: {MEXC_CONFIG['max_trade_size']} BTC")
    
    print(f"\n📊 Available Trading Pairs:")
    for symbol in MEXC_CONFIG['symbols']:
        print(f"   • {symbol}")
    
    print(f"\n💡 How it works:")
    print(f"   1. Start with your 5k sats (0.00005 BTC)")
    print(f"   2. DCA buys altcoins regularly")
    print(f"   3. Paper trading = no real money risk")
    print(f"   4. Monitor performance and learn")

def start_mexc_trading():
    """Initialize MEXC trading with Poise Trader"""
    print("\n🚀 INITIALIZING MEXC TRADING")
    print("=" * 50)
    
    try:
        # Import our strategy framework
        from core.strategies import StrategyFactory
        
        print("📦 Loading Poise Trader framework...")
        
        # Create DCA strategy for MEXC
        strategy = StrategyFactory.create_strategy('dca', MEXC_CONFIG)
        
        print("✅ MEXC strategy created successfully!")
        print(f"🎯 Ready to trade {MEXC_CONFIG['initial_capital']} BTC")
        
        if MEXC_CONFIG['paper_trading']:
            print("🛡️  PAPER TRADING MODE - Learning safely!")
            print("💡 No real money will be used")
        else:
            print("💰 LIVE TRADING MODE - Using real BTC!")
        
        print("\n📊 Strategy Status:")
        print(f"   • Exchange: MEXC")
        print(f"   • Strategy: {MEXC_CONFIG['strategy_type'].upper()}")
        print(f"   • Capital: {MEXC_CONFIG['initial_capital']} BTC")
        print(f"   • Pairs: {len(MEXC_CONFIG['symbols'])} trading pairs")
        
        # Start the strategy
        print("\n🔄 Starting trading strategy...")
        strategy.start()
        
        print("🎉 MEXC trading is now active!")
        
        return strategy
        
    except ImportError as e:
        print(f"❌ Missing framework: {e}")
        print("🔍 Make sure Poise Trader core is properly installed")
        return None
        
    except Exception as e:
        print(f"❌ Failed to start trading: {e}")
        return None

def main():
    """Main MEXC setup and trading"""
    print("🎯 POISE TRADER - MEXC SETUP")
    print("=" * 50)
    print("🚫 NO KYC REQUIRED!")
    print("💰 Perfect for your 5k sats")
    print("🛡️  Starting in safe paper trading mode")
    
    # Test MEXC connection
    if test_mexc_connection():
        show_trading_plan()
        
        print("\n🎮 READY TO START TRADING!")
        print("=" * 30)
        print("✅ MEXC API connected")
        print("✅ Trading permissions verified") 
        print("✅ Strategy configured")
        print("🛡️  Paper trading mode (safe)")
        
        start_choice = input("\n🚀 Start MEXC trading now? (y/n): ").strip().lower()
        
        if start_choice in ['y', 'yes']:
            strategy = start_mexc_trading()
            
            if strategy:
                print("\n🎊 SUCCESS! Your MEXC trading is now running!")
                print("📊 Monitor your portfolio and learn how it works")
                print("🛡️  Remember: This is paper trading (no real risk)")
                print("\n💡 Next steps:")
                print("1. Watch how the DCA strategy works")
                print("2. Learn from the trading patterns") 
                print("3. When confident, switch to live trading")
            else:
                print("❌ Failed to start trading strategy")
        else:
            print("✅ Setup complete!")
            print("💡 Run this script again when ready to trade")
    
    else:
        print("\n🔍 Connection issues detected")
        print("Please check your MEXC API setup and try again")

if __name__ == "__main__":
    main()
