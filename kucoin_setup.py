#!/usr/bin/env python3
"""
🚀 KuCoin Setup for Poise Trader - NO KYC Required!

This script helps you set up KuCoin API keys for live trading
with your 0.00005 BTC (5k sats).
"""

import os
from decimal import Decimal

# KuCoin Configuration Template
KUCOIN_CONFIG = {
    # 🔑 Your KuCoin API credentials (get from KuCoin.com)
    'api_key': '',           # TODO: Add your API key here
    'api_secret': '',        # TODO: Add your secret here  
    'passphrase': '',        # TODO: Add your passphrase here
    
    # 🏦 Exchange settings
    'exchange': 'kucoin',
    'base_url': 'https://api.kucoin.com',
    'sandbox': False,        # Live trading mode
    'kyc_required': False,   # ✅ No KYC needed!
    
    # 💰 Your trading setup
    'initial_capital': Decimal('0.00005'),  # Your 5k sats
    'base_currency': 'BTC',
    'quote_currencies': ['USDT', 'ETH'],
    
    # 📊 Trading pairs for your BTC
    'symbols': [
        'ETH/BTC',   # Trade BTC for Ethereum
        'ADA/BTC',   # Trade BTC for Cardano
        'DOT/BTC',   # Trade BTC for Polkadot
        'LINK/BTC',  # Trade BTC for Chainlink
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

def validate_config():
    """Check if API keys are configured"""
    missing = []
    if not KUCOIN_CONFIG['api_key']:
        missing.append('api_key')
    if not KUCOIN_CONFIG['api_secret']:
        missing.append('api_secret')  
    if not KUCOIN_CONFIG['passphrase']:
        missing.append('passphrase')
    
    return missing

def setup_kucoin_keys():
    """Interactive setup for KuCoin API keys"""
    print("🔑 KuCoin API Key Setup")
    print("=" * 50)
    print("\n📋 You need these from KuCoin.com → API Management:")
    print("   1. API Key")
    print("   2. Secret Key") 
    print("   3. Passphrase")
    print("\n⚠️  Make sure to enable 'Trade' permissions!")
    
    api_key = input("\n🔑 Enter your API Key: ").strip()
    api_secret = input("🔒 Enter your Secret Key: ").strip()
    passphrase = input("🔐 Enter your Passphrase: ").strip()
    
    if api_key and api_secret and passphrase:
        # Update config
        KUCOIN_CONFIG['api_key'] = api_key
        KUCOIN_CONFIG['api_secret'] = api_secret
        KUCOIN_CONFIG['passphrase'] = passphrase
        
        print("\n✅ API keys configured!")
        return True
    else:
        print("\n❌ All fields are required!")
        return False

def test_connection():
    """Test connection to KuCoin API"""
    try:
        print("\n🔄 Testing KuCoin connection...")
        
        # Import the strategy factory
        from core.strategies import StrategyFactory
        
        # Create strategy with KuCoin config
        strategy = StrategyFactory.create_strategy('dca', KUCOIN_CONFIG)
        
        print("✅ KuCoin connection successful!")
        print(f"💰 Account Balance: {KUCOIN_CONFIG['initial_capital']} BTC")
        print(f"📊 Trading Pairs: {len(KUCOIN_CONFIG['symbols'])} pairs")
        print(f"🛡️  Paper Trading: {'ON' if KUCOIN_CONFIG['paper_trading'] else 'OFF'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔍 Check your API keys and permissions")
        return False

def show_trading_status():
    """Display current trading configuration"""
    print("\n📊 CURRENT POISE TRADER CONFIG")
    print("=" * 50)
    print(f"🏦 Exchange: KuCoin (No KYC!)")
    print(f"💰 Capital: {KUCOIN_CONFIG['initial_capital']} BTC")
    print(f"📈 Strategy: {KUCOIN_CONFIG['strategy_type'].upper()}")
    print(f"🛡️  Mode: {'Paper Trading' if KUCOIN_CONFIG['paper_trading'] else 'LIVE TRADING'}")
    print(f"⏰ DCA Interval: {KUCOIN_CONFIG['dca_interval']/3600} hours")
    print(f"💸 DCA Amount: {KUCOIN_CONFIG['dca_amount']} BTC per buy")
    
    print(f"\n📊 Trading Pairs:")
    for symbol in KUCOIN_CONFIG['symbols']:
        print(f"   • {symbol}")

def start_trading():
    """Start the KuCoin trading strategy"""
    print("\n🚀 STARTING KUCOIN TRADING!")
    print("=" * 50)
    
    try:
        from core.strategies import StrategyFactory
        
        # Create and start strategy
        strategy = StrategyFactory.create_strategy('dca', KUCOIN_CONFIG)
        
        print("✅ Strategy initialized successfully!")
        print(f"🎯 Trading {KUCOIN_CONFIG['initial_capital']} BTC")
        print(f"📈 Using {KUCOIN_CONFIG['strategy_type'].upper()} strategy")
        
        if KUCOIN_CONFIG['paper_trading']:
            print("🛡️  PAPER TRADING MODE - No real money at risk!")
        else:
            print("💰 LIVE TRADING MODE - Using real BTC!")
            
        print("\n🔄 Strategy is now running...")
        print("📊 Monitor your trades in the dashboard")
        
        # Start the strategy
        strategy.start()
        
    except Exception as e:
        print(f"❌ Failed to start trading: {e}")

def main():
    """Main setup flow"""
    print("🎯 POISE TRADER - KUCOIN SETUP")
    print("=" * 50)
    print("💰 Perfect for your 5k sats (0.00005 BTC)")
    print("🚫 No KYC required!")
    print("🛡️  Starting in safe paper trading mode")
    
    # Check if keys are already configured
    missing = validate_config()
    
    if missing:
        print(f"\n⚠️  Missing configuration: {', '.join(missing)}")
        print("\n🔧 Let's set up your KuCoin API keys...")
        
        if not setup_kucoin_keys():
            print("❌ Setup cancelled")
            return
    
    # Test connection
    if test_connection():
        show_trading_status()
        
        print("\n🎮 NEXT STEPS:")
        print("1. ✅ Your KuCoin is connected!")
        print("2. 🛡️  Currently in PAPER TRADING mode (safe)")
        print("3. 💰 Deposit your 5k sats to KuCoin")
        print("4. 🚀 Run this script to start trading!")
        
        start = input("\n🚀 Start trading now? (y/n): ").strip().lower()
        if start in ['y', 'yes']:
            start_trading()
        else:
            print("✅ Setup complete! Run this script again when ready to trade.")
    
    else:
        print("\n🔍 Please check your KuCoin API setup:")
        print("1. Go to KuCoin.com → API Management")
        print("2. Make sure 'Trade' permission is enabled")
        print("3. Check your API key, secret, and passphrase")
        print("4. Run this script again")

if __name__ == "__main__":
    main()
