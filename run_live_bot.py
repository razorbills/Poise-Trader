#!/usr/bin/env python3
"""
🔥 RUN POISE TRADER WITH LIVE DATA
Launch the bot with real-time MEXC market data feeds

FEATURES:
✅ Live MEXC price feeds  
✅ Real-time market data
✅ Paper trading mode (safe)
✅ Current 2025 prices
❌ No more fake 2022 data!
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

_REAL_TRADING_ENABLED = str(os.getenv('REAL_TRADING', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
_STRICT_REAL_DATA = str(os.getenv('STRICT_REAL_DATA', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
if _REAL_TRADING_ENABLED or _STRICT_REAL_DATA:
    raise RuntimeError(
        "run_live_bot.py is a paper-trading launcher (it forces PAPER_TRADING_MODE=true). "
        "It cannot be used with REAL_TRADING/STRICT_REAL_DATA."
    )

# Load environment with live data settings
load_dotenv()

# Force live data mode
os.environ['USE_LIVE_DATA'] = 'true'
os.environ['USE_DEMO_DATA'] = 'false'
os.environ['PAPER_TRADING_MODE'] = 'true'  # Keep safe for testing
os.environ['USE_TESTNET'] = 'false'

print("🚀 POISE TRADER - LIVE DATA MODE")
print("=" * 50)
print("🔥 Using REAL-TIME MEXC market data")
print("📡 Live prices from actual markets")
print("✅ Paper trading mode (safe)")
print("❌ No more fake 2022 prices!")
print()

# Import after setting environment variables
sys.path.append(str(Path(__file__).parent))

try:
    # First test live data connection
    print("🧪 Testing live data connection...")
    
    from live_paper_trading_test import LiveMexcDataFeed
    
    async def test_connection():
        feed = LiveMexcDataFeed()
        prices = await feed.get_multiple_prices(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
        
        if prices:
            print("✅ Live data connection successful!")
            print("📊 Current live prices:")
            for symbol, price in prices.items():
                print(f"   {symbol:10} ${price:>10,.2f}")
            print()
            return True
        else:
            print("❌ Failed to get live data")
            return False
    
    # Test connection first
    if asyncio.run(test_connection()):
        
        print("🤖 Starting Poise Master Bot with LIVE data...")
        print("=" * 50)
        
        # Import the main bot
        from poise_master_bot import main as run_main_bot
        
        # Run the main bot with live data
        asyncio.run(run_main_bot())
        
    else:
        print("❌ Cannot start bot - live data connection failed")
        print("💡 Check your internet connection and MEXC API status")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all dependencies are installed")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Check logs for detailed error information")
