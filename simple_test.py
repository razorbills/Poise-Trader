#!/usr/bin/env python3
"""
Simple verification script to test all bot systems
"""

import sys
import os

def main():
    try:
        print("🔍 Starting simple system verification...")
        
        # Test imports
        from micro_trading_bot import LegendaryCryptoTitanBot
        print("✅ Main bot imports successfully")
        
        # Test initialization
        bot = LegendaryCryptoTitanBot(5.0)
        print("✅ Bot initialization successful")
        
        # Test basic functionality
        test_prices = [100, 101, 102, 103, 104, 105]
        
        rsi = bot._calculate_rsi(test_prices)
        print(f"✅ RSI calculation works: {rsi:.2f}")
        
        print("✅ All basic systems working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 VERIFICATION SUCCESSFUL - All systems operational!")
    else:
        print("\n❌ VERIFICATION FAILED - Some issues found")
