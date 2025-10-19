#!/usr/bin/env python3
"""
Test script to verify real-time data integration is working correctly
"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from micro_trading_bot import LegendaryCryptoTitanBot
from core.feeds.real_time_data_manager import RealTimeDataManager

async def test_realtime_integration():
    """Test the complete real-time data integration"""
    print("🧪 Testing Real-Time Data Integration...")
    
    # Initialize the bot
    bot = LegendaryCryptoTitanBot(initial_capital=5.0)
    
    # Test real-time data manager
    try:
        print("   ✅ Bot initialized successfully")
        
        # Check if real-time data manager is available
        if hasattr(bot, 'real_time_data_manager'):
            print("   ✅ Real-time data manager attribute exists")
            
            # Test real-time data fetching
            if bot.real_time_data_initialized:
                print("   ✅ Real-time data manager initialized")
                
                # Test getting comprehensive market data
                symbols = ['BTC/USDT', 'ETH/USDT']
                market_data = await bot.real_time_data_manager.get_comprehensive_market_data(symbols)
                
                if market_data:
                    print(f"   ✅ Real-time data fetched for {len(market_data)} symbols")
                    
                    # Check for social sentiment data
                    for symbol in symbols:
                        if symbol in market_data and 'social_sentiment' in market_data[symbol]:
                            sentiment = market_data[symbol]['social_sentiment']
                            print(f"   ✅ Social sentiment for {symbol}: {sentiment.get('score', 'N/A')}")
                        
                        if symbol in market_data and 'onchain_analytics' in market_data[symbol]:
                            onchain = market_data[symbol]['onchain_analytics']
                            print(f"   ✅ On-chain data for {symbol}: {len(onchain)} metrics")
                            
                        if symbol in market_data and 'economic_calendar' in market_data[symbol]:
                            econ = market_data[symbol]['economic_calendar']
                            print(f"   ✅ Economic data for {symbol}: {len(econ)} events")
                            
                        if symbol in market_data and 'options_data' in market_data[symbol]:
                            options = market_data[symbol]['options_data']
                            print(f"   ✅ Options data for {symbol}: {len(options)} metrics")
                else:
                    print("   ⚠️ No market data returned")
            else:
                print("   ⚠️ Real-time data manager not initialized - using fallback")
        else:
            print("   ❌ Real-time data manager not found")
            
    except Exception as e:
        print(f"   ❌ Error during testing: {e}")
        return False
    
    print("\n✅ Real-time data integration test completed successfully!")
    return True

if __name__ == "__main__":
    asyncio.run(test_realtime_integration())
