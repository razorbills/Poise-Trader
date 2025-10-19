"""
Demo script for Poise Trader Multi-Exchange Feed System

This script demonstrates how to:
- Connect to multiple exchanges
- Subscribe to market data streams
- Handle real-time data updates
- Use the unified feed system
"""

import asyncio
import logging
import json
from typing import Dict, Any

from . import FeedFactory, create_unified_feed
from ..framework.event_system import EventBus, Event


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)


async def demo_single_exchange():
    """Demo connecting to a single exchange (Binance)"""
    print("\n=== Single Exchange Demo (Binance) ===")
    
    # Configuration for Binance (public data only)
    config = {
        'api_key': '',  # Leave empty for public data
        'api_secret': '',
        'sandbox': True
    }
    
    # Create Binance feed
    feed = FeedFactory.create_feed('binance', config)
    
    if not feed:
        print("Failed to create Binance feed")
        return
    
    try:
        # Connect to the feed
        if await feed.connect():
            print("✅ Connected to Binance")
            
            # Subscribe to some popular symbols
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            await feed.subscribe(symbols)
            print(f"📡 Subscribed to: {symbols}")
            
            # Listen to market data for 30 seconds
            print("📊 Listening to market data...")
            count = 0
            async for data in feed.get_market_data():
                print(f"💰 {data.symbol}: ${data.price} (Vol: {data.volume}) [{data.exchange}]")
                count += 1
                
                # Stop after 10 messages for demo
                if count >= 10:
                    break
            
        else:
            print("❌ Failed to connect to Binance")
            
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await feed.disconnect()
        print("🔌 Disconnected from Binance")


async def demo_unified_feed():
    """Demo using unified feed with multiple exchanges"""
    print("\n=== Unified Feed Demo (Multiple Exchanges) ===")
    
    # Configuration for multiple exchanges
    exchange_configs = {
        'binance': {
            'api_key': '',
            'api_secret': '',
            'sandbox': True
        },
        'coinbase': {
            'api_key': '',
            'api_secret': '',
            'passphrase': '',
            'sandbox': True
        }
    }
    
    # Create event bus for system events
    event_bus = EventBus()
    await event_bus.start()
    
    try:
        # Create unified feed
        unified_feed = create_unified_feed(exchange_configs)
        
        # Manually add feeds (in real usage, this would be automatic)
        for exchange_name, config in exchange_configs.items():
            feed = FeedFactory.create_feed(exchange_name, config)
            if feed:
                await unified_feed.add_feed(feed)
        
        # Connect to all feeds
        if await unified_feed.connect():
            print("✅ Connected to unified feed")
            
            # Subscribe to symbols across all exchanges
            symbols = ['BTC/USD', 'ETH/USD']  # Use USD for Coinbase compatibility
            await unified_feed.subscribe(symbols)
            print(f"📡 Subscribed to: {symbols} across all exchanges")
            
            # Listen to aggregated data
            print("📊 Listening to unified market data...")
            count = 0
            async for data in unified_feed.get_market_data():
                print(f"💰 {data.symbol}: ${data.price} (Vol: {data.volume}) [{data.exchange}]")
                count += 1
                
                # Stop after 20 messages for demo
                if count >= 20:
                    break
            
            # Show feed metrics
            metrics = unified_feed.get_metrics()
            print(f"\n📈 Feed Metrics: {json.dumps(metrics, indent=2)}")
            
        else:
            print("❌ Failed to connect to unified feed")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await unified_feed.disconnect()
        await event_bus.stop()
        print("🔌 Disconnected from unified feed")


async def demo_historical_data():
    """Demo fetching historical data"""
    print("\n=== Historical Data Demo ===")
    
    config = {
        'api_key': '',
        'api_secret': '',
        'sandbox': True
    }
    
    # Create Binance feed
    feed = FeedFactory.create_feed('binance', config)
    
    if not feed:
        print("Failed to create feed")
        return
    
    try:
        if await feed.connect():
            print("✅ Connected to exchange")
            
            # Get historical data for the last hour
            import time
            end_time = int(time.time() * 1000)  # Current time in milliseconds
            start_time = end_time - (60 * 60 * 1000)  # 1 hour ago
            
            print("📜 Fetching historical data for BTC/USDT...")
            historical_data = await feed.get_historical_data('BTC/USDT', start_time, end_time)
            
            if historical_data:
                print(f"📊 Retrieved {len(historical_data)} historical records")
                
                # Show first and last few records
                for i, data in enumerate(historical_data[:3]):
                    print(f"🕐 {data.timestamp}: ${data.price} (Vol: {data.volume})")
                
                if len(historical_data) > 6:
                    print("... (skipped middle records) ...")
                    for data in historical_data[-3:]:
                        print(f"🕐 {data.timestamp}: ${data.price} (Vol: {data.volume})")
            else:
                print("❌ No historical data retrieved")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await feed.disconnect()
        print("🔌 Disconnected")


async def demo_event_system():
    """Demo integrating feeds with event system"""
    print("\n=== Event System Integration Demo ===")
    
    # Create event bus
    event_bus = EventBus(max_workers=2)
    await event_bus.start()
    
    # Event handler for market data
    async def handle_market_data(event: Event):
        data = event.data
        symbol = data.get('symbol')
        price = data.get('price')
        exchange = data.get('exchange')
        print(f"🔔 Event: {symbol} = ${price} on {exchange}")
    
    # Subscribe to market data events
    event_bus.subscribe('market.data', handle_market_data)
    
    try:
        config = {'api_key': '', 'api_secret': '', 'sandbox': True}
        
        # Create unified feed with event bus
        unified_feed = create_unified_feed({'binance': config})
        unified_feed.event_bus = event_bus  # Connect event bus
        
        # Add feed manually for demo
        feed = FeedFactory.create_feed('binance', config)
        if feed:
            await unified_feed.add_feed(feed)
        
        if await unified_feed.connect():
            print("✅ Connected with event system")
            
            await unified_feed.subscribe(['BTC/USDT'])
            print("📡 Subscribed to BTC/USDT")
            
            # Let it run for a bit to see events
            print("🔔 Listening for events...")
            await asyncio.sleep(10)  # Listen for 10 seconds
            
        else:
            print("❌ Failed to connect")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await unified_feed.disconnect()
        await event_bus.stop()
        print("🔌 Disconnected and stopped event system")


async def main():
    """Run all demos"""
    print("🚀 Poise Trader Feed System Demo\n")
    
    try:
        # Demo 1: Single exchange
        await demo_single_exchange()
        
        # Demo 2: Unified feed (multiple exchanges)
        await demo_unified_feed()
        
        # Demo 3: Historical data
        await demo_historical_data()
        
        # Demo 4: Event system integration
        await demo_event_system()
        
        print("\n✅ All demos completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
