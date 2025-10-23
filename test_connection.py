import aiohttp
import asyncio
import ssl
import certifi

async def test_mexc_connection():
    """Test connection to MEXC API"""
    print("Testing MEXC API connection...")
    
    # Test URLs
    urls = [
        "https://api.mexc.com/api/v3/ping",
        "https://api.mexc.com/api/v3/ticker/price?symbol=BTCUSDT",
        "https://www.mexc.com",
    ]
    
    # Create SSL context
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    # Test with different configurations
    configs = [
        {"ssl": ssl_context},
        {"ssl": False},  # No SSL verification
        {},  # Default
    ]
    
    for url in urls:
        print(f"\nTesting: {url}")
        for i, config in enumerate(configs):
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, **config) as response:
                        if response.status == 200:
                            print(f"  ✅ Config {i+1}: Success! Status: {response.status}")
                            if 'ticker' in url:
                                data = await response.json()
                                print(f"     BTC Price: ${float(data['price']):,.2f}")
                        else:
                            print(f"  ⚠️ Config {i+1}: Status {response.status}")
            except Exception as e:
                print(f"  ❌ Config {i+1}: {str(e)[:100]}")
    
    # Test DNS resolution
    print("\nTesting DNS resolution...")
    import socket
    try:
        ip = socket.gethostbyname("api.mexc.com")
        print(f"  ✅ api.mexc.com resolves to: {ip}")
    except Exception as e:
        print(f"  ❌ DNS resolution failed: {e}")
    
    # Test alternative API endpoints
    print("\nTesting alternative endpoints...")
    alt_urls = [
        "https://www.binance.com/api/v3/ping",
        "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
    ]
    
    for url in alt_urls:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        print(f"  ✅ {url.split('/')[2]}: Connected")
                    else:
                        print(f"  ⚠️ {url.split('/')[2]}: Status {response.status}")
        except Exception as e:
            print(f"  ❌ {url.split('/')[2]}: {str(e)[:50]}")

if __name__ == "__main__":
    asyncio.run(test_mexc_connection())
