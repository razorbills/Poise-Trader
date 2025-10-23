#!/usr/bin/env python3
"""
Test REAL MEXC connection
"""
import asyncio
import aiohttp
import ssl
import certifi

async def test_real_mexc():
    """Test getting REAL prices from MEXC"""
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    base_url = "https://api.mexc.com"
    
    print("🔍 Testing REAL MEXC Connection...")
    print("=" * 50)
    
    # Method 1: Basic request (no SSL verification)
    print("\n📡 Method 1: Basic request (no SSL verification)")
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        connector = aiohttp.TCPConnector(ssl=False)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for symbol in symbols:
                try:
                    url = f"{base_url}/api/v3/ticker/price?symbol={symbol}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            price = float(data['price'])
                            print(f"   ✅ {symbol}: ${price:,.2f}")
                        else:
                            print(f"   ❌ {symbol}: Status {response.status}")
                except Exception as e:
                    print(f"   ❌ {symbol}: {str(e)[:50]}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 2: With SSL context
    print("\n📡 Method 2: With SSL context")
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        timeout = aiohttp.ClientTimeout(total=5)
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for symbol in symbols:
                try:
                    url = f"{base_url}/api/v3/ticker/price?symbol={symbol}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            price = float(data['price'])
                            print(f"   ✅ {symbol}: ${price:,.2f}")
                        else:
                            print(f"   ❌ {symbol}: Status {response.status}")
                except Exception as e:
                    print(f"   ❌ {symbol}: {str(e)[:50]}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 3: Using requests library as fallback
    print("\n📡 Method 3: Using requests library")
    try:
        import requests
        for symbol in symbols:
            try:
                url = f"{base_url}/api/v3/ticker/price?symbol={symbol}"
                response = requests.get(url, timeout=5, verify=False)
                if response.status_code == 200:
                    data = response.json()
                    price = float(data['price'])
                    print(f"   ✅ {symbol}: ${price:,.2f}")
                else:
                    print(f"   ❌ {symbol}: Status {response.status_code}")
            except Exception as e:
                print(f"   ❌ {symbol}: {str(e)[:50]}")
    except ImportError:
        print("   ⚠️ requests library not available")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("Test complete!")

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    asyncio.run(test_real_mexc())
