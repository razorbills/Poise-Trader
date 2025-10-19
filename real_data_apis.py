#!/usr/bin/env python3
"""
🌐 REAL DATA APIs - No Simulations!
Integrates FREE real-world data sources for sentiment, on-chain, and macro data
"""

import asyncio
import aiohttp
import time
from typing import Dict, Optional
import json

class RealDataAPIs:
    """Real data from free APIs - no simulations!"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        cache_time = self.cache[cache_key].get('timestamp', 0)
        return (time.time() - cache_time) < self.cache_duration
    
    # ==================== SOCIAL SENTIMENT APIs ====================
    
    async def get_fear_greed_index(self) -> Dict:
        """
        Get Crypto Fear & Greed Index (FREE API)
        Source: Alternative.me - No API key required
        """
        cache_key = 'fear_greed'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            session = await self._get_session()
            url = "https://api.alternative.me/fng/?limit=1"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and len(data['data']) > 0:
                        fg_data = data['data'][0]
                        
                        result = {
                            'index': int(fg_data['value']),  # 0-100
                            'classification': fg_data['value_classification'],  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
                            'timestamp': int(fg_data['timestamp']),
                            'source': 'Alternative.me (REAL)',
                            'data_type': 'REAL'
                        }
                        
                        self.cache[cache_key] = {'data': result, 'timestamp': time.time()}
                        print(f"📊 Fear & Greed Index: {result['index']}/100 ({result['classification']}) - REAL DATA")
                        return result
        except Exception as e:
            print(f"⚠️ Fear & Greed API error: {e}")
        
        return {'error': 'Failed to get fear & greed index'}
    
    async def get_coingecko_sentiment(self, symbol: str) -> Dict:
        """
        Get coin sentiment from CoinGecko (FREE API)
        Source: CoinGecko - No API key required
        """
        cache_key = f'coingecko_sentiment_{symbol}'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Map symbol to CoinGecko ID
            coin_map = {
                'BTC/USDT': 'bitcoin',
                'ETH/USDT': 'ethereum',
                'SOL/USDT': 'solana',
                'BNB/USDT': 'binancecoin',
                'AVAX/USDT': 'avalanche-2'
            }
            
            coin_id = coin_map.get(symbol)
            if not coin_id:
                return {'error': 'Symbol not mapped'}
            
            session = await self._get_session()
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract sentiment indicators
                    sentiment_votes = data.get('sentiment_votes_up_percentage', 50)
                    community_score = data.get('community_score', 0)
                    developer_score = data.get('developer_score', 0)
                    
                    # Calculate overall sentiment (-1 to 1)
                    sentiment_score = (sentiment_votes - 50) / 50  # Convert 0-100 to -1 to 1
                    
                    result = {
                        'score': sentiment_score,
                        'classification': 'bullish' if sentiment_score > 0.1 else 'bearish' if sentiment_score < -0.1 else 'neutral',
                        'sentiment_votes_up': sentiment_votes,
                        'community_score': community_score,
                        'developer_score': developer_score,
                        'confidence': abs(sentiment_score),
                        'source': f'CoinGecko {coin_id} (REAL)',
                        'data_type': 'REAL',
                        'timestamp': time.time()
                    }
                    
                    self.cache[cache_key] = {'data': result, 'timestamp': time.time()}
                    print(f"📊 {symbol} Sentiment: {sentiment_score:+.2f} ({result['classification']}) - REAL DATA")
                    return result
                    
        except Exception as e:
            print(f"⚠️ CoinGecko sentiment error for {symbol}: {e}")
        
        return {'error': 'Failed to get CoinGecko sentiment'}
    
    async def get_combined_sentiment(self, symbol: str) -> Dict:
        """Combine multiple sentiment sources for better accuracy"""
        try:
            # Get both sources
            fear_greed = await self.get_fear_greed_index()
            coingecko = await self.get_coingecko_sentiment(symbol)
            
            if 'error' not in fear_greed and 'error' not in coingecko:
                # Combine scores (60% CoinGecko, 40% Fear & Greed)
                fg_normalized = (fear_greed['index'] - 50) / 50  # Convert to -1 to 1
                combined_score = (coingecko['score'] * 0.6) + (fg_normalized * 0.4)
                
                return {
                    'score': combined_score,
                    'classification': 'bullish' if combined_score > 0.1 else 'bearish' if combined_score < -0.1 else 'neutral',
                    'confidence': abs(combined_score),
                    'fear_greed_index': fear_greed['index'],
                    'coingecko_sentiment': coingecko['sentiment_votes_up'],
                    'sources': ['CoinGecko', 'Alternative.me'],
                    'data_type': 'REAL',
                    'timestamp': time.time()
                }
            elif 'error' not in coingecko:
                return coingecko
            elif 'error' not in fear_greed:
                fg_normalized = (fear_greed['index'] - 50) / 50
                return {
                    'score': fg_normalized,
                    'classification': fear_greed['classification'].lower(),
                    'confidence': abs(fg_normalized),
                    'source': 'Alternative.me (REAL)',
                    'data_type': 'REAL',
                    'timestamp': time.time()
                }
        except Exception as e:
            print(f"⚠️ Combined sentiment error: {e}")
        
        return {'error': 'All sentiment sources failed'}
    
    # ==================== ON-CHAIN DATA APIs ====================
    
    async def get_blockchain_stats(self, symbol: str) -> Dict:
        """
        Get blockchain statistics (FREE API)
        Source: Blockchain.com for BTC, CoinGecko for others
        """
        cache_key = f'blockchain_{symbol}'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            if 'BTC' in symbol:
                # Use Blockchain.com API (free, no key)
                session = await self._get_session()
                url = "https://blockchain.info/stats?format=json"
                
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        result = {
                            'network_metrics': {
                                'hash_rate': data.get('hash_rate', 0) / 1e9,  # Convert to EH/s
                                'difficulty': data.get('difficulty', 0),
                                'total_btc': data.get('totalbc', 0) / 1e8,
                                'market_price_usd': data.get('market_price_usd', 0),
                                'trade_volume_usd': data.get('trade_volume_usd', 0),
                                'miners_revenue_usd': data.get('miners_revenue_usd', 0)
                            },
                            'transactions': {
                                'n_tx': data.get('n_tx', 0),
                                'total_fees': data.get('total_fees_btc', 0) / 1e8
                            },
                            'source': 'Blockchain.com (REAL)',
                            'data_type': 'REAL',
                            'timestamp': time.time()
                        }
                        
                        self.cache[cache_key] = {'data': result, 'timestamp': time.time()}
                        print(f"⛓️ BTC On-Chain: Hash Rate {result['network_metrics']['hash_rate']:.2f} EH/s - REAL DATA")
                        return result
            else:
                # Use CoinGecko for other coins
                coin_map = {
                    'ETH/USDT': 'ethereum',
                    'SOL/USDT': 'solana',
                    'BNB/USDT': 'binancecoin'
                }
                coin_id = coin_map.get(symbol)
                if not coin_id:
                    return {'error': 'Symbol not supported'}
                
                session = await self._get_session()
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        market_data = data.get('market_data', {})
                        
                        result = {
                            'network_metrics': {
                                'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                                'total_volume': market_data.get('total_volume', {}).get('usd', 0),
                                'circulating_supply': market_data.get('circulating_supply', 0),
                                'price_change_24h': market_data.get('price_change_percentage_24h', 0)
                            },
                            'source': f'CoinGecko {coin_id} (REAL)',
                            'data_type': 'REAL',
                            'timestamp': time.time()
                        }
                        
                        self.cache[cache_key] = {'data': result, 'timestamp': time.time()}
                        print(f"⛓️ {symbol} On-Chain: Market Cap ${result['network_metrics']['market_cap']:,.0f} - REAL DATA")
                        return result
                        
        except Exception as e:
            print(f"⚠️ Blockchain stats error for {symbol}: {e}")
        
        return {'error': 'Failed to get blockchain stats'}
    
    # ==================== MACRO ECONOMIC DATA APIs ====================
    
    async def get_fear_greed_macro(self) -> Dict:
        """
        Get macro fear & greed as proxy for VIX
        This is crypto-specific but correlates with traditional markets
        """
        fg = await self.get_fear_greed_index()
        if 'error' not in fg:
            # Use fear & greed as proxy for market volatility
            vix_proxy = 100 - fg['index']  # Invert: high fear = high VIX
            
            return {
                'vix_proxy': vix_proxy,
                'market_sentiment': fg['classification'],
                'fear_greed_index': fg['index'],
                'source': 'Alternative.me Fear & Greed (REAL)',
                'data_type': 'REAL',
                'timestamp': time.time()
            }
        return {'error': 'Failed to get macro indicators'}
    
    async def get_bitcoin_dominance(self) -> Dict:
        """
        Get Bitcoin dominance (FREE API)
        Important macro indicator for crypto markets
        """
        cache_key = 'btc_dominance'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            session = await self._get_session()
            url = "https://api.coingecko.com/api/v3/global"
            
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    market_data = data.get('data', {})
                    
                    result = {
                        'btc_dominance': market_data.get('market_cap_percentage', {}).get('btc', 0),
                        'eth_dominance': market_data.get('market_cap_percentage', {}).get('eth', 0),
                        'total_market_cap': market_data.get('total_market_cap', {}).get('usd', 0),
                        'total_volume_24h': market_data.get('total_volume', {}).get('usd', 0),
                        'active_cryptocurrencies': market_data.get('active_cryptocurrencies', 0),
                        'source': 'CoinGecko Global (REAL)',
                        'data_type': 'REAL',
                        'timestamp': time.time()
                    }
                    
                    self.cache[cache_key] = {'data': result, 'timestamp': time.time()}
                    print(f"📊 BTC Dominance: {result['btc_dominance']:.2f}% - REAL DATA")
                    return result
                    
        except Exception as e:
            print(f"⚠️ Bitcoin dominance error: {e}")
        
        return {'error': 'Failed to get BTC dominance'}
    
    async def get_macro_indicators(self) -> Dict:
        """
        Get comprehensive macro indicators
        Combines multiple free sources
        """
        try:
            fear_greed = await self.get_fear_greed_macro()
            btc_dom = await self.get_bitcoin_dominance()
            
            if 'error' not in fear_greed and 'error' not in btc_dom:
                return {
                    'fear_greed_index': fear_greed.get('fear_greed_index', 50),
                    'vix_proxy': fear_greed.get('vix_proxy', 50),
                    'market_sentiment': fear_greed.get('market_sentiment', 'neutral'),
                    'btc_dominance': btc_dom.get('btc_dominance', 0),
                    'total_market_cap': btc_dom.get('total_market_cap', 0),
                    'total_volume_24h': btc_dom.get('total_volume_24h', 0),
                    'source': 'Multiple Free APIs (REAL)',
                    'data_type': 'REAL',
                    'timestamp': time.time()
                }
        except Exception as e:
            print(f"⚠️ Macro indicators error: {e}")
        
        return {'error': 'Failed to get macro indicators'}


# Global instance
real_data_apis = RealDataAPIs()


async def test_real_apis():
    """Test all real API integrations"""
    print("\n" + "="*70)
    print("🌐 TESTING REAL DATA APIs")
    print("="*70)
    
    apis = RealDataAPIs()
    
    # Test Fear & Greed
    print("\n📊 Testing Fear & Greed Index...")
    fg = await apis.get_fear_greed_index()
    print(f"   Result: {json.dumps(fg, indent=2)}")
    
    # Test CoinGecko Sentiment
    print("\n📊 Testing CoinGecko Sentiment (BTC)...")
    sentiment = await apis.get_coingecko_sentiment('BTC/USDT')
    print(f"   Result: {json.dumps(sentiment, indent=2)}")
    
    # Test Combined Sentiment
    print("\n📊 Testing Combined Sentiment...")
    combined = await apis.get_combined_sentiment('BTC/USDT')
    print(f"   Result: {json.dumps(combined, indent=2)}")
    
    # Test Blockchain Stats
    print("\n⛓️ Testing Blockchain Stats (BTC)...")
    blockchain = await apis.get_blockchain_stats('BTC/USDT')
    print(f"   Result: {json.dumps(blockchain, indent=2, default=str)}")
    
    # Test Bitcoin Dominance
    print("\n📊 Testing Bitcoin Dominance...")
    dominance = await apis.get_bitcoin_dominance()
    print(f"   Result: {json.dumps(dominance, indent=2)}")
    
    # Test Macro Indicators
    print("\n📊 Testing Macro Indicators...")
    macro = await apis.get_macro_indicators()
    print(f"   Result: {json.dumps(macro, indent=2)}")
    
    await apis.close()
    
    print("\n" + "="*70)
    print("✅ ALL REAL API TESTS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_real_apis())
