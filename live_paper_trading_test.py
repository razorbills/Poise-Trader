#!/usr/bin/env python3
"""
 LIVE PAPER TRADING TEST
Test paper trading with REAL live MEXC prices instead of demo data
"""

import asyncio
import aiohttp
import time
import warnings
import urllib3
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import random
import sys
from pathlib import Path
import requests
import urllib3
import os
import time
import uuid

try:
    import ccxt.async_support as ccxt  # type: ignore
except Exception:
    ccxt = None

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


async def _requests_get_async(url: str, timeout: float = 5, verify: bool = False, headers: Dict[str, str] = None):
    """Run requests.get in a worker thread so async loops don't stall."""
    try:
        return await asyncio.to_thread(requests.get, url, timeout=timeout, verify=verify, headers=headers)
    except Exception:
        # Fallback for environments without to_thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: requests.get(url, timeout=timeout, verify=verify, headers=headers))

# Add core modules to path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
import os
load_dotenv()

# MEXC API Configuration
MEXC_API_KEY = os.getenv('MEXC_API_KEY', '')
MEXC_SECRET_KEY = os.getenv('MEXC_SECRET_KEY', '') or os.getenv('MEXC_API_SECRET', '')
PAPER_TRADING_MODE = True  # Set to False for real trading

_REAL_TRADING_ENABLED = str(os.getenv('REAL_TRADING', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
_STRICT_REAL_DATA = str(os.getenv('STRICT_REAL_DATA', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
ALLOW_SIMULATED_FEATURES = (
    str(os.getenv('ALLOW_SIMULATED_FEATURES', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
    and not _REAL_TRADING_ENABLED
    and not _STRICT_REAL_DATA
)

class LiveMexcDataFeed:
    """Live MEXC data feed with comprehensive market data for AI learning"""
    
    def __init__(self):
        self.base_url = "https://api.mexc.com"
        self.prices_cache = {}
        self.orderbook_cache = {}  # Order book depth data
        self.trades_cache = {}  # Recent trades
        self.klines_cache = {}  # OHLCV candles
        self.ticker_24h_cache = {}  # 24h statistics
        self.last_update = None
        # Try alternative endpoints if main one fails
        self.alternative_urls = [
            "https://api.mexc.com",
            "https://www.mexc.com",
            "https://contract.mexc.com"
        ]
        self.exchange_symbols = None
        self.unsupported_symbols = set()

        try:
            self._is_render = bool(os.getenv('RENDER_EXTERNAL_URL') or os.getenv('RENDER_SERVICE_NAME'))
        except Exception:
            self._is_render = False

        # Hard caps to avoid Render free-tier OOM (keeps features, bounds memory)
        self._max_cache_items = 200
        self._max_kline_items = 60
        try:
            if self._is_render:
                self._max_cache_items = 40
                self._max_kline_items = 30
        except Exception:
            pass

        self.last_ok_time = None
        self.last_error_time = None
        self.last_error = None
        self.last_latency_ms = None
        self.consecutive_failures = 0
        self.total_requests = 0
        self.total_failures = 0

        self._adaptive_delay_s = 0.0
        self._adaptive_min_delay_s = 0.05
        self._adaptive_max_delay_s = 2.5
        self._adaptive_next_ts = 0.0
        self._adaptive_last_reason = None
        try:
            url = f"{self.base_url}/api/v3/exchangeInfo"
            resp = requests.get(url, timeout=8, verify=False)
            if resp.status_code == 200:
                data = resp.json() or {}
                symbols = set()
                for s in (data.get('symbols') or []):
                    sym = (s or {}).get('symbol')
                    if sym:
                        symbols.add(str(sym).upper())
                if symbols:
                    self.exchange_symbols = symbols
        except Exception:
            self.exchange_symbols = None

    async def _adaptive_wait(self):
        try:
            now = time.time()
            next_ts = float(getattr(self, '_adaptive_next_ts', 0.0) or 0.0)
            if next_ts > now:
                await asyncio.sleep(next_ts - now)
        except Exception:
            pass

    def _adaptive_note(self, ok: bool, status_code: int = None, latency_ms: int = None):
        try:
            delay = float(getattr(self, '_adaptive_delay_s', 0.0) or 0.0)
            min_d = float(getattr(self, '_adaptive_min_delay_s', 0.05) or 0.05)
            max_d = float(getattr(self, '_adaptive_max_delay_s', 2.5) or 2.5)

            if ok:
                if latency_ms is not None and int(latency_ms) > 1200:
                    delay = min(max_d, max(min_d, delay * 1.15 + 0.05))
                    self._adaptive_last_reason = 'high_latency'
                else:
                    delay = max(0.0, delay * 0.85 - 0.02)
                    self._adaptive_last_reason = None
            else:
                if int(status_code or 0) == 429:
                    delay = min(max_d, max(1.0, delay * 2.5 + 0.5))
                    self._adaptive_last_reason = 'rate_limit_429'
                else:
                    delay = min(max_d, max(min_d, delay * 1.7 + 0.10))
                    self._adaptive_last_reason = f'error_{int(status_code or 0)}'

            self._adaptive_delay_s = float(delay)
            self._adaptive_next_ts = float(time.time()) + float(delay)
        except Exception:
            try:
                self._adaptive_next_ts = float(time.time()) + 0.1
            except Exception:
                pass

    def get_health(self) -> Dict[str, Any]:
        try:
            now = datetime.now()
            age_ok_s = (now - self.last_ok_time).total_seconds() if self.last_ok_time else None
            age_err_s = (now - self.last_error_time).total_seconds() if self.last_error_time else None
            connected = bool(self.last_ok_time and age_ok_s is not None and age_ok_s <= 30)
            return {
                'connected': connected,
                'last_ok_time': self.last_ok_time.isoformat() if self.last_ok_time else None,
                'last_ok_age_sec': age_ok_s,
                'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
                'last_error_age_sec': age_err_s,
                'last_error': self.last_error,
                'last_latency_ms': self.last_latency_ms,
                'consecutive_failures': self.consecutive_failures,
                'total_requests': self.total_requests,
                'total_failures': self.total_failures,
                'unsupported_symbols_count': len(getattr(self, 'unsupported_symbols', set()) or set()),
                'exchange_symbols_loaded': self.exchange_symbols is not None
            }
        except Exception:
            return {
                'connected': False,
                'last_error': 'health_check_failed'
            }

    def _normalize_symbol(self, symbol: str) -> str:
        try:
            s = str(symbol).strip().upper()
            s = s.replace('/', '').replace('-', '').replace('_', '').replace(' ', '')
            return s
        except Exception:
            return str(symbol)

    def is_symbol_supported(self, symbol: str) -> bool:
        try:
            mexc_symbol = self._normalize_symbol(symbol)
            if not mexc_symbol:
                return False
            if mexc_symbol in self.unsupported_symbols:
                return False
            if self.exchange_symbols is None:
                return True
            return mexc_symbol in self.exchange_symbols
        except Exception:
            return True
        
    async def get_live_price(self, symbol: str) -> float:
        """Get current live price from MEXC using requests (works reliably)"""
        
        # Convert symbol format (BTC/USDT -> BTCUSDT)
        mexc_symbol = self._normalize_symbol(symbol)
        if not self.is_symbol_supported(symbol):
            print(f" Unsupported symbol on MEXC spot: {symbol}")
            return None
        
        # Use requests library which works reliably
        await self._adaptive_wait()
        start = time.time()
        self.total_requests += 1
        try:
            url = f"{self.base_url}/api/v3/ticker/price?symbol={mexc_symbol}"
            response = await _requests_get_async(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                price = float(data['price'])

                self.last_ok_time = datetime.now()
                self.last_latency_ms = int((time.time() - start) * 1000)
                self.last_error = None
                self.consecutive_failures = 0
                
                # Cache the price
                self.prices_cache[symbol] = {
                    'price': price,
                    'timestamp': datetime.now()
                }

                self._adaptive_note(True, status_code=200, latency_ms=int((time.time() - start) * 1000))
                
                return price
            else:
                if response.status_code in (400, 404):
                    self.unsupported_symbols.add(mexc_symbol)

                self.total_failures += 1
                self.consecutive_failures += 1
                self.last_error_time = datetime.now()
                self.last_latency_ms = int((time.time() - start) * 1000)
                self.last_error = f"HTTP_{response.status_code}"

                print(f" Failed to get {symbol} price: Status {response.status_code}")

                self._adaptive_note(False, status_code=int(response.status_code), latency_ms=int((time.time() - start) * 1000))

                cached = self.prices_cache.get(symbol)
                if cached and isinstance(cached, dict):
                    try:
                        age = (datetime.now() - cached.get('timestamp')).total_seconds()
                        if age <= 30 and cached.get('price'):
                            return float(cached.get('price'))
                    except Exception:
                        pass
                return None
                
        except requests.exceptions.RequestException as e:
            self.total_failures += 1
            self.consecutive_failures += 1
            self.last_error_time = datetime.now()
            self.last_latency_ms = int((time.time() - start) * 1000)
            self.last_error = str(e)[:120]

            self._adaptive_note(False, status_code=0, latency_ms=int((time.time() - start) * 1000))
            print(f" Error getting {symbol} price: {str(e)[:50]}")

            cached = self.prices_cache.get(symbol)
            if cached and isinstance(cached, dict):
                try:
                    age = (datetime.now() - cached.get('timestamp')).total_seconds()
                    if age <= 30 and cached.get('price'):
                        return float(cached.get('price'))
                except Exception:
                    pass
            return None
        except Exception as e:
            self.total_failures += 1
            self.consecutive_failures += 1
            self.last_error_time = datetime.now()
            self.last_latency_ms = int((time.time() - start) * 1000)
            self.last_error = str(e)[:120]

            try:
                self._adaptive_note(False, status_code=0, latency_ms=int((time.time() - start) * 1000))
            except Exception:
                pass
            print(f" Unexpected error getting {symbol} price: {str(e)[:50]}")

            cached = self.prices_cache.get(symbol)
            if cached and isinstance(cached, dict):
                try:
                    age = (datetime.now() - cached.get('timestamp')).total_seconds()
                    if age <= 30 and cached.get('price'):
                        return float(cached.get('price'))
                except Exception:
                    pass
            return None
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get multiple live prices at once"""
        prices = {}

        if not symbols:
            return prices

        # Try a single bulk endpoint first for speed (if supported), then fall back.
        if len(symbols) >= 8:
            start = time.time()
            self.total_requests += 1
            try:
                await self._adaptive_wait()
                url = f"{self.base_url}/api/v3/ticker/price"
                response = await _requests_get_async(url, timeout=8, verify=False)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        lookup = {}
                        for item in data:
                            try:
                                sym = str((item or {}).get('symbol') or '').upper()
                                px = float((item or {}).get('price') or 0)
                                if sym and px > 0:
                                    lookup[sym] = px
                            except Exception:
                                continue

                        for symbol in symbols:
                            mexc_symbol = self._normalize_symbol(symbol)
                            px = lookup.get(mexc_symbol)
                            if px:
                                prices[symbol] = px
                                self.prices_cache[symbol] = {'price': px, 'timestamp': datetime.now()}

                        self.last_ok_time = datetime.now()
                        self.last_latency_ms = int((time.time() - start) * 1000)
                        self.last_error = None
                        self.consecutive_failures = 0
                        self._adaptive_note(True, status_code=200, latency_ms=int((time.time() - start) * 1000))
                        return prices
            except Exception as e:
                self.total_failures += 1
                self.consecutive_failures += 1
                self.last_error_time = datetime.now()
                self.last_latency_ms = int((time.time() - start) * 1000)
                self.last_error = str(e)[:120]
                self._adaptive_note(False, status_code=0, latency_ms=int((time.time() - start) * 1000))

        for symbol in symbols:
            price = await self.get_live_price(symbol)
            if price:
                prices[symbol] = price

        return prices
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Get order book depth data - critical for AI learning!"""
        mexc_symbol = self._normalize_symbol(symbol)
        if not self.is_symbol_supported(symbol):
            return None
        
        try:
            await self._adaptive_wait()
            start = time.time()
            url = f"{self.base_url}/api/v3/depth?symbol={mexc_symbol}&limit={limit}"
            response = await _requests_get_async(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process orderbook data
                orderbook = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'bids': [[float(p), float(q)] for p, q in data.get('bids', [])[:limit]],
                    'asks': [[float(p), float(q)] for p, q in data.get('asks', [])[:limit]],
                }
                
                # Calculate orderbook metrics for AI
                if orderbook['bids'] and orderbook['asks']:
                    orderbook['spread'] = orderbook['asks'][0][0] - orderbook['bids'][0][0]
                    orderbook['spread_pct'] = (orderbook['spread'] / orderbook['bids'][0][0]) * 100
                    orderbook['bid_volume'] = sum(q for p, q in orderbook['bids'])
                    orderbook['ask_volume'] = sum(q for p, q in orderbook['asks'])
                    orderbook['volume_imbalance'] = (orderbook['bid_volume'] - orderbook['ask_volume']) / (orderbook['bid_volume'] + orderbook['ask_volume'])
                    orderbook['mid_price'] = (orderbook['bids'][0][0] + orderbook['asks'][0][0]) / 2
                
                self.orderbook_cache[symbol] = orderbook
                try:
                    if isinstance(self.orderbook_cache, dict) and len(self.orderbook_cache) > int(self._max_cache_items):
                        self.orderbook_cache.pop(next(iter(self.orderbook_cache)), None)
                except Exception:
                    pass
                try:
                    self._adaptive_note(True, status_code=200, latency_ms=int((time.time() - start) * 1000))
                except Exception:
                    pass
                return orderbook
            
            try:
                self._adaptive_note(False, status_code=int(response.status_code), latency_ms=int((time.time() - start) * 1000))
            except Exception:
                pass
            return None
        except Exception as e:
            print(f"⚠️ Error getting orderbook for {symbol}: {str(e)[:50]}")
            try:
                self._adaptive_note(False, status_code=0, latency_ms=None)
            except Exception:
                pass
            return None
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades - shows market momentum and pressure"""
        mexc_symbol = self._normalize_symbol(symbol)
        if not self.is_symbol_supported(symbol):
            return None
        
        try:
            await self._adaptive_wait()
            start = time.time()
            url = f"{self.base_url}/api/v3/trades?symbol={mexc_symbol}&limit={limit}"
            response = await _requests_get_async(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                trades = []
                for trade in data:
                    trades.append({
                        'price': float(trade['price']),
                        'qty': float(trade['qty']),
                        'time': trade['time'],
                        'is_buyer_maker': trade.get('isBuyerMaker', True)
                    })
                
                # Calculate trade metrics for AI
                if trades:
                    buy_volume = sum(t['qty'] for t in trades if not t['is_buyer_maker'])
                    sell_volume = sum(t['qty'] for t in trades if t['is_buyer_maker'])
                    total_volume = buy_volume + sell_volume
                    
                    trade_data = {
                        'symbol': symbol,
                        'trades': trades,
                        'buy_volume': buy_volume,
                        'sell_volume': sell_volume,
                        'buy_sell_ratio': buy_volume / sell_volume if sell_volume > 0 else 999,
                        'volume_pressure': (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0,
                        'avg_trade_size': total_volume / len(trades),
                        'timestamp': datetime.now()
                    }
                    
                    self.trades_cache[symbol] = trade_data
                    try:
                        if isinstance(self.trades_cache, dict) and len(self.trades_cache) > int(self._max_cache_items):
                            self.trades_cache.pop(next(iter(self.trades_cache)), None)
                    except Exception:
                        pass
                    try:
                        self._adaptive_note(True, status_code=200, latency_ms=int((time.time() - start) * 1000))
                    except Exception:
                        pass
                    return trade_data
            
            try:
                self._adaptive_note(False, status_code=int(response.status_code), latency_ms=int((time.time() - start) * 1000))
            except Exception:
                pass
            return None
        except Exception as e:
            print(f"⚠️ Error getting trades for {symbol}: {str(e)[:50]}")
            try:
                self._adaptive_note(False, status_code=0, latency_ms=None)
            except Exception:
                pass
            return None
    
    async def get_24h_ticker(self, symbol: str) -> Dict:
        """Get 24h ticker statistics - volume, price change, high/low"""
        mexc_symbol = self._normalize_symbol(symbol)
        if not self.is_symbol_supported(symbol):
            return None
        
        try:
            await self._adaptive_wait()
            start = time.time()
            url = f"{self.base_url}/api/v3/ticker/24hr?symbol={mexc_symbol}"
            response = await _requests_get_async(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                # Safely parse values with None handling
                count_value = data.get('count', 0)
                if count_value is None:
                    count_value = 0
                
                ticker = {
                    'symbol': symbol,
                    'price_change': float(data.get('priceChange') or 0),
                    'price_change_pct': float(data.get('priceChangePercent') or 0),
                    'high_24h': float(data.get('highPrice') or 0),
                    'low_24h': float(data.get('lowPrice') or 0),
                    'volume_24h': float(data.get('volume') or 0),
                    'quote_volume_24h': float(data.get('quoteVolume') or 0),
                    'trades_count': int(float(count_value)) if count_value else 0,
                    'open_price': float(data.get('openPrice') or 0),
                    'close_price': float(data.get('lastPrice') or 0),
                    'timestamp': datetime.now()
                }
                
                # Calculate additional metrics
                if ticker['high_24h'] > 0 and ticker['low_24h'] > 0:
                    ticker['volatility_24h'] = ((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h']) * 100
                    ticker['price_position'] = ((ticker['close_price'] - ticker['low_24h']) / (ticker['high_24h'] - ticker['low_24h'])) if ticker['high_24h'] != ticker['low_24h'] else 0.5
                
                self.ticker_24h_cache[symbol] = ticker
                try:
                    if isinstance(self.ticker_24h_cache, dict) and len(self.ticker_24h_cache) > int(self._max_cache_items):
                        self.ticker_24h_cache.pop(next(iter(self.ticker_24h_cache)), None)
                except Exception:
                    pass
                try:
                    self._adaptive_note(True, status_code=200, latency_ms=int((time.time() - start) * 1000))
                except Exception:
                    pass
                return ticker
            
            try:
                self._adaptive_note(False, status_code=int(response.status_code), latency_ms=int((time.time() - start) * 1000))
            except Exception:
                pass
            return None
        except Exception as e:
            print(f"⚠️ Error getting 24h ticker for {symbol}: {str(e)[:50]}")
            try:
                self._adaptive_note(False, status_code=0, latency_ms=None)
            except Exception:
                pass
            return None
    
    async def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100) -> List[Dict]:
        """Get candlestick/kline data - OHLCV for pattern recognition"""
        mexc_symbol = self._normalize_symbol(symbol)
        if not self.is_symbol_supported(symbol):
            return None
        
        try:
            await self._adaptive_wait()
            start = time.time()
            url = f"{self.base_url}/api/v3/klines?symbol={mexc_symbol}&interval={interval}&limit={limit}"
            response = await _requests_get_async(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate data is a list
                if not isinstance(data, list) or len(data) == 0:
                    return []
                
                klines = []
                for k in data:
                    # Validate kline has enough elements
                    if not isinstance(k, list) or len(k) < 9:
                        continue
                    
                    try:
                        klines.append({
                            'open_time': k[0],
                            'open': float(k[1]),
                            'high': float(k[2]),
                            'low': float(k[3]),
                            'close': float(k[4]),
                            'volume': float(k[5]),
                            'close_time': k[6],
                            'quote_volume': float(k[7]),
                            'trades_count': int(float(k[8])) if k[8] else 0
                        })
                    except (ValueError, TypeError, IndexError) as e:
                        # Skip invalid klines
                        continue
                
                try:
                    if isinstance(klines, list) and len(klines) > int(self._max_kline_items):
                        klines = klines[-int(self._max_kline_items):]
                except Exception:
                    pass

                self.klines_cache[f"{symbol}_{interval}"] = klines
                try:
                    if isinstance(self.klines_cache, dict) and len(self.klines_cache) > int(self._max_cache_items):
                        self.klines_cache.pop(next(iter(self.klines_cache)), None)
                except Exception:
                    pass
                try:
                    self._adaptive_note(True, status_code=200, latency_ms=int((time.time() - start) * 1000))
                except Exception:
                    pass
                return klines if klines else []
            
            if response.status_code in (400, 404):
                self.unsupported_symbols.add(mexc_symbol)
            try:
                self._adaptive_note(False, status_code=int(response.status_code), latency_ms=int((time.time() - start) * 1000))
            except Exception:
                pass
            return None
        except Exception as e:
            print(f"⚠️ Error getting klines for {symbol}: {str(e)[:50]}")
            try:
                self._adaptive_note(False, status_code=0, latency_ms=None)
            except Exception:
                pass
            return None
    
    async def get_comprehensive_market_data(self, symbol: str) -> Dict:
        """🔥 Get ALL market data at once - feed AI everything!"""
        print(f"📊 Fetching comprehensive data for {symbol}...")
        
        # Fetch all data types in parallel would be ideal, but sequentially works
        price = await self.get_live_price(symbol)
        orderbook = await self.get_orderbook(symbol)
        trades = await self.get_recent_trades(symbol)
        ticker_24h = await self.get_24h_ticker(symbol)
        klines_1m = await self.get_klines(symbol, '1m', 100)
        klines_5m = await self.get_klines(symbol, '5m', 50)
        klines_15m = await self.get_klines(symbol, '15m', 30)
        
        comprehensive_data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price': price,
            'orderbook': orderbook,
            'recent_trades': trades,
            'ticker_24h': ticker_24h,
            'klines': {
                '1m': klines_1m,
                '5m': klines_5m,
                '15m': klines_15m
            },
            # Derived metrics for AI
            'market_sentiment': self._calculate_market_sentiment(orderbook, trades, ticker_24h),
            'liquidity_score': self._calculate_liquidity(orderbook, trades),
            'momentum_score': self._calculate_momentum(klines_1m, ticker_24h)
        }
        
        return comprehensive_data
    
    def _calculate_market_sentiment(self, orderbook: Dict, trades: Dict, ticker: Dict) -> float:
        """Calculate overall market sentiment from multiple indicators"""
        sentiment_score = 0.0
        factors = 0
        
        # Factor 1: Volume imbalance from orderbook
        if orderbook and 'volume_imbalance' in orderbook:
            sentiment_score += orderbook['volume_imbalance']
            factors += 1
        
        # Factor 2: Buy/sell pressure from trades
        if trades and 'volume_pressure' in trades:
            sentiment_score += trades['volume_pressure']
            factors += 1
        
        # Factor 3: Price change trend
        if ticker and 'price_change_pct' in ticker:
            sentiment_score += (ticker['price_change_pct'] / 10)  # Normalize
            factors += 1
        
        return sentiment_score / factors if factors > 0 else 0.0
    
    def _calculate_liquidity(self, orderbook: Dict, trades: Dict) -> float:
        """Calculate market liquidity score"""
        if not orderbook:
            return 0.0
        
        # Tight spread + high volume = high liquidity
        spread_score = 1 / (1 + orderbook.get('spread_pct', 1.0))  # Lower spread = higher score
        volume_score = min(orderbook.get('bid_volume', 0) + orderbook.get('ask_volume', 0), 1000) / 1000
        
        return (spread_score + volume_score) / 2
    
    def _calculate_momentum(self, klines: List[Dict], ticker: Dict) -> float:
        """Calculate price momentum score"""
        if not klines or len(klines) < 5:
            return 0.0
        
        # Recent price movement
        recent_closes = [k['close'] for k in klines[-5:]]
        momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100
        
        return max(min(momentum, 5), -5) / 5  # Normalize to [-1, 1]


class MexcFuturesDataFeed:
    def __init__(self):
        self.base_url = "https://contract.mexc.com"
        self.prices_cache = {}
        self.contract_cache = {}
        self.contract_last_loaded = None
        self.unsupported_symbols = set()

        self.last_ok_time = None
        self.last_error_time = None
        self.last_error = None
        self.last_latency_ms = None
        self.consecutive_failures = 0
        self.total_requests = 0
        self.total_failures = 0

        self._adaptive_delay_s = 0.0
        self._adaptive_min_delay_s = 0.05
        self._adaptive_max_delay_s = 2.5
        self._adaptive_next_ts = 0.0
        self._adaptive_last_reason = None

        try:
            if not hasattr(self, '_normalize_contract_symbol'):
                def _compat_normalize_contract_symbol(sym: str) -> str:
                    try:
                        s = str(sym).strip().upper()
                        if not s:
                            return ''
                        s = s.replace(' ', '')
                        s2 = s.replace('-', '_').replace('/', '_')
                        while '__' in s2:
                            s2 = s2.replace('__', '_')
                        if '_' in s2:
                            return s2
                        compact = s.replace('-', '').replace('/', '').replace('_', '').replace(' ', '')
                        for quote in ('USDT', 'USDC', 'USD'):
                            if compact.endswith(quote) and len(compact) > len(quote):
                                return f"{compact[:-len(quote)]}_{quote}"
                        return compact
                    except Exception:
                        return str(sym)

                self._normalize_contract_symbol = _compat_normalize_contract_symbol
        except Exception:
            pass

        try:
            if not hasattr(self, '_adaptive_wait'):
                async def _compat_adaptive_wait():
                    try:
                        now = time.time()
                        next_ts = float(getattr(self, '_adaptive_next_ts', 0.0) or 0.0)
                        if next_ts > now:
                            await asyncio.sleep(next_ts - now)
                    except Exception:
                        pass

                self._adaptive_wait = _compat_adaptive_wait
        except Exception:
            pass

        try:
            if not hasattr(self, '_adaptive_note'):
                def _compat_adaptive_note(ok: bool, status_code: int = None, latency_ms: int = None):
                    try:
                        delay = float(getattr(self, '_adaptive_delay_s', 0.0) or 0.0)
                        min_d = float(getattr(self, '_adaptive_min_delay_s', 0.05) or 0.05)
                        max_d = float(getattr(self, '_adaptive_max_delay_s', 2.5) or 2.5)

                        if ok:
                            if latency_ms is not None and int(latency_ms) > 1200:
                                delay = min(max_d, max(min_d, delay * 1.15 + 0.05))
                                self._adaptive_last_reason = 'high_latency'
                            else:
                                delay = max(0.0, delay * 0.85 - 0.02)
                                self._adaptive_last_reason = None
                        else:
                            if int(status_code or 0) == 429:
                                delay = min(max_d, max(1.0, delay * 2.5 + 0.5))
                                self._adaptive_last_reason = 'rate_limit_429'
                            else:
                                delay = min(max_d, max(min_d, delay * 1.7 + 0.10))
                                self._adaptive_last_reason = f'error_{int(status_code or 0)}'

                        self._adaptive_delay_s = float(delay)
                        self._adaptive_next_ts = float(time.time()) + float(delay)
                    except Exception:
                        try:
                            self._adaptive_next_ts = float(time.time()) + 0.1
                        except Exception:
                            pass

                self._adaptive_note = _compat_adaptive_note
        except Exception:
            pass

        try:
            loop = asyncio.get_running_loop()
        except Exception:
            loop = None

        if loop is not None and getattr(loop, 'is_running', lambda: False)():
            try:
                loop.create_task(self._load_contracts_if_needed(force=True))
            except Exception:
                pass
        else:
            try:
                asyncio.run(self._load_contracts_if_needed(force=True))
            except Exception:
                pass

    async def _adaptive_wait(self):
        try:
            now = time.time()
            next_ts = float(getattr(self, '_adaptive_next_ts', 0.0) or 0.0)
            if next_ts > now:
                await asyncio.sleep(next_ts - now)
        except Exception:
            pass

    def _adaptive_note(self, ok: bool, status_code: int = None, latency_ms: int = None):
        try:
            delay = float(getattr(self, '_adaptive_delay_s', 0.0) or 0.0)
            min_d = float(getattr(self, '_adaptive_min_delay_s', 0.05) or 0.05)
            max_d = float(getattr(self, '_adaptive_max_delay_s', 2.5) or 2.5)

            if ok:
                if latency_ms is not None and int(latency_ms) > 1200:
                    delay = min(max_d, max(min_d, delay * 1.15 + 0.05))
                    self._adaptive_last_reason = 'high_latency'
                else:
                    delay = max(0.0, delay * 0.85 - 0.02)
                    self._adaptive_last_reason = None
            else:
                if int(status_code or 0) == 429:
                    delay = min(max_d, max(1.0, delay * 2.5 + 0.5))
                    self._adaptive_last_reason = 'rate_limit_429'
                else:
                    delay = min(max_d, max(min_d, delay * 1.7 + 0.10))
                    self._adaptive_last_reason = f'error_{int(status_code or 0)}'

            self._adaptive_delay_s = float(delay)
            self._adaptive_next_ts = float(time.time()) + float(delay)
        except Exception:
            try:
                self._adaptive_next_ts = float(time.time()) + 0.1
            except Exception:
                pass

    def _normalize_contract_symbol(self, symbol: str) -> str:
        try:
            s = str(symbol).strip().upper()
            if not s:
                return ''

            s = s.replace(' ', '')
            s2 = s.replace('-', '_').replace('/', '_')
            while '__' in s2:
                s2 = s2.replace('__', '_')
            if '_' in s2:
                return s2

            compact = s.replace('-', '').replace('/', '').replace('_', '').replace(' ', '')
            for quote in ('USDT', 'USDC', 'USD'):
                if compact.endswith(quote) and len(compact) > len(quote):
                    return f"{compact[:-len(quote)]}_{quote}"
            return compact
        except Exception:
            return str(symbol)

    def _kick_contracts_load(self, force: bool = False):
        try:
            try:
                loop = asyncio.get_running_loop()
            except Exception:
                loop = None

            if loop is not None and getattr(loop, 'is_running', lambda: False)():
                try:
                    loop.create_task(self._load_contracts_if_needed(force=bool(force)))
                except Exception:
                    pass
            else:
                try:
                    asyncio.run(self._load_contracts_if_needed(force=bool(force)))
                except Exception:
                    pass
        except Exception:
            pass

    async def _refresh_contracts(self):
        try:
            now = time.time()
            next_ts = float(getattr(self, '_adaptive_next_ts', 0.0) or 0.0)
            if next_ts > now:
                await asyncio.sleep(next_ts - now)
        except Exception:
            pass

    async def _load_contracts_if_needed(self, force: bool = False):
        try:
            now = datetime.now()
            if not force and self.contract_last_loaded and (now - self.contract_last_loaded).total_seconds() < 300:
                return

            start = time.time()
            self.total_requests += 1
            await self._refresh_contracts()
            url = f"{self.base_url}/api/v1/contract/detail"
            response = await _requests_get_async(url, timeout=8, verify=False)
            if response.status_code != 200:
                self.total_failures += 1
                self.consecutive_failures += 1
                self.last_error_time = datetime.now()
                self.last_latency_ms = int((time.time() - start) * 1000)
                self.last_error = f"HTTP_{response.status_code}"
                self._adaptive_note(False, status_code=int(response.status_code), latency_ms=int((time.time() - start) * 1000))
                return

            payload = response.json() or {}
            data = payload.get('data') or []
            if not isinstance(data, list):
                return

            lookup = {}
            for item in data:
                try:
                    sym = str((item or {}).get('symbol') or '').upper()
                    if sym:
                        lookup[sym] = item
                except Exception:
                    continue

            if lookup:
                self.contract_cache = lookup
                self.contract_last_loaded = datetime.now()
                self.last_ok_time = datetime.now()
                self.last_latency_ms = int((time.time() - start) * 1000)
                self.last_error = None
                self.consecutive_failures = 0
                self._adaptive_note(True, status_code=200, latency_ms=int((time.time() - start) * 1000))
        except Exception as e:
            self.total_failures += 1
            self.consecutive_failures += 1
            self.last_error_time = datetime.now()
            self.last_error = str(e)[:120]
            self._adaptive_note(False, status_code=0, latency_ms=None)

    def is_symbol_supported(self, symbol: str) -> bool:
        try:
            self._kick_contracts_load(force=False)
            sym = self._normalize_contract_symbol(symbol)
            if not sym:
                return False
            if sym in self.unsupported_symbols:
                return False
            if not self.contract_cache:
                return True
            return sym in self.contract_cache
        except Exception:
            return True

    def get_contract_meta(self, symbol: str) -> Dict[str, Any]:
        self._kick_contracts_load(force=False)
        sym = self._normalize_contract_symbol(symbol)
        return dict(self.contract_cache.get(sym) or {})

    def get_health(self) -> Dict[str, Any]:
        try:
            now = datetime.now()
            age_ok_s = (now - self.last_ok_time).total_seconds() if self.last_ok_time else None
            age_err_s = (now - self.last_error_time).total_seconds() if self.last_error_time else None
            connected = bool(self.last_ok_time and age_ok_s is not None and age_ok_s <= 30)
            return {
                'connected': connected,
                'last_ok_time': self.last_ok_time.isoformat() if self.last_ok_time else None,
                'last_ok_age_sec': age_ok_s,
                'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
                'last_error_age_sec': age_err_s,
                'last_error': self.last_error,
                'last_latency_ms': self.last_latency_ms,
                'consecutive_failures': self.consecutive_failures,
                'total_requests': self.total_requests,
                'total_failures': self.total_failures,
                'unsupported_symbols_count': len(getattr(self, 'unsupported_symbols', set()) or set()),
                'contracts_cached_count': len(getattr(self, 'contract_cache', {}) or {})
            }
        except Exception:
            return {
                'connected': False,
                'last_error': 'health_check_failed'
            }

    async def get_live_price(self, symbol: str) -> float:
        mexc_symbol = self._normalize_contract_symbol(symbol)
        if not self.is_symbol_supported(symbol):
            return None
        if mexc_symbol in self.unsupported_symbols:
            return None

        start = time.time()
        self.total_requests += 1
        try:
            await self._adaptive_wait()
            url = f"{self.base_url}/api/v1/contract/ticker?symbol={mexc_symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; PoiseTrader/1.0)'
            }
            response = await _requests_get_async(url, timeout=6, verify=False, headers=headers)

            if response.status_code == 200:
                try:
                    payload = response.json() or {}
                except Exception:
                    self.total_failures += 1
                    self.consecutive_failures += 1
                    self.last_error_time = datetime.now()
                    self.last_latency_ms = int((time.time() - start) * 1000)
                    self.last_error = 'INVALID_JSON'
                    try:
                        self._adaptive_note(False, status_code=200, latency_ms=int((time.time() - start) * 1000))
                    except Exception:
                        pass
                    cached = self.prices_cache.get(symbol)
                    if cached and isinstance(cached, dict):
                        try:
                            age = (datetime.now() - cached.get('timestamp')).total_seconds()
                            if age <= 30 and cached.get('price'):
                                return float(cached.get('price'))
                        except Exception:
                            pass
                    return None

                data = payload.get('data') if isinstance(payload, dict) else None

                try:
                    success = payload.get('success', True) if isinstance(payload, dict) else True
                    code = payload.get('code', 0) if isinstance(payload, dict) else 0
                    if success is False:
                        raise ValueError('api_success_false')
                    try:
                        if code is not None and int(code) != 0:
                            raise ValueError(f"api_code_{int(code)}")
                    except ValueError:
                        raise
                    except Exception:
                        pass
                except Exception as e:
                    self.total_failures += 1
                    self.consecutive_failures += 1
                    self.last_error_time = datetime.now()
                    self.last_latency_ms = int((time.time() - start) * 1000)
                    self.last_error = str(e)[:120]
                    try:
                        self._adaptive_note(False, status_code=200, latency_ms=int((time.time() - start) * 1000))
                    except Exception:
                        pass
                    cached = self.prices_cache.get(symbol)
                    if cached and isinstance(cached, dict):
                        try:
                            age = (datetime.now() - cached.get('timestamp')).total_seconds()
                            if age <= 30 and cached.get('price'):
                                return float(cached.get('price'))
                        except Exception:
                            pass
                    return None

                if isinstance(data, list):
                    chosen = None
                    for item in data:
                        try:
                            if isinstance(item, dict) and str((item or {}).get('symbol') or '').upper() == str(mexc_symbol).upper():
                                chosen = item
                                break
                        except Exception:
                            continue
                    if chosen is None and data:
                        chosen = data[0]
                    data = chosen

                if not isinstance(data, dict):
                    data = {}

                px = 0.0
                for k in ('lastPrice', 'fairPrice', 'markPrice', 'indexPrice', 'last', 'price', 'ask1', 'bid1'):
                    try:
                        v = data.get(k)
                        if v is None or v == '':
                            continue
                        px = float(v)
                        if px > 0:
                            break
                    except Exception:
                        continue

                if px > 0:
                    self.last_ok_time = datetime.now()
                    self.last_latency_ms = int((time.time() - start) * 1000)
                    self.last_error = None
                    self.consecutive_failures = 0

                    self.prices_cache[symbol] = {
                        'price': px,
                        'timestamp': datetime.now()
                    }
                    self._adaptive_note(True, status_code=200, latency_ms=int((time.time() - start) * 1000))
                    return px

                try:
                    await self._adaptive_wait()
                    url2 = f"{self.base_url}/api/v1/contract/ticker"
                    resp2 = await _requests_get_async(url2, timeout=8, verify=False, headers=headers)
                    if resp2.status_code == 200:
                        try:
                            payload2 = resp2.json() or {}
                        except Exception:
                            payload2 = {}
                        data2 = payload2.get('data') if isinstance(payload2, dict) else None
                        if isinstance(data2, list):
                            chosen2 = None
                            for item2 in data2:
                                try:
                                    if isinstance(item2, dict) and str((item2 or {}).get('symbol') or '').upper() == str(mexc_symbol).upper():
                                        chosen2 = item2
                                        break
                                except Exception:
                                    continue
                            if chosen2 is not None:
                                px2 = 0.0
                                for k2 in ('lastPrice', 'fairPrice', 'markPrice', 'indexPrice', 'last', 'price', 'ask1', 'bid1'):
                                    try:
                                        v2 = (chosen2 or {}).get(k2)
                                        if v2 is None or v2 == '':
                                            continue
                                        px2 = float(v2)
                                        if px2 > 0:
                                            break
                                    except Exception:
                                        continue
                                if px2 > 0:
                                    self.last_ok_time = datetime.now()
                                    self.last_latency_ms = int((time.time() - start) * 1000)
                                    self.last_error = None
                                    self.consecutive_failures = 0

                                    self.prices_cache[symbol] = {
                                        'price': px2,
                                        'timestamp': datetime.now()
                                    }
                                    try:
                                        self._adaptive_note(True, status_code=200, latency_ms=int((time.time() - start) * 1000))
                                    except Exception:
                                        pass
                                    return px2
                except Exception:
                    pass

                try:
                    await self._adaptive_wait()
                    spot_symbol = str(mexc_symbol).replace('_', '')
                    url3 = f"https://api.mexc.com/api/v3/ticker/price?symbol={spot_symbol}"
                    resp3 = await _requests_get_async(url3, timeout=6, verify=False, headers=headers)
                    if resp3.status_code == 200:
                        try:
                            payload3 = resp3.json() or {}
                        except Exception:
                            payload3 = {}
                        try:
                            px3 = float((payload3 or {}).get('price') or 0)
                        except Exception:
                            px3 = 0.0
                        if px3 > 0:
                            self.last_ok_time = datetime.now()
                            self.last_latency_ms = int((time.time() - start) * 1000)
                            self.last_error = None
                            self.consecutive_failures = 0

                            self.prices_cache[symbol] = {
                                'price': px3,
                                'timestamp': datetime.now()
                            }
                            try:
                                self._adaptive_note(True, status_code=200, latency_ms=int((time.time() - start) * 1000))
                            except Exception:
                                pass
                            return px3
                except Exception:
                    pass

                self.total_failures += 1
                self.consecutive_failures += 1
                self.last_error_time = datetime.now()
                self.last_latency_ms = int((time.time() - start) * 1000)
                try:
                    keys = sorted(list(data.keys()))[:12]
                    self.last_error = f"NO_PRICE_FIELDS keys={','.join([str(x) for x in keys])}"[:120]
                except Exception:
                    self.last_error = 'NO_PRICE_FIELDS'
                try:
                    self._adaptive_note(False, status_code=200, latency_ms=int((time.time() - start) * 1000))
                except Exception:
                    pass

                cached = self.prices_cache.get(symbol)
                if cached and isinstance(cached, dict):
                    try:
                        age = (datetime.now() - cached.get('timestamp')).total_seconds()
                        if age <= 30 and cached.get('price'):
                            return float(cached.get('price'))
                    except Exception:
                        pass
                return None

            if response.status_code in (400, 404):
                self.unsupported_symbols.add(mexc_symbol)

            self.total_failures += 1
            self.consecutive_failures += 1
            self.last_error_time = datetime.now()
            self.last_latency_ms = int((time.time() - start) * 1000)
            self.last_error = f"HTTP_{response.status_code}"
            self._adaptive_note(False, status_code=int(response.status_code), latency_ms=int((time.time() - start) * 1000))

            cached = self.prices_cache.get(symbol)
            if cached and isinstance(cached, dict):
                try:
                    age = (datetime.now() - cached.get('timestamp')).total_seconds()
                    if age <= 30 and cached.get('price'):
                        return float(cached.get('price'))
                except Exception:
                    pass
            return None
        except Exception as e:
            self.total_failures += 1
            self.consecutive_failures += 1
            self.last_error_time = datetime.now()
            self.last_latency_ms = int((time.time() - start) * 1000)
            self.last_error = str(e)[:120]

            try:
                self._adaptive_note(False, status_code=0, latency_ms=int((time.time() - start) * 1000))
            except Exception:
                pass

            cached = self.prices_cache.get(symbol)
            if cached and isinstance(cached, dict):
                try:
                    age = (datetime.now() - cached.get('timestamp')).total_seconds()
                    if age <= 30 and cached.get('price'):
                        return float(cached.get('price'))
                except Exception:
                    pass
            return None

    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        prices = {}
        for symbol in symbols or []:
            px = await self.get_live_price(symbol)
            if px:
                prices[symbol] = px
        return prices

class LivePaperTradingManager:
    """Paper trading manager using LIVE market prices"""
    
    def __init__(self, initial_capital: float = 5.0):
        state_dir = Path("data")
        state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = str(state_dir / "trading_state.json")
        self.initial_capital = initial_capital
        
        # Try to load existing state
        if self._load_state():
            print("💾 Loaded existing trading state")
        else:
            # Initialize fresh state
            self.cash_balance = initial_capital
            self.positions = {}
            self.trade_history = []
            self.total_trades = 0
            self.winning_trades = 0
            print(f"🆕 NEW trading session with ${initial_capital:,.2f}")
        
        try:
            self.market_type = str(os.getenv('PAPER_MARKET_TYPE', 'futures') or 'futures').strip().lower()
        except Exception:
            self.market_type = 'futures'

        if self.market_type not in ['spot', 'futures']:
            self.market_type = 'futures'

        if self.market_type == 'futures':
            self.data_feed = MexcFuturesDataFeed()
        else:
            self.data_feed = LiveMexcDataFeed()

        try:
            shorting_env = str(os.getenv('ENABLE_PAPER_SHORTING', '1') or '1').strip().lower()
            self.enable_shorting = shorting_env not in ['0', 'false', 'no', 'off']
        except Exception:
            self.enable_shorting = True

        try:
            self.leverage = float(os.getenv('PAPER_LEVERAGE', '10') or 10)
            if self.leverage < 1:
                self.leverage = 1.0
        except Exception:
            self.leverage = 10.0

        try:
            self.fee_rate = float(os.getenv('PAPER_FEE_RATE', '0.0002') or 0.0002)
        except Exception:
            self.fee_rate = 0.0002

        try:
            self.paper_execution_model = str(os.getenv('PAPER_EXECUTION_MODEL', 'ideal') or 'ideal').strip().lower()
        except Exception:
            self.paper_execution_model = 'ideal'
        if self.paper_execution_model not in ['ideal', 'realistic']:
            self.paper_execution_model = 'ideal'

        try:
            self.paper_spread_bps = float(os.getenv('PAPER_SPREAD_BPS', '1.5') or 1.5)
        except Exception:
            self.paper_spread_bps = 1.5

        try:
            self.paper_slippage_bps = float(os.getenv('PAPER_SLIPPAGE_BPS', '2.0') or 2.0)
        except Exception:
            self.paper_slippage_bps = 2.0

        try:
            self.paper_latency_ms_min = int(float(os.getenv('PAPER_LATENCY_MS_MIN', '0') or 0))
        except Exception:
            self.paper_latency_ms_min = 0

        try:
            self.paper_latency_ms_max = int(float(os.getenv('PAPER_LATENCY_MS_MAX', '0') or 0))
        except Exception:
            self.paper_latency_ms_max = 0

        if self.paper_latency_ms_max < self.paper_latency_ms_min:
            self.paper_latency_ms_max = self.paper_latency_ms_min

        try:
            self.paper_partial_fill_prob = float(os.getenv('PAPER_PARTIAL_FILL_PROB', '0') or 0)
        except Exception:
            self.paper_partial_fill_prob = 0.0
        self.paper_partial_fill_prob = max(0.0, min(1.0, self.paper_partial_fill_prob))

        try:
            self.paper_partial_fill_min_pct = float(os.getenv('PAPER_PARTIAL_FILL_MIN_PCT', '0.6') or 0.6)
        except Exception:
            self.paper_partial_fill_min_pct = 0.6

        try:
            self.paper_partial_fill_max_pct = float(os.getenv('PAPER_PARTIAL_FILL_MAX_PCT', '0.95') or 0.95)
        except Exception:
            self.paper_partial_fill_max_pct = 0.95

        if self.paper_partial_fill_max_pct < self.paper_partial_fill_min_pct:
            self.paper_partial_fill_max_pct = self.paper_partial_fill_min_pct

        try:
            real_env = str(os.getenv('REAL_TRADING', '0') or '0').strip().lower()
            self.real_trading_enabled = real_env in ['1', 'true', 'yes', 'on']
        except Exception:
            self.real_trading_enabled = False

        self._ccxt_exchange = None
        self.last_real_order_error = None
        self.last_real_order = None
        
        print(f"🔥 LIVE Paper Trading Manager active")
        print(f"   💰 Current Balance: ${self.cash_balance:.2f}")
        print(f"   📊 Active Positions: {len(self.positions)}")
        print("📡 Using REAL-TIME MEXC market prices!")
        
        # Save state immediately
        self._save_state()

    def _real_trading_ready(self) -> bool:
        if not self.real_trading_enabled:
            return False
        if self.market_type != 'futures':
            return False
        if ccxt is None:
            return False
        api_key = os.getenv('MEXC_API_KEY', '')
        api_secret = os.getenv('MEXC_API_SECRET', '') or os.getenv('MEXC_SECRET_KEY', '')
        if not api_key or not api_secret:
            return False
        return True

    async def _get_ccxt_exchange(self):
        if self._ccxt_exchange is not None:
            return self._ccxt_exchange

        api_key = os.getenv('MEXC_API_KEY', '')
        api_secret = os.getenv('MEXC_API_SECRET', '') or os.getenv('MEXC_SECRET_KEY', '')

        exchange = ccxt.mexc({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': 'swap'
            }
        })

        await exchange.load_markets()
        self._ccxt_exchange = exchange
        return exchange

    def _to_ccxt_swap_symbol(self, symbol: str) -> str:
        try:
            s = str(symbol or '').strip()
            if ':' in s:
                return s
            if '/' not in s:
                return s
            base, quote = s.split('/', 1)
            base = base.strip().upper()
            quote = quote.strip().upper()
            if quote == 'USDT':
                return f"{base}/{quote}:{quote}"
            return s
        except Exception:
            return symbol

    def _make_client_order_id(self, symbol: str, side: str, reduce_only: bool) -> str:
        try:
            base = str(symbol or '').replace('/', '').replace(':', '').upper()
        except Exception:
            base = 'SYMBOL'
        try:
            s = str(side or '').lower()[:1]
        except Exception:
            s = 'x'
        try:
            ro = 'r' if reduce_only else 'o'
        except Exception:
            ro = 'o'
        try:
            ts = int(time.time() * 1000)
        except Exception:
            ts = int(time.time())
        try:
            u = uuid.uuid4().hex[:10]
        except Exception:
            u = str(random.randint(1000000000, 9999999999))
        cid = f"POISE{ro}{s}{base}{ts}{u}"
        return cid[:32]

    async def _sync_from_exchange(self, symbol: str = None):
        exchange = await self._get_ccxt_exchange()

        try:
            balance = await exchange.fetch_balance()
            usdt = balance.get('USDT') or {}
            free = usdt.get('free')
            if free is None:
                free = balance.get('free', {}).get('USDT') if isinstance(balance.get('free'), dict) else None
            if free is not None:
                self.cash_balance = float(free)
        except Exception:
            pass

        if not symbol:
            return

        ccxt_symbol = self._to_ccxt_swap_symbol(symbol)
        try:
            positions = await exchange.fetch_positions([ccxt_symbol])
        except Exception:
            positions = []

        chosen = None
        for p in positions or []:
            if str(p.get('symbol') or '').upper() == str(ccxt_symbol).upper():
                chosen = p
                break
        if not chosen and positions:
            chosen = positions[0]

        if not chosen:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'total_cost': 0,
                'take_profit': None,
                'stop_loss': None,
                'action': 'BUY'
            }
            return

        contracts = float(chosen.get('contracts') or 0)
        side = str(chosen.get('side') or '').lower()
        entry_px = float(chosen.get('entryPrice') or 0)

        market = None
        try:
            market = exchange.markets.get(ccxt_symbol)
        except Exception:
            market = None

        contract_size = None
        try:
            contract_size = float((market or {}).get('contractSize') or 0) or None
        except Exception:
            contract_size = None

        base_qty = contracts
        if contract_size:
            base_qty = contracts * contract_size

        margin = None
        for k in ['initialMargin', 'margin', 'collateral', 'notional']:
            try:
                v = chosen.get(k)
                if v is not None:
                    margin = float(v)
                    break
            except Exception:
                pass
        if margin is None:
            margin = 0.0

        if contracts <= 0:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'total_cost': 0,
                'take_profit': None,
                'stop_loss': None,
                'action': 'BUY'
            }
            return

        self.positions[symbol] = {
            'quantity': float(base_qty),
            'avg_price': float(entry_px),
            'total_cost': float(margin),
            'take_profit': self.positions.get(symbol, {}).get('take_profit') if isinstance(self.positions.get(symbol), dict) else None,
            'stop_loss': self.positions.get(symbol, {}).get('stop_loss') if isinstance(self.positions.get(symbol), dict) else None,
            'action': 'SELL' if side == 'short' else 'BUY'
        }

    async def _execute_real_trade(self, symbol: str, action: str, amount_usd: float, strategy: str = "live") -> Dict[str, Any]:
        self.last_real_order_error = None

        exchange = await self._get_ccxt_exchange()
        ccxt_symbol = self._to_ccxt_swap_symbol(symbol)

        try:
            leverage_int = int(float(self.leverage))
        except Exception:
            leverage_int = 1
        if leverage_int < 1:
            leverage_int = 1

        params = {}
        try:
            margin_mode = str(os.getenv('MEXC_MARGIN_MODE', 'isolated') or 'isolated').strip().lower()
            if margin_mode:
                params['marginMode'] = margin_mode
        except Exception:
            pass

        try:
            await exchange.set_leverage(leverage_int, ccxt_symbol, params)
        except Exception:
            pass

        ticker = await exchange.fetch_ticker(ccxt_symbol)
        last_price = float(ticker.get('last') or ticker.get('close') or 0)
        if last_price <= 0:
            raise RuntimeError('Unable to fetch last price for live order sizing')

        market = exchange.markets.get(ccxt_symbol) or {}
        try:
            contract_size = float(market.get('contractSize') or 0) or None
        except Exception:
            contract_size = None

        order_side = 'buy' if str(action).upper() == 'BUY' else 'sell'
        order_type = 'market'

        existing_contracts = 0.0
        existing_side = None
        try:
            positions = await exchange.fetch_positions([ccxt_symbol])
            for p in positions or []:
                if str(p.get('symbol') or '').upper() == str(ccxt_symbol).upper():
                    existing_contracts = float(p.get('contracts') or 0)
                    existing_side = str(p.get('side') or '').lower() or None
                    break
        except Exception:
            pass

        is_reduce = False
        if existing_contracts > 0 and existing_side in ['long', 'short']:
            if existing_side == 'long' and order_side == 'sell':
                is_reduce = True
            elif existing_side == 'short' and order_side == 'buy':
                is_reduce = True

        order_params = {}
        try:
            order_params.update(params)
        except Exception:
            pass

        if is_reduce:
            order_params['reduceOnly'] = True

        client_order_id = None
        try:
            client_order_id = self._make_client_order_id(symbol, order_side, bool(is_reduce))
            order_params['clientOrderId'] = client_order_id
        except Exception:
            client_order_id = None

        # Sizing rules match the paper engine:
        # - OPEN/ADD: amount_usd is margin -> notional = margin * leverage
        # - CLOSE/REDUCE: amount_usd is notional to close (no leverage multiplier)
        amount_usd_f = float(amount_usd or 0)
        if amount_usd_f <= 0:
            amount_usd_f = 0.0

        if is_reduce:
            # Close up to the existing position
            position_base_qty = existing_contracts
            if contract_size:
                position_base_qty = existing_contracts * contract_size

            position_notional = position_base_qty * last_price
            close_notional = amount_usd_f if amount_usd_f > 0 else position_notional
            close_notional = min(close_notional, position_notional)
            if close_notional <= 0:
                return {'success': False, 'error': 'No position to reduce'}

            close_base_qty = close_notional / last_price
            contracts = close_base_qty
            if contract_size:
                contracts = close_base_qty / contract_size

            # Clamp to existing
            contracts = min(float(contracts or 0), float(existing_contracts or 0))
        else:
            notional = amount_usd_f * float(self.leverage or 1)
            if notional <= 0:
                return {'success': False, 'error': 'Amount must be > 0'}
            base_qty = notional / last_price
            contracts = base_qty
            if contract_size:
                contracts = base_qty / contract_size

        try:
            contracts = float(exchange.amount_to_precision(ccxt_symbol, contracts))
        except Exception:
            pass

        if contracts <= 0:
            return {'success': False, 'error': 'Order size too small after precision/limits'}

        order = None
        attempts = 0
        last_err = None
        max_attempts = 4
        base_sleep_s = 0.35
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                order = await exchange.create_order(ccxt_symbol, order_type, order_side, contracts, None, order_params)
                last_err = None
                break
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                retryable = any(k in msg for k in ['timeout', 'timed out', 'tempor', 'overload', 'busy', 'network', 'rate limit', '429', 'too many'])
                if not retryable or attempt >= max_attempts:
                    break
                sleep_s = base_sleep_s * (2 ** (attempt - 1))
                try:
                    sleep_s += random.uniform(0.0, base_sleep_s)
                except Exception:
                    pass
                await asyncio.sleep(min(5.0, float(sleep_s)))

        if order is None:
            self.last_real_order_error = str(last_err) if last_err is not None else 'create_order failed'
            return {'success': False, 'error': self.last_real_order_error}

        try:
            oid = order.get('id') if isinstance(order, dict) else None
            if oid:
                fetched = await exchange.fetch_order(oid, ccxt_symbol)
                if fetched:
                    order = fetched
        except Exception:
            pass

        self.last_real_order = {
            'id': order.get('id'),
            'client_order_id': client_order_id,
            'symbol': symbol,
            'ccxt_symbol': ccxt_symbol,
            'side': order_side,
            'type': order_type,
            'contracts': contracts,
            'strategy': strategy,
            'attempts': attempts,
            'timestamp': datetime.now().isoformat(),
        }

        try:
            await self._sync_from_exchange(symbol)
        except Exception:
            pass

        try:
            self.total_trades = int(getattr(self, 'total_trades', 0) or 0) + 1
        except Exception:
            pass

        try:
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': str(action).upper(),
                'amount_usd': float(amount_usd or 0),
                'mode': 'REAL_TRADING',
                'order_id': order.get('id'),
                'client_order_id': client_order_id,
                'attempts': attempts,
                'strategy': strategy,
                'reduce_only': bool(is_reduce),
            }
            if isinstance(getattr(self, 'trade_history', None), list):
                self.trade_history.append(trade_record)
        except Exception:
            pass

        try:
            self._save_state()
        except Exception:
            pass

        return {'success': True, 'mode': 'REAL_TRADING', 'order': order}
    
    async def execute_live_trade(self, symbol: str, action: str, amount_usd: float, strategy: str = "test", stop_loss: float = None, take_profit: float = None, *args, **kwargs):
        """Execute trade using live market prices.
        Accepts optional stop_loss/take_profit and extra args for compatibility.
        """
        
        print(f"\n🎯 EXECUTING LIVE TRADE: {action} ${amount_usd} of {symbol}")

        real_enabled_now = False
        try:
            real_env = str(os.getenv('REAL_TRADING', '0') or '0').strip().lower()
            real_enabled_now = real_env in ['1', 'true', 'yes', 'on']
        except Exception:
            real_enabled_now = False

        try:
            if not bool(getattr(self, 'real_trading_enabled', False)) and real_enabled_now:
                self.real_trading_enabled = True
        except Exception:
            pass

        if real_enabled_now and not self._real_trading_ready():
            raise RuntimeError(
                "REAL_TRADING enabled but futures real trading is not ready (missing API keys/ccxt, or market_type not futures). Refusing to paper trade."
            )

        if self._real_trading_ready():
            try:
                return await self._execute_real_trade(symbol, action, amount_usd, strategy=strategy)
            except Exception as e:
                self.last_real_order_error = str(e)
                return {"success": False, "error": f"REAL_TRADING failed: {e}"}
        
        # Get current live price
        current_price = await self.data_feed.get_live_price(symbol)
        
        if not current_price:
            return {"success": False, "error": "Could not get live price"}
        
        print(f"📈 LIVE {symbol} Price: ${current_price:,.2f}")

        latency_ms = 0
        try:
            if self.paper_execution_model == 'realistic' and self.paper_latency_ms_max > 0:
                latency_ms = random.randint(int(self.paper_latency_ms_min), int(self.paper_latency_ms_max))
        except Exception:
            latency_ms = 0
        if latency_ms and latency_ms > 0:
            await asyncio.sleep(float(latency_ms) / 1000.0)

        fill_ratio = 1.0
        try:
            if self.paper_execution_model == 'realistic' and self.paper_partial_fill_prob > 0:
                if random.random() < float(self.paper_partial_fill_prob):
                    fill_ratio = random.uniform(float(self.paper_partial_fill_min_pct), float(self.paper_partial_fill_max_pct))
        except Exception:
            fill_ratio = 1.0
        fill_ratio = max(0.05, min(1.0, float(fill_ratio)))

        spread_pct = 0.0
        slippage_pct = 0.0

        if self.paper_execution_model == 'realistic':
            spread_pct = max(0.0, float(self.paper_spread_bps) / 10000.0)
            slip_bound = max(0.0, float(self.paper_slippage_bps) / 10000.0)
            try:
                slippage_pct = abs(random.gauss(0.0, slip_bound / 2.0))
            except Exception:
                slippage_pct = random.uniform(0.0, slip_bound)
            slippage_pct = min(slippage_pct, slip_bound * 3.0)
        else:
            if ALLOW_SIMULATED_FEATURES:
                slippage_pct = random.uniform(-0.0002, 0.0002)

        action_u = str(action).upper()
        if action_u not in ['BUY', 'SELL']:
            return {"success": False, "error": f"Invalid action: {action}"}

        half_spread = spread_pct / 2.0
        if self.paper_execution_model == 'realistic':
            if action_u == 'BUY':
                execution_price = current_price * (1.0 + half_spread + slippage_pct)
            else:
                execution_price = current_price * (1.0 - half_spread - slippage_pct)
        else:
            execution_price = current_price * (1.0 + slippage_pct)

        contract_meta = None
        contract_size = None
        min_vol = 1
        vol_unit = 1
        effective_fee_rate = self.fee_rate
        try:
            if self.market_type == 'futures' and hasattr(self.data_feed, 'get_contract_meta'):
                contract_meta = self.data_feed.get_contract_meta(symbol) or None
                if contract_meta:
                    contract_size = float(contract_meta.get('contractSize') or 0) or None
                    min_vol = int(float(contract_meta.get('minVol') or 1) or 1)
                    vol_unit = int(float(contract_meta.get('volUnit') or 1) or 1)
                    effective_fee_rate = float(contract_meta.get('takerFeeRate') or self.fee_rate or 0)
        except Exception:
            contract_meta = None
            contract_size = None
            min_vol = 1
            vol_unit = 1
            effective_fee_rate = self.fee_rate

        existing = self.positions.get(symbol) if isinstance(getattr(self, 'positions', None), dict) else None
        existing_qty = float(existing.get('quantity', 0) or 0) if isinstance(existing, dict) else 0.0
        existing_side = str(existing.get('action', 'BUY') or 'BUY').upper() if isinstance(existing, dict) else ''

        trade_event = None
        realized_pnl = None
        notional_usd = None
        
        if action_u == 'BUY':
            if existing_qty > 0 and existing_side == 'SELL':
                max_notional = existing_qty * execution_price
                close_notional = float(amount_usd or 0)
                if close_notional <= 0:
                    close_notional = max_notional
                close_notional = min(close_notional, max_notional)

                if self.paper_execution_model == 'realistic' and fill_ratio < 1.0:
                    close_notional = close_notional * fill_ratio

                trade_event = 'COVER_SHORT'
                notional_usd = close_notional

                quantity = close_notional / execution_price
                if contract_size:
                    try:
                        max_contracts = int(existing_qty / contract_size) if contract_size > 0 else 0
                        desired_contracts = int(quantity / contract_size) if contract_size > 0 else 0
                        if vol_unit > 1 and desired_contracts > 0:
                            desired_contracts = desired_contracts - (desired_contracts % vol_unit)
                        if desired_contracts <= 0 and max_contracts > 0:
                            desired_contracts = max_contracts
                        desired_contracts = min(desired_contracts, max_contracts)
                        if desired_contracts > 0:
                            quantity = desired_contracts * contract_size
                            close_notional = quantity * execution_price
                    except Exception:
                        pass
                commission = close_notional * effective_fee_rate

                entry_px = float(existing.get('avg_price', 0) or 0)
                pnl = (entry_px - execution_price) * quantity

                old_margin = float(existing.get('total_cost', 0) or 0)
                margin_released = old_margin * (quantity / existing_qty) if existing_qty > 0 else 0.0

                self.cash_balance += margin_released + pnl - commission
                realized_pnl = pnl - commission
                new_qty = existing_qty - quantity
                new_margin = max(0.0, old_margin - margin_released)

                existing_tp = existing.get('take_profit')
                existing_sl = existing.get('stop_loss')

                if new_qty <= 0.0000001:
                    self.positions[symbol] = {
                        "quantity": 0,
                        "avg_price": 0,
                        "total_cost": 0,
                        "take_profit": None,
                        "stop_loss": None,
                        "action": "BUY"
                    }
                else:
                    self.positions[symbol] = {
                        "quantity": new_qty,
                        "avg_price": entry_px,
                        "total_cost": new_margin,
                        "take_profit": take_profit if take_profit is not None else existing_tp,
                        "stop_loss": stop_loss if stop_loss is not None else existing_sl,
                        "action": "SELL"
                    }
            else:
                margin = float(amount_usd or 0)
                if margin <= 0:
                    return {"success": False, "error": "Amount must be > 0"}

                if self.paper_execution_model == 'realistic' and fill_ratio < 1.0:
                    margin = margin * fill_ratio

                notional = margin * (self.leverage if self.enable_shorting else 1.0)
                quantity = notional / execution_price
                if contract_size:
                    try:
                        desired_contracts = int(quantity / contract_size) if contract_size > 0 else 0
                        if vol_unit > 1 and desired_contracts > 0:
                            desired_contracts = desired_contracts - (desired_contracts % vol_unit)
                        if desired_contracts < min_vol:
                            desired_contracts = min_vol
                        quantity = desired_contracts * contract_size
                        notional = quantity * execution_price
                        margin = notional / (self.leverage if self.enable_shorting else 1.0)
                    except Exception:
                        pass

                commission = notional * effective_fee_rate
                required = margin + commission
                if required > self.cash_balance:
                    return {"success": False, "error": f"Insufficient funds: ${self.cash_balance:.2f}"}


                trade_event = 'OPEN_LONG' if not (existing_qty > 0 and existing_side == 'BUY') else 'ADD_LONG'
                notional_usd = notional

                if symbol not in self.positions:
                    self.positions[symbol] = {
                        "quantity": 0,
                        "avg_price": 0,
                        "total_cost": 0,
                        "take_profit": None,
                        "stop_loss": None,
                        "action": "BUY"
                    }

                if existing_qty > 0 and existing_side == 'BUY':
                    old_quantity = existing_qty
                    old_margin = float(existing.get('total_cost', 0) or 0)
                    old_avg = float(existing.get('avg_price', 0) or 0)
                    new_quantity = old_quantity + quantity
                    new_avg = (old_avg * old_quantity + execution_price * quantity) / new_quantity if new_quantity > 0 else 0
                    new_margin = old_margin + margin
                    existing_tp = existing.get('take_profit')
                    existing_sl = existing.get('stop_loss')
                else:
                    new_quantity = quantity
                    new_avg = execution_price
                    new_margin = margin
                    existing_tp = existing.get('take_profit') if isinstance(existing, dict) else None
                    existing_sl = existing.get('stop_loss') if isinstance(existing, dict) else None

                self.positions[symbol] = {
                    "quantity": new_quantity,
                    "avg_price": new_avg,
                    "total_cost": new_margin,
                    "take_profit": take_profit if take_profit is not None else existing_tp,
                    "stop_loss": stop_loss if stop_loss is not None else existing_sl,
                    "action": "BUY"
                }

                self.cash_balance -= required
                realized_pnl = -commission

        else:  # SELL
            if existing_qty > 0 and existing_side == 'BUY':
                max_notional = existing_qty * execution_price
                close_notional = float(amount_usd or 0)
                if close_notional <= 0:
                    close_notional = max_notional
                close_notional = min(close_notional, max_notional)

                if self.paper_execution_model == 'realistic' and fill_ratio < 1.0:
                    close_notional = close_notional * fill_ratio

                trade_event = 'CLOSE_LONG'
                notional_usd = close_notional

                quantity = close_notional / execution_price
                if contract_size:
                    try:
                        max_contracts = int(existing_qty / contract_size) if contract_size > 0 else 0
                        desired_contracts = int(quantity / contract_size) if contract_size > 0 else 0
                        if vol_unit > 1 and desired_contracts > 0:
                            desired_contracts = desired_contracts - (desired_contracts % vol_unit)
                        if desired_contracts <= 0 and max_contracts > 0:
                            desired_contracts = max_contracts
                        desired_contracts = min(desired_contracts, max_contracts)
                        if desired_contracts > 0:
                            quantity = desired_contracts * contract_size
                            close_notional = quantity * execution_price
                            notional_usd = close_notional
                    except Exception:
                        pass
                commission = close_notional * effective_fee_rate

                entry_px = float(existing.get('avg_price', 0) or 0)
                pnl = (execution_price - entry_px) * quantity

                old_margin = float(existing.get('total_cost', 0) or 0)
                margin_released = old_margin * (quantity / existing_qty) if existing_qty > 0 else 0.0

                self.cash_balance += margin_released + pnl - commission
                realized_pnl = pnl - commission
                new_qty = existing_qty - quantity
                new_margin = max(0.0, old_margin - margin_released)

                existing_tp = existing.get('take_profit')
                existing_sl = existing.get('stop_loss')

                if new_qty <= 0.0000001:
                    self.positions[symbol] = {
                        "quantity": 0,
                        "avg_price": 0,
                        "total_cost": 0,
                        "take_profit": None,
                        "stop_loss": None,
                        "action": "BUY"
                    }
                else:
                    self.positions[symbol] = {
                        "quantity": new_qty,
                        "avg_price": entry_px,
                        "total_cost": new_margin,
                        "take_profit": take_profit if take_profit is not None else existing_tp,
                        "stop_loss": stop_loss if stop_loss is not None else existing_sl,
                        "action": "BUY"
                    }
            else:
                if not self.enable_shorting:
                    return {"success": False, "error": "Shorting disabled"}

                margin = float(amount_usd or 0)
                if margin <= 0:
                    return {"success": False, "error": "Amount must be > 0"}

                if self.paper_execution_model == 'realistic' and fill_ratio < 1.0:
                    margin = margin * fill_ratio

                notional = margin * self.leverage
                quantity = notional / execution_price
                if contract_size:
                    try:
                        desired_contracts = int(quantity / contract_size) if contract_size > 0 else 0
                        if vol_unit > 1 and desired_contracts > 0:
                            desired_contracts = desired_contracts - (desired_contracts % vol_unit)
                        if desired_contracts < min_vol:
                            desired_contracts = min_vol
                        quantity = desired_contracts * contract_size
                        notional = quantity * execution_price
                        margin = notional / self.leverage
                    except Exception:
                        pass

                commission = notional * effective_fee_rate
                required = margin + commission
                if required > self.cash_balance:
                    return {"success": False, "error": f"Insufficient funds: ${self.cash_balance:.2f}"}


                trade_event = 'OPEN_SHORT' if not (existing_qty > 0 and existing_side == 'SELL') else 'ADD_SHORT'
                notional_usd = notional

                if symbol not in self.positions:
                    self.positions[symbol] = {
                        "quantity": 0,
                        "avg_price": 0,
                        "total_cost": 0,
                        "take_profit": None,
                        "stop_loss": None,
                        "action": "BUY"
                    }

                if existing_qty > 0 and existing_side == 'SELL':
                    old_quantity = existing_qty
                    old_margin = float(existing.get('total_cost', 0) or 0)
                    old_avg = float(existing.get('avg_price', 0) or 0)
                    new_quantity = old_quantity + quantity
                    new_avg = (old_avg * old_quantity + execution_price * quantity) / new_quantity if new_quantity > 0 else 0
                    new_margin = old_margin + margin
                    existing_tp = existing.get('take_profit')
                    existing_sl = existing.get('stop_loss')
                else:
                    new_quantity = quantity
                    new_avg = execution_price
                    new_margin = margin
                    existing_tp = existing.get('take_profit') if isinstance(existing, dict) else None
                    existing_sl = existing.get('stop_loss') if isinstance(existing, dict) else None

                self.positions[symbol] = {
                    "quantity": new_quantity,
                    "avg_price": new_avg,
                    "total_cost": new_margin,
                    "take_profit": take_profit if take_profit is not None else existing_tp,
                    "stop_loss": stop_loss if stop_loss is not None else existing_sl,
                    "action": "SELL"
                }

                self.cash_balance -= required
                realized_pnl = -commission
        
        # Record trade
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action.upper(),
            "amount_usd": amount_usd,
            "notional_usd": notional_usd,
            "quantity": quantity,
            "live_price": current_price,
            "execution_price": execution_price,
            "slippage_pct": slippage_pct * 100,
            "spread_bps": spread_pct * 10000,
            "latency_ms": latency_ms,
            "fill_ratio": fill_ratio,
            "fee_rate": effective_fee_rate,
            "commission": commission,
            "pnl": float(realized_pnl or 0.0),
            "event": trade_event,
            "strategy": strategy,
            "success": True
        }
        
        self.trade_history.append(trade_record)
        self.total_trades += 1
        
        print(f"✅ TRADE EXECUTED:")
        print(f"   💰 Quantity: {quantity:.6f} {symbol.split('/')[0]}")
        print(f"   💵 Price: ${execution_price:,.2f} (slippage: {slippage_pct*100:.2f}%)")
        print(f"   💸 Commission: ${commission:.2f}")
        print(f"   🏦 Cash Balance: ${self.cash_balance:,.2f}")
        
        # Save state after each trade
        self._save_state()
        
        return {"success": True, "trade": trade_record}

    async def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        try:
            symbol = signal.get('symbol')
            action_u = str(signal.get('action') or '').upper()
            if not symbol or action_u not in ['BUY', 'SELL']:
                return {'success': False, 'error': 'Invalid signal'}

            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            strategy = signal.get('strategy_name', 'paper')

            qty = 0.0
            try:
                qty = float(signal.get('position_size') or 0)
            except Exception:
                qty = 0.0

            current_price = None
            try:
                current_price = await self.data_feed.get_live_price(symbol)
            except Exception:
                current_price = None
            if not current_price:
                try:
                    current_price = float(signal.get('entry_price') or 0) or None
                except Exception:
                    current_price = None

            if not current_price or float(current_price) <= 0:
                return {'success': False, 'error': 'Unable to fetch price for sizing'}
            current_price = float(current_price)

            existing = self.positions.get(symbol) if isinstance(getattr(self, 'positions', None), dict) else None
            existing_qty = float(existing.get('quantity', 0) or 0) if isinstance(existing, dict) else 0.0
            existing_side = str(existing.get('action', 'BUY') or 'BUY').upper() if isinstance(existing, dict) else 'BUY'

            if qty <= 0:
                portfolio = self.get_portfolio_value_sync() or {}
                total_value = float(portfolio.get('total_value') or self.cash_balance or 0)
                target_notional = total_value * 0.05
                qty = target_notional / current_price if current_price > 0 else 0.0
                if qty <= 0:
                    return {'success': False, 'error': 'Unable to size position'}

            is_close = False
            if existing_qty > 0:
                if action_u == 'SELL' and existing_side == 'BUY':
                    is_close = True
                elif action_u == 'BUY' and existing_side == 'SELL':
                    is_close = True

            leverage_factor = float(self.leverage if getattr(self, 'enable_shorting', False) else 1.0)
            if leverage_factor <= 0:
                leverage_factor = 1.0

            if is_close:
                amount_usd = qty * current_price
            else:
                amount_usd = (qty * current_price) / leverage_factor

            result = await self.execute_live_trade(
                symbol,
                action_u,
                float(amount_usd or 0),
                strategy=strategy,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            if not isinstance(result, dict):
                return {'success': False, 'error': 'Unexpected paper trader response'}
            if not result.get('success'):
                return result

            trade = result.get('trade') if isinstance(result.get('trade'), dict) else {}

            portfolio_value = 0.0
            total_pnl = 0.0
            try:
                portfolio = await self.get_portfolio_value()
                if isinstance(portfolio, dict):
                    portfolio_value = float(portfolio.get('total_value') or 0)
                    total_pnl = float(portfolio.get('total_pnl') or 0)
            except Exception:
                pass

            return {
                'success': True,
                'execution_price': trade.get('execution_price'),
                'position_size': trade.get('quantity'),
                'commission': trade.get('commission', 0),
                'portfolio_value': portfolio_value,
                'total_pnl': total_pnl,
                'trade': trade,
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_performance_summary(self) -> Dict[str, Any]:
        portfolio = self.get_portfolio_value_sync() or {}
        open_positions = 0
        try:
            if isinstance(getattr(self, 'positions', None), dict):
                open_positions = sum(1 for p in self.positions.values() if isinstance(p, dict) and float(p.get('quantity', 0) or 0) > 0)
        except Exception:
            open_positions = 0

        portfolio_value = float(portfolio.get('total_value') or 0)
        total_pnl = float(portfolio.get('total_pnl') or 0)

        return {
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'cash_balance': float(getattr(self, 'cash_balance', 0) or 0),
            'open_positions': open_positions,
            'total_trades': int(getattr(self, 'total_trades', 0) or 0),
        }
    
    async def get_portfolio_value(self):
        """Calculate total portfolio value using live prices"""
        
        portfolio_value = self.cash_balance
        position_values = {}
        
        # Get live prices for all positions
        symbols_with_positions = [symbol for symbol, pos in self.positions.items() if pos["quantity"] > 0]
        
        live_prices = {}
        if symbols_with_positions:
            fetched = await self.data_feed.get_multiple_prices(symbols_with_positions)
            live_prices = fetched or {}
        
        for symbol, position in self.positions.items():
            if position.get("quantity", 0) > 0:
                if symbol in live_prices:
                    current_price = live_prices[symbol]
                elif position.get("current_price") and position.get("current_price") > 0:
                    current_price = position["current_price"]
                else:
                    current_price = position.get("avg_price", 0)

                side = str(position.get('action', 'BUY') or 'BUY').upper()
                qty = float(position.get('quantity', 0) or 0)
                avg_px = float(position.get('avg_price', current_price) or current_price)
                margin = float(position.get('total_cost', 0) or 0)

                notional_value = qty * current_price
                unrealized_pnl = (current_price - avg_px) * qty if side == 'BUY' else (avg_px - current_price) * qty
                equity_value = margin + unrealized_pnl

                portfolio_value += equity_value

                position["current_price"] = current_price
                position["current_value"] = notional_value
                position["cost_basis"] = margin
                position["unrealized_pnl"] = unrealized_pnl
                position["avg_price"] = avg_px
                position["equity_value"] = equity_value
                if "take_profit" not in position:
                    position["take_profit"] = None
                if "stop_loss" not in position:
                    position["stop_loss"] = None
                
                position_values[symbol] = {
                    "quantity": qty,
                    "current_price": current_price,
                    "current_value": notional_value,
                    "cost_basis": margin,
                    "unrealized_pnl": unrealized_pnl,
                    "avg_price": avg_px,
                    "equity_value": equity_value,
                    "action": side,
                    "take_profit": position.get("take_profit"),
                    "stop_loss": position.get("stop_loss")
                }
        
        return {
            "total_value": portfolio_value,
            "cash": self.cash_balance,
            "positions": position_values,
            "total_return": (portfolio_value - self.initial_capital) / self.initial_capital,
            "total_pnl": portfolio_value - self.initial_capital
        }
    
    def get_portfolio_value_sync(self):
        """SYNC version for dashboard - uses last known/avg prices instead of live lookup"""
        portfolio_value = self.cash_balance
        position_values = {}
        
        for symbol, position in self.positions.items():
            if position.get("quantity", 0) > 0:
                current_price = position.get('current_price', position.get("avg_price", 0))
                side = str(position.get('action', 'BUY') or 'BUY').upper()
                qty = float(position.get('quantity', 0) or 0)
                avg_px = float(position.get('avg_price', current_price) or current_price)
                margin = float(position.get('total_cost', 0) or 0)

                notional_value = qty * current_price
                unrealized_pnl = (current_price - avg_px) * qty if side == 'BUY' else (avg_px - current_price) * qty
                equity_value = margin + unrealized_pnl

                portfolio_value += equity_value
                pnl_percentage = (unrealized_pnl / margin) * 100 if margin > 0 else 0
                position_values[symbol] = {
                    "quantity": qty,
                    "current_price": current_price,
                    "current_value": notional_value,
                    "cost_basis": margin,
                    "unrealized_pnl": unrealized_pnl,
                    "pnl_percentage": pnl_percentage,
                    "avg_price": avg_px,
                    "equity_value": equity_value,
                    "action": side,
                    "take_profit": position.get('take_profit'),
                    "stop_loss": position.get('stop_loss')
                }
        
        return {
            "total_value": portfolio_value,
            "cash": self.cash_balance,
            "positions": position_values,
            "total_return": (portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0,
            "total_pnl": portfolio_value - self.initial_capital
        }

    def _save_state(self):
        """Save trading state to file"""
        try:
            state = {
                "cash_balance": self.cash_balance,
                "positions": self.positions,
                "trade_history": self.trade_history[-100:],  # Keep last 100 trades
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "initial_capital": self.initial_capital,
                "last_save_time": datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            print(f"⚠️ Failed to save state: {e}")
            return False
    
    def _load_state(self):
        """Load trading state from file - with automatic stale data detection"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Check if state is stale (older than 30 minutes = probably from old deployment)
                last_save_time = state.get('last_save_time')
                is_stale = False
                
                if last_save_time:
                    try:
                        last_save = datetime.fromisoformat(last_save_time)
                        age_minutes = (datetime.now() - last_save).total_seconds() / 60
                        
                        max_age_minutes = float(os.getenv('PAPER_STATE_MAX_AGE_MINUTES', '10080') or 10080)
                        if max_age_minutes > 0 and age_minutes > max_age_minutes:
                            is_stale = True
                            print(f"⚠️ State is {age_minutes:.1f} minutes old - STALE DATA DETECTED")
                    except:
                        pass
                
                # Check for unrealistic position prices (indicates old/corrupted data)
                positions = state.get("positions", {})
                has_unrealistic_prices = False
                
                for symbol, pos in positions.items():
                    if pos.get('quantity', 0) > 0:
                        avg_price = pos.get('avg_price', 0)
                        # Check if prices are wildly unrealistic
                        if symbol == "BTC/USDT" and (avg_price > 120000 or avg_price < 20000):
                            has_unrealistic_prices = True
                            print(f"⚠️ Unrealistic BTC price detected: ${avg_price:,.2f}")
                        elif symbol == "ETH/USDT" and (avg_price > 10000 or avg_price < 1000):
                            has_unrealistic_prices = True
                            print(f"⚠️ Unrealistic ETH price detected: ${avg_price:,.2f}")
                
                # If data is stale or has unrealistic prices, start fresh
                if is_stale or has_unrealistic_prices:
                    print("🔄 STARTING FRESH - Stale/corrupted data detected")
                    print(f"   Reason: {'Stale timestamp' if is_stale else 'Unrealistic prices'}")
                    
                    # Delete the stale state file
                    try:
                        os.remove(self.state_file)
                        print(f"   🗑️ Deleted stale state file")
                    except:
                        pass
                    
                    return False  # Start fresh
                
                # Data looks good - load it
                self.cash_balance = state.get("cash_balance", self.initial_capital)
                self.positions = state.get("positions", {})
                self.trade_history = state.get("trade_history", [])
                self.total_trades = state.get("total_trades", 0)
                self.winning_trades = state.get("winning_trades", 0)
                
                # Keep initial capital from constructor if not in state
                if "initial_capital" in state:
                    self.initial_capital = state["initial_capital"]
                
                print(f"✅ Loaded valid state from {state.get('last_save_time', 'unknown time')}")
                return True
            return False
        except Exception as e:
            print(f"⚠️ Failed to load state: {e}")
            return False

async def run_live_paper_trading_test():
    """Test paper trading with live MEXC prices"""
    
    print("🚀 LIVE PAPER TRADING TEST")
    print("🔥 Using REAL-TIME market prices from MEXC!")
    print("=" * 60)
    
    # Initialize live paper trading manager
    trader = LivePaperTradingManager(5000.0)
    
    # Test symbols with live prices
    test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT"]
    
    print(f"\n📊 CURRENT LIVE PRICES:")
    live_prices = await trader.data_feed.get_multiple_prices(test_symbols)
    for symbol, price in live_prices.items():
        print(f"   {symbol:10} ${price:>10,.2f}")
    
    print(f"\n🎯 EXECUTING TEST TRADES WITH LIVE PRICES:")
    
    # Test 1: Buy some BTC with live price
    await trader.execute_live_trade("BTC/USDT", "BUY", 1000, "live_test_btc")
    
    # Test 2: Buy some ETH with live price
    await trader.execute_live_trade("ETH/USDT", "BUY", 1500, "live_test_eth")
    
    # Test 3: Buy some SOL with live price
    await trader.execute_live_trade("SOL/USDT", "BUY", 800, "live_test_sol")
    
    # Test 4: Sell some ETH
    await trader.execute_live_trade("ETH/USDT", "SELL", 500, "live_test_eth_sell")
    
    print(f"\n📈 FINAL PORTFOLIO (with LIVE prices):")
    portfolio = await trader.get_portfolio_value()
    
    print(f"💰 Total Portfolio Value: ${portfolio['total_value']:,.2f}")
    print(f"💵 Cash Balance: ${portfolio['cash']:,.2f}")
    print(f"📈 Total Return: {portfolio['total_return']*100:+.2f}%")
    print(f"💎 Total P&L: ${portfolio['total_pnl']:+,.2f}")
    
    print(f"\n🎯 ACTIVE POSITIONS (with LIVE market values):")
    for symbol, pos in portfolio['positions'].items():
        pnl_pct = (pos['unrealized_pnl'] / pos['cost_basis']) * 100 if pos['cost_basis'] > 0 else 0
        print(f"   {symbol:10} {pos['quantity']:>8.4f} @ ${pos['current_price']:>8,.2f} = ${pos['current_value']:>8,.2f} (PnL: ${pos['unrealized_pnl']:+7,.2f} / {pnl_pct:+5.1f}%)")
    
    print(f"\n🔥 COMPARISON:")
    print(f"   Old Paper Trading: Used 2022 prices (BTC ~$42k)")
    print(f"   Live Paper Trading: Uses 2025 prices (BTC ~$111k)")
    print(f"   This is the REAL market data your bot should use!")
    
    return portfolio

if __name__ == "__main__":
    asyncio.run(run_live_paper_trading_test())
