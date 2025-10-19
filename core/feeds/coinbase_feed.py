"""
Coinbase Pro Exchange Feed for Poise Trader

Provides real-time market data from Coinbase Pro using WebSocket and REST API.
Supports comprehensive market data with proper authentication.
"""

import asyncio
import aiohttp
import json
import time
import hmac
import hashlib
import base64
from typing import Dict, Any, List, Optional
from decimal import Decimal
from urllib.parse import urlencode

from ..framework.base_classes import MarketData
from .base_feed import WebSocketFeed, FeedConfig


class CoinbaseFeed(WebSocketFeed):
    """
    Coinbase Pro market data feed with WebSocket streaming
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Set Coinbase-specific defaults
        feed_config = FeedConfig(
            exchange_name="coinbase",
            base_url=config.get('base_url', 'https://api.pro.coinbase.com'),
            websocket_url=config.get('websocket_url', 'wss://ws-feed.pro.coinbase.com'),
            rate_limit=config.get('rate_limit', 1000),  # Conservative rate limit
            timeout=config.get('timeout', 30),
            heartbeat_interval=config.get('heartbeat_interval', 30),
            **{k: v for k, v in config.items() if k not in ['base_url', 'websocket_url']}
        )
        
        super().__init__(feed_config)
        
        # Coinbase-specific setup
        self.session = None
        self.channels = ['ticker', 'trades', 'level2']  # Default channels
        
        # Symbol mapping
        self.symbol_map = {}
    
    async def connect(self) -> bool:
        """Connect to Coinbase Pro WebSocket"""
        self.session = aiohttp.ClientSession()
        
        # Connect to WebSocket
        return await super().connect()
    
    async def disconnect(self) -> None:
        """Disconnect and cleanup"""
        if self.session:
            await self.session.close()
            self.session = None
        
        await super().disconnect()
    
    async def subscribe(self, symbols: List[str]) -> None:
        """
        Subscribe to market data for specified symbols
        
        Args:
            symbols: List of normalized symbols (e.g., ['BTC/USD', 'ETH/USD'])
        """
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket")
        
        # Convert symbols to Coinbase format
        coinbase_symbols = []
        for symbol in symbols:
            coinbase_symbol = self._normalize_symbol_to_coinbase(symbol)
            self.symbol_map[symbol] = coinbase_symbol
            coinbase_symbols.append(coinbase_symbol)
        
        # Create subscription message
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": coinbase_symbols,
            "channels": self.channels
        }
        
        # Add authentication if credentials are provided
        if self.feed_config.api_key and self.feed_config.api_secret:
            timestamp = str(int(time.time()))
            message = timestamp + 'GET' + '/users/self/verify'
            
            signature = base64.b64encode(
                hmac.new(
                    base64.b64decode(self.feed_config.api_secret),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            
            subscribe_msg.update({
                "signature": signature,
                "key": self.feed_config.api_key,
                "passphrase": self.feed_config.passphrase,
                "timestamp": timestamp
            })
        
        await self.websocket.send(json.dumps(subscribe_msg))
        
        self.subscribed_symbols.update(symbols)
        self.logger.info("Subscribed to %d symbols on Coinbase Pro", len(symbols))
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        if not self.websocket:
            return
        
        # Convert to Coinbase format
        coinbase_symbols = []
        for symbol in symbols:
            if symbol in self.symbol_map:
                coinbase_symbols.append(self.symbol_map[symbol])
                del self.symbol_map[symbol]
        
        if coinbase_symbols:
            unsubscribe_msg = {
                "type": "unsubscribe",
                "product_ids": coinbase_symbols,
                "channels": self.channels
            }
            
            await self.websocket.send(json.dumps(unsubscribe_msg))
        
        self.subscribed_symbols.difference_update(symbols)
        self.logger.info("Unsubscribed from %d symbols on Coinbase Pro", len(symbols))
    
    async def _parse_market_data(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse Coinbase Pro WebSocket message into MarketData"""
        try:
            message_type = data.get('type')
            
            if message_type == 'ticker':
                return self._parse_ticker_data(data)
            elif message_type == 'match':
                return self._parse_trade_data(data)
            elif message_type == 'snapshot' or message_type == 'l2update':
                return self._parse_orderbook_data(data)
            elif message_type == 'heartbeat':
                # Just acknowledge heartbeat, don't create market data
                return None
            elif message_type == 'subscriptions':
                # Subscription confirmation
                self.logger.debug("Subscription confirmed: %s", data)
                return None
            
            return None
            
        except Exception as e:
            self.logger.error("Error parsing Coinbase data: %s", e)
            return None
    
    def _parse_ticker_data(self, data: Dict[str, Any]) -> MarketData:
        """Parse ticker data"""
        product_id = data['product_id']
        symbol = self._normalize_symbol_from_coinbase(product_id)
        
        return MarketData(
            symbol=symbol,
            timestamp=int(self._parse_timestamp(data['time'])),
            price=Decimal(str(data['price'])),
            volume=Decimal(str(data.get('last_size', '0'))),
            bid=Decimal(str(data.get('best_bid', data['price']))),
            ask=Decimal(str(data.get('best_ask', data['price']))),
            high_24h=Decimal(str(data.get('high_24h', data['price']))),
            low_24h=Decimal(str(data.get('low_24h', data['price']))),
            volume_24h=Decimal(str(data.get('volume_24h', '0'))),
            exchange="coinbase",
            raw_data=data
        )
    
    def _parse_trade_data(self, data: Dict[str, Any]) -> MarketData:
        """Parse trade/match data"""
        product_id = data['product_id']
        symbol = self._normalize_symbol_from_coinbase(product_id)
        
        return MarketData(
            symbol=symbol,
            timestamp=int(self._parse_timestamp(data['time'])),
            price=Decimal(str(data['price'])),
            volume=Decimal(str(data['size'])),
            exchange="coinbase",
            raw_data=data
        )
    
    def _parse_orderbook_data(self, data: Dict[str, Any]) -> MarketData:
        """Parse orderbook snapshot/update data"""
        product_id = data['product_id']
        symbol = self._normalize_symbol_from_coinbase(product_id)
        
        # For orderbook data, we'll use the best bid/ask to create market data
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        if not bids or not asks:
            return None
        
        best_bid = Decimal(str(bids[0][0])) if bids else None
        best_ask = Decimal(str(asks[0][0])) if asks else None
        
        # Use mid-price as the main price
        if best_bid and best_ask:
            price = (best_bid + best_ask) / Decimal('2')
        elif best_bid:
            price = best_bid
        elif best_ask:
            price = best_ask
        else:
            return None
        
        return MarketData(
            symbol=symbol,
            timestamp=int(time.time() * 1000),  # Current timestamp
            price=price,
            volume=Decimal('0'),  # No volume in orderbook updates
            bid=best_bid,
            ask=best_ask,
            exchange="coinbase",
            raw_data=data
        )
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse Coinbase timestamp format to Unix timestamp in milliseconds"""
        try:
            # Coinbase uses ISO format: "2023-08-27T10:30:45.123456Z"
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.timestamp() * 1000
        except Exception:
            # Fallback to current time
            return time.time() * 1000
    
    def _normalize_symbol_to_coinbase(self, symbol: str) -> str:
        """
        Convert normalized symbol format to Coinbase format
        
        Examples:
        - 'BTC/USD' -> 'BTC-USD'
        - 'ETH/BTC' -> 'ETH-BTC'
        """
        if '/' in symbol:
            base, quote = symbol.split('/')
            return f"{base}-{quote}"
        return symbol
    
    def _normalize_symbol_from_coinbase(self, coinbase_symbol: str) -> str:
        """
        Convert Coinbase symbol format to normalized format
        """
        # Find in reverse mapping first
        for normalized, coinbase in self.symbol_map.items():
            if coinbase == coinbase_symbol:
                return normalized
        
        # Convert from Coinbase format
        if '-' in coinbase_symbol:
            base, quote = coinbase_symbol.split('-')
            return f"{base}/{quote}"
        
        return coinbase_symbol
    
    async def get_historical_data(self, symbol: str, start_time: int, 
                                 end_time: int) -> List[MarketData]:
        """
        Get historical candle data from Coinbase Pro REST API
        """
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
        
        coinbase_symbol = self._normalize_symbol_to_coinbase(symbol)
        
        # Convert timestamps to ISO format
        start_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start_time / 1000))
        end_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end_time / 1000))
        
        params = {
            'start': start_iso,
            'end': end_iso,
            'granularity': 60  # 1 minute candles
        }
        
        url = f"{self.feed_config.base_url}/products/{coinbase_symbol}/candles"
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    historical_data = []
                    for candle in data:
                        # Coinbase candle format: [timestamp, low, high, open, close, volume]
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=int(candle[0]) * 1000,  # Convert to milliseconds
                            price=Decimal(str(candle[4])),  # Close price
                            volume=Decimal(str(candle[5])),  # Volume
                            high_24h=Decimal(str(candle[2])),  # High
                            low_24h=Decimal(str(candle[1])),  # Low
                            exchange="coinbase",
                            raw_data={
                                'timestamp': candle[0],
                                'low': candle[1],
                                'high': candle[2],
                                'open': candle[3],
                                'close': candle[4],
                                'volume': candle[5]
                            }
                        )
                        historical_data.append(market_data)
                    
                    # Sort by timestamp (Coinbase returns newest first)
                    historical_data.sort(key=lambda x: x.timestamp)
                    
                    return historical_data
                    
                else:
                    self.logger.error("Failed to get historical data: %s", response.status)
                    return []
                    
        except Exception as e:
            self.logger.error("Error getting historical data: %s", e)
            return []
    
    async def _send_heartbeat(self):
        """Coinbase Pro handles heartbeat automatically, just ping WebSocket"""
        if self.websocket:
            await self.websocket.ping()
    
    async def get_products(self) -> List[Dict[str, Any]]:
        """Get all available trading pairs from Coinbase Pro"""
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
        
        url = f"{self.feed_config.base_url}/products"
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    products = await response.json()
                    return products
                else:
                    self.logger.error("Failed to get products: %s", response.status)
                    return []
                    
        except Exception as e:
            self.logger.error("Error getting products: %s", e)
            return []
    
    async def get_order_book(self, symbol: str, level: int = 1) -> Optional[Dict[str, Any]]:
        """
        Get order book for a symbol
        
        Args:
            symbol: Trading pair symbol
            level: 1 = best bid/ask, 2 = top 50, 3 = full order book
        """
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
        
        coinbase_symbol = self._normalize_symbol_to_coinbase(symbol)
        url = f"{self.feed_config.base_url}/products/{coinbase_symbol}/book"
        
        params = {'level': level}
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error("Failed to get order book: %s", response.status)
                    return None
                    
        except Exception as e:
            self.logger.error("Error getting order book: %s", e)
            return None
