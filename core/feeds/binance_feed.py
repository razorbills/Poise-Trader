"""
Binance Exchange Feed for Poise Trader

Provides real-time market data from Binance using WebSocket and REST API.
Supports spot and futures markets with comprehensive data normalization.
"""

import asyncio
import aiohttp
import json
import time
import hmac
import hashlib
from typing import Dict, Any, List, Optional
from decimal import Decimal
from urllib.parse import urlencode

from ..framework.base_classes import MarketData
from .base_feed import WebSocketFeed, FeedConfig, RateLimiter


class BinanceFeed(WebSocketFeed):
    """
    Binance market data feed with WebSocket streaming
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Set Binance-specific defaults
        feed_config = FeedConfig(
            exchange_name="binance",
            base_url=config.get('base_url', 'https://api.binance.com'),
            websocket_url=config.get('websocket_url', 'wss://stream.binance.com:9443/ws'),
            rate_limit=config.get('rate_limit', 1200),  # Binance allows 1200 requests/minute
            timeout=config.get('timeout', 30),
            heartbeat_interval=config.get('heartbeat_interval', 180),  # 3 minutes
            **{k: v for k, v in config.items() if k not in ['base_url', 'websocket_url']}
        )
        
        super().__init__(feed_config)
        
        # Binance-specific setup
        self.session = None
        self.listen_key = None
        self.stream_id = 1
        
        # Symbol mapping (Binance uses different format)
        self.symbol_map = {}  # Maps normalized symbols to Binance format
        
    async def connect(self) -> bool:
        """Connect to Binance WebSocket with proper authentication if needed"""
        # Create HTTP session for REST API calls
        self.session = aiohttp.ClientSession()
        
        # Get listen key if we have API credentials
        if self.feed_config.api_key:
            await self._get_listen_key()
        
        # Connect to WebSocket
        return await super().connect()
    
    async def disconnect(self) -> None:
        """Disconnect and cleanup"""
        if self.session:
            await self.session.close()
            self.session = None
        
        if self.listen_key:
            await self._close_listen_key()
            self.listen_key = None
        
        await super().disconnect()
    
    async def subscribe(self, symbols: List[str]) -> None:
        """
        Subscribe to ticker streams for specified symbols
        
        Args:
            symbols: List of normalized symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
        """
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket")
        
        # Convert symbols to Binance format
        binance_symbols = []
        for symbol in symbols:
            binance_symbol = self._normalize_symbol_to_binance(symbol)
            self.symbol_map[symbol] = binance_symbol
            binance_symbols.append(binance_symbol.lower())
        
        # Create stream names for multiple data types
        streams = []
        for binance_symbol in binance_symbols:
            # Ticker stream (24hr statistics)
            streams.append(f"{binance_symbol}@ticker")
            # Trade stream (individual trades)
            streams.append(f"{binance_symbol}@trade")
            # Book ticker (best bid/ask)
            streams.append(f"{binance_symbol}@bookTicker")
            # Kline stream (1-minute candles)
            streams.append(f"{binance_symbol}@kline_1m")
        
        # Subscribe message
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": self.stream_id
        }
        
        await self.websocket.send(json.dumps(subscribe_msg))
        self.stream_id += 1
        
        self.subscribed_symbols.update(symbols)
        self.logger.info("Subscribed to %d symbols on Binance", len(symbols))
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        if not self.websocket:
            return
        
        # Convert to Binance format and create streams
        streams = []
        for symbol in symbols:
            if symbol in self.symbol_map:
                binance_symbol = self.symbol_map[symbol].lower()
                streams.extend([
                    f"{binance_symbol}@ticker",
                    f"{binance_symbol}@trade",
                    f"{binance_symbol}@bookTicker",
                    f"{binance_symbol}@kline_1m"
                ])
                del self.symbol_map[symbol]
        
        # Unsubscribe message
        unsubscribe_msg = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": self.stream_id
        }
        
        await self.websocket.send(json.dumps(unsubscribe_msg))
        self.stream_id += 1
        
        self.subscribed_symbols.difference_update(symbols)
        self.logger.info("Unsubscribed from %d symbols on Binance", len(symbols))
    
    async def _parse_market_data(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse Binance WebSocket message into MarketData"""
        try:
            # Handle different message types
            if 'stream' in data and 'data' in data:
                stream_name = data['stream']
                stream_data = data['data']
                
                if '@ticker' in stream_name:
                    return self._parse_ticker_data(stream_data)
                elif '@trade' in stream_name:
                    return self._parse_trade_data(stream_data)
                elif '@bookTicker' in stream_name:
                    return self._parse_book_ticker_data(stream_data)
                elif '@kline' in stream_name:
                    return self._parse_kline_data(stream_data)
            
            # Handle single stream messages (legacy format)
            elif 'e' in data:  # Event type field
                event_type = data['e']
                
                if event_type == '24hrTicker':
                    return self._parse_ticker_data(data)
                elif event_type == 'trade':
                    return self._parse_trade_data(data)
                elif event_type == 'bookTicker':
                    return self._parse_book_ticker_data(data)
                elif event_type == 'kline':
                    return self._parse_kline_data(data['k'])
            
            return None
            
        except Exception as e:
            self.logger.error("Error parsing Binance data: %s", e)
            return None
    
    def _parse_ticker_data(self, data: Dict[str, Any]) -> MarketData:
        """Parse 24hr ticker statistics"""
        symbol = self._normalize_symbol_from_binance(data['s'])
        
        return MarketData(
            symbol=symbol,
            timestamp=int(data['E']),  # Event time
            price=Decimal(str(data['c'])),  # Last price
            volume=Decimal(str(data['v'])),  # Volume
            bid=Decimal(str(data['b'])) if 'b' in data else None,  # Best bid
            ask=Decimal(str(data['a'])) if 'a' in data else None,  # Best ask
            high_24h=Decimal(str(data['h'])) if 'h' in data else None,  # 24h high
            low_24h=Decimal(str(data['l'])) if 'l' in data else None,  # 24h low
            volume_24h=Decimal(str(data['v'])) if 'v' in data else None,  # 24h volume
            exchange="binance",
            raw_data=data
        )
    
    def _parse_trade_data(self, data: Dict[str, Any]) -> MarketData:
        """Parse individual trade data"""
        symbol = self._normalize_symbol_from_binance(data['s'])
        
        return MarketData(
            symbol=symbol,
            timestamp=int(data['T']),  # Trade time
            price=Decimal(str(data['p'])),  # Price
            volume=Decimal(str(data['q'])),  # Quantity
            exchange="binance",
            raw_data=data
        )
    
    def _parse_book_ticker_data(self, data: Dict[str, Any]) -> MarketData:
        """Parse book ticker (best bid/ask) data"""
        symbol = self._normalize_symbol_from_binance(data['s'])
        
        # Use bid price as the main price (could also use midpoint)
        price = Decimal(str(data['b']))
        
        return MarketData(
            symbol=symbol,
            timestamp=int(time.time() * 1000),  # Current time as Binance doesn't provide timestamp
            price=price,
            volume=Decimal(str(data['B'])),  # Best bid quantity
            bid=Decimal(str(data['b'])),  # Best bid price
            ask=Decimal(str(data['a'])),  # Best ask price
            exchange="binance",
            raw_data=data
        )
    
    def _parse_kline_data(self, data: Dict[str, Any]) -> MarketData:
        """Parse kline/candlestick data"""
        symbol = self._normalize_symbol_from_binance(data['s'])
        
        return MarketData(
            symbol=symbol,
            timestamp=int(data['T']),  # Close time
            price=Decimal(str(data['c'])),  # Close price
            volume=Decimal(str(data['v'])),  # Volume
            high_24h=Decimal(str(data['h'])),  # High price
            low_24h=Decimal(str(data['l'])),  # Low price
            exchange="binance",
            raw_data=data
        )
    
    def _normalize_symbol_to_binance(self, symbol: str) -> str:
        """
        Convert normalized symbol format to Binance format
        
        Examples:
        - 'BTC/USDT' -> 'BTCUSDT'
        - 'ETH/BTC' -> 'ETHBTC'
        """
        if '/' in symbol:
            base, quote = symbol.split('/')
            return f"{base}{quote}"
        return symbol
    
    def _normalize_symbol_from_binance(self, binance_symbol: str) -> str:
        """
        Convert Binance symbol format to normalized format
        
        This is a simple implementation - in production, you'd want
        to use Binance's exchange info API to get accurate mappings
        """
        # Find the symbol in our reverse mapping
        for normalized, binance in self.symbol_map.items():
            if binance == binance_symbol:
                return normalized
        
        # Fallback: try to split common quote currencies
        common_quotes = ['USDT', 'USDC', 'BTC', 'ETH', 'BNB', 'BUSD']
        
        for quote in common_quotes:
            if binance_symbol.endswith(quote):
                base = binance_symbol[:-len(quote)]
                return f"{base}/{quote}"
        
        # If we can't parse it, return as-is
        return binance_symbol
    
    async def get_historical_data(self, symbol: str, start_time: int, 
                                 end_time: int) -> List[MarketData]:
        """
        Get historical kline data from Binance REST API
        """
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
        
        binance_symbol = self._normalize_symbol_to_binance(symbol)
        
        params = {
            'symbol': binance_symbol,
            'interval': '1m',  # 1 minute intervals
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000  # Max 1000 candles per request
        }
        
        url = f"{self.feed_config.base_url}/api/v3/klines"
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    historical_data = []
                    for kline in data:
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=int(kline[6]),  # Close time
                            price=Decimal(str(kline[4])),  # Close price
                            volume=Decimal(str(kline[5])),  # Volume
                            high_24h=Decimal(str(kline[2])),  # High
                            low_24h=Decimal(str(kline[3])),  # Low
                            exchange="binance",
                            raw_data={
                                'open_time': kline[0],
                                'open_price': kline[1],
                                'high_price': kline[2],
                                'low_price': kline[3],
                                'close_price': kline[4],
                                'volume': kline[5],
                                'close_time': kline[6],
                                'quote_asset_volume': kline[7],
                                'number_of_trades': kline[8],
                                'taker_buy_base_asset_volume': kline[9],
                                'taker_buy_quote_asset_volume': kline[10]
                            }
                        )
                        historical_data.append(market_data)
                    
                    return historical_data
                    
                else:
                    self.logger.error("Failed to get historical data: %s", response.status)
                    return []
                    
        except Exception as e:
            self.logger.error("Error getting historical data: %s", e)
            return []
    
    async def _get_listen_key(self):
        """Get listen key for user data stream"""
        if not self.feed_config.api_key:
            return
        
        headers = {'X-MBX-APIKEY': self.feed_config.api_key}
        url = f"{self.feed_config.base_url}/api/v3/userDataStream"
        
        try:
            async with self.session.post(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.listen_key = data['listenKey']
                    self.logger.info("Obtained Binance listen key")
                    
                    # Start keep-alive task
                    asyncio.create_task(self._keep_alive_listen_key())
                    
        except Exception as e:
            self.logger.error("Failed to get listen key: %s", e)
    
    async def _keep_alive_listen_key(self):
        """Keep the listen key alive by sending periodic PUT requests"""
        while self.is_running and self.listen_key:
            try:
                await asyncio.sleep(1800)  # 30 minutes
                
                headers = {'X-MBX-APIKEY': self.feed_config.api_key}
                params = {'listenKey': self.listen_key}
                url = f"{self.feed_config.base_url}/api/v3/userDataStream"
                
                async with self.session.put(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        self.logger.debug("Listen key refreshed")
                    else:
                        self.logger.warning("Failed to refresh listen key")
                        
            except Exception as e:
                self.logger.error("Error in listen key keep-alive: %s", e)
    
    async def _close_listen_key(self):
        """Close the listen key"""
        if not self.listen_key:
            return
        
        try:
            headers = {'X-MBX-APIKEY': self.feed_config.api_key}
            params = {'listenKey': self.listen_key}
            url = f"{self.feed_config.base_url}/api/v3/userDataStream"
            
            async with self.session.delete(url, headers=headers, params=params) as response:
                if response.status == 200:
                    self.logger.info("Listen key closed")
                    
        except Exception as e:
            self.logger.error("Error closing listen key: %s", e)
    
    async def _send_heartbeat(self):
        """Binance doesn't require explicit heartbeat, connection is kept alive automatically"""
        # The connection stays alive as long as we receive messages
        # We can still send a ping to test connectivity
        if self.websocket:
            await self.websocket.ping()
