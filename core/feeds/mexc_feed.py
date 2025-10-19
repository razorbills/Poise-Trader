#!/usr/bin/env python3
"""
🏛️ MEXC EXCHANGE INTEGRATION
High-performance MEXC exchange connector for Poise Trader

Features:
• Real-time market data via WebSocket
• Advanced order execution
• Portfolio management
• Risk controls
• Auto-reconnection and error handling
"""

import asyncio
import logging
import json
import hmac
import hashlib
import time
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import websockets
from urllib.parse import urlencode

from .base_feed import BaseDataFeed


@dataclass
class MexcTicker:
    """MEXC ticker data structure"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    change_percent_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime


@dataclass
class MexcOrderBook:
    """MEXC order book data"""
    symbol: str
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]
    timestamp: datetime


@dataclass
class MexcTrade:
    """MEXC trade data"""
    symbol: str
    id: str
    price: float
    quantity: float
    side: str
    timestamp: datetime


class MexcConnector(BaseDataFeed):
    """
    🏛️ MEXC Exchange Connector
    
    Provides full integration with MEXC Global exchange including:
    • Real-time market data
    • Order execution
    • Account management
    • Portfolio tracking
    """
    
    BASE_URL = "https://api.mexc.com"
    WS_URL = "wss://wbs.mexc.com/ws"
    
    def __init__(self, api_key: str, secret_key: str, passphrase: str = "", testnet: bool = False):
        super().__init__("mexc")
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.testnet = testnet
        
        if testnet:
            self.BASE_URL = "https://api.mexc.com"  # MEXC doesn't have separate testnet URL
            
        self.logger = logging.getLogger("MexcConnector")
        
        # Connection management
        self.session = None
        self.ws_connection = None
        self.subscribed_symbols = set()
        
        # Data storage
        self.tickers: Dict[str, MexcTicker] = {}
        self.orderbooks: Dict[str, MexcOrderBook] = {}
        self.recent_trades: Dict[str, List[MexcTrade]] = {}
        
        # Account data
        self.account_info = {}
        self.positions = {}
        self.active_orders = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        
        # Status
        self.is_connected = False
        self.is_authenticated = False
        
    async def initialize(self) -> bool:
        """Initialize the MEXC connector"""
        try:
            self.logger.info("🔄 Initializing MEXC connector...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Test API connection
            if await self._test_connection():
                self.logger.info("✅ MEXC API connection successful")
                
                # Get account info to verify authentication
                account = await self.get_account_info()
                if account:
                    self.is_authenticated = True
                    self.logger.info("✅ MEXC authentication successful")
                    
                    # Start WebSocket connection
                    asyncio.create_task(self._start_websocket())
                    
                    # Start data monitoring
                    asyncio.create_task(self._monitor_connection())
                    
                    return True
                else:
                    self.logger.error("❌ MEXC authentication failed")
                    return False
            else:
                self.logger.error("❌ MEXC API connection failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing MEXC connector: {e}")
            return False
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generate MEXC API signature"""
        message = timestamp + method + request_path + body
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """Generate headers for MEXC API requests"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        return {
            'Content-Type': 'application/json',
            'X-MEXC-APIKEY': self.api_key,
            'X-MEXC-TIMESTAMP': timestamp,
            'X-MEXC-SIGN': signature,
        }
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, data: Dict = None) -> Optional[Dict]:
        """Make authenticated request to MEXC API"""
        if not self.session:
            return None
            
        url = f"{self.BASE_URL}{endpoint}"
        
        # Rate limiting
        await self._handle_rate_limit()
        
        try:
            # Prepare request
            request_path = endpoint
            body = ""
            
            if method == "GET" and params:
                query_string = urlencode(params)
                url += f"?{query_string}"
                request_path += f"?{query_string}"
            elif method == "POST" and data:
                body = json.dumps(data)
            
            headers = self._get_headers(method, request_path, body)
            
            # Make request
            async with self.session.request(method, url, headers=headers, data=body) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    self.logger.error(f"MEXC API error {response.status}: {await response.text()}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Request error: {e}")
            return None
    
    async def _handle_rate_limit(self):
        """Handle MEXC API rate limiting"""
        current_time = time.time()
        
        # Reset counter every second
        if current_time - self.last_request_time >= 1:
            self.request_count = 0
            self.last_request_time = current_time
        
        # MEXC allows ~10 requests per second
        if self.request_count >= 8:
            sleep_time = 1 - (current_time - self.last_request_time)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
        self.request_count += 1
    
    async def _test_connection(self) -> bool:
        """Test connection to MEXC API"""
        try:
            result = await self._make_request("GET", "/api/v3/time")
            return result is not None
        except:
            return False
    
    async def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        result = await self._make_request("GET", "/api/v3/account")
        if result:
            self.account_info = result
            return result
        return None
    
    async def get_ticker(self, symbol: str) -> Optional[MexcTicker]:
        """Get ticker for symbol"""
        result = await self._make_request("GET", "/api/v3/ticker/24hr", {"symbol": symbol})
        if result:
            ticker = MexcTicker(
                symbol=result["symbol"],
                price=float(result["lastPrice"]),
                bid=float(result["bidPrice"]),
                ask=float(result["askPrice"]),
                volume_24h=float(result["volume"]),
                change_24h=float(result["priceChange"]),
                change_percent_24h=float(result["priceChangePercent"]),
                high_24h=float(result["highPrice"]),
                low_24h=float(result["lowPrice"]),
                timestamp=datetime.now()
            )
            self.tickers[symbol] = ticker
            return ticker
        return None
    
    async def get_all_tickers(self) -> List[MexcTicker]:
        """Get all tickers"""
        result = await self._make_request("GET", "/api/v3/ticker/24hr")
        tickers = []
        
        if result:
            for ticker_data in result:
                ticker = MexcTicker(
                    symbol=ticker_data["symbol"],
                    price=float(ticker_data["lastPrice"]),
                    bid=float(ticker_data["bidPrice"]),
                    ask=float(ticker_data["askPrice"]),
                    volume_24h=float(ticker_data["volume"]),
                    change_24h=float(ticker_data["priceChange"]),
                    change_percent_24h=float(ticker_data["priceChangePercent"]),
                    high_24h=float(ticker_data["highPrice"]),
                    low_24h=float(ticker_data["lowPrice"]),
                    timestamp=datetime.now()
                )
                tickers.append(ticker)
                self.tickers[ticker.symbol] = ticker
        
        return tickers
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Optional[MexcOrderBook]:
        """Get order book for symbol"""
        params = {"symbol": symbol, "limit": limit}
        result = await self._make_request("GET", "/api/v3/depth", params)
        
        if result:
            bids = [(float(bid[0]), float(bid[1])) for bid in result["bids"]]
            asks = [(float(ask[0]), float(ask[1])) for ask in result["asks"]]
            
            orderbook = MexcOrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.now()
            )
            self.orderbooks[symbol] = orderbook
            return orderbook
        
        return None
    
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, 
                         price: float = None, stop_price: float = None) -> Optional[Dict]:
        """Place order on MEXC"""
        order_data = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        
        if price:
            order_data["price"] = str(price)
        if stop_price:
            order_data["stopPrice"] = str(stop_price)
        
        # Add timestamp
        order_data["timestamp"] = int(time.time() * 1000)
        
        result = await self._make_request("POST", "/api/v3/order", data=order_data)
        
        if result:
            self.logger.info(f"✅ Order placed: {side} {quantity} {symbol} @ {price}")
            return result
        else:
            self.logger.error(f"❌ Failed to place order: {side} {quantity} {symbol}")
            return None
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order"""
        params = {
            "symbol": symbol,
            "orderId": order_id,
            "timestamp": int(time.time() * 1000)
        }
        
        result = await self._make_request("DELETE", "/api/v3/order", params=params)
        return result is not None
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders"""
        params = {"timestamp": int(time.time() * 1000)}
        if symbol:
            params["symbol"] = symbol
            
        result = await self._make_request("GET", "/api/v3/openOrders", params=params)
        return result if result else []
    
    async def _start_websocket(self):
        """Start WebSocket connection for real-time data"""
        try:
            self.logger.info("🔌 Starting MEXC WebSocket connection...")
            
            async with websockets.connect(self.WS_URL) as websocket:
                self.ws_connection = websocket
                self.is_connected = True
                
                self.logger.info("✅ MEXC WebSocket connected")
                
                # Start handling messages
                asyncio.create_task(self._handle_ws_messages())
                
                # Keep connection alive
                while True:
                    try:
                        # Send ping every 30 seconds
                        await asyncio.sleep(30)
                        if self.ws_connection:
                            await self.ws_connection.ping()
                    except:
                        self.logger.warning("WebSocket ping failed, reconnecting...")
                        break
                        
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            self.is_connected = False
            
        # Auto-reconnect after delay
        await asyncio.sleep(5)
        asyncio.create_task(self._start_websocket())
    
    async def _handle_ws_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            if not self.ws_connection:
                return
                
            async for message in self.ws_connection:
                try:
                    data = json.loads(message)
                    await self._process_ws_message(data)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing WebSocket message: {e}")
                    
        except Exception as e:
            self.logger.error(f"WebSocket message handler error: {e}")
    
    async def _process_ws_message(self, data: Dict):
        """Process WebSocket message"""
        try:
            # Handle different message types
            if "stream" in data:
                stream = data["stream"]
                stream_data = data["data"]
                
                if "@ticker" in stream:
                    await self._handle_ticker_update(stream_data)
                elif "@depth" in stream:
                    await self._handle_orderbook_update(stream_data)
                elif "@trade" in stream:
                    await self._handle_trade_update(stream_data)
                    
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
    
    async def _handle_ticker_update(self, data: Dict):
        """Handle ticker updates"""
        symbol = data.get("s", "")
        if symbol:
            ticker = MexcTicker(
                symbol=symbol,
                price=float(data.get("c", 0)),
                bid=float(data.get("b", 0)),
                ask=float(data.get("a", 0)),
                volume_24h=float(data.get("v", 0)),
                change_24h=float(data.get("P", 0)),
                change_percent_24h=float(data.get("p", 0)),
                high_24h=float(data.get("h", 0)),
                low_24h=float(data.get("l", 0)),
                timestamp=datetime.now()
            )
            self.tickers[symbol] = ticker
    
    async def _handle_orderbook_update(self, data: Dict):
        """Handle order book updates"""
        symbol = data.get("s", "")
        if symbol:
            bids = [(float(bid[0]), float(bid[1])) for bid in data.get("b", [])]
            asks = [(float(ask[0]), float(ask[1])) for ask in data.get("a", [])]
            
            orderbook = MexcOrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.now()
            )
            self.orderbooks[symbol] = orderbook
    
    async def _handle_trade_update(self, data: Dict):
        """Handle trade updates"""
        symbol = data.get("s", "")
        if symbol:
            trade = MexcTrade(
                symbol=symbol,
                id=data.get("t", ""),
                price=float(data.get("p", 0)),
                quantity=float(data.get("q", 0)),
                side="buy" if data.get("m", False) else "sell",
                timestamp=datetime.now()
            )
            
            if symbol not in self.recent_trades:
                self.recent_trades[symbol] = []
            
            self.recent_trades[symbol].append(trade)
            
            # Keep only last 100 trades
            if len(self.recent_trades[symbol]) > 100:
                self.recent_trades[symbol] = self.recent_trades[symbol][-100:]
    
    async def subscribe_ticker(self, symbol: str):
        """Subscribe to ticker updates"""
        if self.ws_connection:
            subscribe_msg = {
                "method": "SUBSCRIPTION",
                "params": [f"{symbol.lower()}@ticker"]
            }
            await self.ws_connection.send(json.dumps(subscribe_msg))
            self.subscribed_symbols.add(symbol)
    
    async def subscribe_orderbook(self, symbol: str):
        """Subscribe to order book updates"""
        if self.ws_connection:
            subscribe_msg = {
                "method": "SUBSCRIPTION", 
                "params": [f"{symbol.lower()}@depth20"]
            }
            await self.ws_connection.send(json.dumps(subscribe_msg))
    
    async def subscribe_trades(self, symbol: str):
        """Subscribe to trade updates"""
        if self.ws_connection:
            subscribe_msg = {
                "method": "SUBSCRIPTION",
                "params": [f"{symbol.lower()}@trade"]
            }
            await self.ws_connection.send(json.dumps(subscribe_msg))
    
    async def _monitor_connection(self):
        """Monitor connection health"""
        while True:
            try:
                # Check WebSocket connection
                if not self.is_connected and self.ws_connection:
                    self.logger.warning("WebSocket disconnected, reconnecting...")
                    asyncio.create_task(self._start_websocket())
                
                # Refresh account info periodically
                if self.is_authenticated:
                    await self.get_account_info()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Connection monitor error: {e}")
                await asyncio.sleep(30)
    
    async def get_trading_pairs(self) -> List[str]:
        """Get all available trading pairs"""
        result = await self._make_request("GET", "/api/v3/exchangeInfo")
        
        pairs = []
        if result and "symbols" in result:
            for symbol_info in result["symbols"]:
                if symbol_info["status"] == "TRADING":
                    pairs.append(symbol_info["symbol"])
        
        return pairs
    
    async def get_balance(self, asset: str = None) -> Dict:
        """Get account balance"""
        account = await self.get_account_info()
        if not account:
            return {}
        
        balances = {}
        for balance in account.get("balances", []):
            asset_name = balance["asset"]
            balances[asset_name] = {
                "free": float(balance["free"]),
                "locked": float(balance["locked"]),
                "total": float(balance["free"]) + float(balance["locked"])
            }
        
        if asset:
            return balances.get(asset, {"free": 0.0, "locked": 0.0, "total": 0.0})
        
        return balances
    
    async def cleanup(self):
        """Cleanup connections"""
        try:
            self.is_connected = False
            
            if self.ws_connection:
                await self.ws_connection.close()
                
            if self.session:
                await self.session.close()
                
            self.logger.info("✅ MEXC connector cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    # BaseDataFeed interface methods
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        ticker = await self.get_ticker(symbol)
        return ticker.price if ticker else None
    
    async def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple symbols"""
        prices = {}
        for symbol in symbols:
            price = await self.get_price(symbol)
            if price:
                prices[symbol] = price
        return prices
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data"""
        ticker = await self.get_ticker(symbol)
        orderbook = await self.get_orderbook(symbol, 20)
        
        if not ticker:
            return {}
        
        data = {
            "symbol": symbol,
            "price": ticker.price,
            "bid": ticker.bid,
            "ask": ticker.ask,
            "volume_24h": ticker.volume_24h,
            "change_24h": ticker.change_24h,
            "change_percent_24h": ticker.change_percent_24h,
            "high_24h": ticker.high_24h,
            "low_24h": ticker.low_24h,
            "timestamp": ticker.timestamp
        }
        
        if orderbook:
            data["orderbook"] = {
                "bids": orderbook.bids[:10],  # Top 10 bids
                "asks": orderbook.asks[:10]   # Top 10 asks
            }
        
        return data
