"""
Base Feed System for Poise Trader

Provides unified interface for all market data feeds with:
- WebSocket and REST API support
- Connection management and reconnection logic
- Rate limiting and throttling
- Data normalization and validation
"""

import asyncio
import aiohttp
import websockets
import logging
import time
import json
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable
from decimal import Decimal
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import hmac
import base64
from urllib.parse import urlencode

from ..framework.base_classes import BaseDataFeed, MarketData
from ..framework.event_system import EventBus, Event, EventPriority, create_event


class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class FeedConfig:
    """Configuration for market data feeds"""
    exchange_name: str
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    base_url: str = ""
    websocket_url: str = ""
    rate_limit: int = 1000  # requests per minute
    timeout: int = 30
    max_retries: int = 5
    retry_delay: float = 1.0
    heartbeat_interval: int = 30
    sandbox: bool = True


class UnifiedFeed(BaseDataFeed):
    """
    Unified market data feed that can aggregate multiple exchange feeds
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus = None):
        super().__init__(config)
        self.event_bus = event_bus
        self.feeds: Dict[str, BaseDataFeed] = {}
        self.active_subscriptions: Dict[str, set] = {}
        self.data_queue = asyncio.Queue()
        
        # Performance metrics
        self.metrics = {
            'messages_received': 0,
            'messages_per_second': 0,
            'last_message_time': 0,
            'connection_uptime': 0,
            'reconnections': 0
        }
        
    async def add_feed(self, feed: BaseDataFeed) -> None:
        """Add an exchange feed to the unified feed"""
        exchange_name = feed.config.get('exchange_name', 'unknown')
        self.feeds[exchange_name] = feed
        self.active_subscriptions[exchange_name] = set()
        
        self.logger.info("Added feed for exchange: %s", exchange_name)
    
    async def connect(self) -> bool:
        """Connect all registered feeds"""
        success = True
        
        for exchange_name, feed in self.feeds.items():
            try:
                if await feed.connect():
                    self.logger.info("Connected to %s", exchange_name)
                else:
                    self.logger.error("Failed to connect to %s", exchange_name)
                    success = False
            except Exception as e:
                self.logger.error("Error connecting to %s: %s", exchange_name, e)
                success = False
        
        if success:
            self.is_running = True
            # Start data aggregation task
            asyncio.create_task(self._aggregate_data())
        
        return success
    
    async def disconnect(self) -> None:
        """Disconnect all feeds"""
        self.is_running = False
        
        for exchange_name, feed in self.feeds.items():
            try:
                await feed.disconnect()
                self.logger.info("Disconnected from %s", exchange_name)
            except Exception as e:
                self.logger.error("Error disconnecting from %s: %s", exchange_name, e)
    
    async def subscribe(self, symbols: List[str], exchanges: List[str] = None) -> None:
        """
        Subscribe to symbols across specified exchanges
        
        Args:
            symbols: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
            exchanges: List of exchange names (if None, subscribe to all)
        """
        target_exchanges = exchanges or list(self.feeds.keys())
        
        for exchange_name in target_exchanges:
            if exchange_name in self.feeds:
                try:
                    feed = self.feeds[exchange_name]
                    await feed.subscribe(symbols)
                    self.active_subscriptions[exchange_name].update(symbols)
                    
                    self.logger.info("Subscribed to %s on %s", symbols, exchange_name)
                except Exception as e:
                    self.logger.error("Failed to subscribe to %s on %s: %s", 
                                    symbols, exchange_name, e)
    
    async def unsubscribe(self, symbols: List[str], exchanges: List[str] = None) -> None:
        """Unsubscribe from symbols"""
        target_exchanges = exchanges or list(self.feeds.keys())
        
        for exchange_name in target_exchanges:
            if exchange_name in self.feeds:
                try:
                    feed = self.feeds[exchange_name]
                    await feed.unsubscribe(symbols)
                    self.active_subscriptions[exchange_name].difference_update(symbols)
                    
                    self.logger.info("Unsubscribed from %s on %s", symbols, exchange_name)
                except Exception as e:
                    self.logger.error("Failed to unsubscribe from %s on %s: %s", 
                                    symbols, exchange_name, e)
    
    async def get_market_data(self) -> AsyncGenerator[MarketData, None]:
        """Get unified market data stream"""
        while self.is_running:
            try:
                # Wait for data with timeout
                data = await asyncio.wait_for(self.data_queue.get(), timeout=1.0)
                
                # Update metrics
                self.metrics['messages_received'] += 1
                self.metrics['last_message_time'] = time.time()
                
                # Publish to event bus if available
                if self.event_bus:
                    event = create_event("market.data", 
                                       symbol=data.symbol,
                                       exchange=data.exchange,
                                       price=float(data.price),
                                       timestamp=data.timestamp)
                    await self.event_bus.publish(event)
                
                yield data
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error("Error in market data stream: %s", e)
    
    async def _aggregate_data(self):
        """Aggregate data from all feeds into unified stream"""
        tasks = []
        
        # Create tasks for each feed
        for exchange_name, feed in self.feeds.items():
            task = asyncio.create_task(
                self._process_feed_data(exchange_name, feed)
            )
            tasks.append(task)
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error("Error in data aggregation: %s", e)
    
    async def _process_feed_data(self, exchange_name: str, feed: BaseDataFeed):
        """Process data from a specific feed"""
        try:
            async for data in feed.get_market_data():
                # Ensure exchange is set
                data.exchange = exchange_name
                await self.data_queue.put(data)
        except Exception as e:
            self.logger.error("Error processing data from %s: %s", exchange_name, e)
    
    async def get_historical_data(self, symbol: str, start_time: int, 
                                 end_time: int, exchange: str = None) -> List[MarketData]:
        """Get historical data from specified exchange"""
        if exchange and exchange in self.feeds:
            return await self.feeds[exchange].get_historical_data(
                symbol, start_time, end_time
            )
        
        # If no exchange specified, try all feeds and return first successful result
        for exchange_name, feed in self.feeds.items():
            try:
                return await feed.get_historical_data(symbol, start_time, end_time)
            except Exception as e:
                self.logger.debug("Failed to get historical data from %s: %s", 
                                exchange_name, e)
        
        return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feed performance metrics"""
        current_time = time.time()
        
        # Calculate messages per second
        if self.metrics['last_message_time'] > 0:
            time_diff = current_time - (self.metrics['last_message_time'] - 60)
            if time_diff > 0:
                self.metrics['messages_per_second'] = (
                    self.metrics['messages_received'] / time_diff
                )
        
        return dict(self.metrics)
    
    def get_subscriptions(self) -> Dict[str, List[str]]:
        """Get current subscriptions per exchange"""
        return {
            exchange: list(symbols) 
            for exchange, symbols in self.active_subscriptions.items()
        }


class WebSocketFeed(BaseDataFeed):
    """
    Base class for WebSocket-based feeds
    """
    
    def __init__(self, config: FeedConfig, event_bus: EventBus = None):
        super().__init__(asdict(config))
        self.feed_config = config
        self.event_bus = event_bus
        
        # Connection management
        self.websocket = None
        self.status = ConnectionStatus.DISCONNECTED
        self.reconnect_count = 0
        self.last_heartbeat = 0
        
        # Rate limiting
        self.rate_limiter = RateLimiter(config.rate_limit)
        
        # Message queue
        self.message_queue = asyncio.Queue()
        
        # Subscription management
        self.subscribed_symbols: set = set()
        
    async def connect(self) -> bool:
        """Connect to WebSocket"""
        if self.status == ConnectionStatus.CONNECTED:
            return True
            
        self.status = ConnectionStatus.CONNECTING
        
        try:
            self.websocket = await websockets.connect(
                self.feed_config.websocket_url,
                timeout=self.feed_config.timeout
            )
            
            self.status = ConnectionStatus.CONNECTED
            self.is_running = True
            self.reconnect_count = 0
            
            # Start message handling
            asyncio.create_task(self._handle_messages())
            asyncio.create_task(self._heartbeat_task())
            
            self.logger.info("Connected to %s WebSocket", self.feed_config.exchange_name)
            return True
            
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.logger.error("Failed to connect to WebSocket: %s", e)
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket"""
        self.is_running = False
        self.status = ConnectionStatus.DISCONNECTED
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.logger.info("Disconnected from %s WebSocket", self.feed_config.exchange_name)
    
    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        while self.is_running and self.websocket:
            try:
                message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=self.feed_config.timeout
                )
                
                # Parse and process message
                await self._process_message(message)
                
                self.last_heartbeat = time.time()
                
            except asyncio.TimeoutError:
                self.logger.warning("WebSocket receive timeout")
                await self._reconnect()
                break
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket connection closed")
                await self._reconnect()
                break
            except Exception as e:
                self.logger.error("Error handling WebSocket message: %s", e)
    
    async def _process_message(self, message: str):
        """Process raw WebSocket message - to be implemented by subclasses"""
        try:
            data = json.loads(message)
            market_data = await self._parse_market_data(data)
            
            if market_data:
                await self.message_queue.put(market_data)
                
        except Exception as e:
            self.logger.error("Error processing message: %s", e)
    
    async def _parse_market_data(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Parse exchange-specific data into MarketData - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _parse_market_data")
    
    async def _heartbeat_task(self):
        """Send periodic heartbeat/ping messages"""
        while self.is_running:
            try:
                if self.websocket and self.status == ConnectionStatus.CONNECTED:
                    await self._send_heartbeat()
                
                await asyncio.sleep(self.feed_config.heartbeat_interval)
                
            except Exception as e:
                self.logger.error("Error in heartbeat task: %s", e)
    
    async def _send_heartbeat(self):
        """Send heartbeat message - to be implemented by subclasses"""
        # Default implementation - just send ping
        if self.websocket:
            await self.websocket.ping()
    
    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_count >= self.feed_config.max_retries:
            self.logger.error("Max reconnection attempts reached")
            self.status = ConnectionStatus.ERROR
            return
        
        self.status = ConnectionStatus.RECONNECTING
        self.reconnect_count += 1
        
        delay = min(
            self.feed_config.retry_delay * (2 ** (self.reconnect_count - 1)),
            60  # Max 60 seconds
        )
        
        self.logger.info("Reconnecting in %s seconds (attempt %d)", 
                        delay, self.reconnect_count)
        
        await asyncio.sleep(delay)
        
        # Close old connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        # Attempt reconnection
        if await self.connect():
            # Resubscribe to symbols
            if self.subscribed_symbols:
                await self.subscribe(list(self.subscribed_symbols))
    
    async def get_market_data(self) -> AsyncGenerator[MarketData, None]:
        """Get market data stream"""
        while self.is_running:
            try:
                data = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                yield data
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error("Error in market data stream: %s", e)


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        async with self.lock:
            now = time.time()
            
            # Remove old requests (older than 1 minute)
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < 60]
            
            if len(self.requests) >= self.requests_per_minute:
                # Need to wait
                oldest_request = min(self.requests)
                wait_time = 60 - (now - oldest_request)
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            # Record this request
            self.requests.append(now)
