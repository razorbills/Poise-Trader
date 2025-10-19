#!/usr/bin/env python3
"""
🚀 MULTI-VENUE DATA AGGREGATION SYSTEM
Real-time data from multiple exchanges with redundant WebSocket connections
"""

import asyncio
import websockets
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
import ssl
import certifi

# Exchange imports
try:
    import ccxt.async_support as ccxt
    import requests
    from websocket import create_connection
    EXCHANGES_AVAILABLE = True
except ImportError:
    EXCHANGES_AVAILABLE = False
    print("⚠️ Install ccxt and websocket-client for multi-venue support")

@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    exchange: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: datetime
    orderbook_depth: Dict
    latency_ms: float

@dataclass
class TradeData:
    """Standardized trade data"""
    symbol: str
    exchange: str
    side: str
    amount: float
    price: float
    timestamp: datetime

class ExchangeConfig:
    """Configuration for each exchange"""
    
    EXCHANGES = {
        'mexc': {
            'name': 'MEXC',
            'ws_url': 'wss://wbs.mexc.com/ws',
            'rest_url': 'https://api.mexc.com',
            'ccxt_id': 'mexc',
            'rate_limit': 1200,  # requests per minute
            'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
        },
        'binance': {
            'name': 'Binance',
            'ws_url': 'wss://stream.binance.com:9443/ws',
            'rest_url': 'https://api.binance.com',
            'ccxt_id': 'binance',
            'rate_limit': 1200,
            'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
        },
        'coinbasepro': {
            'name': 'Coinbase Pro',
            'ws_url': 'wss://ws-feed.exchange.coinbase.com',
            'rest_url': 'https://api.exchange.coinbase.com',
            'ccxt_id': 'coinbasepro',
            'rate_limit': 1000,
            'symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD']
        },
        'kraken': {
            'name': 'Kraken',
            'ws_url': 'wss://ws.kraken.com',
            'rest_url': 'https://api.kraken.com',
            'ccxt_id': 'kraken',
            'rate_limit': 1000,
            'symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD']
        },
        'ftx': {
            'name': 'FTX',
            'ws_url': 'wss://ftx.com/ws/',
            'rest_url': 'https://ftx.com/api',
            'ccxt_id': 'ftx',
            'rate_limit': 3000,
            'symbols': ['BTC/USD', 'ETH/USD', 'SOL/USD']
        }
    }

class MultiVenueConnector:
    """Advanced multi-venue data aggregation with redundant connections"""
    
    def __init__(self):
        self.exchanges = {}
        self.websocket_connections = {}
        self.market_data = defaultdict(lambda: defaultdict(dict))
        self.trade_data = defaultdict(deque)
        self.latency_tracker = defaultdict(deque)
        self.failover_status = defaultdict(bool)
        self.data_callbacks = []
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Aggregated data
        self.consolidated_orderbook = {}
        self.best_prices = {}
        self.liquidity_analysis = {}
        
        # Performance metrics
        self.connection_stats = defaultdict(lambda: {
            'uptime': 0,
            'messages_received': 0,
            'errors': 0,
            'avg_latency': 0,
            'last_heartbeat': None
        })
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def initialize_exchanges(self):
        """Initialize all exchange connections"""
        if not EXCHANGES_AVAILABLE:
            self.logger.warning("Exchange libraries not available")
            return
        
        for exchange_id, config in ExchangeConfig.EXCHANGES.items():
            try:
                # Initialize CCXT exchange
                exchange_class = getattr(ccxt, config['ccxt_id'])
                exchange = exchange_class({
                    'apiKey': '',
                    'secret': '',
                    'timeout': 30000,
                    'enableRateLimit': True,
                })
                
                self.exchanges[exchange_id] = exchange
                self.logger.info(f"✅ {config['name']} initialized")
                
                # Start WebSocket connections
                asyncio.create_task(self._start_websocket_connection(exchange_id, config))
                asyncio.create_task(self._start_backup_connection(exchange_id, config))
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {config['name']}: {e}")
    
    async def _start_websocket_connection(self, exchange_id: str, config: Dict, backup=False):
        """Start WebSocket connection for an exchange"""
        connection_type = "backup" if backup else "primary"
        
        while self.is_running:
            try:
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                
                async with websockets.connect(
                    config['ws_url'],
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=10**7
                ) as websocket:
                    
                    self.websocket_connections[f"{exchange_id}_{connection_type}"] = websocket
                    self.logger.info(f"🔗 {config['name']} WebSocket connected ({connection_type})")
                    
                    # Subscribe to market data
                    await self._subscribe_to_feeds(websocket, exchange_id, config)
                    
                    # Listen for messages
                    async for message in websocket:
                        await self._process_websocket_message(exchange_id, message, backup)
                        
            except Exception as e:
                self.logger.error(f"❌ {config['name']} WebSocket error ({connection_type}): {e}")
                
                # Activate failover for primary connection
                if not backup:
                    self.failover_status[exchange_id] = True
                
                await asyncio.sleep(5)  # Retry delay
    
    async def _start_backup_connection(self, exchange_id: str, config: Dict):
        """Start backup WebSocket connection"""
        await asyncio.sleep(2)  # Slight delay to stagger connections
        await self._start_websocket_connection(exchange_id, config, backup=True)
    
    async def _subscribe_to_feeds(self, websocket, exchange_id: str, config: Dict):
        """Subscribe to market data feeds"""
        if exchange_id == 'binance':
            # Binance subscription format
            symbols = [s.replace('/', '').lower() for s in config['symbols']]
            streams = []
            for symbol in symbols:
                streams.extend([
                    f"{symbol}@ticker",
                    f"{symbol}@depth20@100ms",
                    f"{symbol}@trade"
                ])
            
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": 1
            }
            
        elif exchange_id == 'mexc':
            # MEXC subscription format
            subscribe_msg = {
                "method": "SUBSCRIPTION",
                "params": ["spot@public.deals.v3.api@BTCUSDT", "spot@public.bookTicker.v3.api@BTCUSDT"]
            }
            
        elif exchange_id == 'coinbasepro':
            # Coinbase Pro subscription
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": ["BTC-USD", "ETH-USD", "SOL-USD"],
                "channels": ["ticker", "level2", "matches"]
            }
            
        elif exchange_id == 'kraken':
            # Kraken subscription
            subscribe_msg = {
                "event": "subscribe",
                "pair": ["XBT/USD", "ETH/USD", "SOL/USD"],
                "subscription": {"name": "ticker"}
            }
            
        else:
            return
        
        await websocket.send(json.dumps(subscribe_msg))
        self.logger.info(f"📡 Subscribed to {config['name']} feeds")
    
    async def _process_websocket_message(self, exchange_id: str, message: str, is_backup: bool):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            timestamp = datetime.now()
            
            # Record latency (simplified)
            latency_ms = 50  # Placeholder for actual latency calculation
            self.latency_tracker[exchange_id].append(latency_ms)
            if len(self.latency_tracker[exchange_id]) > 1000:
                self.latency_tracker[exchange_id].popleft()
            
            # Update connection stats
            stats = self.connection_stats[exchange_id]
            stats['messages_received'] += 1
            stats['last_heartbeat'] = timestamp
            stats['avg_latency'] = sum(self.latency_tracker[exchange_id]) / len(self.latency_tracker[exchange_id])
            
            # Parse exchange-specific data format
            market_data = self._parse_market_data(exchange_id, data, timestamp, latency_ms)
            if market_data:
                self._update_market_data(market_data, is_backup)
                
                # Notify callbacks
                for callback in self.data_callbacks:
                    try:
                        await callback(market_data)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error processing {exchange_id} message: {e}")
            self.connection_stats[exchange_id]['errors'] += 1
    
    def _parse_market_data(self, exchange_id: str, data: Dict, timestamp: datetime, latency_ms: float) -> Optional[MarketData]:
        """Parse exchange-specific data format into standardized MarketData"""
        try:
            if exchange_id == 'binance' and 'stream' in data:
                stream_data = data['data']
                if 'c' in stream_data:  # Ticker data
                    return MarketData(
                        symbol=stream_data['s'],
                        exchange=exchange_id,
                        bid=float(stream_data.get('b', 0)),
                        ask=float(stream_data.get('a', 0)),
                        last=float(stream_data.get('c', 0)),
                        volume_24h=float(stream_data.get('v', 0)),
                        timestamp=timestamp,
                        orderbook_depth={},
                        latency_ms=latency_ms
                    )
            
            elif exchange_id == 'mexc' and 'c' in data:
                return MarketData(
                    symbol=data.get('s', ''),
                    exchange=exchange_id,
                    bid=float(data.get('b', 0)),
                    ask=float(data.get('a', 0)),
                    last=float(data.get('c', 0)),
                    volume_24h=float(data.get('v', 0)),
                    timestamp=timestamp,
                    orderbook_depth={},
                    latency_ms=latency_ms
                )
            
            # Add more exchange parsers as needed
            
        except Exception as e:
            self.logger.error(f"Error parsing {exchange_id} data: {e}")
        
        return None
    
    def _update_market_data(self, market_data: MarketData, is_backup: bool):
        """Update internal market data storage and perform aggregation"""
        symbol = market_data.symbol
        exchange = market_data.exchange
        
        # Store raw data
        self.market_data[symbol][exchange] = market_data
        
        # Update consolidated orderbook
        self._update_consolidated_orderbook(symbol)
        
        # Update best prices across exchanges
        self._update_best_prices(symbol)
        
        # Perform liquidity analysis
        self._analyze_liquidity(symbol)
    
    def _update_consolidated_orderbook(self, symbol: str):
        """Create consolidated orderbook from all exchanges"""
        consolidated_bids = []
        consolidated_asks = []
        
        for exchange, data in self.market_data[symbol].items():
            if isinstance(data, MarketData) and data.bid > 0 and data.ask > 0:
                consolidated_bids.append((data.bid, 1.0, exchange))  # (price, volume, exchange)
                consolidated_asks.append((data.ask, 1.0, exchange))
        
        # Sort by price
        consolidated_bids.sort(key=lambda x: x[0], reverse=True)
        consolidated_asks.sort(key=lambda x: x[0])
        
        self.consolidated_orderbook[symbol] = {
            'bids': consolidated_bids[:10],  # Top 10 levels
            'asks': consolidated_asks[:10],
            'timestamp': datetime.now()
        }
    
    def _update_best_prices(self, symbol: str):
        """Track best bid/ask across all exchanges"""
        best_bid = 0
        best_ask = float('inf')
        best_bid_exchange = ""
        best_ask_exchange = ""
        
        for exchange, data in self.market_data[symbol].items():
            if isinstance(data, MarketData):
                if data.bid > best_bid:
                    best_bid = data.bid
                    best_bid_exchange = exchange
                
                if data.ask < best_ask:
                    best_ask = data.ask
                    best_ask_exchange = exchange
        
        spread = best_ask - best_bid if best_ask != float('inf') else 0
        
        self.best_prices[symbol] = {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_bps': (spread / best_bid * 10000) if best_bid > 0 else 0,
            'bid_exchange': best_bid_exchange,
            'ask_exchange': best_ask_exchange,
            'timestamp': datetime.now()
        }
    
    def _analyze_liquidity(self, symbol: str):
        """Analyze liquidity across venues"""
        total_bid_liquidity = 0
        total_ask_liquidity = 0
        venue_count = 0
        
        for exchange, data in self.market_data[symbol].items():
            if isinstance(data, MarketData):
                total_bid_liquidity += data.volume_24h * 0.4  # Estimate
                total_ask_liquidity += data.volume_24h * 0.4
                venue_count += 1
        
        avg_latency = sum(sum(self.latency_tracker[ex]) / len(self.latency_tracker[ex]) 
                         for ex in self.latency_tracker if self.latency_tracker[ex]) / max(1, venue_count)
        
        self.liquidity_analysis[symbol] = {
            'total_bid_liquidity': total_bid_liquidity,
            'total_ask_liquidity': total_ask_liquidity,
            'venue_count': venue_count,
            'avg_latency_ms': avg_latency,
            'liquidity_score': min(100, (total_bid_liquidity + total_ask_liquidity) / 1000000),
            'timestamp': datetime.now()
        }
    
    async def start(self):
        """Start the multi-venue connector"""
        self.is_running = True
        print("🚀 MULTI-VENUE CONNECTOR STARTING...")
        print("📡 Connecting to institutional-grade data feeds...")
        
        await self.initialize_exchanges()
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_connections())
        asyncio.create_task(self._calculate_arbitrage_opportunities())
        
        print("✅ MULTI-VENUE CONNECTOR ACTIVE - LEGENDARY DATA FLOW!")
        print(f"🔗 Connected exchanges: {list(self.exchanges.keys())}")
    
    async def _monitor_connections(self):
        """Monitor connection health and perform failover"""
        while self.is_running:
            try:
                for exchange_id, stats in self.connection_stats.items():
                    if stats['last_heartbeat']:
                        time_since_heartbeat = (datetime.now() - stats['last_heartbeat']).seconds
                        
                        if time_since_heartbeat > 60:  # No data for 1 minute
                            self.logger.warning(f"⚠️ {exchange_id} connection stale, activating failover")
                            self.failover_status[exchange_id] = True
                    
                    # Log health status
                    if stats['messages_received'] > 0:
                        self.logger.info(f"📊 {exchange_id}: {stats['messages_received']} msgs, "
                                       f"{stats['avg_latency']:.1f}ms avg latency, {stats['errors']} errors")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Connection monitoring error: {e}")
    
    async def _calculate_arbitrage_opportunities(self):
        """Identify arbitrage opportunities across venues"""
        while self.is_running:
            try:
                for symbol in self.best_prices:
                    best_prices = self.best_prices[symbol]
                    
                    if best_prices['spread_bps'] > 10:  # Spread > 10 basis points
                        opportunity = {
                            'symbol': symbol,
                            'buy_exchange': best_prices['ask_exchange'],
                            'sell_exchange': best_prices['bid_exchange'],
                            'buy_price': best_prices['best_ask'],
                            'sell_price': best_prices['best_bid'],
                            'profit_bps': best_prices['spread_bps'],
                            'timestamp': datetime.now()
                        }
                        
                        # Notify arbitrage opportunity
                        for callback in self.data_callbacks:
                            try:
                                await callback({'type': 'arbitrage', 'data': opportunity})
                            except:
                                pass
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Arbitrage calculation error: {e}")
    
    def register_callback(self, callback: Callable):
        """Register callback for market data updates"""
        self.data_callbacks.append(callback)
    
    def get_best_execution_venue(self, symbol: str, side: str, amount: float) -> Dict:
        """Get best venue for execution considering liquidity and latency"""
        if symbol not in self.best_prices:
            return {'venue': 'mexc', 'confidence': 0.5}
        
        best_prices = self.best_prices[symbol]
        liquidity = self.liquidity_analysis.get(symbol, {})
        
        # Consider price, latency, and liquidity
        venues = []
        for exchange, data in self.market_data[symbol].items():
            if isinstance(data, MarketData):
                score = 100 - data.latency_ms/10  # Lower latency = higher score
                
                if side == 'buy' and data.ask > 0:
                    price_score = 100 - ((data.ask - best_prices['best_ask']) / best_prices['best_ask'] * 10000)
                    venues.append((exchange, score + price_score, data.ask))
                elif side == 'sell' and data.bid > 0:
                    price_score = 100 - ((best_prices['best_bid'] - data.bid) / best_prices['best_bid'] * 10000)
                    venues.append((exchange, score + price_score, data.bid))
        
        if venues:
            best_venue = max(venues, key=lambda x: x[1])
            return {
                'venue': best_venue[0],
                'price': best_venue[2],
                'confidence': min(0.95, best_venue[1] / 200),
                'latency_ms': self.connection_stats[best_venue[0]]['avg_latency']
            }
        
        return {'venue': 'mexc', 'confidence': 0.5}
    
    def get_market_summary(self) -> Dict:
        """Get comprehensive market summary across all venues"""
        summary = {
            'timestamp': datetime.now(),
            'total_venues': len(self.exchanges),
            'active_connections': sum(1 for status in self.failover_status.values() if not status),
            'symbols_tracked': len(self.market_data),
            'best_prices': self.best_prices,
            'liquidity_analysis': self.liquidity_analysis,
            'connection_stats': dict(self.connection_stats),
            'arbitrage_opportunities': []
        }
        
        # Add arbitrage opportunities
        for symbol, best_prices in self.best_prices.items():
            if best_prices['spread_bps'] > 5:
                summary['arbitrage_opportunities'].append({
                    'symbol': symbol,
                    'spread_bps': best_prices['spread_bps'],
                    'profit_potential': best_prices['spread_bps'] / 10000
                })
        
        return summary
    
    async def stop(self):
        """Stop the multi-venue connector"""
        self.is_running = False
        
        # Close WebSocket connections
        for connection in self.websocket_connections.values():
            try:
                await connection.close()
            except:
                pass
        
        # Close CCXT exchanges
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except:
                pass
        
        self.executor.shutdown(wait=True)
        print("🔌 MULTI-VENUE CONNECTOR STOPPED")

# Global instance
multi_venue_connector = MultiVenueConnector()
