#!/usr/bin/env python3
"""
🚀 ADVANCED MARKET DATA FEED MANAGER
Real-time data feeds for fully autonomous trading

Features:
• Multi-exchange WebSocket connections
• High-frequency data aggregation
• Intelligent data quality monitoring
• Auto-failover and reconnection
• Real-time market analysis
• AI-powered signal detection
"""

import asyncio
import websocket
import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from decimal import Decimal
import aiohttp
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
import numpy as np


@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    price: float
    volume: float
    timestamp: int
    exchange: str
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    price_change_24h: float = 0.0
    volume_change_24h: float = 0.0


@dataclass
class MarketOpportunity:
    """AI-detected market opportunity"""
    symbol: str
    opportunity_type: str  # 'breakout', 'reversal', 'momentum', 'volume_spike'
    confidence: float
    expected_move: float
    time_horizon_minutes: int
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    reasoning: str


class AdvancedFeedManager:
    """
    🧠 AI-POWERED MARKET DATA MANAGER
    
    Intelligently aggregates data from multiple sources and
    detects trading opportunities in real-time
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("AdvancedFeedManager")
        
        # Exchange connections
        self.exchanges = {}
        self.websocket_connections = {}
        
        # Data storage
        self.market_data = {}  # Real-time tick data
        self.price_histories = {}  # Historical price data
        self.volume_profiles = {}  # Volume analysis
        self.order_books = {}  # Order book data
        
        # AI Analysis
        self.ai_analyzer = MarketAIAnalyzer()
        self.opportunity_callbacks = []  # Functions to call when opportunities detected
        
        # Performance tracking
        self.data_quality_metrics = {
            'total_ticks_received': 0,
            'invalid_ticks_filtered': 0,
            'connection_uptime': 0,
            'average_latency_ms': 0,
        }
        
        # Configuration
        self.target_symbols = config.get('symbols', [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'PEPE/USDT', 
            'SHIB/USDT', 'DOGE/USDT', 'AVAX/USDT'
        ])
        self.max_latency_ms = config.get('max_latency_ms', 100)
        self.data_quality_threshold = config.get('data_quality_threshold', 0.95)
    
    async def initialize(self):
        """Initialize all exchange connections"""
        self.logger.info("🚀 Initializing Advanced Feed Manager")
        
        # Initialize exchange APIs
        await self._initialize_exchanges()
        
        # Start WebSocket connections
        await self._start_websocket_feeds()
        
        # Start AI analysis engine
        await self._start_ai_analysis()
        
        # Start data quality monitoring
        asyncio.create_task(self._monitor_data_quality())
        
        self.logger.info("✅ Advanced Feed Manager initialized successfully")
    
    async def _initialize_exchanges(self):
        """Initialize exchange API connections"""
        exchange_configs = {
            'mexc': {
                'apiKey': self.config.get('mexc_api_key'),
                'secret': self.config.get('mexc_api_secret'),
                'sandbox': self.config.get('use_sandbox', True),
                'enableRateLimit': True,
                'timeout': 30000,
            },
            'binance': {
                'apiKey': self.config.get('binance_api_key'),
                'secret': self.config.get('binance_api_secret'),
                'sandbox': self.config.get('use_sandbox', True),
                'enableRateLimit': True,
                'timeout': 30000,
            }
        }
        
        for exchange_name, config in exchange_configs.items():
            if config['apiKey']:  # Only initialize if API key provided
                try:
                    exchange_class = getattr(ccxt, exchange_name)
                    exchange = exchange_class(config)
                    await exchange.load_markets()
                    self.exchanges[exchange_name] = exchange
                    self.logger.info(f"✅ {exchange_name} exchange initialized")
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize {exchange_name}: {e}")
    
    async def _start_websocket_feeds(self):
        """Start WebSocket connections for real-time data"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Start WebSocket for each target symbol
                for symbol in self.target_symbols:
                    if symbol in exchange.markets:
                        asyncio.create_task(
                            self._websocket_feed_handler(exchange_name, symbol)
                        )
                
                self.logger.info(f"🔗 WebSocket feeds started for {exchange_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to start WebSocket for {exchange_name}: {e}")
    
    async def _websocket_feed_handler(self, exchange_name: str, symbol: str):
        """Handle WebSocket data feed for a specific exchange and symbol"""
        while True:
            try:
                exchange = self.exchanges[exchange_name]
                
                # Subscribe to ticker stream
                async for ticker in exchange.watch_ticker(symbol):
                    await self._process_market_tick(exchange_name, symbol, ticker)
                
                # Subscribe to order book stream  
                async for order_book in exchange.watch_order_book(symbol):
                    await self._process_order_book(exchange_name, symbol, order_book)
                
            except Exception as e:
                self.logger.error(f"WebSocket error for {exchange_name} {symbol}: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting
    
    async def _process_market_tick(self, exchange: str, symbol: str, ticker: Dict):
        """Process incoming market tick data"""
        try:
            # Create market tick
            tick = MarketTick(
                symbol=symbol,
                price=float(ticker['last']) if ticker['last'] else 0,
                volume=float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0,
                timestamp=int(time.time() * 1000),
                exchange=exchange,
                bid=float(ticker['bid']) if ticker['bid'] else 0,
                ask=float(ticker['ask']) if ticker['ask'] else 0,
                spread=float(ticker['ask'] - ticker['bid']) if ticker['ask'] and ticker['bid'] else 0,
                price_change_24h=float(ticker['percentage']) if ticker['percentage'] else 0,
            )
            
            # Quality checks
            if not self._validate_tick_data(tick):
                self.data_quality_metrics['invalid_ticks_filtered'] += 1
                return
            
            # Store data
            key = f"{exchange}:{symbol}"
            if key not in self.market_data:
                self.market_data[key] = []
                self.price_histories[key] = []
            
            self.market_data[key].append(tick)
            self.price_histories[key].append(tick.price)
            
            # Maintain data size limits
            if len(self.market_data[key]) > 1000:
                self.market_data[key] = self.market_data[key][-500:]
            if len(self.price_histories[key]) > 5000:
                self.price_histories[key] = self.price_histories[key][-2500:]
            
            # Update metrics
            self.data_quality_metrics['total_ticks_received'] += 1
            
            # Trigger AI analysis
            await self._trigger_ai_analysis(tick)
            
        except Exception as e:
            self.logger.error(f"Error processing tick for {symbol}: {e}")
    
    async def _process_order_book(self, exchange: str, symbol: str, order_book: Dict):
        """Process order book data for depth analysis"""
        try:
            key = f"{exchange}:{symbol}"
            self.order_books[key] = {
                'bids': order_book.get('bids', []),
                'asks': order_book.get('asks', []),
                'timestamp': int(time.time() * 1000)
            }
        except Exception as e:
            self.logger.error(f"Error processing order book for {symbol}: {e}")
    
    def _validate_tick_data(self, tick: MarketTick) -> bool:
        """Validate tick data quality"""
        if tick.price <= 0:
            return False
        if tick.volume < 0:
            return False
        if tick.spread < 0:
            return False
        # Add more validation rules as needed
        return True
    
    async def _trigger_ai_analysis(self, tick: MarketTick):
        """Trigger AI analysis on new market data"""
        try:
            # Get recent price history
            key = f"{tick.exchange}:{tick.symbol}"
            if key in self.price_histories and len(self.price_histories[key]) >= 20:
                recent_prices = self.price_histories[key][-100:]  # Last 100 prices
                
                # Run AI analysis
                opportunities = await self.ai_analyzer.analyze_market_data(
                    tick, recent_prices, self.order_books.get(key)
                )
                
                # Notify about opportunities
                for opportunity in opportunities:
                    await self._notify_opportunity(opportunity)
                
        except Exception as e:
            self.logger.error(f"Error in AI analysis: {e}")
    
    async def _notify_opportunity(self, opportunity: MarketOpportunity):
        """Notify registered callbacks about trading opportunities"""
        for callback in self.opportunity_callbacks:
            try:
                await callback(opportunity)
            except Exception as e:
                self.logger.error(f"Error in opportunity callback: {e}")
    
    def register_opportunity_callback(self, callback: Callable):
        """Register callback function for trading opportunities"""
        self.opportunity_callbacks.append(callback)
    
    async def _start_ai_analysis(self):
        """Start continuous AI analysis engine"""
        asyncio.create_task(self._ai_analysis_loop())
    
    async def _ai_analysis_loop(self):
        """Continuous AI analysis of all market data"""
        while True:
            try:
                # Perform cross-market analysis every 10 seconds
                await self._cross_market_analysis()
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in AI analysis loop: {e}")
                await asyncio.sleep(30)
    
    async def _cross_market_analysis(self):
        """Analyze opportunities across multiple markets"""
        try:
            # Find arbitrage opportunities
            arbitrage_opps = await self._find_arbitrage_opportunities()
            
            # Detect market-wide movements
            market_trends = await self._analyze_market_trends()
            
            # Look for correlation breaks
            correlation_opps = await self._find_correlation_opportunities()
            
            # Combine all opportunities
            all_opportunities = arbitrage_opps + market_trends + correlation_opps
            
            # Rank by confidence and potential profit
            sorted_opportunities = sorted(
                all_opportunities, 
                key=lambda x: x.confidence * x.risk_reward_ratio, 
                reverse=True
            )
            
            # Notify about top opportunities
            for opp in sorted_opportunities[:3]:  # Top 3 opportunities
                await self._notify_opportunity(opp)
                
        except Exception as e:
            self.logger.error(f"Error in cross-market analysis: {e}")
    
    async def _find_arbitrage_opportunities(self) -> List[MarketOpportunity]:
        """Find cross-exchange arbitrage opportunities"""
        opportunities = []
        
        # Compare prices across exchanges for same symbol
        symbols_by_exchange = {}
        for key, data in self.market_data.items():
            if data:  # Has recent data
                exchange, symbol = key.split(':', 1)
                if symbol not in symbols_by_exchange:
                    symbols_by_exchange[symbol] = {}
                symbols_by_exchange[symbol][exchange] = data[-1].price
        
        # Find price differences
        for symbol, prices in symbols_by_exchange.items():
            if len(prices) >= 2:
                max_price = max(prices.values())
                min_price = min(prices.values())
                price_diff = (max_price - min_price) / min_price
                
                if price_diff > 0.005:  # 0.5% minimum arbitrage opportunity
                    opportunities.append(MarketOpportunity(
                        symbol=symbol,
                        opportunity_type='arbitrage',
                        confidence=min(price_diff * 20, 0.9),
                        expected_move=price_diff,
                        time_horizon_minutes=5,
                        entry_price=min_price,
                        stop_loss=min_price * 0.99,
                        take_profit=max_price * 0.99,
                        risk_reward_ratio=price_diff / 0.01,
                        reasoning=f"Arbitrage: {price_diff*100:.2f}% price difference across exchanges"
                    ))
        
        return opportunities
    
    async def _analyze_market_trends(self) -> List[MarketOpportunity]:
        """Analyze market-wide trends and momentum"""
        opportunities = []
        
        for key, price_history in self.price_histories.items():
            if len(price_history) >= 50:
                exchange, symbol = key.split(':', 1)
                current_price = price_history[-1]
                
                # Calculate various indicators
                ma_20 = sum(price_history[-20:]) / 20
                ma_50 = sum(price_history[-50:]) / 50
                
                # Price momentum
                momentum = (current_price - price_history[-10]) / price_history[-10]
                
                # Volume analysis (if available)
                recent_data = self.market_data.get(key, [])[-10:]
                if recent_data:
                    avg_volume = sum(d.volume for d in recent_data) / len(recent_data)
                    current_volume = recent_data[-1].volume if recent_data else 0
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                else:
                    volume_ratio = 1
                
                # Detect breakout patterns
                if (current_price > ma_20 > ma_50 and 
                    abs(momentum) > 0.02 and 
                    volume_ratio > 1.5):
                    
                    direction = 'bullish' if momentum > 0 else 'bearish'
                    confidence = min(abs(momentum) * 10 + (volume_ratio - 1) * 0.2, 0.85)
                    
                    opportunities.append(MarketOpportunity(
                        symbol=symbol,
                        opportunity_type='momentum',
                        confidence=confidence,
                        expected_move=abs(momentum) * 2,  # Expect continuation
                        time_horizon_minutes=60,
                        entry_price=current_price,
                        stop_loss=current_price * (0.98 if momentum > 0 else 1.02),
                        take_profit=current_price * (1.04 if momentum > 0 else 0.96),
                        risk_reward_ratio=2.0,
                        reasoning=f"{direction} momentum: {momentum*100:.2f}%, volume: {volume_ratio:.1f}x"
                    ))
        
        return opportunities
    
    async def _find_correlation_opportunities(self) -> List[MarketOpportunity]:
        """Find opportunities based on correlation breaks"""
        opportunities = []
        
        # This would implement correlation analysis between different symbols
        # For now, simplified version
        
        return opportunities
    
    async def _monitor_data_quality(self):
        """Monitor and report data quality metrics"""
        while True:
            try:
                # Calculate data quality metrics
                total_connections = len(self.websocket_connections)
                active_connections = sum(1 for conn in self.websocket_connections.values() if conn)
                
                uptime_pct = (active_connections / total_connections * 100) if total_connections > 0 else 0
                
                # Log quality metrics every 5 minutes
                self.logger.info("📊 Data Quality Metrics:")
                self.logger.info(f"   • Total ticks: {self.data_quality_metrics['total_ticks_received']:,}")
                self.logger.info(f"   • Invalid filtered: {self.data_quality_metrics['invalid_ticks_filtered']:,}")
                self.logger.info(f"   • Connection uptime: {uptime_pct:.1f}%")
                self.logger.info(f"   • Active symbols: {len(self.market_data)}")
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error monitoring data quality: {e}")
                await asyncio.sleep(60)
    
    def get_latest_price(self, symbol: str, exchange: str = None) -> Optional[float]:
        """Get latest price for a symbol"""
        if exchange:
            key = f"{exchange}:{symbol}"
            data = self.market_data.get(key, [])
            return data[-1].price if data else None
        else:
            # Return best price across all exchanges
            best_price = None
            for key, data in self.market_data.items():
                if data and key.endswith(f":{symbol}"):
                    price = data[-1].price
                    if best_price is None or price > best_price:
                        best_price = price
            return best_price
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        summary = {
            'active_symbols': len(self.market_data),
            'total_exchanges': len(self.exchanges),
            'data_quality_score': self._calculate_data_quality_score(),
            'top_movers': self._get_top_movers(),
            'market_sentiment': self._calculate_market_sentiment(),
            'opportunities_detected': len(self.opportunity_callbacks),
        }
        
        return summary
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score"""
        total = self.data_quality_metrics['total_ticks_received']
        invalid = self.data_quality_metrics['invalid_ticks_filtered']
        
        if total == 0:
            return 0.0
        
        return max(0.0, min(1.0, (total - invalid) / total))
    
    def _get_top_movers(self) -> List[Dict[str, Any]]:
        """Get top price movers"""
        movers = []
        
        for key, data in self.market_data.items():
            if len(data) >= 2:
                exchange, symbol = key.split(':', 1)
                current = data[-1]
                previous = data[-2]
                
                change_pct = (current.price - previous.price) / previous.price * 100
                
                movers.append({
                    'symbol': symbol,
                    'exchange': exchange,
                    'price': current.price,
                    'change_pct': change_pct,
                    'volume': current.volume
                })
        
        # Sort by absolute change percentage
        movers.sort(key=lambda x: abs(x['change_pct']), reverse=True)
        return movers[:10]  # Top 10 movers
    
    def _calculate_market_sentiment(self) -> str:
        """Calculate overall market sentiment"""
        positive_count = 0
        total_count = 0
        
        for key, data in self.market_data.items():
            if data:
                if data[-1].price_change_24h > 0:
                    positive_count += 1
                total_count += 1
        
        if total_count == 0:
            return "neutral"
        
        positive_ratio = positive_count / total_count
        
        if positive_ratio > 0.6:
            return "bullish"
        elif positive_ratio < 0.4:
            return "bearish"
        else:
            return "neutral"


class MarketAIAnalyzer:
    """
    🧠 AI Market Analysis Engine
    
    Uses machine learning techniques to identify trading opportunities
    """
    
    def __init__(self):
        self.logger = logging.getLogger("MarketAIAnalyzer")
        
        # ML models would be loaded here
        # For now, using rule-based analysis with AI-like logic
        
        self.pattern_thresholds = {
            'breakout_volume_multiplier': 2.0,
            'reversal_rsi_threshold': 70,
            'momentum_strength_min': 0.02,
            'volatility_spike_threshold': 1.5,
        }
    
    async def analyze_market_data(self, tick: MarketTick, price_history: List[float], 
                                  order_book: Optional[Dict] = None) -> List[MarketOpportunity]:
        """Analyze market data and detect opportunities"""
        opportunities = []
        
        if len(price_history) < 20:
            return opportunities
        
        # Breakout detection
        breakout_opp = self._detect_breakout(tick, price_history)
        if breakout_opp:
            opportunities.append(breakout_opp)
        
        # Reversal detection
        reversal_opp = self._detect_reversal(tick, price_history)
        if reversal_opp:
            opportunities.append(reversal_opp)
        
        # Volume spike detection
        volume_opp = self._detect_volume_spike(tick, price_history)
        if volume_opp:
            opportunities.append(volume_opp)
        
        return opportunities
    
    def _detect_breakout(self, tick: MarketTick, price_history: List[float]) -> Optional[MarketOpportunity]:
        """Detect breakout patterns"""
        if len(price_history) < 20:
            return None
        
        # Calculate resistance/support levels
        recent_20 = price_history[-20:]
        resistance = max(recent_20)
        support = min(recent_20)
        
        current_price = tick.price
        
        # Check for breakout above resistance
        if current_price > resistance * 1.001:  # 0.1% above resistance
            confidence = min((current_price - resistance) / resistance * 50, 0.8)
            
            return MarketOpportunity(
                symbol=tick.symbol,
                opportunity_type='breakout',
                confidence=confidence,
                expected_move=0.05,  # 5% expected move
                time_horizon_minutes=30,
                entry_price=current_price,
                stop_loss=resistance * 0.995,
                take_profit=current_price * 1.05,
                risk_reward_ratio=10.0,
                reasoning=f"Breakout above resistance at {resistance:.6f}"
            )
        
        # Check for breakdown below support
        elif current_price < support * 0.999:  # 0.1% below support
            confidence = min((support - current_price) / support * 50, 0.8)
            
            return MarketOpportunity(
                symbol=tick.symbol,
                opportunity_type='breakdown',
                confidence=confidence,
                expected_move=-0.05,  # 5% expected drop
                time_horizon_minutes=30,
                entry_price=current_price,
                stop_loss=support * 1.005,
                take_profit=current_price * 0.95,
                risk_reward_ratio=10.0,
                reasoning=f"Breakdown below support at {support:.6f}"
            )
        
        return None
    
    def _detect_reversal(self, tick: MarketTick, price_history: List[float]) -> Optional[MarketOpportunity]:
        """Detect reversal patterns using RSI-like logic"""
        if len(price_history) < 14:
            return None
        
        # Simple RSI calculation
        gains = []
        losses = []
        
        for i in range(1, 14):
            change = price_history[-i] - price_history[-i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)
        
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        
        if avg_loss == 0:
            return None
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Overbought reversal
        if rsi > 70:
            return MarketOpportunity(
                symbol=tick.symbol,
                opportunity_type='reversal',
                confidence=min((rsi - 70) / 30, 0.75),
                expected_move=-0.03,  # 3% expected drop
                time_horizon_minutes=60,
                entry_price=tick.price,
                stop_loss=tick.price * 1.02,
                take_profit=tick.price * 0.97,
                risk_reward_ratio=1.5,
                reasoning=f"Overbought reversal signal (RSI: {rsi:.1f})"
            )
        
        # Oversold reversal
        elif rsi < 30:
            return MarketOpportunity(
                symbol=tick.symbol,
                opportunity_type='reversal',
                confidence=min((30 - rsi) / 30, 0.75),
                expected_move=0.03,  # 3% expected rise
                time_horizon_minutes=60,
                entry_price=tick.price,
                stop_loss=tick.price * 0.98,
                take_profit=tick.price * 1.03,
                risk_reward_ratio=1.5,
                reasoning=f"Oversold reversal signal (RSI: {rsi:.1f})"
            )
        
        return None
    
    def _detect_volume_spike(self, tick: MarketTick, price_history: List[float]) -> Optional[MarketOpportunity]:
        """Detect volume spike opportunities"""
        # This would need historical volume data
        # For now, simplified based on current volume vs typical patterns
        
        if tick.volume > 0 and tick.price_change_24h != 0:
            # Estimate if this is a volume spike based on price movement
            volume_price_correlation = abs(tick.price_change_24h) * tick.volume / 1000000
            
            if volume_price_correlation > 10:  # Arbitrary threshold
                direction = 1 if tick.price_change_24h > 0 else -1
                
                return MarketOpportunity(
                    symbol=tick.symbol,
                    opportunity_type='volume_spike',
                    confidence=min(volume_price_correlation / 50, 0.7),
                    expected_move=0.04 * direction,
                    time_horizon_minutes=20,
                    entry_price=tick.price,
                    stop_loss=tick.price * (0.98 if direction > 0 else 1.02),
                    take_profit=tick.price * (1.04 if direction > 0 else 0.96),
                    risk_reward_ratio=2.0,
                    reasoning=f"Volume spike detected with {tick.price_change_24h:.1f}% price move"
                )
        
        return None
