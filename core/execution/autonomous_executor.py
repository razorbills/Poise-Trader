#!/usr/bin/env python3
"""
🚀 AUTONOMOUS TRADE EXECUTOR
Advanced automated order management and execution system

Features:
• Fully automated trade execution
• Intelligent order management
• Risk-aware position sizing  
• Real-time portfolio monitoring
• Advanced order types and logic
• AI-powered execution optimization
• Zero manual intervention required
"""

import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
try:
    import ccxt.async_support as ccxt
except ImportError:
    ccxt = None
import numpy as np
from .paper_trading_manager import PaperTradingManager

try:
    from live_paper_trading_test import LivePaperTradingManager
except Exception:
    LivePaperTradingManager = None

_REAL_TRADING_ENABLED = str(os.getenv('REAL_TRADING', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
_STRICT_REAL_DATA = str(os.getenv('STRICT_REAL_DATA', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
ALLOW_SIMULATED_FEATURES = (
    str(os.getenv('ALLOW_SIMULATED_FEATURES', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
    and not _REAL_TRADING_ENABLED
    and not _STRICT_REAL_DATA
)


class OrderType(Enum):
    """Order types supported by the executor"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other


class OrderStatus(Enum):
    """Order status tracking"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExecutionStrategy(Enum):
    """Execution strategy types"""
    AGGRESSIVE = "aggressive"      # Market orders, immediate execution
    CONSERVATIVE = "conservative"  # Limit orders, better prices
    STEALTH = "stealth"           # Hidden orders, minimal market impact
    OPTIMAL = "optimal"           # AI-optimized execution


@dataclass
class TradeOrder:
    """Trade order with full lifecycle management"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    order_type: OrderType
    strategy_name: str
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    
    # Order management
    exchange_order_id: Optional[str] = None
    filled_amount: float = 0.0
    average_price: float = 0.0
    fees: Dict[str, float] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_slippage: float = 0.001  # 0.1% default
    timeout_seconds: int = 300   # 5 minutes default
    
    # Execution parameters
    execution_strategy: ExecutionStrategy = ExecutionStrategy.OPTIMAL
    priority: int = 5  # 1-10, higher = more priority
    retries_remaining: int = 3
    
    # AI optimization
    confidence_score: float = 0.0
    expected_profit: float = 0.0
    risk_score: float = 0.0
    
    # Tracking
    created_at: datetime = None
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.fees is None:
            self.fees = {}


@dataclass
class Position:
    """Portfolio position tracking"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_loss_amount: Optional[float] = None
    
    # Strategy tracking
    strategy_name: str = ""
    confidence_score: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.size * self.current_price
    
    @property
    def pnl_percentage(self) -> float:
        if self.entry_price == 0:
            return 0.0
        
        if self.side == 'long':
            return (self.current_price - self.entry_price) / self.entry_price
        else:  # short
            return (self.entry_price - self.current_price) / self.entry_price


class AutonomousExecutor:
    """
    🤖 FULLY AUTONOMOUS TRADE EXECUTOR
    
    Executes all trades automatically with:
    • Zero manual intervention
    • AI-optimized execution timing
    • Advanced risk management
    • Real-time portfolio monitoring
    • Intelligent order management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("AutonomousExecutor")
        
        # Paper trading support
        self.paper_trading = config.get('paper_trading', True)
        self.paper_trader = None

        if _REAL_TRADING_ENABLED and self.paper_trading:
            raise RuntimeError("Paper trading cannot be enabled when REAL_TRADING is active")
        
        if self.paper_trading:
            initial_capital = config.get('initial_capital', 5000)
            if LivePaperTradingManager is not None:
                self.paper_trader = LivePaperTradingManager(float(initial_capital))
                self.logger.info(f"📊 Live-price paper trading enabled with ${float(initial_capital):,.2f} virtual capital")
            else:
                self.paper_trader = PaperTradingManager(initial_capital)
                self.logger.info(f"📊 Paper trading mode enabled with ${initial_capital:,.2f} virtual capital")
        
        # Exchange connections
        self.exchanges = {}
        self.active_exchange = None
        
        # Order and position tracking
        self.active_orders: Dict[str, TradeOrder] = {}
        self.order_history: List[TradeOrder] = []
        self.positions: Dict[str, Position] = {}
        
        # Portfolio management
        self.total_capital = Decimal(str(config.get('initial_capital', 5000)))
        self.available_balance = {}
        self.portfolio_value = self.total_capital
        
        # AI components
        self.execution_optimizer = ExecutionOptimizer()
        self.risk_manager = AdvancedRiskManager(config.get('risk_config', {}))
        self.portfolio_monitor = PortfolioMonitor()
        
        # Performance tracking
        self.execution_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_volume': 0.0,
            'total_fees': 0.0,
            'average_execution_time_ms': 0.0,
            'slippage_stats': [],
        }
        
        # Configuration
        self.max_concurrent_orders = config.get('max_concurrent_orders', 10)
        self.default_timeout = config.get('default_order_timeout', 300)  # 5 minutes
        self.max_position_size = config.get('max_position_size_pct', 0.1)  # 10% of portfolio
        self.emergency_stop_loss = config.get('emergency_stop_loss_pct', 0.05)  # 5% emergency stop
        
        # Execution settings
        self.execution_delay_ms = config.get('execution_delay_ms', 50)
        self.retry_delays = [1, 3, 5, 10]  # Progressive retry delays
        self.slippage_tolerance = config.get('slippage_tolerance', 0.002)  # 0.2%
    
    async def initialize(self):
        """Initialize the autonomous executor"""
        self.logger.info("🚀 Initializing Autonomous Executor")
        
        # Initialize exchange connections
        await self._initialize_exchanges()
        
        # Initialize AI components
        await self.execution_optimizer.initialize()
        await self.risk_manager.initialize(self.total_capital)
        
        # Start monitoring loops
        asyncio.create_task(self._order_management_loop())
        asyncio.create_task(self._position_monitoring_loop())
        asyncio.create_task(self._portfolio_rebalancing_loop())
        asyncio.create_task(self._performance_tracking_loop())
        
        # Load existing positions if any
        await self._load_existing_positions()
        
        self.logger.info("✅ Autonomous Executor initialized successfully")
    
    async def _initialize_exchanges(self):
        """Initialize exchange API connections"""
        use_sandbox = bool(self.config.get('use_sandbox', False if (_REAL_TRADING_ENABLED or _STRICT_REAL_DATA) else True))
        if (_REAL_TRADING_ENABLED or _STRICT_REAL_DATA) and use_sandbox:
            raise RuntimeError("use_sandbox cannot be enabled when REAL_TRADING/STRICT_REAL_DATA is active")

        exchange_configs = {
            'mexc': {
                'apiKey': self.config.get('mexc_api_key'),
                'secret': self.config.get('mexc_api_secret'),
                'sandbox': use_sandbox,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000,
                }
            },
            'binance': {
                'apiKey': self.config.get('binance_api_key'),
                'secret': self.config.get('binance_api_secret'),
                'sandbox': use_sandbox,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 5000,
                }
            }
        }
        
        for exchange_name, config in exchange_configs.items():
            if config['apiKey']:
                try:
                    exchange_class = getattr(ccxt, exchange_name)
                    exchange = exchange_class(config)
                    await exchange.load_markets()
                    
                    # Test connection
                    balance = await exchange.fetch_balance()
                    self.available_balance[exchange_name] = balance
                    
                    self.exchanges[exchange_name] = exchange
                    
                    # Set primary exchange
                    if not self.active_exchange:
                        self.active_exchange = exchange_name
                    
                    self.logger.info(f"✅ {exchange_name} exchange connected successfully")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize {exchange_name}: {e}")
    
    async def execute_autonomous_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade completely autonomously
        
        Args:
            signal: Trading signal with all necessary information
            
        Returns:
            Execution result with detailed information
        """
        start_time = time.time()
        
        try:
            # Handle paper trading mode
            if self.paper_trading and self.paper_trader:
                self.logger.info(f"📊 PAPER TRADING: Executing {signal['symbol']} {signal['action']} signal")
                
                # Execute in paper trading simulation
                result = await self.paper_trader.execute_trade(signal)
                
                # Add execution metadata
                result['mode'] = 'paper_trading'
                result['execution_time_ms'] = (time.time() - start_time) * 1000
                
                # Update our local statistics
                if result['success']:
                    self.execution_stats['successful_trades'] += 1
                    self.execution_stats['total_trades'] += 1
                    
                    # Get paper trading performance
                    paper_summary = self.paper_trader.get_performance_summary()
                    self.portfolio_value = paper_summary['portfolio_value']
                    self.execution_stats['total_unrealized_pnl'] = paper_summary['total_pnl']
                    
                    self.logger.info(f"✅ PAPER TRADE SUCCESS: {signal['symbol']} - Portfolio: ${self.portfolio_value:,.2f}")
                else:
                    self.execution_stats['failed_trades'] += 1
                    self.execution_stats['total_trades'] += 1
                    self.logger.warning(f"⚠️ PAPER TRADE FAILED: {signal['symbol']} - {result.get('error', 'Unknown error')}")
                
                return result
            
            # Real trading execution (for when paper trading is disabled)
            # Create trade order from signal
            order = await self._create_order_from_signal(signal)
            
            # AI-powered pre-execution analysis
            execution_analysis = await self.execution_optimizer.analyze_execution(order, self.positions)
            
            # Risk management checks
            risk_check = await self.risk_manager.validate_trade(order, self.portfolio_value, self.positions)
            if not risk_check['approved']:
                return {
                    'success': False,
                    'error': f"Risk check failed: {risk_check['reason']}",
                    'order_id': order.id
                }
            
            # Optimize execution parameters based on market conditions
            optimized_order = await self.execution_optimizer.optimize_order(order, execution_analysis)
            
            # Execute the trade
            result = await self._execute_order_with_intelligence(optimized_order)
            
            # Update statistics
            execution_time_ms = (time.time() - start_time) * 1000
            await self._update_execution_stats(result, execution_time_ms)
            
            # Log autonomous decision
            self.logger.info(f"🤖 AUTONOMOUS EXECUTION: {result['status']} - "
                           f"{order.symbol} {order.side} {order.amount} @ {result.get('avg_price', order.price)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Autonomous execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': (time.time() - start_time) * 1000
            }
    
    async def _create_order_from_signal(self, signal: Dict[str, Any]) -> TradeOrder:
        """Create a trade order from a trading signal"""
        
        # Generate unique order ID
        order_id = f"AUTO_{int(time.time() * 1000)}_{signal['symbol'].replace('/', '')}"
        
        # Determine execution strategy based on signal characteristics
        if signal.get('urgency') == 'EXTREME':
            exec_strategy = ExecutionStrategy.AGGRESSIVE
        elif signal.get('confidence', 0.5) > 0.8:
            exec_strategy = ExecutionStrategy.OPTIMAL
        else:
            exec_strategy = ExecutionStrategy.CONSERVATIVE
        
        # Create order
        order = TradeOrder(
            id=order_id,
            symbol=signal['symbol'],
            side='buy' if signal['action'].upper() == 'BUY' else 'sell',
            amount=signal['position_size'],
            price=signal['entry_price'],
            order_type=OrderType.LIMIT,  # Default to limit orders
            strategy_name=signal.get('strategy_name', 'unknown'),
            timestamp=datetime.now(),
            
            # Risk management
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit'),
            max_slippage=signal.get('max_slippage', self.slippage_tolerance),
            timeout_seconds=signal.get('timeout_seconds', self.default_timeout),
            
            # Execution parameters
            execution_strategy=exec_strategy,
            priority=int(signal.get('confidence', 0.5) * 10),
            
            # AI metrics
            confidence_score=signal.get('confidence', 0.0),
            expected_profit=signal.get('expected_profit', 0.0),
            risk_score=signal.get('risk_score', 0.5),
        )
        
        return order
    
    async def _execute_order_with_intelligence(self, order: TradeOrder) -> Dict[str, Any]:
        """Execute order with AI-driven intelligence"""
        
        self.logger.info(f"🧠 Executing intelligent order: {order.id}")
        
        # Add to active orders
        self.active_orders[order.id] = order
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        
        try:
            exchange = self.exchanges[self.active_exchange]
            
            # Choose execution method based on strategy
            if order.execution_strategy == ExecutionStrategy.AGGRESSIVE:
                result = await self._execute_market_order(exchange, order)
            
            elif order.execution_strategy == ExecutionStrategy.STEALTH:
                result = await self._execute_stealth_order(exchange, order)
                
            elif order.execution_strategy == ExecutionStrategy.OPTIMAL:
                result = await self._execute_optimal_order(exchange, order)
                
            else:  # CONSERVATIVE
                result = await self._execute_conservative_order(exchange, order)
            
            # Set up automatic risk management if successful
            if result['success']:
                await self._setup_automatic_risk_management(order, result)
            
            # Update order status
            if result['success']:
                order.status = OrderStatus.FILLED
                order.completed_at = datetime.now()
                order.filled_amount = result.get('filled_amount', order.amount)
                order.average_price = result.get('avg_price', order.price)
                
                # Update positions
                await self._update_positions(order, result)
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = result.get('error', 'Unknown error')
            
            # Move to history
            self.order_history.append(order)
            if order.id in self.active_orders:
                del self.active_orders[order.id]
            
            return result
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.logger.error(f"❌ Order execution failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'order_id': order.id
            }
    
    async def _execute_market_order(self, exchange, order: TradeOrder) -> Dict[str, Any]:
        """Execute aggressive market order for immediate fill"""
        
        try:
            # Place market order
            exchange_order = await exchange.create_market_order(
                symbol=order.symbol,
                side=order.side,
                amount=order.amount
            )
            
            # Wait for fill (market orders should fill immediately)
            await asyncio.sleep(0.1)  
            
            # Fetch order details
            order_info = await exchange.fetch_order(exchange_order['id'], order.symbol)
            
            return {
                'success': True,
                'order_id': order.id,
                'exchange_order_id': exchange_order['id'],
                'status': 'filled',
                'filled_amount': order_info.get('filled', order.amount),
                'avg_price': order_info.get('price', order.price),
                'fees': order_info.get('fees', []),
                'execution_method': 'market'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'order_id': order.id,
                'execution_method': 'market'
            }
    
    async def _execute_conservative_order(self, exchange, order: TradeOrder) -> Dict[str, Any]:
        """Execute conservative limit order with better pricing"""
        
        try:
            # Get current market price for better limit price
            ticker = await exchange.fetch_ticker(order.symbol)
            current_price = ticker['last']
            
            # Adjust limit price for better execution probability
            if order.side == 'buy':
                # Bid slightly above best bid
                limit_price = current_price * 1.001  # 0.1% above market
            else:
                # Ask slightly below best ask  
                limit_price = current_price * 0.999  # 0.1% below market
            
            # Place limit order
            exchange_order = await exchange.create_limit_order(
                symbol=order.symbol,
                side=order.side,
                amount=order.amount,
                price=limit_price
            )
            
            # Monitor for fill with timeout
            fill_result = await self._monitor_order_fill(exchange, exchange_order['id'], order.symbol, order.timeout_seconds)
            
            return {
                'success': fill_result['filled'],
                'order_id': order.id,
                'exchange_order_id': exchange_order['id'],
                'status': 'filled' if fill_result['filled'] else 'timeout',
                'filled_amount': fill_result.get('filled_amount', 0),
                'avg_price': fill_result.get('avg_price', limit_price),
                'fees': fill_result.get('fees', []),
                'execution_method': 'conservative_limit'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'order_id': order.id,
                'execution_method': 'conservative_limit'
            }
    
    async def _execute_stealth_order(self, exchange, order: TradeOrder) -> Dict[str, Any]:
        """Execute stealth order to minimize market impact"""
        
        try:
            # Break large orders into smaller chunks
            chunk_size = min(order.amount * 0.2, order.amount)  # Max 20% chunks
            chunks = []
            remaining = order.amount
            
            while remaining > 0:
                chunk_amount = min(chunk_size, remaining)
                chunks.append(chunk_amount)
                remaining -= chunk_amount
            
            filled_chunks = []
            total_filled = 0.0
            weighted_avg_price = 0.0
            
            # Execute chunks with delays
            for i, chunk_amount in enumerate(chunks):
                try:
                    # Place limit order for chunk
                    ticker = await exchange.fetch_ticker(order.symbol)
                    current_price = ticker['last']
                    
                    if order.side == 'buy':
                        limit_price = current_price * 0.9995  # Slightly below market
                    else:
                        limit_price = current_price * 1.0005  # Slightly above market
                    
                    chunk_order = await exchange.create_limit_order(
                        symbol=order.symbol,
                        side=order.side,
                        amount=chunk_amount,
                        price=limit_price
                    )
                    
                    # Monitor chunk fill (shorter timeout for stealth)
                    fill_result = await self._monitor_order_fill(
                        exchange, chunk_order['id'], order.symbol, 60  # 1 minute timeout
                    )
                    
                    if fill_result['filled']:
                        filled_chunks.append(fill_result)
                        total_filled += fill_result['filled_amount']
                        weighted_avg_price += fill_result['avg_price'] * fill_result['filled_amount']
                    
                    # Delay between chunks (stealth)
                    if i < len(chunks) - 1:
                        delay_s = 20.0
                        if ALLOW_SIMULATED_FEATURES:
                            delay_s = float(np.random.uniform(10, 30))  # 10-30 second random delay
                        await asyncio.sleep(delay_s)
                        
                except Exception as e:
                    self.logger.warning(f"Stealth chunk {i+1} failed: {e}")
                    continue
            
            if total_filled > 0:
                weighted_avg_price = weighted_avg_price / total_filled
                success = total_filled >= order.amount * 0.8  # 80% fill threshold
                
                return {
                    'success': success,
                    'order_id': order.id,
                    'status': 'filled' if success else 'partial',
                    'filled_amount': total_filled,
                    'avg_price': weighted_avg_price,
                    'execution_method': 'stealth',
                    'chunks_executed': len(filled_chunks),
                    'fill_rate': total_filled / order.amount
                }
            else:
                return {
                    'success': False,
                    'error': 'No chunks filled successfully',
                    'order_id': order.id,
                    'execution_method': 'stealth'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'order_id': order.id,
                'execution_method': 'stealth'
            }
    
    async def _execute_optimal_order(self, exchange, order: TradeOrder) -> Dict[str, Any]:
        """Execute AI-optimized order with best execution logic"""
        
        try:
            # Get market conditions for optimization
            orderbook = await exchange.fetch_order_book(order.symbol, limit=20)
            ticker = await exchange.fetch_ticker(order.symbol)
            
            # Analyze optimal execution approach
            market_analysis = {
                'spread': ticker['ask'] - ticker['bid'],
                'bid_depth': sum([bid[1] for bid in orderbook['bids'][:5]]),
                'ask_depth': sum([ask[1] for ask in orderbook['asks'][:5]]),
                'volume_24h': ticker['quoteVolume'],
                'volatility': abs(ticker['percentage']) / 100 if ticker['percentage'] else 0,
            }
            
            # Choose optimal execution method based on analysis
            if market_analysis['spread'] / ticker['last'] < 0.001:  # Tight spread
                # Use limit order close to mid-price
                mid_price = (ticker['bid'] + ticker['ask']) / 2
                
                if order.side == 'buy':
                    optimal_price = mid_price + market_analysis['spread'] * 0.3
                else:
                    optimal_price = mid_price - market_analysis['spread'] * 0.3
                
                return await self._execute_optimal_limit_order(exchange, order, optimal_price)
                
            elif order.amount * ticker['last'] > market_analysis['bid_depth'] * 0.1:  # Large order
                # Use TWAP (Time-Weighted Average Price) execution
                return await self._execute_twap_order(exchange, order)
                
            else:  # Normal conditions
                # Use smart limit order with aggressive timeout
                return await self._execute_smart_limit_order(exchange, order)
                
        except Exception as e:
            # Fallback to conservative execution
            self.logger.warning(f"Optimal execution analysis failed, using conservative: {e}")
            return await self._execute_conservative_order(exchange, order)
    
    async def _execute_optimal_limit_order(self, exchange, order: TradeOrder, optimal_price: float) -> Dict[str, Any]:
        """Execute limit order at optimal price"""
        
        try:
            exchange_order = await exchange.create_limit_order(
                symbol=order.symbol,
                side=order.side,
                amount=order.amount,
                price=optimal_price
            )
            
            # Aggressive monitoring with price updates
            fill_result = await self._monitor_order_with_updates(exchange, exchange_order['id'], order)
            
            return {
                'success': fill_result['filled'],
                'order_id': order.id,
                'exchange_order_id': exchange_order['id'],
                'status': 'filled' if fill_result['filled'] else 'timeout',
                'filled_amount': fill_result.get('filled_amount', 0),
                'avg_price': fill_result.get('avg_price', optimal_price),
                'fees': fill_result.get('fees', []),
                'execution_method': 'optimal_limit'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'order_id': order.id,
                'execution_method': 'optimal_limit'
            }
    
    async def _execute_twap_order(self, exchange, order: TradeOrder) -> Dict[str, Any]:
        """Execute Time-Weighted Average Price order for large sizes"""
        
        try:
            # Split into time-based chunks
            duration_minutes = min(30, order.timeout_seconds / 60)  # Max 30 minutes
            num_chunks = max(3, int(duration_minutes / 5))  # Chunk every 5 minutes minimum
            chunk_size = order.amount / num_chunks
            interval_seconds = (duration_minutes * 60) / num_chunks
            
            executed_chunks = []
            total_filled = 0.0
            weighted_avg_price = 0.0
            
            for i in range(num_chunks):
                try:
                    # Get current best price
                    ticker = await exchange.fetch_ticker(order.symbol)
                    
                    if order.side == 'buy':
                        chunk_price = ticker['ask'] * 0.9999  # Just below ask
                    else:
                        chunk_price = ticker['bid'] * 1.0001  # Just above bid
                    
                    # Execute chunk
                    chunk_order = await exchange.create_limit_order(
                        symbol=order.symbol,
                        side=order.side,
                        amount=chunk_size,
                        price=chunk_price
                    )
                    
                    # Monitor chunk (short timeout for TWAP)
                    fill_result = await self._monitor_order_fill(
                        exchange, chunk_order['id'], order.symbol, min(180, interval_seconds)
                    )
                    
                    if fill_result['filled']:
                        executed_chunks.append(fill_result)
                        total_filled += fill_result['filled_amount']
                        weighted_avg_price += fill_result['avg_price'] * fill_result['filled_amount']
                    
                    # Wait for next chunk
                    if i < num_chunks - 1:
                        await asyncio.sleep(interval_seconds)
                        
                except Exception as e:
                    self.logger.warning(f"TWAP chunk {i+1} failed: {e}")
                    continue
            
            if total_filled > 0:
                weighted_avg_price = weighted_avg_price / total_filled
                success = total_filled >= order.amount * 0.7  # 70% fill threshold for TWAP
                
                return {
                    'success': success,
                    'order_id': order.id,
                    'status': 'filled' if success else 'partial',
                    'filled_amount': total_filled,
                    'avg_price': weighted_avg_price,
                    'execution_method': 'twap',
                    'chunks_executed': len(executed_chunks),
                    'fill_rate': total_filled / order.amount
                }
            else:
                return {
                    'success': False,
                    'error': 'No TWAP chunks filled',
                    'order_id': order.id,
                    'execution_method': 'twap'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'order_id': order.id,
                'execution_method': 'twap'
            }
    
    async def _execute_smart_limit_order(self, exchange, order: TradeOrder) -> Dict[str, Any]:
        """Execute smart limit order with dynamic price updates"""
        
        try:
            # Start with optimal limit price
            ticker = await exchange.fetch_ticker(order.symbol)
            
            if order.side == 'buy':
                initial_price = ticker['bid'] * 1.0005  # Slightly above best bid
            else:
                initial_price = ticker['ask'] * 0.9995  # Slightly below best ask
            
            exchange_order = await exchange.create_limit_order(
                symbol=order.symbol,
                side=order.side,
                amount=order.amount,
                price=initial_price
            )
            
            # Smart monitoring with price adjustments
            fill_result = await self._monitor_order_with_smart_updates(exchange, exchange_order['id'], order)
            
            return {
                'success': fill_result['filled'],
                'order_id': order.id,
                'exchange_order_id': exchange_order['id'],
                'status': 'filled' if fill_result['filled'] else 'timeout',
                'filled_amount': fill_result.get('filled_amount', 0),
                'avg_price': fill_result.get('avg_price', initial_price),
                'fees': fill_result.get('fees', []),
                'execution_method': 'smart_limit',
                'price_updates': fill_result.get('price_updates', 0)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'order_id': order.id,
                'execution_method': 'smart_limit'
            }
    
    async def _monitor_order_fill(self, exchange, order_id: str, symbol: str, timeout_seconds: int) -> Dict[str, Any]:
        """Monitor order fill with timeout"""
        
        start_time = time.time()
        check_interval = min(5, timeout_seconds / 10)  # Check every 5 seconds or 10% of timeout
        
        while time.time() - start_time < timeout_seconds:
            try:
                order_info = await exchange.fetch_order(order_id, symbol)
                
                if order_info['status'] == 'closed':
                    return {
                        'filled': True,
                        'filled_amount': order_info['filled'],
                        'avg_price': order_info['price'],
                        'fees': order_info.get('fees', []),
                        'status': order_info['status']
                    }
                elif order_info['status'] in ['canceled', 'rejected']:
                    return {
                        'filled': False,
                        'error': f"Order {order_info['status']}",
                        'status': order_info['status']
                    }
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error monitoring order {order_id}: {e}")
                await asyncio.sleep(check_interval)
        
        # Timeout reached - try to cancel order
        try:
            await exchange.cancel_order(order_id, symbol)
        except:
            pass  # Order might have filled just as we tried to cancel
        
        return {
            'filled': False,
            'error': 'Order timeout',
            'status': 'timeout'
        }
    
    async def _monitor_order_with_updates(self, exchange, order_id: str, order: TradeOrder) -> Dict[str, Any]:
        """Monitor order with dynamic price updates for better fills"""
        
        start_time = time.time()
        last_update_time = start_time
        price_updates = 0
        
        while time.time() - start_time < order.timeout_seconds:
            try:
                # Check order status
                order_info = await exchange.fetch_order(order_id, symbol=order.symbol)
                
                if order_info['status'] == 'closed':
                    return {
                        'filled': True,
                        'filled_amount': order_info['filled'],
                        'avg_price': order_info['price'],
                        'fees': order_info.get('fees', []),
                        'price_updates': price_updates
                    }
                elif order_info['status'] in ['canceled', 'rejected']:
                    return {
                        'filled': False,
                        'error': f"Order {order_info['status']}",
                        'price_updates': price_updates
                    }
                
                # Update price if order is aging
                elapsed = time.time() - last_update_time
                if elapsed > 30 and price_updates < 3:  # Update every 30 seconds, max 3 times
                    try:
                        # Cancel current order
                        await exchange.cancel_order(order_id, order.symbol)
                        await asyncio.sleep(1)
                        
                        # Place new order with updated price
                        ticker = await exchange.fetch_ticker(order.symbol)
                        
                        if order.side == 'buy':
                            new_price = ticker['bid'] * 1.002  # More aggressive
                        else:
                            new_price = ticker['ask'] * 0.998  # More aggressive
                        
                        new_exchange_order = await exchange.create_limit_order(
                            symbol=order.symbol,
                            side=order.side,
                            amount=order.amount,
                            price=new_price
                        )
                        
                        order_id = new_exchange_order['id']
                        last_update_time = time.time()
                        price_updates += 1
                        
                        self.logger.info(f"📈 Updated order price: {order.symbol} {new_price}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to update order price: {e}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring order with updates: {e}")
                await asyncio.sleep(5)
        
        # Timeout reached
        try:
            await exchange.cancel_order(order_id, order.symbol)
        except:
            pass
        
        return {
            'filled': False,
            'error': 'Order timeout with updates',
            'price_updates': price_updates
        }
    
    async def _monitor_order_with_smart_updates(self, exchange, order_id: str, order: TradeOrder) -> Dict[str, Any]:
        """Monitor with AI-driven smart price updates"""
        
        start_time = time.time()
        price_updates = 0
        update_history = []
        
        while time.time() - start_time < order.timeout_seconds:
            try:
                # Check order status
                order_info = await exchange.fetch_order(order_id, symbol=order.symbol)
                
                if order_info['status'] == 'closed':
                    return {
                        'filled': True,
                        'filled_amount': order_info['filled'],
                        'avg_price': order_info['price'],
                        'fees': order_info.get('fees', []),
                        'price_updates': price_updates
                    }
                
                # Smart update decision based on market movement
                should_update, new_price = await self._should_update_order_price(
                    exchange, order, update_history
                )
                
                if should_update and price_updates < 5:  # Max 5 updates
                    try:
                        # Cancel and replace with new price
                        await exchange.cancel_order(order_id, order.symbol)
                        await asyncio.sleep(0.5)
                        
                        new_exchange_order = await exchange.create_limit_order(
                            symbol=order.symbol,
                            side=order.side,
                            amount=order.amount,
                            price=new_price
                        )
                        
                        order_id = new_exchange_order['id']
                        price_updates += 1
                        update_history.append({
                            'timestamp': time.time(),
                            'price': new_price,
                            'reason': 'smart_update'
                        })
                        
                        self.logger.info(f"🧠 Smart price update: {order.symbol} {new_price}")
                        
                    except Exception as e:
                        self.logger.warning(f"Smart update failed: {e}")
                
                await asyncio.sleep(10)  # Check every 10 seconds for smart updates
                
            except Exception as e:
                self.logger.error(f"Error in smart monitoring: {e}")
                await asyncio.sleep(10)
        
        # Timeout
        try:
            await exchange.cancel_order(order_id, order.symbol)
        except:
            pass
        
        return {
            'filled': False,
            'error': 'Smart order timeout',
            'price_updates': price_updates
        }
    
    async def _should_update_order_price(self, exchange, order: TradeOrder, 
                                        update_history: List[Dict]) -> Tuple[bool, float]:
        """AI logic to determine if order price should be updated"""
        
        try:
            # Get current market state
            ticker = await exchange.fetch_ticker(order.symbol)
            orderbook = await exchange.fetch_order_book(order.symbol, limit=10)
            
            current_price = ticker['last']
            spread = ticker['ask'] - ticker['bid']
            
            # Calculate how far our order is from market
            if order.side == 'buy':
                distance_from_market = (current_price - order.price) / current_price
                market_moving_away = current_price > order.price
            else:
                distance_from_market = (order.price - current_price) / current_price
                market_moving_away = current_price < order.price
            
            # Update conditions:
            # 1. Market moved away significantly (>0.5%)
            # 2. Spread is reasonable (<0.2%)
            # 3. Not updated recently (>60 seconds since last update)
            
            last_update = update_history[-1]['timestamp'] if update_history else 0
            time_since_update = time.time() - last_update
            
            should_update = (
                market_moving_away and
                distance_from_market > 0.005 and  # 0.5% away
                spread / current_price < 0.002 and  # Spread < 0.2%
                time_since_update > 60  # At least 60 seconds
            )
            
            if should_update:
                # Calculate new aggressive price
                if order.side == 'buy':
                    new_price = ticker['bid'] * 1.003  # Above best bid
                else:
                    new_price = ticker['ask'] * 0.997  # Below best ask
                
                return True, new_price
            
            return False, order.price
            
        except Exception as e:
            self.logger.error(f"Error in price update logic: {e}")
            return False, order.price
    
    async def _setup_automatic_risk_management(self, order: TradeOrder, execution_result: Dict):
        """Set up automatic stop loss and take profit orders"""
        
        if not execution_result.get('success'):
            return
        
        try:
            exchange = self.exchanges[self.active_exchange]
            avg_price = execution_result['avg_price']
            filled_amount = execution_result['filled_amount']
            
            # Set up stop loss
            if order.stop_loss:
                try:
                    # Determine stop loss side
                    sl_side = 'sell' if order.side == 'buy' else 'buy'
                    
                    sl_order = await exchange.create_order(
                        symbol=order.symbol,
                        type='stop_loss_limit',
                        side=sl_side,
                        amount=filled_amount,
                        price=order.stop_loss * 0.995,  # Slightly below stop for safety
                        params={'stopPrice': order.stop_loss}
                    )
                    
                    self.logger.info(f"🛡️ Stop loss set: {order.symbol} @ {order.stop_loss}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to set stop loss: {e}")
            
            # Set up take profit
            if order.take_profit:
                try:
                    # Determine take profit side
                    tp_side = 'sell' if order.side == 'buy' else 'buy'
                    
                    tp_order = await exchange.create_limit_order(
                        symbol=order.symbol,
                        side=tp_side,
                        amount=filled_amount,
                        price=order.take_profit
                    )
                    
                    self.logger.info(f"🎯 Take profit set: {order.symbol} @ {order.take_profit}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to set take profit: {e}")
            
        except Exception as e:
            self.logger.error(f"Error setting up automatic risk management: {e}")
    
    async def _update_positions(self, order: TradeOrder, execution_result: Dict):
        """Update portfolio positions after successful trade"""
        
        try:
            symbol = order.symbol
            side = 'long' if order.side == 'buy' else 'short'
            amount = execution_result['filled_amount']
            price = execution_result['avg_price']
            
            # Update or create position
            if symbol in self.positions:
                pos = self.positions[symbol]
                
                if pos.side == side:
                    # Adding to existing position
                    total_value = pos.size * pos.entry_price + amount * price
                    total_size = pos.size + amount
                    pos.entry_price = total_value / total_size
                    pos.size = total_size
                else:
                    # Reducing or flipping position
                    if amount >= pos.size:
                        # Flip or close position
                        remaining = amount - pos.size
                        if remaining > 0:
                            pos.side = side
                            pos.size = remaining
                            pos.entry_price = price
                        else:
                            # Close position completely
                            del self.positions[symbol]
                            return
                    else:
                        # Reduce position
                        pos.size -= amount
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    size=amount,
                    entry_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=datetime.now(),
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    strategy_name=order.strategy_name,
                    confidence_score=order.confidence_score
                )
            
            self.logger.info(f"📊 Position updated: {symbol} {side} {amount} @ {price}")
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _order_management_loop(self):
        """Continuous order management and monitoring"""
        while True:
            try:
                # Monitor active orders
                for order_id, order in list(self.active_orders.items()):
                    if order.status == OrderStatus.SUBMITTED:
                        # Check if order has timed out
                        elapsed = (datetime.now() - order.submitted_at).total_seconds()
                        if elapsed > order.timeout_seconds:
                            await self._handle_order_timeout(order)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in order management loop: {e}")
                await asyncio.sleep(60)
    
    async def _position_monitoring_loop(self):
        """Continuous position monitoring and risk management"""
        while True:
            try:
                if self.paper_trading:
                    await asyncio.sleep(30)
                    continue

                if not self.positions:
                    await asyncio.sleep(30)
                    continue

                if not self.active_exchange:
                    await asyncio.sleep(30)
                    continue
                
                # Update current prices for all positions
                for symbol, position in list(self.positions.items()):
                    try:
                        ticker = await self.exchanges[self.active_exchange].fetch_ticker(symbol)
                        position.current_price = ticker['last']
                        
                        # Calculate unrealized PnL
                        if position.side == 'long':
                            position.unrealized_pnl = (position.current_price - position.entry_price) * position.size
                        else:
                            position.unrealized_pnl = (position.entry_price - position.current_price) * position.size
                        
                        # Check emergency stop loss
                        pnl_pct = position.pnl_percentage
                        if abs(pnl_pct) > self.emergency_stop_loss:
                            await self._emergency_close_position(position)
                        
                    except Exception as e:
                        self.logger.error(f"Error updating position {symbol}: {e}")
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _portfolio_rebalancing_loop(self):
        """Continuous portfolio monitoring and rebalancing"""
        while True:
            try:
                # Update portfolio value
                await self._update_portfolio_value()
                
                # Check for rebalancing needs
                await self._check_portfolio_balance()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in portfolio rebalancing: {e}")
                await asyncio.sleep(600)
    
    async def _performance_tracking_loop(self):
        """Continuous performance tracking and reporting"""
        while True:
            try:
                # Calculate performance metrics
                await self._calculate_performance_metrics()
                
                # Log performance report every hour
                await self._log_performance_report()
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                self.logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(1800)
    
    async def _handle_order_timeout(self, order: TradeOrder):
        """Handle order timeout with intelligent retry logic"""
        
        try:
            # Try to cancel the order first
            try:
                exchange = self.exchanges[self.active_exchange]
                await exchange.cancel_order(order.exchange_order_id, order.symbol)
                self.logger.warning(f"⏰ Order {order.id} timed out and cancelled")
            except:
                pass  # Order might have filled or already cancelled
            
            # Decide whether to retry
            if order.retries_remaining > 0 and order.priority >= 7:  # High priority orders
                order.retries_remaining -= 1
                order.status = OrderStatus.PENDING
                
                # Retry with adjusted parameters
                await self._retry_order_with_adjustments(order)
            else:
                # Move to history as failed
                order.status = OrderStatus.EXPIRED
                self.order_history.append(order)
                del self.active_orders[order.id]
                
        except Exception as e:
            self.logger.error(f"Error handling order timeout: {e}")
    
    async def _retry_order_with_adjustments(self, order: TradeOrder):
        """Retry order with adjusted parameters for better execution"""
        
        try:
            # Get current market conditions
            exchange = self.exchanges[self.active_exchange]
            ticker = await exchange.fetch_ticker(order.symbol)
            
            # Adjust price for better fill probability
            if order.side == 'buy':
                # More aggressive buying price
                adjusted_price = ticker['ask'] * 0.999  # Just below ask
            else:
                # More aggressive selling price
                adjusted_price = ticker['bid'] * 1.001  # Just above bid
            
            order.price = adjusted_price
            order.execution_strategy = ExecutionStrategy.AGGRESSIVE  # More aggressive on retry
            order.timeout_seconds = min(order.timeout_seconds * 1.5, 600)  # Increase timeout
            
            self.logger.info(f"🔄 Retrying order {order.id} with adjusted price: {adjusted_price}")
            
            # Re-execute
            result = await self._execute_order_with_intelligence(order)
            
        except Exception as e:
            self.logger.error(f"Error retrying order: {e}")
    
    async def _emergency_close_position(self, position: Position):
        """Emergency close position due to excessive loss"""
        
        try:
            self.logger.warning(f"🚨 EMERGENCY CLOSE: {position.symbol} at {position.pnl_percentage*100:.1f}% loss")
            
            # Create emergency close order
            close_side = 'sell' if position.side == 'long' else 'buy'
            
            emergency_order = TradeOrder(
                id=f"EMERGENCY_{int(time.time() * 1000)}_{position.symbol.replace('/', '')}",
                symbol=position.symbol,
                side=close_side,
                amount=position.size,
                price=position.current_price,
                order_type=OrderType.MARKET,  # Market order for immediate execution
                strategy_name="emergency_stop",
                timestamp=datetime.now(),
                execution_strategy=ExecutionStrategy.AGGRESSIVE,
                priority=10  # Highest priority
            )
            
            # Execute emergency close
            result = await self._execute_order_with_intelligence(emergency_order)
            
            if result['success']:
                self.logger.warning(f"✅ Emergency close successful: {position.symbol}")
            else:
                self.logger.error(f"❌ Emergency close failed: {position.symbol}")
                
        except Exception as e:
            self.logger.error(f"Error in emergency position close: {e}")
    
    async def _update_portfolio_value(self):
        """Update total portfolio value"""
        
        try:
            if self.paper_trading and self.paper_trader and hasattr(self.paper_trader, 'get_performance_summary'):
                summary = self.paper_trader.get_performance_summary() or {}
                try:
                    self.portfolio_value = float(summary.get('portfolio_value') or self.portfolio_value)
                except Exception:
                    pass
                try:
                    self.execution_stats['total_unrealized_pnl'] = float(summary.get('total_pnl') or self.execution_stats.get('total_unrealized_pnl', 0) or 0)
                except Exception:
                    pass
                return

            total_value = 0.0
            
            # Add cash balances
            for exchange_name, balance_info in self.available_balance.items():
                for currency, amount in balance_info['free'].items():
                    if currency == 'USDT':
                        total_value += amount
                    elif amount > 0:
                        # Convert to USDT value if possible
                        try:
                            symbol = f"{currency}/USDT"
                            exchange = self.exchanges[exchange_name]
                            ticker = await exchange.fetch_ticker(symbol)
                            total_value += amount * ticker['last']
                        except:
                            pass  # Skip if conversion not possible
            
            # Add position values
            for position in self.positions.values():
                total_value += position.market_value
            
            self.portfolio_value = total_value
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    async def _check_portfolio_balance(self):
        """Check if portfolio needs rebalancing"""
        
        try:
            if not self.positions:
                return
            
            # Check position concentration
            largest_position_pct = 0.0
            total_position_value = sum(pos.market_value for pos in self.positions.values())
            
            if total_position_value > 0:
                for position in self.positions.values():
                    position_pct = position.market_value / total_position_value
                    largest_position_pct = max(largest_position_pct, position_pct)
                
                # If any position is > 50% of portfolio, consider rebalancing
                if largest_position_pct > 0.5:
                    self.logger.warning(f"⚠️ Large position concentration detected: {largest_position_pct*100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio balance: {e}")
    
    async def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        
        try:
            if not self.order_history:
                return
            
            # Basic statistics
            successful_orders = [o for o in self.order_history if o.status == OrderStatus.FILLED]
            failed_orders = [o for o in self.order_history if o.status in [OrderStatus.REJECTED, OrderStatus.EXPIRED]]
            
            self.execution_stats['total_trades'] = len(self.order_history)
            self.execution_stats['successful_trades'] = len(successful_orders)
            self.execution_stats['failed_trades'] = len(failed_orders)
            self.execution_stats['success_rate'] = len(successful_orders) / len(self.order_history) if self.order_history else 0
            
            # Calculate slippage statistics
            slippages = []
            for order in successful_orders:
                if order.order_type != OrderType.MARKET:  # Only for limit orders
                    expected_price = order.price
                    actual_price = order.average_price
                    if expected_price > 0 and actual_price > 0:
                        slippage = abs(actual_price - expected_price) / expected_price
                        slippages.append(slippage)
            
            if slippages:
                self.execution_stats['avg_slippage'] = np.mean(slippages)
                self.execution_stats['max_slippage'] = np.max(slippages)
                self.execution_stats['slippage_std'] = np.std(slippages)
            
            # Calculate total PnL
            total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            self.execution_stats['total_unrealized_pnl'] = total_pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
    
    async def _log_performance_report(self):
        """Log comprehensive performance report"""
        
        try:
            stats = self.execution_stats
            
            self.logger.info("📊 AUTONOMOUS EXECUTOR PERFORMANCE REPORT")
            self.logger.info("=" * 50)
            self.logger.info(f"💼 Portfolio Value: ${self.portfolio_value:,.2f}")
            self.logger.info(f"🔄 Total Trades: {stats['total_trades']}")
            self.logger.info(f"✅ Success Rate: {stats.get('success_rate', 0)*100:.1f}%")
            self.logger.info(f"⚡ Active Orders: {len(self.active_orders)}")
            self.logger.info(f"📈 Open Positions: {len(self.positions)}")
            
            if self.positions:
                self.logger.info(f"\n📊 Position Summary:")
                for symbol, pos in self.positions.items():
                    pnl_pct = pos.pnl_percentage * 100
                    self.logger.info(f"   • {symbol}: {pos.side} {pos.size:.6f} @ {pos.entry_price:.6f} "
                                   f"({pnl_pct:+.2f}%)")
            
            if 'avg_slippage' in stats:
                self.logger.info(f"\n📉 Execution Quality:")
                self.logger.info(f"   • Avg Slippage: {stats['avg_slippage']*100:.3f}%")
                self.logger.info(f"   • Max Slippage: {stats['max_slippage']*100:.3f}%")
            
        except Exception as e:
            self.logger.error(f"Error logging performance report: {e}")
    
    async def _load_existing_positions(self):
        """Load existing positions from exchange"""
        
        try:
            if not self.active_exchange:
                return
            
            exchange = self.exchanges[self.active_exchange]
            
            # Fetch account positions
            try:
                positions = await exchange.fetch_positions()
                
                for pos_data in positions:
                    if pos_data['size'] > 0:  # Only non-zero positions
                        symbol = pos_data['symbol']
                        
                        position = Position(
                            symbol=symbol,
                            side='long' if pos_data['side'] == 'long' else 'short',
                            size=pos_data['size'],
                            entry_price=pos_data['entryPrice'],
                            current_price=pos_data['markPrice'],
                            unrealized_pnl=pos_data['unrealizedPnl'],
                            realized_pnl=0.0,
                            timestamp=datetime.now(),
                            strategy_name="existing"
                        )
                        
                        self.positions[symbol] = position
                        self.logger.info(f"📊 Loaded existing position: {symbol} {position.side} {position.size}")
                
            except Exception as e:
                # Some exchanges might not support positions API
                self.logger.debug(f"Could not load positions from {self.active_exchange}: {e}")
                
        except Exception as e:
            self.logger.error(f"Error loading existing positions: {e}")
    
    async def _update_execution_stats(self, result: Dict[str, Any], execution_time_ms: float):
        """Update execution statistics"""
        
        try:
            self.execution_stats['total_trades'] += 1
            
            if result['success']:
                self.execution_stats['successful_trades'] += 1
            else:
                self.execution_stats['failed_trades'] += 1
            
            # Update average execution time
            total_time = self.execution_stats.get('total_execution_time_ms', 0.0)
            total_trades = self.execution_stats['total_trades']
            
            total_time += execution_time_ms
            self.execution_stats['total_execution_time_ms'] = total_time
            self.execution_stats['average_execution_time_ms'] = total_time / total_trades
            
        except Exception as e:
            self.logger.error(f"Error updating execution stats: {e}")
    
    def get_executor_summary(self) -> Dict[str, Any]:
        """Get comprehensive executor summary"""
        
        summary = {
            'portfolio_value': float(self.portfolio_value),
            'total_capital': float(self.total_capital),
            'active_orders': len(self.active_orders),
            'open_positions': len(self.positions),
            'connected_exchanges': list(self.exchanges.keys()),
            'active_exchange': self.active_exchange,
            'execution_stats': self.execution_stats,
            'positions': {
                symbol: {
                    'side': pos.side,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'pnl_percentage': pos.pnl_percentage,
                    'market_value': pos.market_value,
                    'strategy': pos.strategy_name
                }
                for symbol, pos in self.positions.items()
            },
            'recent_orders': [
                {
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'amount': order.amount,
                    'price': order.price,
                    'status': order.status.value,
                    'strategy': order.strategy_name,
                    'timestamp': order.timestamp.isoformat()
                }
                for order in self.order_history[-10:]  # Last 10 orders
            ]
        }
        
        return summary


class ExecutionOptimizer:
    """AI-powered execution optimizer"""
    
    def __init__(self):
        self.logger = logging.getLogger("ExecutionOptimizer")
        
    async def initialize(self):
        """Initialize execution optimizer"""
        self.logger.info("🧠 ExecutionOptimizer initialized")
    
    async def analyze_execution(self, order: TradeOrder, positions: Dict[str, Position]) -> Dict[str, Any]:
        """Analyze optimal execution approach for an order"""
        
        analysis = {
            'recommended_strategy': ExecutionStrategy.OPTIMAL,
            'urgency_score': 0.5,
            'market_impact_estimate': 0.001,
            'optimal_chunk_size': order.amount,
            'recommended_timeout': order.timeout_seconds
        }
        
        # Analyze based on order size relative to typical volumes
        if order.amount * order.price > 10000:  # Large orders
            analysis['recommended_strategy'] = ExecutionStrategy.STEALTH
            analysis['optimal_chunk_size'] = order.amount * 0.3
            analysis['recommended_timeout'] = order.timeout_seconds * 1.5
        
        # Analyze based on confidence score
        if order.confidence_score > 0.9:
            analysis['recommended_strategy'] = ExecutionStrategy.AGGRESSIVE
            analysis['urgency_score'] = 0.9
        
        return analysis
    
    async def optimize_order(self, order: TradeOrder, analysis: Dict[str, Any]) -> TradeOrder:
        """Optimize order parameters based on analysis"""
        
        # Apply optimization recommendations
        order.execution_strategy = analysis['recommended_strategy']
        order.timeout_seconds = int(analysis['recommended_timeout'])
        
        # Adjust amount if chunking recommended
        if analysis['optimal_chunk_size'] < order.amount:
            order.amount = analysis['optimal_chunk_size']
        
        return order


class AdvancedRiskManager:
    """Advanced risk management system"""
    
    def __init__(self, risk_config: Dict[str, Any]):
        self.config = risk_config
        self.logger = logging.getLogger("AdvancedRiskManager")
        
        # Risk limits
        self.max_position_size_pct = risk_config.get('max_position_size_pct', 0.1)
        self.max_portfolio_risk = risk_config.get('max_portfolio_risk', 0.02)
        self.max_daily_trades = risk_config.get('max_daily_trades', 50)
        self.max_correlation_exposure = risk_config.get('max_correlation_exposure', 0.3)
        
        # Tracking
        self.daily_trades_count = 0
        self.last_reset_date = datetime.now().date()
    
    async def initialize(self, total_capital: Decimal):
        """Initialize risk manager"""
        self.total_capital = total_capital
        self.logger.info("🛡️ AdvancedRiskManager initialized")
    
    async def validate_trade(self, order: TradeOrder, portfolio_value: float, 
                           positions: Dict[str, Position]) -> Dict[str, Any]:
        """Comprehensive trade validation"""
        
        # Reset daily counter if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trades_count = 0
            self.last_reset_date = current_date
        
        # Check daily trade limit
        if self.daily_trades_count >= self.max_daily_trades:
            return {
                'approved': False,
                'reason': f'Daily trade limit reached ({self.max_daily_trades})'
            }
        
        # Check position size limit
        position_value = order.amount * order.price
        position_size_pct = position_value / portfolio_value
        
        if position_size_pct > self.max_position_size_pct:
            return {
                'approved': False,
                'reason': f'Position size too large ({position_size_pct*100:.1f}% > {self.max_position_size_pct*100:.1f}%)'
            }
        
        # Check portfolio risk exposure
        total_risk = sum(pos.market_value for pos in positions.values()) + position_value
        portfolio_risk_pct = total_risk / portfolio_value
        
        if portfolio_risk_pct > self.max_portfolio_risk:
            return {
                'approved': False,
                'reason': f'Portfolio risk too high ({portfolio_risk_pct*100:.1f}% > {self.max_portfolio_risk*100:.1f}%)'
            }
        
        # Increment daily trade count
        self.daily_trades_count += 1
        
        return {
            'approved': True,
            'reason': 'Risk checks passed',
            'position_size_pct': position_size_pct,
            'portfolio_risk_pct': portfolio_risk_pct
        }


class PortfolioMonitor:
    """Portfolio monitoring and analytics"""
    
    def __init__(self):
        self.logger = logging.getLogger("PortfolioMonitor")
    
    def calculate_portfolio_metrics(self, positions: Dict[str, Position], 
                                   total_value: float) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        
        if not positions:
            return {'total_positions': 0, 'total_value': total_value}
        
        metrics = {
            'total_positions': len(positions),
            'total_value': total_value,
            'total_unrealized_pnl': sum(pos.unrealized_pnl for pos in positions.values()),
            'long_positions': len([p for p in positions.values() if p.side == 'long']),
            'short_positions': len([p for p in positions.values() if p.side == 'short']),
            'largest_position': max((pos.market_value / total_value for pos in positions.values()), default=0),
            'position_distribution': {
                symbol: pos.market_value / total_value
                for symbol, pos in positions.items()
            }
        }
        
        return metrics
