#!/usr/bin/env python3
"""
🎯 ADVANCED ORDER MANAGEMENT SYSTEM
Professional-grade order types and execution algorithms
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import threading

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    HIDDEN = "hidden"
    TWAP = "twap"
    VWAP = "vwap"
    SMART = "smart"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class OrderExecution:
    """Single order execution record"""
    timestamp: datetime
    price: float
    quantity: float
    fees: float
    execution_id: str

@dataclass
class AdvancedOrder:
    """Advanced order with sophisticated execution logic"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    
    # Advanced parameters
    iceberg_visible_qty: Optional[float] = None
    twap_duration_minutes: Optional[int] = None
    max_participation_rate: float = 0.1  # Max 10% of volume
    min_execution_size: Optional[float] = None
    max_execution_size: Optional[float] = None
    
    # Execution tracking
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: float = 0.0
    total_fees: float = 0.0
    executions: List[OrderExecution] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity

class IcebergOrderManager:
    """Manages iceberg orders with hidden quantity"""
    
    def __init__(self, exchange_client):
        self.exchange_client = exchange_client
        self.active_icebergs: Dict[str, AdvancedOrder] = {}
        self._execution_lock = threading.Lock()
    
    async def submit_iceberg_order(self, order: AdvancedOrder) -> str:
        """Submit iceberg order with visible quantity management"""
        if order.order_type != OrderType.ICEBERG:
            raise ValueError("Order must be ICEBERG type")
        
        visible_qty = order.iceberg_visible_qty or min(order.quantity * 0.1, 100)
        
        # Submit first slice
        slice_order = self._create_order_slice(order, visible_qty)
        
        try:
            order_id = await self.exchange_client.create_order(
                symbol=order.symbol,
                type='limit',
                side=order.side.value,
                amount=slice_order.quantity,
                price=slice_order.price
            )
            
            order.order_id = order_id
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            
            self.active_icebergs[order_id] = order
            
            # Start monitoring
            asyncio.create_task(self._monitor_iceberg_order(order))
            
            logger.info(f"🧊 Iceberg order submitted: {order_id} ({visible_qty}/{order.quantity})")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to submit iceberg order: {e}")
            order.status = OrderStatus.REJECTED
            raise
    
    def _create_order_slice(self, parent_order: AdvancedOrder, quantity: float) -> AdvancedOrder:
        """Create a slice of the iceberg order"""
        return AdvancedOrder(
            symbol=parent_order.symbol,
            side=parent_order.side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=parent_order.price,
            time_in_force=parent_order.time_in_force
        )
    
    async def _monitor_iceberg_order(self, order: AdvancedOrder):
        """Monitor iceberg order execution and submit new slices"""
        while order.remaining_quantity > 0 and order.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            try:
                # Check order status
                order_status = await self.exchange_client.fetch_order(order.order_id, order.symbol)
                
                if order_status['status'] == 'closed':
                    # Current slice filled, submit next slice
                    with self._execution_lock:
                        filled_qty = order_status['filled']
                        order.filled_quantity += filled_qty
                        order.remaining_quantity -= filled_qty
                        
                        if order.remaining_quantity > 0:
                            # Submit next slice
                            visible_qty = min(
                                order.iceberg_visible_qty or order.remaining_quantity * 0.1,
                                order.remaining_quantity
                            )
                            
                            next_slice = self._create_order_slice(order, visible_qty)
                            
                            new_order_id = await self.exchange_client.create_order(
                                symbol=next_slice.symbol,
                                type='limit',
                                side=next_slice.side.value,
                                amount=next_slice.quantity,
                                price=next_slice.price
                            )
                            
                            order.order_id = new_order_id
                            logger.info(f"🧊 Next iceberg slice: {new_order_id} ({visible_qty}/{order.remaining_quantity})")
                        else:
                            order.status = OrderStatus.FILLED
                            order.completed_at = datetime.now()
                            logger.info(f"✅ Iceberg order completed: {order.order_id}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Iceberg monitoring error: {e}")
                await asyncio.sleep(10)

class TWAPExecutor:
    """Time-Weighted Average Price execution algorithm"""
    
    def __init__(self, exchange_client):
        self.exchange_client = exchange_client
        self.active_twaps: Dict[str, AdvancedOrder] = {}
    
    async def execute_twap_order(self, order: AdvancedOrder) -> str:
        """Execute TWAP order over specified time period"""
        if order.order_type != OrderType.TWAP:
            raise ValueError("Order must be TWAP type")
        
        duration_minutes = order.twap_duration_minutes or 60
        num_slices = min(duration_minutes, 20)  # Max 20 slices
        slice_quantity = order.quantity / num_slices
        slice_interval = (duration_minutes * 60) / num_slices
        
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        
        # Start TWAP execution
        asyncio.create_task(self._execute_twap_slices(order, slice_quantity, slice_interval, num_slices))
        
        logger.info(f"⏱️ TWAP order started: {order.symbol} - {num_slices} slices over {duration_minutes}min")
        return f"twap_{int(time.time())}"
    
    async def _execute_twap_slices(self, order: AdvancedOrder, slice_qty: float, 
                                 interval: float, total_slices: int):
        """Execute TWAP order slices"""
        for slice_num in range(total_slices):
            try:
                if order.remaining_quantity <= 0:
                    break
                
                # Adjust slice quantity if remaining is less
                current_slice_qty = min(slice_qty, order.remaining_quantity)
                
                # Get current market price for market orders
                if order.price is None:
                    ticker = await self.exchange_client.fetch_ticker(order.symbol)
                    price = ticker['last']
                else:
                    price = order.price
                
                # Submit slice
                slice_order_id = await self.exchange_client.create_order(
                    symbol=order.symbol,
                    type='market' if order.price is None else 'limit',
                    side=order.side.value,
                    amount=current_slice_qty,
                    price=price if order.price else None
                )
                
                # Wait for fill and update order
                await self._wait_for_slice_fill(order, slice_order_id, current_slice_qty)
                
                logger.info(f"⏱️ TWAP slice {slice_num + 1}/{total_slices} executed: {current_slice_qty}")
                
                # Wait for next slice (except for last slice)
                if slice_num < total_slices - 1:
                    await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"TWAP slice execution error: {e}")
                order.status = OrderStatus.PARTIALLY_FILLED
                break
        
        # Mark as completed
        if order.remaining_quantity <= 0.001:  # Account for rounding
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        order.completed_at = datetime.now()
        logger.info(f"✅ TWAP order completed: {order.symbol}")
    
    async def _wait_for_slice_fill(self, order: AdvancedOrder, slice_order_id: str, slice_qty: float):
        """Wait for slice to fill and update parent order"""
        max_wait_time = 60  # Max 1 minute wait
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                slice_status = await self.exchange_client.fetch_order(slice_order_id, order.symbol)
                
                if slice_status['status'] == 'closed':
                    filled_qty = slice_status['filled']
                    fill_price = slice_status['average']
                    
                    # Update parent order
                    order.filled_quantity += filled_qty
                    order.remaining_quantity -= filled_qty
                    
                    # Update average fill price
                    total_value = order.average_fill_price * (order.filled_quantity - filled_qty)
                    total_value += fill_price * filled_qty
                    order.average_fill_price = total_value / order.filled_quantity
                    
                    # Record execution
                    execution = OrderExecution(
                        timestamp=datetime.now(),
                        price=fill_price,
                        quantity=filled_qty,
                        fees=slice_status.get('fee', {}).get('cost', 0),
                        execution_id=slice_order_id
                    )
                    order.executions.append(execution)
                    
                    break
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error waiting for slice fill: {e}")
                break

class VWAPExecutor:
    """Volume-Weighted Average Price execution algorithm"""
    
    def __init__(self, exchange_client):
        self.exchange_client = exchange_client
        self.volume_profile = {}
    
    async def execute_vwap_order(self, order: AdvancedOrder) -> str:
        """Execute VWAP order based on historical volume patterns"""
        if order.order_type != OrderType.VWAP:
            raise ValueError("Order must be VWAP type")
        
        # Get volume profile for the symbol
        volume_profile = await self._get_volume_profile(order.symbol)
        
        # Calculate execution schedule based on volume
        execution_schedule = self._calculate_vwap_schedule(order, volume_profile)
        
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        
        # Start VWAP execution
        asyncio.create_task(self._execute_vwap_schedule(order, execution_schedule))
        
        logger.info(f"📊 VWAP order started: {order.symbol} - {len(execution_schedule)} scheduled executions")
        return f"vwap_{int(time.time())}"
    
    async def _get_volume_profile(self, symbol: str) -> Dict[int, float]:
        """Get historical volume profile for symbol"""
        try:
            # Get recent OHLCV data
            ohlcv = await self.exchange_client.fetch_ohlcv(symbol, '1h', limit=24)
            
            volume_by_hour = {}
            for candle in ohlcv:
                hour = datetime.fromtimestamp(candle[0] / 1000).hour
                volume = candle[5]
                
                if hour not in volume_by_hour:
                    volume_by_hour[hour] = []
                volume_by_hour[hour].append(volume)
            
            # Calculate average volume per hour
            avg_volume_by_hour = {}
            for hour, volumes in volume_by_hour.items():
                avg_volume_by_hour[hour] = np.mean(volumes)
            
            return avg_volume_by_hour
            
        except Exception as e:
            logger.error(f"Failed to get volume profile: {e}")
            # Return uniform distribution as fallback
            return {hour: 1.0 for hour in range(24)}
    
    def _calculate_vwap_schedule(self, order: AdvancedOrder, volume_profile: Dict[int, float]) -> List[Dict]:
        """Calculate VWAP execution schedule"""
        duration_hours = order.twap_duration_minutes // 60 or 1
        current_hour = datetime.now().hour
        
        # Get volume weights for execution period
        total_volume_weight = 0
        hour_weights = {}
        
        for i in range(duration_hours):
            hour = (current_hour + i) % 24
            weight = volume_profile.get(hour, 1.0)
            hour_weights[hour] = weight
            total_volume_weight += weight
        
        # Calculate quantity per hour based on volume
        schedule = []
        remaining_qty = order.quantity
        
        for i, (hour, weight) in enumerate(hour_weights.items()):
            if i == len(hour_weights) - 1:
                # Last execution gets remaining quantity
                hour_qty = remaining_qty
            else:
                proportion = weight / total_volume_weight
                hour_qty = order.quantity * proportion
                remaining_qty -= hour_qty
            
            schedule.append({
                'hour': hour,
                'quantity': hour_qty,
                'weight': weight,
                'execution_time': datetime.now() + timedelta(hours=i)
            })
        
        return schedule
    
    async def _execute_vwap_schedule(self, order: AdvancedOrder, schedule: List[Dict]):
        """Execute VWAP schedule"""
        for i, execution in enumerate(schedule):
            try:
                # Wait until execution time
                now = datetime.now()
                if execution['execution_time'] > now:
                    wait_seconds = (execution['execution_time'] - now).total_seconds()
                    await asyncio.sleep(wait_seconds)
                
                if order.remaining_quantity <= 0:
                    break
                
                # Execute this portion
                exec_qty = min(execution['quantity'], order.remaining_quantity)
                
                # Limit participation rate
                market_volume = await self._get_current_market_volume(order.symbol)
                max_qty = market_volume * order.max_participation_rate
                exec_qty = min(exec_qty, max_qty)
                
                if exec_qty > 0:
                    exec_order_id = await self.exchange_client.create_order(
                        symbol=order.symbol,
                        type='market',
                        side=order.side.value,
                        amount=exec_qty
                    )
                    
                    await self._wait_for_execution(order, exec_order_id, exec_qty)
                    
                    logger.info(f"📊 VWAP execution {i+1}/{len(schedule)}: {exec_qty}")
                
            except Exception as e:
                logger.error(f"VWAP execution error: {e}")
                break
        
        order.status = OrderStatus.FILLED if order.remaining_quantity <= 0.001 else OrderStatus.PARTIALLY_FILLED
        order.completed_at = datetime.now()
    
    async def _get_current_market_volume(self, symbol: str) -> float:
        """Get current market volume for participation rate calculation"""
        try:
            ticker = await self.exchange_client.fetch_ticker(symbol)
            return ticker.get('quoteVolume', 1000000)  # Fallback to 1M
        except:
            return 1000000  # Safe fallback

class SmartOrderRouter:
    """Intelligent order routing with dynamic strategy selection"""
    
    def __init__(self, exchange_client):
        self.exchange_client = exchange_client
        self.iceberg_manager = IcebergOrderManager(exchange_client)
        self.twap_executor = TWAPExecutor(exchange_client)
        self.vwap_executor = VWAPExecutor(exchange_client)
        
        # Performance tracking
        self.strategy_performance = {
            'market': {'executions': 0, 'avg_slippage': 0.0},
            'limit': {'executions': 0, 'avg_slippage': 0.0},
            'iceberg': {'executions': 0, 'avg_slippage': 0.0},
            'twap': {'executions': 0, 'avg_slippage': 0.0},
            'vwap': {'executions': 0, 'avg_slippage': 0.0}
        }
    
    async def submit_smart_order(self, order: AdvancedOrder) -> str:
        """Submit order with intelligent routing"""
        if order.order_type != OrderType.SMART:
            return await self._submit_standard_order(order)
        
        # Analyze market conditions
        market_analysis = await self._analyze_market_conditions(order.symbol)
        
        # Select optimal execution strategy
        optimal_strategy = self._select_execution_strategy(order, market_analysis)
        
        # Execute with selected strategy
        order.order_type = optimal_strategy
        
        logger.info(f"🧠 Smart routing selected: {optimal_strategy.value} for {order.symbol}")
        
        return await self._execute_with_strategy(order, optimal_strategy)
    
    async def _analyze_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Analyze current market conditions"""
        try:
            # Get ticker data
            ticker = await self.exchange_client.fetch_ticker(symbol)
            
            # Get order book
            order_book = await self.exchange_client.fetch_order_book(symbol, limit=20)
            
            # Calculate metrics
            spread = (order_book['asks'][0][0] - order_book['bids'][0][0]) / order_book['bids'][0][0]
            
            bid_depth = sum(order[1] for order in order_book['bids'][:5])
            ask_depth = sum(order[1] for order in order_book['asks'][:5])
            liquidity = min(bid_depth, ask_depth)
            
            volatility = abs(ticker['change']) / ticker['open'] if ticker['open'] else 0
            
            return {
                'spread': spread,
                'liquidity': liquidity,
                'volatility': volatility,
                'volume': ticker.get('quoteVolume', 0),
                'price': ticker['last']
            }
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return {'spread': 0.001, 'liquidity': 1000, 'volatility': 0.01, 'volume': 1000000}
    
    def _select_execution_strategy(self, order: AdvancedOrder, market_analysis: Dict) -> OrderType:
        """Select optimal execution strategy based on order and market conditions"""
        spread = market_analysis['spread']
        liquidity = market_analysis['liquidity']
        volatility = market_analysis['volatility']
        volume = market_analysis['volume']
        
        order_size_ratio = order.quantity * market_analysis['price'] / volume
        
        # Decision logic based on conditions
        if order_size_ratio > 0.05:  # Large order (>5% of volume)
            if spread < 0.001 and liquidity > order.quantity * 10:
                return OrderType.ICEBERG  # Good liquidity, use iceberg
            else:
                return OrderType.VWAP  # Poor liquidity, use VWAP
        
        elif volatility > 0.02:  # High volatility
            return OrderType.TWAP  # Spread out execution
        
        elif spread < 0.0005:  # Tight spread
            return OrderType.MARKET  # Fast execution
        
        else:
            return OrderType.LIMIT  # Standard limit order
    
    async def _execute_with_strategy(self, order: AdvancedOrder, strategy: OrderType) -> str:
        """Execute order with selected strategy"""
        if strategy == OrderType.ICEBERG:
            return await self.iceberg_manager.submit_iceberg_order(order)
        elif strategy == OrderType.TWAP:
            return await self.twap_executor.execute_twap_order(order)
        elif strategy == OrderType.VWAP:
            return await self.vwap_executor.execute_vwap_order(order)
        else:
            return await self._submit_standard_order(order)
    
    async def _submit_standard_order(self, order: AdvancedOrder) -> str:
        """Submit standard market or limit order"""
        try:
            order_id = await self.exchange_client.create_order(
                symbol=order.symbol,
                type=order.order_type.value,
                side=order.side.value,
                amount=order.quantity,
                price=order.price
            )
            
            order.order_id = order_id
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            
            return order_id
            
        except Exception as e:
            logger.error(f"Standard order submission failed: {e}")
            order.status = OrderStatus.REJECTED
            raise

# Global advanced order system
advanced_order_system = None

def initialize_advanced_order_system(exchange_client):
    """Initialize the advanced order system"""
    global advanced_order_system
    advanced_order_system = SmartOrderRouter(exchange_client)
    return advanced_order_system
