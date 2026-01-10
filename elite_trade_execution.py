#!/usr/bin/env python3
"""
⚡ ELITE TRADE EXECUTION ENGINE
Advanced order execution with smart algorithms and optimal timing
"""

import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
import logging
import json
import os

logger = logging.getLogger(__name__)

_REAL_TRADING_ENABLED = str(os.getenv('REAL_TRADING', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
_STRICT_REAL_DATA = str(os.getenv('STRICT_REAL_DATA', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
ALLOW_SIMULATED_FEATURES = (
    str(os.getenv('ALLOW_SIMULATED_FEATURES', '0') or '0').strip().lower() in ['1', 'true', 'yes', 'on']
    and not _REAL_TRADING_ENABLED
    and not _STRICT_REAL_DATA
)


def _ensure_simulated_allowed(feature_name: str):
    if not ALLOW_SIMULATED_FEATURES:
        raise RuntimeError(
            f"{feature_name} is simulation-only and is disabled in REAL_TRADING/STRICT_REAL_DATA. "
            f"Refusing to simulate execution."
        )

@dataclass
class ExecutionOrder:
    """Advanced execution order"""
    order_id: str
    symbol: str
    side: str  # 'BUY', 'SELL'
    quantity: float
    order_type: str  # 'MARKET', 'LIMIT', 'TWAP', 'VWAP', 'ICEBERG'
    target_price: Optional[float]
    time_in_force: str  # 'GTC', 'IOC', 'FOK'
    execution_strategy: str
    priority: int  # 1-10, 10 being highest
    max_slippage: float
    execution_window: timedelta
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Trade execution result"""
    order_id: str
    status: str  # 'filled', 'partially_filled', 'failed', 'cancelled'
    filled_quantity: float  # Also accessible as executed_quantity for compatibility
    fill_price: float  # Also accessible as executed_price for compatibility
    commission: float  # Also accessible as total_fees for compatibility
    slippage: float
    execution_time: float  # in seconds
    strategy_used: str  # Also accessible as algorithm_used for compatibility
    execution_quality: float  # 0-100 execution quality score
    market_impact: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Compatibility properties
    @property
    def executed_quantity(self) -> float:
        return self.filled_quantity
    
    @property
    def executed_price(self) -> float:
        return self.fill_price
    
    @property
    def total_fees(self) -> float:
        return self.commission
    
    @property
    def algorithm_used(self) -> str:
        return self.strategy_used
    
    @property
    def quality_score(self) -> float:
        return self.execution_quality / 100.0

class EliteTradeExecutionEngine:
    """⚡ Elite trade execution with advanced algorithms"""
    
    def __init__(self, capital: float = 1000.0, max_position_size: float = 300.0, risk_per_trade: float = 0.02):
        _ensure_simulated_allowed('EliteTradeExecutionEngine')
        self.capital = capital
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        
        self.pending_orders = {}
        self.execution_history = deque(maxlen=1000)
        self.execution_stats = defaultdict(lambda: {'count': 0, 'avg_slippage': 0, 'success_rate': 0})
        
        # Execution algorithms
        self.algorithms = {
            'SMART': SmartOrderRouter(),
            'TWAP': TWAPExecutor(),
            'VWAP': VWAPExecutor(),
            'ICEBERG': IcebergExecutor(),
            'SNIPER': SniperExecutor(),
            'STEALTH': StealthExecutor()
        }
        
        # Market monitoring
        self.market_monitor = MarketConditionMonitor()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.timing_optimizer = ExecutionTimingOptimizer()
        
        # Performance tracking
        self.total_orders = 0
        self.successful_executions = 0
        self.total_slippage = 0.0
        self.avg_execution_time = 0.0
        
        print("⚡ Elite Trade Execution Engine initialized!")
        print(f"   💰 Capital: ${capital:.2f}")
        print(f"   📊 Max Position: ${max_position_size:.2f}")
        print(f"   🛡️ Risk per Trade: {risk_per_trade:.1%}")
        print("   🎯 Features: Smart routing, TWAP/VWAP, stealth execution")
        print("   📊 Goal: Minimize slippage, maximize execution quality")
    
    async def execute_order(self, order: ExecutionOrder, market_data: Dict) -> ExecutionResult:
        """Execute order with optimal algorithm"""
        try:
            self.total_orders += 1
            start_time = datetime.now()
            
            print(f"   ⚡ Executing {order.side} {order.quantity} {order.symbol} - Strategy: {order.execution_strategy}")
            
            # Add to pending orders
            self.pending_orders[order.order_id] = order
            
            # Analyze market conditions
            market_conditions = await self.market_monitor.analyze_conditions(order.symbol, market_data)
            
            # Select optimal execution algorithm
            algorithm = await self._select_execution_algorithm(order, market_conditions)
            
            # Execute using selected algorithm
            execution_result = await algorithm.execute(order, market_data, market_conditions)
            
            # Calculate execution metrics
            execution_time_delta = datetime.now() - start_time
            execution_result.execution_time = execution_time_delta.total_seconds()  # Convert to float seconds
            execution_result.strategy_used = algorithm.__class__.__name__
            execution_result.execution_quality = self._calculate_execution_quality(execution_result, order) * 100  # Convert to 0-100 scale
            
            # Update statistics
            self._update_execution_stats(execution_result, order)
            
            # Remove from pending
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]
            
            # Record execution
            self.execution_history.append(execution_result)
            
            if execution_result.status == 'filled':
                self.successful_executions += 1
                print(f"   ✅ Order executed - Price: ${execution_result.fill_price:.6f}, "
                      f"Slippage: {execution_result.slippage:.3%}, Quality: {execution_result.execution_quality:.1f}%")
            else:
                print(f"   ⚠️ Order {execution_result.status} - {order.order_id}")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return ExecutionResult(
                order_id=order.order_id,
                status='failed',
                filled_quantity=0,
                fill_price=0,
                commission=0,
                slippage=0,
                execution_time=0.0,
                strategy_used='ERROR',
                execution_quality=0.0
            )
    
    async def _select_execution_algorithm(self, order: ExecutionOrder, market_conditions: Dict) -> 'BaseExecutionAlgorithm':
        """Select optimal execution algorithm based on conditions"""
        try:
            # Get market characteristics
            volatility = market_conditions.get('volatility', 0.5)
            liquidity = market_conditions.get('liquidity', 0.5)
            spread = market_conditions.get('spread', 0.001)
            volume = market_conditions.get('volume_activity', 1.0)
            
            # Algorithm selection logic
            if order.execution_strategy == 'SMART':
                # Auto-select best algorithm
                if volatility > 0.7 and liquidity < 0.3:
                    # High volatility, low liquidity - use stealth
                    return self.algorithms['STEALTH']
                elif volume > 2.0 and spread < 0.0005:
                    # High volume, tight spread - use VWAP
                    return self.algorithms['VWAP']
                elif order.quantity > 1000:  # Large order
                    # Use TWAP or Iceberg for large orders
                    return self.algorithms['ICEBERG']
                elif volatility < 0.3 and spread < 0.001:
                    # Stable conditions - use sniper for best price
                    return self.algorithms['SNIPER']
                else:
                    # Default to TWAP
                    return self.algorithms['TWAP']
            else:
                # Use specified strategy
                return self.algorithms.get(order.execution_strategy, self.algorithms['SMART'])
            
        except Exception as e:
            logger.error(f"Algorithm selection error: {e}")
            return self.algorithms['TWAP']  # Fallback
    
    def _calculate_execution_quality(self, result: ExecutionResult, order: ExecutionOrder) -> float:
        """Calculate execution quality score (0-1)"""
        try:
            if result.status != 'filled':
                return 0.0
            
            # Slippage score (lower slippage = better)
            slippage_score = max(0, 1 - abs(result.slippage) * 100)  # Penalize high slippage
            
            # Speed score (faster = better for urgent orders)
            target_time = order.execution_window.total_seconds()
            actual_time = result.execution_time  # Now a float in seconds
            speed_score = max(0, 1 - actual_time / target_time) if target_time > 0 else 0.5
            
            # Fee efficiency (lower fees = better)
            fee_score = max(0, 1 - result.commission / (result.filled_quantity * result.fill_price))
            
            # Quantity filled score
            quantity_score = result.executed_quantity / order.quantity
            
            # Weighted combination
            quality_score = (
                slippage_score * 0.4 +
                speed_score * 0.2 +
                fee_score * 0.2 +
                quantity_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Execution quality calculation error: {e}")
            return 0.5
    
    def _update_execution_stats(self, result: ExecutionResult, order: ExecutionOrder):
        """Update execution statistics"""
        try:
            algorithm = result.algorithm_used
            stats = self.execution_stats[algorithm]
            
            # Update counts
            stats['count'] += 1
            
            # Update slippage
            old_avg_slippage = stats['avg_slippage']
            stats['avg_slippage'] = (old_avg_slippage * (stats['count'] - 1) + result.slippage) / stats['count']
            
            # Update success rate
            if result.status == 'filled':
                successes = stats.get('successes', 0) + 1
                stats['successes'] = successes
                stats['success_rate'] = successes / stats['count']
            
            # Update total slippage
            self.total_slippage += abs(result.slippage)
            
        except Exception as e:
            logger.error(f"Stats update error: {e}")
    
    def get_execution_stats(self) -> Dict:
        """Get comprehensive execution statistics"""
        try:
            if self.total_orders == 0:
                return {'total_orders': 0, 'success_rate': 0, 'avg_slippage': 0}
            
            overall_success_rate = self.successful_executions / self.total_orders
            avg_slippage = self.total_slippage / self.total_orders
            
            # Recent performance (last 50 orders)
            recent_executions = list(self.execution_history)[-50:]
            recent_quality = np.mean([exec.quality_score for exec in recent_executions]) if recent_executions else 0
            
            return {
                'total_orders': self.total_orders,
                'successful_executions': self.successful_executions,
                'overall_success_rate': overall_success_rate,
                'avg_slippage': avg_slippage,
                'recent_avg_quality': recent_quality,
                'algorithm_stats': dict(self.execution_stats),
                'pending_orders': len(self.pending_orders)
            }
            
        except Exception as e:
            logger.error(f"Execution stats error: {e}")
            return {}

class BaseExecutionAlgorithm:
    """Base class for execution algorithms"""
    
    async def execute(self, order: ExecutionOrder, market_data: Dict, conditions: Dict) -> ExecutionResult:
        """Execute order - to be implemented by subclasses"""
        raise NotImplementedError

class SmartOrderRouter(BaseExecutionAlgorithm):
    """🧠 Smart order routing algorithm"""
    
    async def execute(self, order: ExecutionOrder, market_data: Dict, conditions: Dict) -> ExecutionResult:
        """Smart order execution with dynamic routing"""
        try:
            _ensure_simulated_allowed('SmartOrderRouter')
            # Simulate smart execution
            target_price = order.target_price or market_data.get('current_price', 0)
            
            # Smart routing adjustments
            market_impact = self._estimate_market_impact(order, conditions)
            optimal_price = target_price * (1 + market_impact * (1 if order.side == 'BUY' else -1))
            
            # Calculate slippage
            slippage = (optimal_price - target_price) / target_price if target_price > 0 else 0
            
            # Simulate execution
            executed_quantity = order.quantity
            executed_price = optimal_price
            fees = executed_quantity * executed_price * 0.001  # 0.1% fee
            
            return ExecutionResult(
                order_id=order.order_id,
                status='filled',
                filled_quantity=executed_quantity,
                fill_price=executed_price,
                commission=fees,
                slippage=slippage,
                execution_time=2.0,
                strategy_used='SmartOrderRouter',
                execution_quality=85.0
            )
            
        except Exception as e:
            logger.error(f"Smart router execution error: {e}")
            return self._failed_execution(order)
    
    def _estimate_market_impact(self, order: ExecutionOrder, conditions: Dict) -> float:
        """Estimate market impact of order"""
        try:
            liquidity = conditions.get('liquidity', 0.5)
            volume_ratio = order.quantity / conditions.get('avg_volume', 1000)
            
            # Market impact increases with order size and decreases with liquidity
            impact = volume_ratio * (1 - liquidity) * 0.002  # Base 0.2% impact
            
            return min(0.01, impact)  # Cap at 1%
            
        except Exception as e:
            logger.error(f"Market impact estimation error: {e}")
            return 0.001
    
    def _failed_execution(self, order: ExecutionOrder) -> ExecutionResult:
        """Return failed execution result"""
        return ExecutionResult(
            order_id=order.order_id,
            status='failed',
            filled_quantity=0,
            fill_price=0,
            commission=0,
            slippage=0,
            execution_time=0.0,
            strategy_used=self.__class__.__name__,
            execution_quality=0.0
        )

class TWAPExecutor(BaseExecutionAlgorithm):
    """📊 Time-Weighted Average Price execution"""
    
    async def execute(self, order: ExecutionOrder, market_data: Dict, conditions: Dict) -> ExecutionResult:
        """Execute using TWAP algorithm"""
        try:
            _ensure_simulated_allowed('TWAPExecutor')
            # Simulate TWAP execution over time
            execution_slices = 5  # Split into 5 slices
            slice_size = order.quantity / execution_slices
            
            total_executed = 0
            weighted_price = 0
            total_fees = 0
            execution_times = []
            
            current_price = market_data.get('current_price', order.target_price or 100)
            
            # Execute slices over time
            for i in range(execution_slices):
                # Simulate price movement during execution
                price_drift = np.random.normal(0, 0.0005)  # Small random drift
                slice_price = current_price * (1 + price_drift)
                
                # Execute slice
                slice_executed = slice_size
                slice_fees = slice_executed * slice_price * 0.001
                
                total_executed += slice_executed
                weighted_price += slice_price * slice_executed
                total_fees += slice_fees
                
                # Simulate execution time
                execution_times.append(np.random.uniform(0.5, 2.0))  # 0.5-2 seconds per slice
                
                await asyncio.sleep(0.1)  # Small delay between slices
            
            # Calculate results
            avg_executed_price = weighted_price / total_executed if total_executed > 0 else 0
            target_price = order.target_price or current_price
            slippage = (avg_executed_price - target_price) / target_price if target_price > 0 else 0
            
            return ExecutionResult(
                order_id=order.order_id,
                status='filled',
                filled_quantity=total_executed,
                fill_price=avg_executed_price,
                commission=total_fees,
                slippage=slippage,
                execution_time=sum(execution_times),
                strategy_used='TWAPExecutor',
                execution_quality=80.0
            )
            
        except Exception as e:
            logger.error(f"TWAP execution error: {e}")
            return self._failed_execution(order)

class VWAPExecutor(BaseExecutionAlgorithm):
    """📈 Volume-Weighted Average Price execution"""
    
    async def execute(self, order: ExecutionOrder, market_data: Dict, conditions: Dict) -> ExecutionResult:
        """Execute using VWAP algorithm"""
        try:
            _ensure_simulated_allowed('VWAPExecutor')
            # Get volume profile
            volume_profile = market_data.get('volume_profile', [])
            if not volume_profile:
                # Fallback to TWAP
                twap_executor = TWAPExecutor()
                return await twap_executor.execute(order, market_data, conditions)
            
            # Calculate VWAP execution schedule
            total_market_volume = sum(volume_profile)
            target_participation = 0.1  # 10% participation rate
            
            execution_slices = []
            remaining_quantity = order.quantity
            current_price = market_data.get('current_price', order.target_price or 100)
            
            for volume_slice in volume_profile[:10]:  # Use first 10 volume slices
                participation_qty = volume_slice * target_participation
                slice_qty = min(participation_qty, remaining_quantity)
                
                if slice_qty > 0:
                    execution_slices.append(slice_qty)
                    remaining_quantity -= slice_qty
                
                if remaining_quantity <= 0:
                    break
            
            # Execute slices
            total_executed = 0
            weighted_price = 0
            total_fees = 0
            
            for i, slice_qty in enumerate(execution_slices):
                # Simulate VWAP price
                price_variation = np.random.normal(0, 0.0003)  # Small variation
                slice_price = current_price * (1 + price_variation)
                
                # Execute slice
                slice_fees = slice_qty * slice_price * 0.001
                
                total_executed += slice_qty
                weighted_price += slice_price * slice_qty
                total_fees += slice_fees
                
                await asyncio.sleep(0.05)  # Quick execution
            
            # Handle remaining quantity with market order
            if remaining_quantity > 0:
                remaining_price = current_price * 1.001  # Slight slippage for remainder
                remaining_fees = remaining_quantity * remaining_price * 0.001
                
                total_executed += remaining_quantity
                weighted_price += remaining_price * remaining_quantity
                total_fees += remaining_fees
            
            # Calculate results
            avg_executed_price = weighted_price / total_executed if total_executed > 0 else 0
            target_price = order.target_price or current_price
            slippage = (avg_executed_price - target_price) / target_price if target_price > 0 else 0
            
            return ExecutionResult(
                order_id=order.order_id,
                status='filled',
                filled_quantity=total_executed,
                fill_price=avg_executed_price,
                commission=total_fees,
                slippage=slippage,
                execution_time=len(execution_slices) * 0.5,
                strategy_used='VWAPExecutor',
                execution_quality=82.0
            )
            
        except Exception as e:
            logger.error(f"VWAP execution error: {e}")
            return self._failed_execution(order)

class IcebergExecutor(BaseExecutionAlgorithm):
    """🧊 Iceberg order execution (hide large orders)"""
    
    async def execute(self, order: ExecutionOrder, market_data: Dict, conditions: Dict) -> ExecutionResult:
        """Execute large order as hidden iceberg"""
        try:
            _ensure_simulated_allowed('IcebergExecutor')
            # Break large order into small visible chunks
            visible_size = min(order.quantity * 0.1, 100)  # 10% or 100 units max
            num_icebergs = int(np.ceil(order.quantity / visible_size))
            
            total_executed = 0
            weighted_price = 0
            total_fees = 0
            current_price = market_data.get('current_price', order.target_price or 100)
            
            print(f"      🧊 Iceberg execution: {num_icebergs} chunks of {visible_size:.2f}")
            
            for i in range(num_icebergs):
                # Calculate chunk size
                chunk_size = min(visible_size, order.quantity - total_executed)
                
                if chunk_size <= 0:
                    break
                
                # Random timing between chunks to avoid detection
                await asyncio.sleep(np.random.uniform(1, 5))
                
                # Execute chunk with minimal market impact
                price_impact = chunk_size / conditions.get('avg_volume', 1000) * 0.0005
                chunk_price = current_price * (1 + price_impact * (1 if order.side == 'BUY' else -1))
                
                # Add small random variation
                chunk_price *= (1 + np.random.normal(0, 0.0002))
                
                chunk_fees = chunk_size * chunk_price * 0.001
                
                total_executed += chunk_size
                weighted_price += chunk_price * chunk_size
                total_fees += chunk_fees
                
                # Update current price slightly
                current_price = chunk_price
            
            # Calculate results
            avg_executed_price = weighted_price / total_executed if total_executed > 0 else 0
            target_price = order.target_price or market_data.get('current_price', 100)
            slippage = (avg_executed_price - target_price) / target_price if target_price > 0 else 0
            
            return ExecutionResult(
                order_id=order.order_id,
                status='filled',
                filled_quantity=total_executed,
                fill_price=avg_executed_price,
                commission=total_fees,
                slippage=slippage,
                execution_time=num_icebergs * 3.0,
                strategy_used='IcebergExecutor',
                execution_quality=78.0
            )
            
        except Exception as e:
            logger.error(f"Iceberg execution error: {e}")
            return self._failed_execution(order)

class SniperExecutor(BaseExecutionAlgorithm):
    """🎯 Sniper execution for optimal timing"""
    
    async def execute(self, order: ExecutionOrder, market_data: Dict, conditions: Dict) -> ExecutionResult:
        """Execute with sniper precision timing"""
        try:
            _ensure_simulated_allowed('SniperExecutor')
            # Wait for optimal execution moment
            optimal_moment = await self._wait_for_optimal_moment(order, market_data, conditions)
            
            if optimal_moment:
                # Execute at optimal price
                execution_price = optimal_moment['price']
                slippage = optimal_moment['slippage']
                quality_bonus = 0.1  # Bonus for optimal timing
            else:
                # Timeout - execute at market
                execution_price = market_data.get('current_price', order.target_price or 100)
                target_price = order.target_price or execution_price
                slippage = (execution_price - target_price) / target_price if target_price > 0 else 0
                quality_bonus = 0
            
            fees = order.quantity * execution_price * 0.001
            
            return ExecutionResult(
                order_id=order.order_id,
                status='filled',
                filled_quantity=order.quantity,
                fill_price=execution_price,
                commission=fees,
                slippage=slippage,
                execution_time=5.0,
                strategy_used='SniperExecutor',
                execution_quality=(90.0 + quality_bonus * 100)
            )
            
        except Exception as e:
            logger.error(f"Sniper execution error: {e}")
            return self._failed_execution(order)
    
    async def _wait_for_optimal_moment(self, order: ExecutionOrder, market_data: Dict, conditions: Dict) -> Optional[Dict]:
        """Wait for optimal execution moment"""
        try:
            _ensure_simulated_allowed('SniperExecutor')
            target_price = order.target_price or market_data.get('current_price', 100)
            wait_time = 0
            max_wait = 30  # Maximum 30 seconds
            
            while wait_time < max_wait:
                await asyncio.sleep(0.5)
                wait_time += 0.5
                
                # Simulate price movement
                current_price = target_price * (1 + np.random.normal(0, 0.001))
                
                # Check if price is favorable
                if order.side == 'BUY' and current_price <= target_price * 0.999:
                    # Good buying opportunity
                    return {
                        'price': current_price,
                        'slippage': (current_price - target_price) / target_price,
                        'timing_quality': 0.9
                    }
                elif order.side == 'SELL' and current_price >= target_price * 1.001:
                    # Good selling opportunity
                    return {
                        'price': current_price,
                        'slippage': (current_price - target_price) / target_price,
                        'timing_quality': 0.9
                    }
            
            return None  # Timeout
            
        except Exception as e:
            logger.error(f"Optimal moment detection error: {e}")
            return None

class StealthExecutor(BaseExecutionAlgorithm):
    """🥷 Stealth execution to avoid detection"""
    
    async def execute(self, order: ExecutionOrder, market_data: Dict, conditions: Dict) -> ExecutionResult:
        """Execute with stealth to minimize market impact"""
        try:
            _ensure_simulated_allowed('StealthExecutor')
            # Ultra-small slices with random timing
            num_slices = max(10, int(order.quantity / 10))  # At least 10 slices
            slice_size = order.quantity / num_slices
            
            total_executed = 0
            weighted_price = 0
            total_fees = 0
            current_price = market_data.get('current_price', order.target_price or 100)
            
            print(f"      🥷 Stealth execution: {num_slices} micro-slices")
            
            for i in range(num_slices):
                # Random delay between slices
                delay = np.random.uniform(0.5, 3.0)
                await asyncio.sleep(delay)
                
                # Minimal market impact
                micro_impact = slice_size / conditions.get('avg_volume', 10000) * 0.0001
                slice_price = current_price * (1 + micro_impact * (1 if order.side == 'BUY' else -1))
                
                # Add tiny random variation
                slice_price *= (1 + np.random.normal(0, 0.0001))
                
                slice_fees = slice_size * slice_price * 0.001
                
                total_executed += slice_size
                weighted_price += slice_price * slice_size
                total_fees += slice_fees
                
                # Update price slightly
                current_price = slice_price
            
            # Calculate results
            avg_executed_price = weighted_price / total_executed if total_executed > 0 else 0
            target_price = order.target_price or market_data.get('current_price', 100)
            slippage = (avg_executed_price - target_price) / target_price if target_price > 0 else 0
            
            return ExecutionResult(
                order_id=order.order_id,
                status='filled',
                filled_quantity=total_executed,
                fill_price=avg_executed_price,
                commission=total_fees,
                slippage=slippage,
                execution_time=num_slices * 2.0,
                strategy_used='StealthExecutor',
                execution_quality=75.0  # Lower speed, but minimal impact
            )
            
        except Exception as e:
            logger.error(f"Stealth execution error: {e}")
            return self._failed_execution(order)

class MarketConditionMonitor:
    """📊 Real-time market condition monitoring"""
    
    def __init__(self):
        self.condition_history = deque(maxlen=100)
        self.alert_thresholds = {
            'volatility_spike': 0.8,
            'liquidity_drought': 0.2,
            'spread_widening': 0.005
        }
    
    async def analyze_conditions(self, symbol: str, market_data: Dict) -> Dict:
        """Analyze current market conditions for execution"""
        try:
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [])
            
            if len(prices) < 10:
                return {'volatility': 0.5, 'liquidity': 0.5, 'spread': 0.001, 'volume_activity': 1.0}
            
            # Calculate volatility
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            normalized_volatility = min(1.0, volatility * 100)
            
            # Calculate liquidity proxy
            recent_volumes = volumes[-10:] if volumes else [1] * 10
            avg_volume = np.mean(recent_volumes)
            volume_consistency = 1 - (np.std(recent_volumes) / (avg_volume + 1))
            liquidity = max(0.1, min(1.0, volume_consistency))
            
            # Estimate spread
            if len(prices) >= 2:
                price_diff = abs(prices[-1] - prices[-2]) / prices[-1]
                spread = max(0.0001, min(0.01, price_diff))
            else:
                spread = 0.001
            
            # Volume activity
            historical_volume = np.mean(volumes) if volumes else avg_volume
            volume_activity = avg_volume / historical_volume if historical_volume > 0 else 1.0
            
            conditions = {
                'volatility': normalized_volatility,
                'liquidity': liquidity,
                'spread': spread,
                'volume_activity': volume_activity,
                'avg_volume': avg_volume,
                'price_stability': 1 - normalized_volatility
            }
            
            # Record conditions
            self.condition_history.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'conditions': conditions
            })
            
            return conditions
            
        except Exception as e:
            logger.error(f"Market condition analysis error: {e}")
            return {'volatility': 0.5, 'liquidity': 0.5, 'spread': 0.001, 'volume_activity': 1.0}

class LiquidityAnalyzer:
    """💧 Market liquidity analysis"""
    
    def __init__(self):
        self.liquidity_history = deque(maxlen=200)
    
    async def analyze_liquidity(self, symbol: str, market_data: Dict) -> Dict:
        """Analyze market liquidity for optimal execution"""
        try:
            # Get order book data (simulated)
            bid_depth = market_data.get('bid_depth', 1000)
            ask_depth = market_data.get('ask_depth', 1000)
            spread = market_data.get('spread', 0.001)
            
            # Calculate liquidity metrics
            total_depth = bid_depth + ask_depth
            depth_imbalance = abs(bid_depth - ask_depth) / total_depth
            
            # Liquidity score
            liquidity_score = (
                min(1.0, total_depth / 5000) * 0.4 +  # Depth component
                (1 - depth_imbalance) * 0.3 +  # Balance component
                max(0, 1 - spread * 1000) * 0.3  # Spread component
            )
            
            liquidity_analysis = {
                'liquidity_score': liquidity_score,
                'total_depth': total_depth,
                'depth_imbalance': depth_imbalance,
                'spread': spread,
                'execution_capacity': total_depth * 0.1,  # Safe execution size
                'recommended_slice_size': min(total_depth * 0.02, 50)  # 2% of depth
            }
            
            self.liquidity_history.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'analysis': liquidity_analysis
            })
            
            return liquidity_analysis
            
        except Exception as e:
            logger.error(f"Liquidity analysis error: {e}")
            return {'liquidity_score': 0.5, 'execution_capacity': 100, 'recommended_slice_size': 10}

class ExecutionTimingOptimizer:
    """⏰ Optimal execution timing"""
    
    def __init__(self):
        self.timing_patterns = deque(maxlen=1000)
        self.optimal_windows = {
            'volatility_low': (9, 11),  # UTC hours
            'volume_high': (13, 17),
            'spread_tight': (14, 16)
        }
    
    async def find_optimal_timing(self, order: ExecutionOrder, market_conditions: Dict) -> Dict:
        """Find optimal execution timing"""
        try:
            current_hour = datetime.now().hour
            
            # Analyze current timing quality
            timing_factors = {
                'volatility_timing': self._analyze_volatility_timing(current_hour, market_conditions),
                'volume_timing': self._analyze_volume_timing(current_hour, market_conditions),
                'spread_timing': self._analyze_spread_timing(current_hour, market_conditions)
            }
            
            # Calculate overall timing score
            timing_score = np.mean(list(timing_factors.values()))
            
            # Recommend delay if timing is poor
            if timing_score < 0.4 and order.priority < 8:
                delay_recommendation = self._calculate_optimal_delay(timing_factors)
            else:
                delay_recommendation = 0
            
            return {
                'timing_score': timing_score,
                'current_quality': timing_score,
                'recommended_delay': delay_recommendation,
                'timing_factors': timing_factors,
                'execute_immediately': timing_score >= 0.6 or order.priority >= 8
            }
            
        except Exception as e:
            logger.error(f"Timing optimization error: {e}")
            return {'timing_score': 0.5, 'execute_immediately': True, 'recommended_delay': 0}
    
    def _analyze_volatility_timing(self, hour: int, conditions: Dict) -> float:
        """Analyze volatility timing quality"""
        try:
            volatility = conditions.get('volatility', 0.5)
            
            # Lower volatility is better for execution
            volatility_score = 1 - volatility
            
            # Time-based adjustments
            if hour in range(9, 11):  # Typically lower volatility
                volatility_score *= 1.1
            elif hour in range(21, 23):  # Often higher volatility
                volatility_score *= 0.9
            
            return max(0, min(1.0, volatility_score))
            
        except Exception as e:
            logger.error(f"Volatility timing analysis error: {e}")
            return 0.5
    
    def _analyze_volume_timing(self, hour: int, conditions: Dict) -> float:
        """Analyze volume timing quality"""
        try:
            volume_activity = conditions.get('volume_activity', 1.0)
            
            # Higher volume is better for execution
            volume_score = min(1.0, volume_activity / 2)
            
            # Time-based adjustments
            if hour in range(13, 17):  # Typically higher volume
                volume_score *= 1.1
            elif hour in range(0, 6):  # Often lower volume
                volume_score *= 0.8
            
            return max(0, min(1.0, volume_score))
            
        except Exception as e:
            logger.error(f"Volume timing analysis error: {e}")
            return 0.5
    
    def _analyze_spread_timing(self, hour: int, conditions: Dict) -> float:
        """Analyze spread timing quality"""
        try:
            spread = conditions.get('spread', 0.001)
            
            # Tighter spread is better
            spread_score = max(0, 1 - spread * 1000)
            
            # Time-based adjustments
            if hour in range(14, 16):  # Typically tighter spreads
                spread_score *= 1.1
            elif hour in range(22, 2):  # Often wider spreads
                spread_score *= 0.9
            
            return max(0, min(1.0, spread_score))
            
        except Exception as e:
            logger.error(f"Spread timing analysis error: {e}")
            return 0.5
    
    def _calculate_optimal_delay(self, timing_factors: Dict) -> int:
        """Calculate optimal delay in seconds"""
        try:
            # Find worst timing factor
            worst_factor = min(timing_factors.values())
            
            if worst_factor < 0.2:
                return 300  # 5 minutes
            elif worst_factor < 0.4:
                return 120  # 2 minutes
            else:
                return 30  # 30 seconds
            
        except Exception as e:
            logger.error(f"Optimal delay calculation error: {e}")
            return 0

class OrderManagementSystem:
    """📋 Advanced order management system"""
    
    def __init__(self):
        self.active_orders = {}
        self.order_queue = deque()
        self.execution_rules = {
            'max_concurrent_orders': 5,
            'max_order_age': timedelta(minutes=30),
            'auto_cancel_threshold': 0.05  # Cancel if 5% adverse move
        }
    
    async def submit_order(self, order: ExecutionOrder) -> bool:
        """Submit order to execution system"""
        try:
            # Validate order
            if not self._validate_order(order):
                print(f"   ❌ Order validation failed: {order.order_id}")
                return False
            
            # Check capacity
            if len(self.active_orders) >= self.execution_rules['max_concurrent_orders']:
                print(f"   ⚠️ Order queue full, queuing order: {order.order_id}")
                self.order_queue.append(order)
                return True
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            
            print(f"   ✅ Order submitted: {order.side} {order.quantity} {order.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Order submission error: {e}")
            return False
    
    def _validate_order(self, order: ExecutionOrder) -> bool:
        """Validate order parameters"""
        try:
            # Basic validation
            if order.quantity <= 0:
                return False
            if order.side not in ['BUY', 'SELL']:
                return False
            if order.max_slippage < 0 or order.max_slippage > 0.1:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False
    
    async def manage_active_orders(self, market_data: Dict):
        """Manage active orders and queue"""
        try:
            current_time = datetime.now()
            orders_to_cancel = []
            
            # Check for stale orders
            for order_id, order in self.active_orders.items():
                order_age = current_time - order.created_at
                if order_age > self.execution_rules['max_order_age']:
                    orders_to_cancel.append(order_id)
            
            # Cancel stale orders
            for order_id in orders_to_cancel:
                await self._cancel_order(order_id, "Timeout")
            
            # Process queue if capacity available
            while (len(self.active_orders) < self.execution_rules['max_concurrent_orders'] and 
                   self.order_queue):
                next_order = self.order_queue.popleft()
                self.active_orders[next_order.order_id] = next_order
                print(f"   📋 Moved order from queue to active: {next_order.order_id}")
            
        except Exception as e:
            logger.error(f"Order management error: {e}")
    
    async def _cancel_order(self, order_id: str, reason: str):
        """Cancel order"""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                del self.active_orders[order_id]
                print(f"   ❌ Cancelled order {order_id}: {reason}")
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")

# Global execution engine (simulation-only)
execution_engine = EliteTradeExecutionEngine() if ALLOW_SIMULATED_FEATURES else None
order_manager = OrderManagementSystem() if ALLOW_SIMULATED_FEATURES else None

# Helper function to create orders
def create_execution_order(
    symbol: str,
    side: str,
    quantity: float,
    strategy: str = 'SMART',
    target_price: Optional[float] = None,
    max_slippage: float = 0.005,
    priority: int = 5
) -> ExecutionOrder:
    """Create execution order with smart defaults"""

    order_id = f"{symbol}_{side}_{datetime.now().strftime('%H%M%S%f')}"
    
    return ExecutionOrder(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type='LIMIT' if target_price else 'MARKET',
        target_price=target_price,
        time_in_force='GTC',
        execution_strategy=strategy,
        priority=priority,
        max_slippage=max_slippage,
        execution_window=timedelta(minutes=10)
    )

if ALLOW_SIMULATED_FEATURES:
    print("⚡ Elite Trade Execution Engine ready!")
    print("   🎯 Features: Smart routing, TWAP/VWAP, stealth execution")
    print("   📊 Goal: Minimize slippage, maximize execution quality")
