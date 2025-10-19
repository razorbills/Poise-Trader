"""
Grid Trading Strategy for Poise Trader

A systematic trading strategy that places buy and sell orders at predefined price levels
creating a "grid" of orders. Profits from price volatility within a range.

Features:
- Dynamic grid creation based on price ranges
- Automatic grid rebalancing
- Configurable grid spacing and levels
- Risk management with stop losses
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

from ..framework.base_classes import MarketData, TradeSignal, OrderSide
from .base_strategy import AdvancedStrategy, PositionType, SignalStrength


class GridDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class GridLevel:
    """Represents a single grid level"""
    price: Decimal
    is_buy: bool  # True for buy orders, False for sell orders
    is_filled: bool = False
    order_size: Decimal = Decimal('0')
    created_time: int = 0


@dataclass
class GridState:
    """State tracking for grid strategy"""
    symbol: str
    direction: GridDirection
    center_price: Decimal
    grid_spacing: Decimal
    num_levels: int
    total_invested: Decimal = Decimal('0')
    levels: List[GridLevel] = None
    last_rebalance: int = 0
    profit_realized: Decimal = Decimal('0')
    
    def __post_init__(self):
        if self.levels is None:
            self.levels = []


class GridStrategy(AdvancedStrategy):
    """
    Grid Trading Strategy
    
    Places orders at predefined price levels to profit from range-bound price movement.
    Automatically rebalances the grid based on market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Grid-specific parameters
        self.grid_spacing_pct = Decimal(str(config.get('grid_spacing_pct', 0.01)))  # 1% spacing
        self.num_grid_levels = config.get('num_grid_levels', 10)  # 5 levels above/below
        self.max_grid_deviation = Decimal(str(config.get('max_grid_deviation', 0.05)))  # 5% max deviation
        self.rebalance_threshold = Decimal(str(config.get('rebalance_threshold', 0.02)))  # 2% rebalance trigger
        self.order_size_base = Decimal(str(config.get('order_size_base', 100)))  # Base order size
        self.profit_target = Decimal(str(config.get('profit_target', 0.01)))  # 1% profit per trade
        
        # Grid state tracking
        self.grid_states: Dict[str, GridState] = {}
        self.filled_orders: Dict[str, List[GridLevel]] = {}
        
        # Market analysis
        self.price_ranges: Dict[str, Tuple[Decimal, Decimal]] = {}
        self.volatility: Dict[str, Decimal] = {}
        
    async def _custom_initialize(self):
        """Initialize grid states for each symbol"""
        for symbol in self.symbols:
            # Initialize with placeholder values - will be updated with market data
            self.grid_states[symbol] = GridState(
                symbol=symbol,
                direction=GridDirection.NEUTRAL,
                center_price=Decimal('0'),
                grid_spacing=self.grid_spacing_pct,
                num_levels=self.num_grid_levels
            )
            self.filled_orders[symbol] = []
            self.price_ranges[symbol] = (Decimal('0'), Decimal('0'))
        
        self.logger.info("Grid Strategy initialized:")
        self.logger.info("- Grid spacing: %s%%", self.grid_spacing_pct * 100)
        self.logger.info("- Grid levels: %d", self.num_grid_levels)
        self.logger.info("- Max deviation: %s%%", self.max_grid_deviation * 100)
        self.logger.info("- Base order size: %s %s", self.order_size_base, self.portfolio.base_currency)
    
    async def _generate_signal(self, data: MarketData) -> Optional[TradeSignal]:
        """Generate grid trading signals"""
        symbol = data.symbol
        current_price = data.price
        
        grid_state = self.grid_states[symbol]
        
        # Initialize or update grid if needed
        if not grid_state.levels or self._should_rebalance_grid(symbol, current_price):
            await self._setup_grid(symbol, current_price)
        
        # Check for grid level triggers
        signal = self._check_grid_triggers(symbol, current_price)
        
        if signal:
            self.logger.info("Grid signal for %s: %s at %s (confidence: %.2f)", 
                           symbol, signal.action.value, signal.price, signal.confidence)
        
        return signal
    
    async def _setup_grid(self, symbol: str, current_price: Decimal):
        """Setup or rebalance the trading grid"""
        grid_state = self.grid_states[symbol]
        
        # Calculate grid center (can be current price or adjusted based on trend)
        center_price = current_price
        
        # Adjust for trend if we have enough data
        if len(self.market_data[symbol]) > 20:
            recent_prices = [d.price for d in self.market_data[symbol][-20:]]
            trend_factor = self._calculate_trend_factor(recent_prices)
            # Slightly adjust center based on trend
            center_price = current_price * (Decimal('1') + trend_factor * Decimal('0.005'))
        
        grid_state.center_price = center_price
        grid_state.last_rebalance = int(time.time())
        
        # Clear existing levels
        grid_state.levels = []
        
        # Create buy levels (below center)
        for i in range(1, self.num_grid_levels // 2 + 1):
            buy_price = center_price * (Decimal('1') - self.grid_spacing_pct * i)
            order_size = self._calculate_order_size(symbol, buy_price, True)
            
            grid_level = GridLevel(
                price=buy_price,
                is_buy=True,
                order_size=order_size,
                created_time=int(time.time())
            )
            grid_state.levels.append(grid_level)
        
        # Create sell levels (above center)
        for i in range(1, self.num_grid_levels // 2 + 1):
            sell_price = center_price * (Decimal('1') + self.grid_spacing_pct * i)
            order_size = self._calculate_order_size(symbol, sell_price, False)
            
            grid_level = GridLevel(
                price=sell_price,
                is_buy=False,
                order_size=order_size,
                created_time=int(time.time())
            )
            grid_state.levels.append(grid_level)
        
        # Sort levels by price
        grid_state.levels.sort(key=lambda x: x.price)
        
        self.logger.info("Grid setup for %s: %d levels around %s", 
                        symbol, len(grid_state.levels), center_price)
    
    def _should_rebalance_grid(self, symbol: str, current_price: Decimal) -> bool:
        """Check if grid needs rebalancing"""
        grid_state = self.grid_states[symbol]
        
        # If no grid exists, need to create one
        if not grid_state.levels:
            return True
        
        # Check if price has moved too far from center
        if grid_state.center_price > 0:
            price_deviation = abs(current_price - grid_state.center_price) / grid_state.center_price
            if price_deviation > self.max_grid_deviation:
                self.logger.info("Grid rebalance triggered for %s: deviation %.2f%%", 
                               symbol, price_deviation * 100)
                return True
        
        # Check time-based rebalance (e.g., every 4 hours)
        time_since_rebalance = int(time.time()) - grid_state.last_rebalance
        if time_since_rebalance > 4 * 3600:  # 4 hours
            return True
        
        return False
    
    def _check_grid_triggers(self, symbol: str, current_price: Decimal) -> Optional[TradeSignal]:
        """Check if current price triggers any grid levels"""
        grid_state = self.grid_states[symbol]
        
        for level in grid_state.levels:
            if level.is_filled:
                continue
            
            # Check for buy level trigger (price at or below buy level)
            if level.is_buy and current_price <= level.price:
                return self._create_grid_signal(symbol, level, current_price)
            
            # Check for sell level trigger (price at or above sell level)
            elif not level.is_buy and current_price >= level.price:
                return self._create_grid_signal(symbol, level, current_price)
        
        return None
    
    def _create_grid_signal(self, symbol: str, level: GridLevel, current_price: Decimal) -> TradeSignal:
        """Create trading signal for grid level"""
        action = OrderSide.BUY if level.is_buy else OrderSide.SELL
        
        # Calculate stop loss for risk management
        if level.is_buy:
            stop_loss = level.price * (Decimal('1') - self.stop_loss_pct)
            take_profit = level.price * (Decimal('1') + self.profit_target)
        else:
            stop_loss = level.price * (Decimal('1') + self.stop_loss_pct)
            take_profit = level.price * (Decimal('1') - self.profit_target)
        
        # Mark level as filled (will be confirmed by execution)
        level.is_filled = True
        
        return TradeSignal(
            symbol=symbol,
            action=action,
            quantity=level.order_size,
            confidence=SignalStrength.STRONG.value,  # Grid signals are systematic
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=int(time.time()),
            strategy_name=self.__class__.__name__,
            metadata={
                'signal_type': 'grid_trigger',
                'grid_level_price': float(level.price),
                'grid_direction': 'buy' if level.is_buy else 'sell',
                'order_size': float(level.order_size),
                'grid_spacing': float(self.grid_spacing_pct * 100)
            }
        )
    
    def _calculate_order_size(self, symbol: str, price: Decimal, is_buy: bool) -> Decimal:
        """Calculate order size for grid level"""
        # Base order size
        base_size = self.order_size_base / price
        
        # Adjust based on available capital
        available_capital = self.portfolio.get_available_balance(self.portfolio.base_currency)
        max_per_order = available_capital / (self.num_grid_levels * 2)  # Distribute across all levels
        
        if is_buy:
            max_size = max_per_order / price
            return min(base_size, max_size)
        else:
            # For sell orders, check available position
            position = self.portfolio.get_position(symbol)
            if position and position.side == PositionType.LONG:
                available_size = position.size / (self.num_grid_levels // 2)  # Distribute sell orders
                return min(base_size, available_size)
            else:
                return base_size  # Will be validated later
    
    def _calculate_trend_factor(self, prices: List[Decimal]) -> Decimal:
        """Calculate trend factor for grid center adjustment"""
        if len(prices) < 2:
            return Decimal('0')
        
        # Simple linear trend calculation
        price_changes = []
        for i in range(1, len(prices)):
            change = (prices[i] - prices[i-1]) / prices[i-1]
            price_changes.append(change)
        
        if price_changes:
            avg_change = sum(price_changes) / len(price_changes)
            # Limit trend factor to reasonable range
            return max(Decimal('-0.1'), min(Decimal('0.1'), avg_change))
        
        return Decimal('0')
    
    async def _custom_validate_signal(self, signal: TradeSignal) -> bool:
        """Custom validation for grid signals"""
        # Check if we have sufficient capital for buy orders
        if signal.action == OrderSide.BUY:
            required_amount = signal.quantity * signal.price
            available = self.portfolio.get_available_balance(self.portfolio.base_currency)
            
            if available < required_amount:
                self.logger.debug("Insufficient funds for grid buy: need %s, have %s", 
                                required_amount, available)
                # Mark grid level as not filled so it can be retried
                self._unmark_grid_level(signal.symbol, signal.price, True)
                return False
        
        # Check if we have sufficient position for sell orders
        elif signal.action == OrderSide.SELL:
            position = self.portfolio.get_position(signal.symbol)
            if not position or position.size < signal.quantity:
                self.logger.debug("Insufficient position for grid sell: need %s, have %s", 
                                signal.quantity, position.size if position else 0)
                self._unmark_grid_level(signal.symbol, signal.price, False)
                return False
        
        return True
    
    def _unmark_grid_level(self, symbol: str, price: Decimal, is_buy: bool):
        """Unmark a grid level if validation fails"""
        grid_state = self.grid_states[symbol]
        
        for level in grid_state.levels:
            if (abs(level.price - price) < Decimal('0.0001') and 
                level.is_buy == is_buy):
                level.is_filled = False
                break
    
    async def _custom_exit_check(self, data: MarketData, position) -> bool:
        """Custom exit conditions for grid strategy"""
        # Grid strategy manages exits through grid levels
        # Additional safety exits can be implemented here
        
        # Example: Exit if price moves beyond maximum deviation
        symbol = data.symbol
        grid_state = self.grid_states[symbol]
        
        if grid_state.center_price > 0:
            deviation = abs(data.price - grid_state.center_price) / grid_state.center_price
            if deviation > self.max_grid_deviation * 2:  # Emergency exit at 2x max deviation
                self.logger.warning("Emergency exit triggered for %s: deviation %.2f%%", 
                                  symbol, deviation * 100)
                return True
        
        return False
    
    def get_grid_summary(self) -> Dict[str, Any]:
        """Get grid-specific summary information"""
        summary = {}
        
        for symbol, grid_state in self.grid_states.items():
            active_buy_levels = sum(1 for level in grid_state.levels if level.is_buy and not level.is_filled)
            active_sell_levels = sum(1 for level in grid_state.levels if not level.is_buy and not level.is_filled)
            filled_levels = sum(1 for level in grid_state.levels if level.is_filled)
            
            current_price = self.last_prices.get(symbol, Decimal('0'))
            position = self.portfolio.get_position(symbol)
            
            summary[symbol] = {
                'center_price': float(grid_state.center_price),
                'current_price': float(current_price),
                'grid_spacing_pct': float(self.grid_spacing_pct * 100),
                'total_levels': len(grid_state.levels),
                'active_buy_levels': active_buy_levels,
                'active_sell_levels': active_sell_levels,
                'filled_levels': filled_levels,
                'position_size': float(position.size) if position else 0,
                'total_invested': float(grid_state.total_invested),
                'profit_realized': float(grid_state.profit_realized),
                'last_rebalance': grid_state.last_rebalance,
                'grid_levels': [
                    {
                        'price': float(level.price),
                        'is_buy': level.is_buy,
                        'is_filled': level.is_filled,
                        'order_size': float(level.order_size)
                    }
                    for level in grid_state.levels
                ]
            }
        
        return summary
    
    def update_grid_parameters(self, **kwargs):
        """Update grid strategy parameters dynamically"""
        if 'grid_spacing_pct' in kwargs:
            self.grid_spacing_pct = Decimal(str(kwargs['grid_spacing_pct']))
            self.logger.info("Updated grid spacing to %s%%", self.grid_spacing_pct * 100)
        
        if 'num_grid_levels' in kwargs:
            self.num_grid_levels = kwargs['num_grid_levels']
            self.logger.info("Updated grid levels to %d", self.num_grid_levels)
        
        if 'max_grid_deviation' in kwargs:
            self.max_grid_deviation = Decimal(str(kwargs['max_grid_deviation']))
            self.logger.info("Updated max deviation to %s%%", self.max_grid_deviation * 100)
        
        if 'order_size_base' in kwargs:
            self.order_size_base = Decimal(str(kwargs['order_size_base']))
            self.logger.info("Updated base order size to %s", self.order_size_base)
        
        # Force grid rebalance for all symbols
        for grid_state in self.grid_states.values():
            grid_state.levels = []  # Clear existing levels to force rebalance
