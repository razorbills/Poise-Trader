"""
Base Strategy Engine for Poise Trader

Provides the foundation for all trading strategies with:
- Portfolio management and position tracking
- Signal generation and validation
- Performance metrics and risk management
- Backtesting support
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import statistics

from ..framework.base_classes import (
    BaseStrategy, MarketData, TradeSignal, Portfolio, 
    OrderSide, OrderType, OrderStatus
)
from ..framework.event_system import EventBus, Event, create_event


class SignalStrength(Enum):
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9


class PositionType(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    side: PositionType
    size: Decimal
    entry_price: Decimal
    current_price: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    entry_time: int = 0
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_price(self, new_price: Decimal):
        """Update current price and calculate unrealized PnL"""
        self.current_price = new_price
        
        if self.side == PositionType.LONG:
            self.unrealized_pnl = (new_price - self.entry_price) * self.size
        elif self.side == PositionType.SHORT:
            self.unrealized_pnl = (self.entry_price - new_price) * self.size
    
    def get_total_pnl(self) -> Decimal:
        """Get total PnL (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class PortfolioManager:
    """Manages portfolio state, balances, and positions"""
    initial_capital: Decimal
    base_currency: str = "USDT"
    
    def __post_init__(self):
        self.balances: Dict[str, Decimal] = {self.base_currency: self.initial_capital}
        self.positions: Dict[str, Position] = {}
        self.locked_balances: Dict[str, Decimal] = {}
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = Decimal('0')
        self.peak_value = self.initial_capital
        self.max_drawdown = Decimal('0')
    
    def get_available_balance(self, currency: str) -> Decimal:
        """Get available balance for a currency"""
        total = self.balances.get(currency, Decimal('0'))
        locked = self.locked_balances.get(currency, Decimal('0'))
        return max(Decimal('0'), total - locked)
    
    def get_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value in base currency"""
        total_value = Decimal('0')
        
        # Add cash balances
        for currency, balance in self.balances.items():
            if currency == self.base_currency:
                total_value += balance
            # Note: In a real implementation, you'd convert to base currency using current prices
        
        # Add unrealized PnL from positions
        for position in self.positions.values():
            total_value += position.get_total_pnl()
        
        return total_value
    
    def update_max_drawdown(self):
        """Update maximum drawdown tracking"""
        current_value = self.get_portfolio_value()
        
        if current_value > self.peak_value:
            self.peak_value = current_value
        else:
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
    
    def can_open_position(self, symbol: str, side: PositionType, size: Decimal, price: Decimal) -> bool:
        """Check if we can open a position"""
        required_capital = size * price
        available = self.get_available_balance(self.base_currency)
        return available >= required_capital
    
    def open_position(self, symbol: str, side: PositionType, size: Decimal, price: Decimal, 
                     stop_loss: Decimal = None, take_profit: Decimal = None) -> bool:
        """Open a new position"""
        if not self.can_open_position(symbol, side, size, price):
            return False
        
        # Close existing position if any
        if symbol in self.positions:
            self.close_position(symbol, price)
        
        # Create new position
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            current_price=price,
            entry_time=int(time.time()),
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = position
        
        # Lock capital
        required_capital = size * price
        if self.base_currency not in self.locked_balances:
            self.locked_balances[self.base_currency] = Decimal('0')
        self.locked_balances[self.base_currency] += required_capital
        
        return True
    
    def close_position(self, symbol: str, exit_price: Decimal) -> Optional[Decimal]:
        """Close a position and realize PnL"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position.update_price(exit_price)
        
        # Calculate realized PnL
        pnl = position.unrealized_pnl
        position.realized_pnl = pnl
        position.unrealized_pnl = Decimal('0')
        
        # Update balances
        self.balances[self.base_currency] += pnl
        
        # Release locked capital
        locked_capital = position.size * position.entry_price
        self.locked_balances[self.base_currency] -= locked_capital
        
        # Update trade statistics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Remove position
        del self.positions[symbol]
        
        self.update_max_drawdown()
        return pnl
    
    def update_position_price(self, symbol: str, price: Decimal):
        """Update position with current market price"""
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
            self.update_max_drawdown()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get portfolio performance metrics"""
        current_value = self.get_portfolio_value()
        total_return = ((current_value - self.initial_capital) / self.initial_capital) * 100
        
        win_rate = 0.0
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
        
        return {
            'initial_capital': float(self.initial_capital),
            'current_value': float(current_value),
            'total_return_pct': float(total_return),
            'max_drawdown_pct': float(self.max_drawdown * 100),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': win_rate,
            'total_fees': float(self.total_fees)
        }


class AdvancedStrategy(BaseStrategy):
    """
    Advanced strategy base class with portfolio management and enhanced features
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Portfolio management
        initial_capital = Decimal(str(config.get('initial_capital', 10000)))
        base_currency = config.get('base_currency', 'USDT')
        self.portfolio = PortfolioManager(initial_capital, base_currency)
        
        # Strategy parameters
        self.symbols = config.get('symbols', [])
        self.risk_per_trade = Decimal(str(config.get('risk_per_trade', 0.02)))  # 2% risk per trade
        self.max_positions = config.get('max_positions', 5)
        self.stop_loss_pct = Decimal(str(config.get('stop_loss_pct', 0.02)))  # 2% stop loss
        self.take_profit_pct = Decimal(str(config.get('take_profit_pct', 0.04)))  # 4% take profit
        
        # Market data storage
        self.market_data: Dict[str, List[MarketData]] = {}
        self.last_prices: Dict[str, Decimal] = {}
        
        # Event bus for communication
        self.event_bus: Optional[EventBus] = None
        
        # Performance tracking
        self.signals_generated = 0
        self.signals_executed = 0
        self.start_time = time.time()
        
    async def initialize(self) -> bool:
        """Initialize the strategy"""
        try:
            self.logger.info("Initializing %s strategy", self.__class__.__name__)
            self.logger.info("Initial capital: %s %s", 
                           self.portfolio.initial_capital, 
                           self.portfolio.base_currency)
            self.logger.info("Symbols: %s", self.symbols)
            self.logger.info("Risk per trade: %s%%", self.risk_per_trade * 100)
            
            # Initialize market data storage
            for symbol in self.symbols:
                self.market_data[symbol] = []
            
            await self._custom_initialize()
            self.is_active = True
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize strategy: %s", e)
            return False
    
    async def _custom_initialize(self):
        """Override this method for custom initialization logic"""
        pass
    
    async def process_market_data(self, data: MarketData) -> Optional[TradeSignal]:
        """Process incoming market data and generate trading signals"""
        if not self.is_active or data.symbol not in self.symbols:
            return None
        
        try:
            # Store market data
            self.market_data[data.symbol].append(data)
            self.last_prices[data.symbol] = data.price
            
            # Keep only recent data (e.g., last 1000 points)
            max_history = self.config.get('max_history', 1000)
            if len(self.market_data[data.symbol]) > max_history:
                self.market_data[data.symbol] = self.market_data[data.symbol][-max_history:]
            
            # Update position prices
            self.portfolio.update_position_price(data.symbol, data.price)
            
            # Check for exit conditions on existing positions
            await self._check_exit_conditions(data)
            
            # Generate new entry signals
            signal = await self._generate_signal(data)
            
            if signal:
                # Validate signal
                if await self._validate_signal(signal):
                    self.signals_generated += 1
                    
                    # Publish signal event
                    if self.event_bus:
                        event = create_event("trading.signal.generated",
                                           strategy=self.__class__.__name__,
                                           signal=signal,
                                           timestamp=time.time())
                        await self.event_bus.publish(event)
                    
                    return signal
            
            return None
            
        except Exception as e:
            self.logger.error("Error processing market data: %s", e)
            return None
    
    @abstractmethod
    async def _generate_signal(self, data: MarketData) -> Optional[TradeSignal]:
        """Generate trading signal based on market data - implement in subclass"""
        pass
    
    async def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate trading signal against risk rules"""
        try:
            # Check position limits
            if len(self.portfolio.positions) >= self.max_positions:
                if signal.symbol not in self.portfolio.positions:
                    self.logger.debug("Max positions reached, ignoring signal for %s", signal.symbol)
                    return False
            
            # Check available capital
            position_size = self._calculate_position_size(signal)
            if position_size <= 0:
                self.logger.debug("Insufficient capital for %s", signal.symbol)
                return False
            
            # Check signal confidence
            min_confidence = self.config.get('min_confidence', 0.6)
            if signal.confidence < min_confidence:
                self.logger.debug("Signal confidence too low: %s", signal.confidence)
                return False
            
            # Custom validation
            return await self._custom_validate_signal(signal)
            
        except Exception as e:
            self.logger.error("Error validating signal: %s", e)
            return False
    
    async def _custom_validate_signal(self, signal: TradeSignal) -> bool:
        """Override for custom signal validation logic"""
        return True
    
    def _calculate_position_size(self, signal: TradeSignal) -> Decimal:
        """Calculate position size based on risk management rules"""
        if not signal.price:
            return Decimal('0')
        
        # Get available capital
        available_capital = self.portfolio.get_available_balance(self.portfolio.base_currency)
        
        # Calculate risk amount (percentage of portfolio)
        portfolio_value = self.portfolio.get_portfolio_value()
        risk_amount = portfolio_value * self.risk_per_trade
        
        # Calculate position size based on stop loss
        if signal.stop_loss and signal.stop_loss != signal.price:
            price_risk = abs(signal.price - signal.stop_loss)
            if price_risk > 0:
                position_size = risk_amount / price_risk
                max_position = available_capital / signal.price * Decimal('0.9')  # Use 90% of available
                return min(position_size, max_position)
        
        # Fallback: use fixed percentage of available capital
        max_position_pct = Decimal('0.2')  # 20% max position size
        return min(available_capital * max_position_pct / signal.price, 
                  available_capital / signal.price * Decimal('0.9'))
    
    async def _check_exit_conditions(self, data: MarketData):
        """Check if any positions should be exited"""
        position = self.portfolio.get_position(data.symbol)
        if not position:
            return
        
        should_exit = False
        exit_reason = ""
        
        # Check stop loss
        if position.stop_loss:
            if ((position.side == PositionType.LONG and data.price <= position.stop_loss) or
                (position.side == PositionType.SHORT and data.price >= position.stop_loss)):
                should_exit = True
                exit_reason = "stop_loss"
        
        # Check take profit
        if position.take_profit:
            if ((position.side == PositionType.LONG and data.price >= position.take_profit) or
                (position.side == PositionType.SHORT and data.price <= position.take_profit)):
                should_exit = True
                exit_reason = "take_profit"
        
        # Custom exit conditions
        if await self._custom_exit_check(data, position):
            should_exit = True
            exit_reason = "custom"
        
        if should_exit:
            await self._execute_exit(data.symbol, data.price, exit_reason)
    
    async def _custom_exit_check(self, data: MarketData, position: Position) -> bool:
        """Override for custom exit conditions"""
        return False
    
    async def _execute_exit(self, symbol: str, price: Decimal, reason: str):
        """Execute position exit"""
        pnl = self.portfolio.close_position(symbol, price)
        if pnl is not None:
            self.logger.info("Closed position %s at %s (reason: %s) - PnL: %s", 
                           symbol, price, reason, pnl)
            
            # Publish exit event
            if self.event_bus:
                event = create_event("trading.position.closed",
                                   strategy=self.__class__.__name__,
                                   symbol=symbol,
                                   exit_price=float(price),
                                   pnl=float(pnl),
                                   reason=reason,
                                   timestamp=time.time())
                await self.event_bus.publish(event)
    
    async def update_portfolio(self, portfolio: Portfolio) -> None:
        """Update strategy based on current portfolio state"""
        # This is called by the main trading engine
        # We manage our own portfolio internally
        pass
    
    def get_required_symbols(self) -> List[str]:
        """Return list of symbols this strategy needs"""
        return self.symbols
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return strategy performance metrics"""
        base_metrics = self.portfolio.get_performance_metrics()
        
        # Add strategy-specific metrics
        runtime = time.time() - self.start_time
        signals_per_hour = 0
        if runtime > 0:
            signals_per_hour = (self.signals_generated / runtime) * 3600
        
        strategy_metrics = {
            'strategy_name': self.__class__.__name__,
            'runtime_seconds': runtime,
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed,
            'signals_per_hour': signals_per_hour,
            'active_positions': len(self.portfolio.positions),
            'symbols_tracked': len(self.symbols)
        }
        
        # Merge metrics
        base_metrics.update(strategy_metrics)
        return base_metrics
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.portfolio.positions.copy()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        return {
            'balances': {k: float(v) for k, v in self.portfolio.balances.items()},
            'locked_balances': {k: float(v) for k, v in self.portfolio.locked_balances.items()},
            'total_value': float(self.portfolio.get_portfolio_value()),
            'positions': {symbol: {
                'side': pos.side.value,
                'size': float(pos.size),
                'entry_price': float(pos.entry_price),
                'current_price': float(pos.current_price),
                'unrealized_pnl': float(pos.unrealized_pnl),
                'realized_pnl': float(pos.realized_pnl)
            } for symbol, pos in self.portfolio.positions.items()}
        }
    
    def add_funds(self, amount: Decimal, currency: str = None):
        """
        Add funds to the strategy portfolio
        
        Args:
            amount: Amount to add
            currency: Currency (defaults to base currency)
        """
        if currency is None:
            currency = self.portfolio.base_currency
        
        if currency not in self.portfolio.balances:
            self.portfolio.balances[currency] = Decimal('0')
        
        self.portfolio.balances[currency] += amount
        self.logger.info("Added %s %s to portfolio. New balance: %s", 
                        amount, currency, self.portfolio.balances[currency])
    
    def withdraw_funds(self, amount: Decimal, currency: str = None) -> bool:
        """
        Withdraw funds from the strategy portfolio
        
        Args:
            amount: Amount to withdraw
            currency: Currency (defaults to base currency)
            
        Returns:
            True if withdrawal successful, False otherwise
        """
        if currency is None:
            currency = self.portfolio.base_currency
        
        available = self.portfolio.get_available_balance(currency)
        if available >= amount:
            self.portfolio.balances[currency] -= amount
            self.logger.info("Withdrew %s %s from portfolio. New balance: %s", 
                           amount, currency, self.portfolio.balances[currency])
            return True
        else:
            self.logger.error("Insufficient funds for withdrawal. Available: %s, Requested: %s", 
                            available, amount)
            return False
