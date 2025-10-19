"""
Dollar Cost Averaging (DCA) Strategy for Poise Trader

A systematic investment strategy that automatically buys assets at regular intervals
regardless of price, helping to average out volatility over time.

Features:
- Regular interval buying (time-based)
- Price deviation triggers (buy more when price drops significantly)
- Dynamic position sizing based on portfolio allocation
- Profit-taking levels with partial exits
"""

import time
from typing import Dict, Any, Optional
from decimal import Decimal
from dataclasses import dataclass

from ..framework.base_classes import MarketData, TradeSignal, OrderSide
from .base_strategy import AdvancedStrategy, PositionType, SignalStrength


@dataclass
class DCAState:
    """State tracking for DCA strategy"""
    symbol: str
    last_buy_time: int = 0
    target_allocation: Decimal = Decimal('0')
    current_allocation: Decimal = Decimal('0')
    average_buy_price: Decimal = Decimal('0')
    total_invested: Decimal = Decimal('0')
    buy_count: int = 0
    next_buy_amount: Decimal = Decimal('0')


class DCAStrategy(AdvancedStrategy):
    """
    Dollar Cost Averaging Strategy
    
    Systematically buys assets at regular intervals and on significant price drops.
    Implements partial profit-taking when targets are reached.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # DCA-specific parameters
        self.buy_interval = config.get('buy_interval', 3600)  # 1 hour in seconds
        self.base_buy_amount = Decimal(str(config.get('base_buy_amount', 100)))  # Base amount per buy
        self.price_drop_threshold = Decimal(str(config.get('price_drop_threshold', 0.05)))  # 5% drop
        self.max_allocation_per_symbol = Decimal(str(config.get('max_allocation_per_symbol', 0.3)))  # 30%
        self.profit_taking_threshold = Decimal(str(config.get('profit_taking_threshold', 0.2)))  # 20% profit
        self.profit_taking_percentage = Decimal(str(config.get('profit_taking_percentage', 0.3)))  # Take 30%
        
        # State tracking
        self.dca_states: Dict[str, DCAState] = {}
        self.price_history: Dict[str, List[Decimal]] = {}
        self.lookback_period = config.get('lookback_period', 24)  # Hours to look back for price analysis
        
    async def _custom_initialize(self):
        """Initialize DCA states for each symbol"""
        for symbol in self.symbols:
            self.dca_states[symbol] = DCAState(
                symbol=symbol,
                target_allocation=self.max_allocation_per_symbol,
                next_buy_amount=self.base_buy_amount
            )
            self.price_history[symbol] = []
        
        self.logger.info("DCA Strategy initialized:")
        self.logger.info("- Buy interval: %d seconds", self.buy_interval)
        self.logger.info("- Base buy amount: %s %s", self.base_buy_amount, self.portfolio.base_currency)
        self.logger.info("- Price drop threshold: %s%%", self.price_drop_threshold * 100)
        self.logger.info("- Max allocation per symbol: %s%%", self.max_allocation_per_symbol * 100)
    
    async def _generate_signal(self, data: MarketData) -> Optional[TradeSignal]:
        """Generate DCA buy signals based on time intervals and price movements"""
        symbol = data.symbol
        current_time = int(time.time())
        current_price = data.price
        
        # Update price history
        self._update_price_history(symbol, current_price)
        
        dca_state = self.dca_states[symbol]
        signal = None
        
        # Check for time-based buy signal
        time_signal = self._check_time_based_signal(symbol, current_time, current_price)
        
        # Check for price-drop signal
        drop_signal = self._check_price_drop_signal(symbol, current_price)
        
        # Check for profit-taking signal
        profit_signal = self._check_profit_taking_signal(symbol, current_price)
        
        # Prioritize signals: profit-taking > price-drop > time-based
        if profit_signal:
            signal = profit_signal
        elif drop_signal:
            signal = drop_signal
        elif time_signal:
            signal = time_signal
        
        if signal:
            self.logger.info("DCA signal for %s: %s at %s (confidence: %.2f)", 
                           symbol, signal.action.value, signal.price, signal.confidence)
        
        return signal
    
    def _update_price_history(self, symbol: str, price: Decimal):
        """Update price history for trend analysis"""
        self.price_history[symbol].append(price)
        
        # Keep only recent prices (based on lookback period)
        max_history = self.lookback_period * 60  # Assume 1-minute data
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
    
    def _check_time_based_signal(self, symbol: str, current_time: int, current_price: Decimal) -> Optional[TradeSignal]:
        """Check if it's time for a regular DCA buy"""
        dca_state = self.dca_states[symbol]
        
        # Check if enough time has passed since last buy
        if current_time - dca_state.last_buy_time < self.buy_interval:
            return None
        
        # Check if we haven't exceeded max allocation
        current_portfolio_value = self.portfolio.get_portfolio_value()
        position = self.portfolio.get_position(symbol)
        
        if position:
            position_value = position.size * current_price
            current_allocation = position_value / current_portfolio_value
            
            if current_allocation >= self.max_allocation_per_symbol:
                self.logger.debug("Max allocation reached for %s: %.2f%%", 
                                symbol, current_allocation * 100)
                return None
        
        # Calculate buy amount (can be dynamic based on current allocation)
        buy_amount = self._calculate_dca_buy_amount(symbol, current_price, current_portfolio_value)
        
        if buy_amount <= 0:
            return None
        
        # Calculate position size
        position_size = buy_amount / current_price
        
        return TradeSignal(
            symbol=symbol,
            action=OrderSide.BUY,
            quantity=position_size,
            confidence=SignalStrength.MODERATE.value,
            price=current_price,
            stop_loss=None,  # DCA doesn't typically use stop losses
            take_profit=self._calculate_take_profit(current_price),
            timestamp=current_time,
            strategy_name=self.__class__.__name__,
            metadata={
                'signal_type': 'time_based_dca',
                'buy_amount': float(buy_amount),
                'allocation_target': float(self.max_allocation_per_symbol)
            }
        )
    
    def _check_price_drop_signal(self, symbol: str, current_price: Decimal) -> Optional[TradeSignal]:
        """Check for significant price drops that warrant additional buying"""
        if len(self.price_history[symbol]) < 2:
            return None
        
        # Calculate recent high (e.g., highest price in last few hours)
        recent_prices = self.price_history[symbol][-60:]  # Last 60 data points
        if not recent_prices:
            return None
        
        recent_high = max(recent_prices)
        price_drop = (recent_high - current_price) / recent_high
        
        if price_drop < self.price_drop_threshold:
            return None
        
        # Calculate enhanced buy amount for price drops
        base_amount = self._calculate_dca_buy_amount(symbol, current_price, self.portfolio.get_portfolio_value())
        
        # Increase buy amount based on severity of drop
        drop_multiplier = min(Decimal('3.0'), Decimal('1') + price_drop * Decimal('5'))
        enhanced_amount = base_amount * drop_multiplier
        
        position_size = enhanced_amount / current_price
        
        return TradeSignal(
            symbol=symbol,
            action=OrderSide.BUY,
            quantity=position_size,
            confidence=SignalStrength.STRONG.value,  # Higher confidence for dip buying
            price=current_price,
            stop_loss=None,
            take_profit=self._calculate_take_profit(current_price),
            timestamp=int(time.time()),
            strategy_name=self.__class__.__name__,
            metadata={
                'signal_type': 'price_drop_dca',
                'price_drop_pct': float(price_drop * 100),
                'recent_high': float(recent_high),
                'buy_amount': float(enhanced_amount),
                'drop_multiplier': float(drop_multiplier)
            }
        )
    
    def _check_profit_taking_signal(self, symbol: str, current_price: Decimal) -> Optional[TradeSignal]:
        """Check if position has reached profit-taking threshold"""
        position = self.portfolio.get_position(symbol)
        if not position or position.side != PositionType.LONG:
            return None
        
        # Calculate profit percentage
        profit_pct = (current_price - position.entry_price) / position.entry_price
        
        if profit_pct < self.profit_taking_threshold:
            return None
        
        # Calculate how much to sell (partial exit)
        sell_quantity = position.size * self.profit_taking_percentage
        
        return TradeSignal(
            symbol=symbol,
            action=OrderSide.SELL,
            quantity=sell_quantity,
            confidence=SignalStrength.STRONG.value,
            price=current_price,
            timestamp=int(time.time()),
            strategy_name=self.__class__.__name__,
            metadata={
                'signal_type': 'profit_taking',
                'profit_pct': float(profit_pct * 100),
                'sell_percentage': float(self.profit_taking_percentage * 100),
                'remaining_position': float((position.size - sell_quantity))
            }
        )
    
    def _calculate_dca_buy_amount(self, symbol: str, current_price: Decimal, portfolio_value: Decimal) -> Decimal:
        """Calculate the amount to buy for DCA"""
        # Start with base amount
        buy_amount = self.base_buy_amount
        
        # Adjust based on current allocation vs target
        position = self.portfolio.get_position(symbol)
        current_allocation = Decimal('0')
        
        if position:
            position_value = position.size * current_price
            current_allocation = position_value / portfolio_value
        
        allocation_gap = self.max_allocation_per_symbol - current_allocation
        
        if allocation_gap <= 0:
            return Decimal('0')
        
        # Scale buy amount based on allocation gap
        available_capital = self.portfolio.get_available_balance(self.portfolio.base_currency)
        max_buy_amount = min(
            available_capital * Decimal('0.1'),  # Don't use more than 10% of available capital
            portfolio_value * allocation_gap     # Don't exceed allocation target
        )
        
        return min(buy_amount, max_buy_amount)
    
    def _calculate_take_profit(self, entry_price: Decimal) -> Decimal:
        """Calculate take profit level"""
        return entry_price * (Decimal('1') + self.profit_taking_threshold)
    
    async def _custom_validate_signal(self, signal: TradeSignal) -> bool:
        """Custom validation for DCA signals"""
        # For sells (profit taking), always allow
        if signal.action == OrderSide.SELL:
            return True
        
        # For buys, check if we have enough capital
        required_amount = signal.quantity * signal.price
        available = self.portfolio.get_available_balance(self.portfolio.base_currency)
        
        if available < required_amount:
            self.logger.debug("Insufficient funds for DCA buy: need %s, have %s", 
                            required_amount, available)
            return False
        
        # Update last buy time for successful buys
        if signal.metadata and signal.metadata.get('signal_type', '').endswith('_dca'):
            self.dca_states[signal.symbol].last_buy_time = int(time.time())
        
        return True
    
    async def _custom_exit_check(self, data: MarketData, position) -> bool:
        """Custom exit conditions for DCA strategy"""
        # DCA strategy doesn't typically exit positions automatically
        # Exits are handled through profit-taking signals
        return False
    
    def get_dca_summary(self) -> Dict[str, Any]:
        """Get DCA-specific summary information"""
        summary = {}
        
        for symbol, dca_state in self.dca_states.items():
            position = self.portfolio.get_position(symbol)
            current_price = self.last_prices.get(symbol, Decimal('0'))
            
            if position and current_price > 0:
                profit_pct = (current_price - position.entry_price) / position.entry_price * 100
                position_value = position.size * current_price
                portfolio_value = self.portfolio.get_portfolio_value()
                allocation_pct = position_value / portfolio_value * 100
            else:
                profit_pct = 0
                position_value = 0
                allocation_pct = 0
            
            summary[symbol] = {
                'position_size': float(position.size) if position else 0,
                'entry_price': float(position.entry_price) if position else 0,
                'current_price': float(current_price),
                'profit_pct': float(profit_pct),
                'position_value': float(position_value),
                'allocation_pct': float(allocation_pct),
                'target_allocation_pct': float(self.max_allocation_per_symbol * 100),
                'last_buy_time': dca_state.last_buy_time,
                'buy_count': dca_state.buy_count,
                'total_invested': float(dca_state.total_invested)
            }
        
        return summary
    
    def update_dca_parameters(self, **kwargs):
        """Update DCA strategy parameters dynamically"""
        if 'buy_interval' in kwargs:
            self.buy_interval = kwargs['buy_interval']
            self.logger.info("Updated buy interval to %d seconds", self.buy_interval)
        
        if 'base_buy_amount' in kwargs:
            self.base_buy_amount = Decimal(str(kwargs['base_buy_amount']))
            self.logger.info("Updated base buy amount to %s", self.base_buy_amount)
        
        if 'price_drop_threshold' in kwargs:
            self.price_drop_threshold = Decimal(str(kwargs['price_drop_threshold']))
            self.logger.info("Updated price drop threshold to %s%%", self.price_drop_threshold * 100)
        
        if 'max_allocation_per_symbol' in kwargs:
            self.max_allocation_per_symbol = Decimal(str(kwargs['max_allocation_per_symbol']))
            for dca_state in self.dca_states.values():
                dca_state.target_allocation = self.max_allocation_per_symbol
            self.logger.info("Updated max allocation to %s%%", self.max_allocation_per_symbol * 100)
