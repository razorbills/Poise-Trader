"""
Trading Strategies Package for Poise Trader

Provides a comprehensive set of trading strategies with built-in:
- Portfolio management and position tracking
- Risk management and position sizing
- Performance monitoring and metrics
- Fund management (add/withdraw)
"""

from .base_strategy import (
    AdvancedStrategy,
    PortfolioManager, 
    Position,
    PositionType,
    SignalStrength
)
from .dca_strategy import DCAStrategy
from .grid_strategy import GridStrategy

# Strategy registry for dynamic loading
STRATEGY_REGISTRY = {
    'dca': DCAStrategy,
    'dollar_cost_averaging': DCAStrategy,
    'grid': GridStrategy,
    'grid_trading': GridStrategy,
}


class StrategyFactory:
    """Factory for creating strategy instances"""
    
    @staticmethod
    def create_strategy(strategy_type: str, config: dict) -> AdvancedStrategy:
        """
        Create a strategy instance
        
        Args:
            strategy_type: Type of strategy ('dca', 'grid', etc.)
            config: Strategy configuration including funds
            
        Returns:
            Strategy instance
            
        Example:
            config = {
                'initial_capital': 10000,
                'base_currency': 'USDT',
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'buy_interval': 3600,  # 1 hour
                'base_buy_amount': 100
            }
            strategy = StrategyFactory.create_strategy('dca', config)
        """
        strategy_type = strategy_type.lower()
        
        if strategy_type not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = STRATEGY_REGISTRY[strategy_type]
        return strategy_class(config)
    
    @staticmethod
    def get_available_strategies() -> list:
        """Get list of available strategy types"""
        return list(STRATEGY_REGISTRY.keys())
    
    @staticmethod
    def register_strategy(name: str, strategy_class: type):
        """Register a custom strategy"""
        if not issubclass(strategy_class, AdvancedStrategy):
            raise ValueError("Strategy must inherit from AdvancedStrategy")
        
        STRATEGY_REGISTRY[name.lower()] = strategy_class


def create_portfolio_manager(initial_capital: float, base_currency: str = "USDT") -> PortfolioManager:
    """
    Create a standalone portfolio manager
    
    Args:
        initial_capital: Starting capital amount
        base_currency: Base currency (default: USDT)
        
    Returns:
        PortfolioManager instance
        
    Example:
        portfolio = create_portfolio_manager(10000, "USDT")
        portfolio.balances  # {'USDT': Decimal('10000')}
    """
    from decimal import Decimal
    return PortfolioManager(Decimal(str(initial_capital)), base_currency)


# Export commonly used classes and functions
__all__ = [
    # Base classes
    'AdvancedStrategy',
    'PortfolioManager',
    'Position',
    'PositionType',
    'SignalStrength',
    
    # Strategy implementations
    'DCAStrategy',
    'GridStrategy',
    
    # Factory and utilities
    'StrategyFactory',
    'create_portfolio_manager',
    'STRATEGY_REGISTRY'
]
