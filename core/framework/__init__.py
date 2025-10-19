"""
Framework Package for Poise Trader

Core framework components providing the foundation for the modular
trading system architecture.
"""

from .base_classes import (
    BaseDataFeed,
    BaseStrategy,
    BaseExecutor,
    BaseRiskManager,
    BaseMonitor,
    BaseBacktester,
    MarketData,
    Order,
    TradeSignal,
    Portfolio,
    OrderType,
    OrderSide,
    OrderStatus
)

from .event_system import (
    EventBus,
    Event,
    EventPriority,
    SystemEvents,
    event_handler,
    create_event
)

from .plugin_system import (
    PluginManager,
    BasePlugin,
    PluginMetadata
)

from .config_manager import (
    ConfigManager,
    Environment,
    ExchangeConfig,
    StrategyConfig,
    RiskConfig,
    SystemConfig
)

__all__ = [
    # Base classes
    'BaseDataFeed',
    'BaseStrategy',
    'BaseExecutor',
    'BaseRiskManager',
    'BaseMonitor',
    'BaseBacktester',
    
    # Data structures
    'MarketData',
    'Order',
    'TradeSignal',
    'Portfolio',
    
    # Enums
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'EventPriority',
    'Environment',
    
    # Event system
    'EventBus',
    'Event',
    'SystemEvents',
    'event_handler',
    'create_event',
    
    # Plugin system
    'PluginManager',
    'BasePlugin',
    'PluginMetadata',
    
    # Configuration
    'ConfigManager',
    'ExchangeConfig',
    'StrategyConfig',
    'RiskConfig',
    'SystemConfig'
]
