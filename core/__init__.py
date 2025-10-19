"""
Poise Trader Core Package

A modular, high-performance trading bot framework designed for:
- Multi-exchange support (CEX and DEX)
- Real-time strategy execution
- Fleet deployment capabilities
- Stealth operation modes
- Advanced performance optimization
"""

__version__ = "1.0.0"
__author__ = "Poise Trading Systems"

from .framework.plugin_system import PluginManager
from .framework.base_classes import (
    BaseDataFeed,
    BaseStrategy,
    BaseExecutor,
    BaseRiskManager
)
from .framework.event_system import EventBus, Event
from .framework.config_manager import ConfigManager

__all__ = [
    'PluginManager',
    'BaseDataFeed',
    'BaseStrategy', 
    'BaseExecutor',
    'BaseRiskManager',
    'EventBus',
    'Event',
    'ConfigManager'
]
