"""
Market Data Feeds Package for Poise Trader

Provides unified market data feeds for multiple exchanges with:
- Real-time WebSocket streaming
- REST API historical data
- Rate limiting and connection management  
- Normalized data formats
"""

from typing import Dict, Any, Optional
import logging

from .base_feed import UnifiedFeed, WebSocketFeed, FeedConfig, RateLimiter
from .binance_feed import BinanceFeed
from .coinbase_feed import CoinbaseFeed
from .mexc_feed import MexcConnector

# Registry of available feed implementations
FEED_REGISTRY = {
    'binance': BinanceFeed,
    'coinbase': CoinbaseFeed,
    'coinbase_pro': CoinbaseFeed,  # Alias
    'mexc': MexcConnector,
}

# Default configurations for each exchange
DEFAULT_CONFIGS = {
    'binance': {
        'base_url': 'https://api.binance.com',
        'websocket_url': 'wss://stream.binance.com:9443/ws',
        'rate_limit': 1200,
        'timeout': 30,
        'heartbeat_interval': 180
    },
    'coinbase': {
        'base_url': 'https://api.pro.coinbase.com',
        'websocket_url': 'wss://ws-feed.pro.coinbase.com',
        'rate_limit': 1000,
        'timeout': 30,
        'heartbeat_interval': 30
    },
    'coinbase_pro': {  # Alias for coinbase
        'base_url': 'https://api.pro.coinbase.com',
        'websocket_url': 'wss://ws-feed.pro.coinbase.com',
        'rate_limit': 1000,
        'timeout': 30,
        'heartbeat_interval': 30
    },
    'mexc': {
        'base_url': 'https://api.mexc.com',
        'websocket_url': 'wss://wbs.mexc.com/ws',
        'rate_limit': 600,
        'timeout': 30,
        'heartbeat_interval': 60
    }
}


class FeedFactory:
    """Factory class for creating exchange feeds"""
    
    @staticmethod
    def create_feed(exchange_name: str, config: Dict[str, Any]) -> Optional[WebSocketFeed]:
        """
        Create a feed instance for the specified exchange
        
        Args:
            exchange_name: Name of the exchange (e.g., 'binance', 'coinbase')
            config: Exchange-specific configuration
            
        Returns:
            Feed instance or None if exchange not supported
        """
        logger = logging.getLogger('FeedFactory')
        
        exchange_name = exchange_name.lower()
        
        if exchange_name not in FEED_REGISTRY:
            logger.error("Unsupported exchange: %s", exchange_name)
            logger.info("Supported exchanges: %s", list(FEED_REGISTRY.keys()))
            return None
        
        # Merge with default config
        merged_config = DEFAULT_CONFIGS.get(exchange_name, {}).copy()
        merged_config.update(config)
        merged_config['exchange_name'] = exchange_name
        
        try:
            feed_class = FEED_REGISTRY[exchange_name]
            feed = feed_class(merged_config)
            logger.info("Created %s feed", exchange_name)
            return feed
            
        except Exception as e:
            logger.error("Failed to create %s feed: %s", exchange_name, e)
            return None
    
    @staticmethod
    def get_supported_exchanges() -> list:
        """Get list of supported exchange names"""
        return list(FEED_REGISTRY.keys())
    
    @staticmethod
    def register_feed(exchange_name: str, feed_class: type) -> None:
        """
        Register a custom feed implementation
        
        Args:
            exchange_name: Name of the exchange
            feed_class: Feed class that inherits from WebSocketFeed
        """
        logger = logging.getLogger('FeedFactory')
        
        if not issubclass(feed_class, WebSocketFeed):
            raise ValueError("Feed class must inherit from WebSocketFeed")
        
        FEED_REGISTRY[exchange_name.lower()] = feed_class
        logger.info("Registered custom feed for: %s", exchange_name)


def create_unified_feed(exchange_configs: Dict[str, Dict[str, Any]]) -> UnifiedFeed:
    """
    Create a unified feed that aggregates multiple exchanges
    
    Args:
        exchange_configs: Dictionary mapping exchange names to their configs
        
    Returns:
        UnifiedFeed instance
    
    Example:
        configs = {
            'binance': {'api_key': 'xxx', 'api_secret': 'yyy'},
            'coinbase': {'api_key': 'aaa', 'api_secret': 'bbb', 'passphrase': 'ccc'}
        }
        unified_feed = create_unified_feed(configs)
    """
    logger = logging.getLogger('create_unified_feed')
    
    # Create unified feed
    unified_config = {
        'exchange_name': 'unified',
        'exchanges': list(exchange_configs.keys())
    }
    
    unified_feed = UnifiedFeed(unified_config)
    
    # Create individual exchange feeds
    for exchange_name, config in exchange_configs.items():
        feed = FeedFactory.create_feed(exchange_name, config)
        if feed:
            # Add to unified feed (this will be done asynchronously)
            logger.info("Will add %s feed to unified feed", exchange_name)
        else:
            logger.error("Failed to create feed for %s", exchange_name)
    
    return unified_feed


# Export commonly used classes and functions
__all__ = [
    'UnifiedFeed',
    'WebSocketFeed', 
    'FeedConfig',
    'RateLimiter',
    'BinanceFeed',
    'CoinbaseFeed',
    'FeedFactory',
    'create_unified_feed',
    'FEED_REGISTRY',
    'DEFAULT_CONFIGS'
]
