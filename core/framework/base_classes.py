"""
Base Classes for Poise Trader Framework

Defines abstract base classes that all plugins and modules must inherit from.
These provide the core interfaces for modular architecture.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class MarketData:
    """Unified market data structure"""
    symbol: str
    timestamp: int
    price: Decimal
    volume: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    high_24h: Optional[Decimal] = None
    low_24h: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None
    exchange: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class Order:
    """Unified order structure"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    average_price: Optional[Decimal] = None
    timestamp: Optional[int] = None
    exchange: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TradeSignal:
    """Strategy signal structure"""
    symbol: str
    action: OrderSide
    quantity: Decimal
    confidence: float
    price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    timestamp: int = 0
    strategy_name: str = ""
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Portfolio:
    """Portfolio state structure"""
    total_value: Decimal
    available_balance: Dict[str, Decimal]
    positions: Dict[str, Decimal]
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    timestamp: int


class BaseDataFeed(ABC):
    """Abstract base class for all data feeds"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_running = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data for given symbols"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data for given symbols"""
        pass
    
    @abstractmethod
    async def get_market_data(self) -> AsyncGenerator[MarketData, None]:
        """Stream real-time market data"""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        start_time: int, 
        end_time: int
    ) -> List[MarketData]:
        """Retrieve historical market data"""
        pass


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_active = False
        self.performance_metrics = {}
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize strategy with configuration"""
        pass
    
    @abstractmethod
    async def process_market_data(self, data: MarketData) -> Optional[TradeSignal]:
        """Process incoming market data and generate signals"""
        pass
    
    @abstractmethod
    async def update_portfolio(self, portfolio: Portfolio) -> None:
        """Update strategy based on current portfolio state"""
        pass
    
    @abstractmethod
    def get_required_symbols(self) -> List[str]:
        """Return list of symbols this strategy needs"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return strategy performance metrics"""
        pass
    
    async def start(self) -> None:
        """Start the strategy"""
        self.is_active = True
        self.logger.info(f"Strategy {self.__class__.__name__} started")
    
    async def stop(self) -> None:
        """Stop the strategy"""
        self.is_active = False
        self.logger.info(f"Strategy {self.__class__.__name__} stopped")


class BaseExecutor(ABC):
    """Abstract base class for all order executors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_connected = False
        self.pending_orders = {}
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to exchange/execution venue"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """Place an order and return order ID"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get current status of an order"""
        pass
    
    @abstractmethod
    async def get_portfolio(self) -> Portfolio:
        """Get current portfolio/balance information"""
        pass
    
    @abstractmethod
    async def get_order_history(self, symbol: str = None) -> List[Order]:
        """Get historical orders"""
        pass


class BaseRiskManager(ABC):
    """Abstract base class for risk management systems"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.risk_limits = config.get('risk_limits', {})
    
    @abstractmethod
    async def check_signal(self, signal: TradeSignal, portfolio: Portfolio) -> bool:
        """Check if signal passes risk controls"""
        pass
    
    @abstractmethod
    async def check_order(self, order: Order, portfolio: Portfolio) -> bool:
        """Check if order passes risk controls"""
        pass
    
    @abstractmethod
    async def check_portfolio(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Check portfolio against risk limits"""
        pass
    
    @abstractmethod
    async def calculate_position_size(
        self, 
        signal: TradeSignal, 
        portfolio: Portfolio
    ) -> Decimal:
        """Calculate appropriate position size"""
        pass


class BaseMonitor(ABC):
    """Abstract base class for monitoring and alerting systems"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def send_alert(self, message: str, severity: str) -> bool:
        """Send alert/notification"""
        pass
    
    @abstractmethod
    async def log_trade(self, order: Order, signal: TradeSignal) -> None:
        """Log executed trade"""
        pass
    
    @abstractmethod
    async def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics"""
        pass


class BaseBacktester(ABC):
    """Abstract base class for backtesting engines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def run_backtest(
        self, 
        strategy: BaseStrategy,
        historical_data: List[MarketData],
        initial_portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Run backtest and return results"""
        pass
    
    @abstractmethod
    async def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate backtest report"""
        pass
