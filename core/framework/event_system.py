"""
Event System for Poise Trader

Provides asynchronous event-driven communication between components.
Enables loose coupling and high-performance message passing.
"""

import asyncio
import logging
import weakref
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor


class EventPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Base event class for all system events"""
    event_type: str
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.MEDIUM
    timestamp: float = 0.0
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()


class EventBus:
    """
    High-performance event bus for component communication.
    
    Features:
    - Async event handling with priorities
    - Weak references to prevent memory leaks
    - Thread-safe operations
    - Event filtering and routing
    - Performance metrics
    """
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger("EventBus")
        self._handlers: Dict[str, List[weakref.ReferenceType]] = defaultdict(list)
        self._event_queue = asyncio.PriorityQueue()
        self._is_running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._max_workers = max_workers
        self._lock = threading.Lock()
        self._metrics = {
            'events_processed': 0,
            'events_failed': 0,
            'handlers_registered': 0,
            'average_processing_time': 0.0
        }
        self._thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def start(self) -> None:
        """Start the event bus workers"""
        if self._is_running:
            return
            
        self._is_running = True
        self.logger.info("Starting EventBus with %d workers", self._max_workers)
        
        # Start worker tasks
        for i in range(self._max_workers):
            task = asyncio.create_task(self._event_worker(f"worker-{i}"))
            self._worker_tasks.append(task)
    
    async def stop(self) -> None:
        """Stop the event bus and cleanup"""
        if not self._is_running:
            return
            
        self.logger.info("Stopping EventBus")
        self._is_running = False
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        
        # Shutdown thread executor
        self._thread_executor.shutdown(wait=True)
        
        self.logger.info("EventBus stopped")
    
    def subscribe(
        self, 
        event_type: str, 
        handler: Callable[[Event], Any],
        weak_ref: bool = True
    ) -> None:
        """
        Subscribe to events of a specific type
        
        Args:
            event_type: Type of events to subscribe to
            handler: Async or sync handler function
            weak_ref: Use weak reference to prevent memory leaks
        """
        with self._lock:
            if weak_ref:
                ref = weakref.ref(handler)
            else:
                # Store strong reference
                ref = lambda: handler
            
            self._handlers[event_type].append(ref)
            self._metrics['handlers_registered'] += 1
            
        self.logger.debug("Handler registered for event type: %s", event_type)
    
    def unsubscribe(self, event_type: str, handler: Callable[[Event], Any]) -> bool:
        """
        Unsubscribe from events
        
        Returns:
            True if handler was found and removed
        """
        with self._lock:
            handlers = self._handlers[event_type]
            for i, ref in enumerate(handlers):
                if ref() is handler:
                    handlers.pop(i)
                    return True
        return False
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to the bus
        
        Events are queued and processed asynchronously
        """
        if not self._is_running:
            self.logger.warning("EventBus not running, event ignored: %s", event.event_type)
            return
        
        # Priority queue uses negative values for higher priority
        priority_value = -event.priority.value
        await self._event_queue.put((priority_value, event))
        
        self.logger.debug("Event published: %s (priority: %s)", 
                         event.event_type, event.priority.name)
    
    async def publish_sync(self, event: Event) -> List[Any]:
        """
        Publish event and wait for all handlers to complete
        
        Returns:
            List of results from all handlers
        """
        handlers = self._get_active_handlers(event.event_type)
        if not handlers:
            return []
        
        results = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(event)
                else:
                    # Run sync handler in thread pool
                    result = await asyncio.get_event_loop().run_in_executor(
                        self._thread_executor, handler, event
                    )
                results.append(result)
            except Exception as e:
                self.logger.error("Handler failed: %s", e, exc_info=True)
                self._metrics['events_failed'] += 1
        
        return results
    
    async def _event_worker(self, worker_name: str) -> None:
        """Worker coroutine that processes events from the queue"""
        self.logger.info("Event worker %s started", worker_name)
        
        while self._is_running:
            try:
                # Wait for event with timeout
                priority, event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=1.0
                )
                
                start_time = asyncio.get_event_loop().time()
                await self._process_event(event)
                
                # Update metrics
                processing_time = asyncio.get_event_loop().time() - start_time
                self._update_metrics(processing_time)
                
                self._event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error("Event worker %s error: %s", worker_name, e, exc_info=True)
        
        self.logger.info("Event worker %s stopped", worker_name)
    
    async def _process_event(self, event: Event) -> None:
        """Process a single event by calling all registered handlers"""
        handlers = self._get_active_handlers(event.event_type)
        
        if not handlers:
            self.logger.debug("No handlers for event type: %s", event.event_type)
            return
        
        # Process all handlers concurrently
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(asyncio.create_task(handler(event)))
            else:
                # Run sync handler in thread pool
                task = asyncio.create_task(
                    asyncio.get_event_loop().run_in_executor(
                        self._thread_executor, handler, event
                    )
                )
                tasks.append(task)
        
        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _get_active_handlers(self, event_type: str) -> List[Callable]:
        """Get all active handlers for an event type, cleaning up dead references"""
        with self._lock:
            handlers = []
            active_refs = []
            
            for ref in self._handlers[event_type]:
                handler = ref()
                if handler is not None:
                    handlers.append(handler)
                    active_refs.append(ref)
            
            # Update the list to remove dead references
            self._handlers[event_type] = active_refs
            
            return handlers
    
    def _update_metrics(self, processing_time: float) -> None:
        """Update performance metrics"""
        self._metrics['events_processed'] += 1
        
        # Update average processing time
        count = self._metrics['events_processed']
        current_avg = self._metrics['average_processing_time']
        self._metrics['average_processing_time'] = (
            (current_avg * (count - 1) + processing_time) / count
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus performance metrics"""
        return dict(self._metrics)
    
    def clear_handlers(self, event_type: Optional[str] = None) -> None:
        """Clear handlers for a specific event type or all handlers"""
        with self._lock:
            if event_type:
                self._handlers[event_type].clear()
            else:
                self._handlers.clear()


# Predefined system events
class SystemEvents:
    """Standard system event types"""
    
    # Core system events
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    
    # Market data events
    MARKET_DATA = "market.data"
    MARKET_DATA_ERROR = "market.data.error"
    MARKET_CONNECTION_LOST = "market.connection.lost"
    MARKET_CONNECTION_RESTORED = "market.connection.restored"
    
    # Trading events
    SIGNAL_GENERATED = "trading.signal.generated"
    ORDER_PLACED = "trading.order.placed"
    ORDER_FILLED = "trading.order.filled"
    ORDER_CANCELLED = "trading.order.cancelled"
    ORDER_REJECTED = "trading.order.rejected"
    
    # Portfolio events
    PORTFOLIO_UPDATED = "portfolio.updated"
    BALANCE_LOW = "portfolio.balance.low"
    POSITION_OPENED = "portfolio.position.opened"
    POSITION_CLOSED = "portfolio.position.closed"
    
    # Risk events
    RISK_LIMIT_EXCEEDED = "risk.limit.exceeded"
    DRAWDOWN_WARNING = "risk.drawdown.warning"
    EMERGENCY_STOP = "risk.emergency.stop"
    
    # Performance events
    PERFORMANCE_UPDATED = "performance.updated"
    BENCHMARK_COMPLETED = "performance.benchmark.completed"


# Convenience decorators
def event_handler(event_type: str, event_bus: EventBus):
    """Decorator to automatically register event handlers"""
    def decorator(func):
        event_bus.subscribe(event_type, func)
        return func
    return decorator


def create_event(event_type: str, **kwargs) -> Event:
    """Convenience function to create events"""
    return Event(event_type=event_type, data=kwargs)
