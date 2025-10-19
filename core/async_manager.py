#!/usr/bin/env python3
"""
⚡ ADVANCED ASYNC MANAGEMENT & CIRCUIT BREAKER SYSTEM
High-performance async operations with intelligent failure handling
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
import random
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    timeout: float = 30.0
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    jitter: bool = True

@dataclass
class CallStats:
    """Statistics for circuit breaker calls"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    average_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))

class AsyncCircuitBreaker:
    """Advanced circuit breaker for async operations"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CallStats()
        self._lock = asyncio.Lock()
        self._last_state_change = datetime.now()
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    await self._attempt_reset()
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
            
            return await self._execute_call(func, *args, **kwargs)
    
    async def _execute_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the actual function call with error handling"""
        start_time = time.time()
        self.stats.total_calls += 1
        
        try:
            # Add timeout to the call
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            response_time = time.time() - start_time
            await self._record_success(response_time)
            
            return result
            
        except self.config.expected_exception as e:
            await self._record_failure()
            raise e
        except asyncio.TimeoutError:
            await self._record_failure()
            raise CircuitBreakerTimeoutError(f"Call to {self.name} timed out after {self.config.timeout}s")
        except Exception as e:
            await self._record_failure()
            raise e
    
    async def _record_success(self, response_time: float):
        """Record successful call"""
        self.stats.successful_calls += 1
        self.stats.last_success_time = datetime.now()
        self.stats.response_times.append(response_time)
        
        # Update average response time
        if self.stats.response_times:
            self.stats.average_response_time = sum(self.stats.response_times) / len(self.stats.response_times)
        
        # Reset to closed if we were in half-open state
        if self.state == CircuitState.HALF_OPEN:
            await self._change_state(CircuitState.CLOSED)
            logger.info(f"✅ Circuit breaker {self.name} recovered - state: CLOSED")
    
    async def _record_failure(self):
        """Record failed call"""
        self.stats.failed_calls += 1
        self.stats.last_failure_time = datetime.now()
        
        # Check if we should open the circuit
        if (self.state == CircuitState.CLOSED and 
            self.stats.failed_calls >= self.config.failure_threshold):
            await self._change_state(CircuitState.OPEN)
            logger.warning(f"⚠️ Circuit breaker {self.name} OPENED after {self.stats.failed_calls} failures")
        
        elif self.state == CircuitState.HALF_OPEN:
            await self._change_state(CircuitState.OPEN)
            logger.warning(f"⚠️ Circuit breaker {self.name} returned to OPEN state")
    
    async def _change_state(self, new_state: CircuitState):
        """Change circuit breaker state"""
        old_state = self.state
        self.state = new_state
        self._last_state_change = datetime.now()
        
        if old_state != new_state:
            logger.info(f"🔄 Circuit breaker {self.name} state changed: {old_state.value} → {new_state.value}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        time_since_change = datetime.now() - self._last_state_change
        return time_since_change.total_seconds() >= self.config.recovery_timeout
    
    async def _attempt_reset(self):
        """Attempt to reset the circuit to half-open"""
        await self._change_state(CircuitState.HALF_OPEN)
        logger.info(f"🔄 Circuit breaker {self.name} attempting reset - state: HALF_OPEN")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        success_rate = 0
        if self.stats.total_calls > 0:
            success_rate = self.stats.successful_calls / self.stats.total_calls
        
        return {
            'name': self.name,
            'state': self.state.value,
            'total_calls': self.stats.total_calls,
            'successful_calls': self.stats.successful_calls,
            'failed_calls': self.stats.failed_calls,
            'success_rate': success_rate,
            'average_response_time': self.stats.average_response_time,
            'last_failure_time': self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            'last_success_time': self.stats.last_success_time.isoformat() if self.stats.last_success_time else None
        }

class AsyncRetryManager:
    """Advanced retry manager with exponential backoff and jitter"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_multiplier: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        
    async def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"❌ Function failed after {self.max_retries} retries: {e}")
                    raise e
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"⚠️ Function failed (attempt {attempt + 1}/{self.max_retries + 1}), retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry with exponential backoff and jitter"""
        delay = min(self.base_delay * (self.backoff_multiplier ** attempt), self.max_delay)
        
        if self.jitter:
            # Add jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)
        
        return delay

class AsyncConnectionPool:
    """Advanced async connection pool manager"""
    
    def __init__(self, max_connections: int = 100, max_keepalive_connections: int = 20,
                 keepalive_expiry: int = 30, timeout: int = 30):
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry
        self.timeout = timeout
        
        # Connection pools
        self._http_sessions: Dict[str, aiohttp.ClientSession] = {}
        self._pool_stats: Dict[str, Dict] = defaultdict(lambda: {
            'created': 0,
            'reused': 0,
            'errors': 0,
            'last_used': None
        })
        
        # Pool management
        self._cleanup_task = None
        self._lock = asyncio.Lock()
        
    async def get_session(self, base_url: str = None) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling"""
        session_key = base_url or 'default'
        
        async with self._lock:
            if session_key not in self._http_sessions:
                connector = aiohttp.TCPConnector(
                    limit=self.max_connections,
                    limit_per_host=self.max_keepalive_connections,
                    keepalive_timeout=self.keepalive_expiry,
                    enable_cleanup_closed=True
                )
                
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                
                session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={'User-Agent': 'Poise-Trader/3.0'}
                )
                
                self._http_sessions[session_key] = session
                self._pool_stats[session_key]['created'] += 1
                
                logger.debug(f"Created new HTTP session for {session_key}")
            else:
                self._pool_stats[session_key]['reused'] += 1
            
            self._pool_stats[session_key]['last_used'] = datetime.now()
            return self._http_sessions[session_key]
    
    async def make_request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request with connection pooling"""
        session = await self.get_session()
        
        try:
            async with session.request(method, url, **kwargs) as response:
                return response
        except Exception as e:
            # Update error stats
            session_key = kwargs.get('base_url', 'default')
            self._pool_stats[session_key]['errors'] += 1
            raise e
    
    async def cleanup_sessions(self):
        """Clean up old or unused sessions"""
        current_time = datetime.now()
        cleanup_threshold = timedelta(minutes=30)
        
        async with self._lock:
            sessions_to_remove = []
            
            for session_key, stats in self._pool_stats.items():
                last_used = stats.get('last_used')
                if last_used and (current_time - last_used) > cleanup_threshold:
                    sessions_to_remove.append(session_key)
            
            for session_key in sessions_to_remove:
                if session_key in self._http_sessions:
                    await self._http_sessions[session_key].close()
                    del self._http_sessions[session_key]
                    del self._pool_stats[session_key]
                    logger.debug(f"Cleaned up unused session: {session_key}")
    
    async def close_all(self):
        """Close all sessions"""
        async with self._lock:
            for session in self._http_sessions.values():
                await session.close()
            self._http_sessions.clear()
            self._pool_stats.clear()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'active_sessions': len(self._http_sessions),
            'session_stats': dict(self._pool_stats)
        }

class AsyncTaskManager:
    """Advanced async task management with priority queues and rate limiting"""
    
    def __init__(self, max_concurrent_tasks: int = 50, max_queue_size: int = 1000):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size
        
        # Task management
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._task_stats = defaultdict(lambda: {
            'completed': 0,
            'failed': 0,
            'cancelled': 0,
            'average_duration': 0.0
        })
        
        # Priority queues
        self._high_priority_queue = asyncio.Queue(maxsize=max_queue_size)
        self._normal_priority_queue = asyncio.Queue(maxsize=max_queue_size)
        self._low_priority_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Rate limiting
        self._rate_limiters: Dict[str, asyncio.Semaphore] = {}
        
        self._worker_tasks = []
        self._running = False
    
    async def start(self):
        """Start the task manager"""
        if not self._running:
            self._running = True
            
            # Start worker tasks
            for i in range(min(5, self.max_concurrent_tasks)):
                worker = asyncio.create_task(self._worker())
                self._worker_tasks.append(worker)
            
            logger.info(f"⚡ Async task manager started with {len(self._worker_tasks)} workers")
    
    async def stop(self):
        """Stop the task manager"""
        self._running = False
        
        # Cancel all worker tasks
        for worker in self._worker_tasks:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Cancel remaining active tasks
        for task in self._active_tasks.values():
            task.cancel()
        
        logger.info("⚡ Async task manager stopped")
    
    async def submit_task(self, coro, priority: str = 'normal', task_id: str = None) -> str:
        """Submit coroutine as a task with priority"""
        if task_id is None:
            task_id = f"task_{time.time()}_{random.randint(1000, 9999)}"
        
        task_item = {
            'coro': coro,
            'task_id': task_id,
            'created_at': datetime.now()
        }
        
        # Select appropriate queue based on priority
        if priority == 'high':
            await self._high_priority_queue.put(task_item)
        elif priority == 'low':
            await self._low_priority_queue.put(task_item)
        else:
            await self._normal_priority_queue.put(task_item)
        
        logger.debug(f"⚡ Submitted task {task_id} with priority {priority}")
        return task_id
    
    async def _worker(self):
        """Worker coroutine to process tasks"""
        while self._running:
            try:
                # Check queues in priority order
                task_item = await self._get_next_task()
                
                if task_item:
                    await self._execute_task(task_item)
                else:
                    await asyncio.sleep(0.1)  # Short sleep if no tasks
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on persistent errors
    
    async def _get_next_task(self):
        """Get next task from priority queues"""
        # Try high priority first
        try:
            return self._high_priority_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        
        # Try normal priority
        try:
            return self._normal_priority_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        
        # Try low priority
        try:
            return self._low_priority_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        
        return None
    
    async def _execute_task(self, task_item):
        """Execute a task with resource management"""
        task_id = task_item['task_id']
        coro = task_item['coro']
        start_time = time.time()
        
        async with self._task_semaphore:
            try:
                # Create and track the task
                task = asyncio.create_task(coro)
                self._active_tasks[task_id] = task
                
                # Execute the task
                result = await task
                
                # Record success
                duration = time.time() - start_time
                self._record_task_completion(task_id, 'completed', duration)
                
                logger.debug(f"✅ Task {task_id} completed in {duration:.2f}s")
                
            except asyncio.CancelledError:
                self._record_task_completion(task_id, 'cancelled', time.time() - start_time)
                logger.debug(f"🚫 Task {task_id} cancelled")
            except Exception as e:
                self._record_task_completion(task_id, 'failed', time.time() - start_time)
                logger.error(f"❌ Task {task_id} failed: {e}")
            finally:
                # Clean up
                if task_id in self._active_tasks:
                    del self._active_tasks[task_id]
    
    def _record_task_completion(self, task_id: str, status: str, duration: float):
        """Record task completion statistics"""
        stats = self._task_stats[status]
        stats['completed'] += 1
        
        # Update average duration
        current_avg = stats['average_duration']
        new_count = stats['completed']
        stats['average_duration'] = ((current_avg * (new_count - 1)) + duration) / new_count
    
    def get_rate_limiter(self, name: str, max_concurrent: int) -> asyncio.Semaphore:
        """Get or create rate limiter"""
        if name not in self._rate_limiters:
            self._rate_limiters[name] = asyncio.Semaphore(max_concurrent)
        return self._rate_limiters[name]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics"""
        return {
            'active_tasks': len(self._active_tasks),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'queue_sizes': {
                'high_priority': self._high_priority_queue.qsize(),
                'normal_priority': self._normal_priority_queue.qsize(),
                'low_priority': self._low_priority_queue.qsize()
            },
            'task_stats': dict(self._task_stats),
            'workers': len(self._worker_tasks),
            'running': self._running
        }

# Custom exceptions
class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors"""
    pass

class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Raised when circuit breaker call times out"""
    pass

# Global instances
async_manager = None

class AsyncManager:
    """Centralized async management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.circuit_breakers: Dict[str, AsyncCircuitBreaker] = {}
        self.retry_manager = AsyncRetryManager()
        self.connection_pool = AsyncConnectionPool()
        self.task_manager = AsyncTaskManager()
        
        # State
        self._initialized = False
    
    async def initialize(self):
        """Initialize the async manager"""
        if not self._initialized:
            await self.task_manager.start()
            self._initialized = True
            logger.info("⚡ Async manager initialized")
    
    async def shutdown(self):
        """Shutdown the async manager"""
        if self._initialized:
            await self.task_manager.stop()
            await self.connection_pool.close_all()
            self._initialized = False
            logger.info("⚡ Async manager shutdown complete")
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> AsyncCircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = AsyncCircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
    async def protected_call(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Make a protected call with circuit breaker and retry"""
        circuit_breaker = self.get_circuit_breaker(name)
        
        async def wrapped_call():
            return await circuit_breaker.call(func, *args, **kwargs)
        
        return await self.retry_manager.retry(wrapped_call)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        circuit_stats = {}
        for name, cb in self.circuit_breakers.items():
            circuit_stats[name] = cb.get_stats()
        
        return {
            'circuit_breakers': circuit_stats,
            'connection_pool': self.connection_pool.get_pool_stats(),
            'task_manager': self.task_manager.get_stats(),
            'initialized': self._initialized
        }

# Global async manager instance
async_manager = AsyncManager()
