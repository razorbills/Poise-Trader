#!/usr/bin/env python3
"""
🧠 ADVANCED MEMORY MANAGEMENT SYSTEM
Optimized memory cleanup, caching, and resource management for high-performance trading
"""

import gc
import threading
import time
import psutil
import os
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import weakref
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    swap_used_mb: float
    process_memory_mb: float
    cache_size_mb: float

class ResourcePool:
    """Thread-safe resource pool for reusable objects"""
    
    def __init__(self, factory_func, max_size: int = 100):
        self.factory_func = factory_func
        self.max_size = max_size
        self._pool = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._created_count = 0
        self._reused_count = 0
    
    def get(self):
        """Get object from pool or create new one"""
        with self._lock:
            if self._pool:
                self._reused_count += 1
                return self._pool.popleft()
            else:
                self._created_count += 1
                return self.factory_func()
    
    def put(self, obj):
        """Return object to pool"""
        with self._lock:
            if len(self._pool) < self.max_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self._pool.append(obj)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'created_count': self._created_count,
                'reused_count': self._reused_count,
                'efficiency': self._reused_count / max(1, self._created_count + self._reused_count)
            }

class AdvancedMemoryManager:
    """Advanced memory management with automatic cleanup and optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_memory_mb = self.config.get('max_memory_mb', 2048)
        self.cleanup_interval = self.config.get('cleanup_interval_seconds', 30)
        self.cache_max_size = self.config.get('cache_max_size', 10000)
        
        # Memory tracking
        self._price_history_caches: Dict[str, deque] = {}
        self._cache_access_times: Dict[str, datetime] = {}
        self._cache_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Resource pools
        self._resource_pools: Dict[str, ResourcePool] = {}
        
        # Weak references for automatic cleanup
        self._tracked_objects: Dict[str, weakref.ref] = {}
        
        # Memory monitoring
        self._memory_metrics = MemoryMetrics(0, 0, 0, 0, 0, 0, 0)
        self._monitoring_active = False
        self._cleanup_thread = None
        
        # Performance counters
        self._cleanup_count = 0
        self._memory_warnings = 0
        
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background memory monitoring"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._cleanup_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._cleanup_thread.start()
            logger.info("🧠 Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self._monitoring_active = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1)
        logger.info("🧠 Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring and cleanup loop"""
        while self._monitoring_active:
            try:
                self._update_memory_metrics()
                
                # Check if memory usage is high
                if self._memory_metrics.memory_percent > 80:
                    self._emergency_cleanup()
                elif self._memory_metrics.memory_percent > 60:
                    self._routine_cleanup()
                
                # Clean up old cache entries
                self._cleanup_old_caches()
                
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(5)  # Short delay on error
    
    def _update_memory_metrics(self):
        """Update current memory metrics"""
        try:
            # System memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Cache memory estimation
            cache_size = sum(len(cache) for cache in self._price_history_caches.values()) * 8  # Assume 8 bytes per float
            
            self._memory_metrics = MemoryMetrics(
                total_memory_mb=memory.total / 1024 / 1024,
                used_memory_mb=memory.used / 1024 / 1024,
                available_memory_mb=memory.available / 1024 / 1024,
                memory_percent=memory.percent,
                swap_used_mb=swap.used / 1024 / 1024,
                process_memory_mb=process_memory.rss / 1024 / 1024,
                cache_size_mb=cache_size / 1024 / 1024
            )
            
        except Exception as e:
            logger.error(f"Error updating memory metrics: {e}")
    
    def get_price_history_cache(self, symbol: str, max_size: int = 1000) -> deque:
        """Get or create thread-safe price history cache"""
        with self._cache_locks[symbol]:
            if symbol not in self._price_history_caches:
                self._price_history_caches[symbol] = deque(maxlen=max_size)
                logger.debug(f"Created price cache for {symbol} (max_size: {max_size})")
            
            self._cache_access_times[symbol] = datetime.now()
            return self._price_history_caches[symbol]
    
    def add_price_data(self, symbol: str, price: float, timestamp: datetime = None):
        """Thread-safe addition of price data"""
        cache = self.get_price_history_cache(symbol)
        with self._cache_locks[symbol]:
            cache.append({
                'price': price,
                'timestamp': timestamp or datetime.now()
            })
    
    def get_price_history(self, symbol: str, max_points: int = None) -> List[float]:
        """Get price history for symbol"""
        if symbol not in self._price_history_caches:
            return []
        
        with self._cache_locks[symbol]:
            cache = self._price_history_caches[symbol]
            prices = [item['price'] for item in cache]
            
            if max_points and len(prices) > max_points:
                return prices[-max_points:]
            return prices
    
    def register_resource_pool(self, name: str, factory_func, max_size: int = 100):
        """Register a resource pool for reusable objects"""
        self._resource_pools[name] = ResourcePool(factory_func, max_size)
        logger.debug(f"Registered resource pool: {name} (max_size: {max_size})")
    
    def get_from_pool(self, pool_name: str):
        """Get object from resource pool"""
        if pool_name in self._resource_pools:
            return self._resource_pools[pool_name].get()
        raise ValueError(f"Resource pool '{pool_name}' not found")
    
    def return_to_pool(self, pool_name: str, obj):
        """Return object to resource pool"""
        if pool_name in self._resource_pools:
            self._resource_pools[pool_name].put(obj)
        else:
            logger.warning(f"Resource pool '{pool_name}' not found")
    
    def track_object(self, name: str, obj):
        """Track object with weak reference for automatic cleanup"""
        def cleanup_callback(ref):
            if name in self._tracked_objects:
                del self._tracked_objects[name]
            logger.debug(f"Automatically cleaned up tracked object: {name}")
        
        self._tracked_objects[name] = weakref.ref(obj, cleanup_callback)
    
    def _cleanup_old_caches(self):
        """Clean up old, unused caches"""
        current_time = datetime.now()
        max_age = timedelta(hours=1)  # Remove caches older than 1 hour
        
        old_symbols = [
            symbol for symbol, last_access in self._cache_access_times.items()
            if current_time - last_access > max_age
        ]
        
        for symbol in old_symbols:
            self._remove_cache(symbol)
            logger.debug(f"Removed old cache for {symbol}")
    
    def _remove_cache(self, symbol: str):
        """Remove cache for symbol"""
        if symbol in self._price_history_caches:
            with self._cache_locks[symbol]:
                del self._price_history_caches[symbol]
                del self._cache_access_times[symbol]
                del self._cache_locks[symbol]
    
    def _routine_cleanup(self):
        """Routine memory cleanup"""
        self._cleanup_count += 1
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clean up empty caches
        empty_caches = [
            symbol for symbol, cache in self._price_history_caches.items()
            if len(cache) == 0
        ]
        
        for symbol in empty_caches:
            self._remove_cache(symbol)
        
        logger.debug(f"Routine cleanup #{self._cleanup_count}: {collected} objects collected, {len(empty_caches)} empty caches removed")
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup when usage is high"""
        self._memory_warnings += 1
        logger.warning(f"Emergency memory cleanup triggered! Memory usage: {self._memory_metrics.memory_percent:.1f}%")
        
        # More aggressive cleanup
        self._routine_cleanup()
        
        # Remove half of the oldest caches
        if self._cache_access_times:
            sorted_caches = sorted(self._cache_access_times.items(), key=lambda x: x[1])
            caches_to_remove = sorted_caches[:len(sorted_caches)//2]
            
            for symbol, _ in caches_to_remove:
                self._remove_cache(symbol)
                logger.debug(f"Emergency removed cache: {symbol}")
        
        # Force multiple garbage collection passes
        for _ in range(3):
            gc.collect()
        
        logger.warning(f"Emergency cleanup complete. Removed {len(caches_to_remove) if 'caches_to_remove' in locals() else 0} caches")
    
    def optimize_memory(self):
        """Manual memory optimization"""
        logger.info("🧠 Starting manual memory optimization...")
        
        initial_memory = self._memory_metrics.process_memory_mb
        
        # Clear all resource pools
        for pool_name, pool in self._resource_pools.items():
            pool._pool.clear()
            logger.debug(f"Cleared resource pool: {pool_name}")
        
        # Optimize cache sizes
        for symbol, cache in self._price_history_caches.items():
            if len(cache) > cache.maxlen // 2:
                # Keep only recent half
                new_cache = deque(list(cache)[len(cache)//2:], maxlen=cache.maxlen)
                with self._cache_locks[symbol]:
                    self._price_history_caches[symbol] = new_cache
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
        
        self._update_memory_metrics()
        final_memory = self._memory_metrics.process_memory_mb
        
        memory_saved = initial_memory - final_memory
        logger.info(f"✅ Memory optimization complete. Saved: {memory_saved:.1f} MB")
        
        return memory_saved
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        self._update_memory_metrics()
        
        pool_stats = {}
        for name, pool in self._resource_pools.items():
            pool_stats[name] = pool.get_stats()
        
        return {
            'memory_metrics': {
                'total_memory_mb': self._memory_metrics.total_memory_mb,
                'used_memory_mb': self._memory_metrics.used_memory_mb,
                'available_memory_mb': self._memory_metrics.available_memory_mb,
                'memory_percent': self._memory_metrics.memory_percent,
                'process_memory_mb': self._memory_metrics.process_memory_mb,
                'cache_size_mb': self._memory_metrics.cache_size_mb
            },
            'cache_stats': {
                'active_caches': len(self._price_history_caches),
                'total_cache_entries': sum(len(cache) for cache in self._price_history_caches.values()),
                'cache_symbols': list(self._price_history_caches.keys())
            },
            'resource_pools': pool_stats,
            'performance_counters': {
                'cleanup_count': self._cleanup_count,
                'memory_warnings': self._memory_warnings,
                'tracked_objects': len(self._tracked_objects)
            }
        }
    
    def cleanup_and_exit(self):
        """Clean shutdown with memory cleanup"""
        logger.info("🧠 Starting shutdown cleanup...")
        
        self.stop_monitoring()
        
        # Clear all caches
        for symbol in list(self._price_history_caches.keys()):
            self._remove_cache(symbol)
        
        # Clear resource pools
        for pool in self._resource_pools.values():
            pool._pool.clear()
        
        # Force final garbage collection
        gc.collect()
        
        logger.info("✅ Memory cleanup complete")

# Global memory manager instance
memory_manager = AdvancedMemoryManager()

# Context manager for automatic resource management
class ManagedResource:
    """Context manager for automatic resource cleanup"""
    
    def __init__(self, pool_name: str, resource=None):
        self.pool_name = pool_name
        self.resource = resource
        
    def __enter__(self):
        if self.resource is None:
            self.resource = memory_manager.get_from_pool(self.pool_name)
        return self.resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.resource is not None:
            memory_manager.return_to_pool(self.pool_name, self.resource)
