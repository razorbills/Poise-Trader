#!/usr/bin/env python3
"""
🚀 DISTRIBUTED COMPUTING & MESSAGE QUEUE SYSTEM
Horizontal Scaling, Load Distribution & High Availability Architecture
"""

import asyncio
import json
import redis
import zmq
import zmq.asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
import hashlib
import time
from enum import Enum
import threading
from queue import Queue
import uuid

class MessageType(Enum):
    """Message types for distributed communication"""
    MARKET_DATA = "market_data"
    TRADE_SIGNAL = "trade_signal"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_ALERT = "risk_alert"
    SYSTEM_STATUS = "system_status"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    HEARTBEAT = "heartbeat"

@dataclass
class DistributedMessage:
    """Standard message format for distributed communication"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]
    timestamp: datetime
    priority: int  # 1=highest, 10=lowest
    payload: Dict
    ttl_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3

class MessageBroker:
    """Redis-based message broker for distributed communication"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.subscribers = {}
        self.message_handlers = {}
        self.running = False
        
        # Message queues for different priorities
        self.high_priority_queue = "poise_high_priority"
        self.normal_priority_queue = "poise_normal_priority"
        self.low_priority_queue = "poise_low_priority"
        
        # Dead letter queue for failed messages
        self.dead_letter_queue = "poise_dead_letter"
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def publish_message(self, message: DistributedMessage):
        """Publish message to appropriate queue based on priority"""
        
        # Serialize message
        message_data = {
            'message_id': message.message_id,
            'message_type': message.message_type.value,
            'sender_id': message.sender_id,
            'recipient_id': message.recipient_id,
            'timestamp': message.timestamp.isoformat(),
            'priority': message.priority,
            'payload': message.payload,
            'ttl_seconds': message.ttl_seconds,
            'retry_count': message.retry_count,
            'max_retries': message.max_retries
        }
        
        serialized_message = json.dumps(message_data)
        
        # Select queue based on priority
        if message.priority <= 3:
            queue_name = self.high_priority_queue
        elif message.priority <= 6:
            queue_name = self.normal_priority_queue
        else:
            queue_name = self.low_priority_queue
        
        # Publish to Redis list (FIFO queue)
        self.redis_client.lpush(queue_name, serialized_message)
        
        # Set TTL for message
        message_key = f"message:{message.message_id}"
        self.redis_client.setex(message_key, message.ttl_seconds, serialized_message)
        
        self.logger.info(f"📨 Published message {message.message_id} to {queue_name}")
    
    async def subscribe_to_messages(self, handler: Callable[[DistributedMessage], None], 
                                  message_types: List[MessageType] = None):
        """Subscribe to messages and process with handler"""
        
        self.running = True
        
        while self.running:
            try:
                # Process high priority queue first
                for queue_name in [self.high_priority_queue, self.normal_priority_queue, self.low_priority_queue]:
                    message_data = self.redis_client.brpop(queue_name, timeout=1)
                    
                    if message_data:
                        _, serialized_message = message_data
                        message_dict = json.loads(serialized_message)
                        
                        # Reconstruct message object
                        message = DistributedMessage(
                            message_id=message_dict['message_id'],
                            message_type=MessageType(message_dict['message_type']),
                            sender_id=message_dict['sender_id'],
                            recipient_id=message_dict.get('recipient_id'),
                            timestamp=datetime.fromisoformat(message_dict['timestamp']),
                            priority=message_dict['priority'],
                            payload=message_dict['payload'],
                            ttl_seconds=message_dict['ttl_seconds'],
                            retry_count=message_dict['retry_count'],
                            max_retries=message_dict['max_retries']
                        )
                        
                        # Check if message is still valid (TTL)
                        if (datetime.now() - message.timestamp).total_seconds() > message.ttl_seconds:
                            self.logger.warning(f"⏰ Message {message.message_id} expired, discarding")
                            continue
                        
                        # Filter by message types if specified
                        if message_types and message.message_type not in message_types:
                            continue
                        
                        # Process message
                        try:
                            await handler(message)
                            self.logger.info(f"✅ Processed message {message.message_id}")
                        except Exception as e:
                            self.logger.error(f"❌ Error processing message {message.message_id}: {e}")
                            
                            # Retry logic
                            if message.retry_count < message.max_retries:
                                message.retry_count += 1
                                await self.publish_message(message)
                            else:
                                # Send to dead letter queue
                                self.redis_client.lpush(self.dead_letter_queue, serialized_message)
                                self.logger.error(f"💀 Message {message.message_id} sent to dead letter queue")
                
            except Exception as e:
                self.logger.error(f"❌ Error in message subscriber: {e}")
                await asyncio.sleep(1)
    
    def stop_subscriber(self):
        """Stop message subscriber"""
        self.running = False

class TaskDistributor:
    """Distribute computational tasks across multiple processes"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        self.task_queue = Queue()
        self.results = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def submit_cpu_intensive_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit CPU-intensive task to process pool"""
        
        task_id = str(uuid.uuid4())
        
        try:
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(self.process_executor, func, *args, **kwargs)
            
            self.results[task_id] = {
                'future': future,
                'status': 'running',
                'submitted_at': datetime.now()
            }
            
            self.logger.info(f"🔄 Submitted CPU task {task_id} to process pool")
            return task_id
            
        except Exception as e:
            self.logger.error(f"❌ Error submitting CPU task: {e}")
            return None
    
    async def submit_io_intensive_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit I/O-intensive task to thread pool"""
        
        task_id = str(uuid.uuid4())
        
        try:
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(self.thread_executor, func, *args, **kwargs)
            
            self.results[task_id] = {
                'future': future,
                'status': 'running',
                'submitted_at': datetime.now()
            }
            
            self.logger.info(f"🔄 Submitted I/O task {task_id} to thread pool")
            return task_id
            
        except Exception as e:
            self.logger.error(f"❌ Error submitting I/O task: {e}")
            return None
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """Get result of distributed task"""
        
        if task_id not in self.results:
            return None
        
        task_info = self.results[task_id]
        future = task_info['future']
        
        try:
            if timeout:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future
            
            task_info['status'] = 'completed'
            task_info['completed_at'] = datetime.now()
            
            self.logger.info(f"✅ Task {task_id} completed successfully")
            return result
            
        except asyncio.TimeoutError:
            task_info['status'] = 'timeout'
            self.logger.warning(f"⏰ Task {task_id} timed out")
            return None
        except Exception as e:
            task_info['status'] = 'failed'
            task_info['error'] = str(e)
            self.logger.error(f"❌ Task {task_id} failed: {e}")
            return None
    
    def get_task_status(self, task_id: str) -> Dict:
        """Get status of distributed task"""
        return self.results.get(task_id, {'status': 'not_found'})

class LoadBalancer:
    """Load balancer for distributing work across multiple instances"""
    
    def __init__(self, instances: List[str]):
        self.instances = instances
        self.instance_loads = {instance: 0 for instance in instances}
        self.instance_health = {instance: True for instance in instances}
        self.round_robin_index = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_least_loaded_instance(self) -> Optional[str]:
        """Get instance with lowest load"""
        
        healthy_instances = [inst for inst in self.instances if self.instance_health[inst]]
        
        if not healthy_instances:
            return None
        
        return min(healthy_instances, key=lambda x: self.instance_loads[x])
    
    def get_round_robin_instance(self) -> Optional[str]:
        """Get next instance using round-robin"""
        
        healthy_instances = [inst for inst in self.instances if self.instance_health[inst]]
        
        if not healthy_instances:
            return None
        
        instance = healthy_instances[self.round_robin_index % len(healthy_instances)]
        self.round_robin_index += 1
        
        return instance
    
    def update_instance_load(self, instance: str, load: float):
        """Update load metric for instance"""
        if instance in self.instance_loads:
            self.instance_loads[instance] = load
    
    def mark_instance_unhealthy(self, instance: str):
        """Mark instance as unhealthy"""
        if instance in self.instance_health:
            self.instance_health[instance] = False
            self.logger.warning(f"🚨 Instance {instance} marked as unhealthy")
    
    def mark_instance_healthy(self, instance: str):
        """Mark instance as healthy"""
        if instance in self.instance_health:
            self.instance_health[instance] = True
            self.logger.info(f"✅ Instance {instance} marked as healthy")

class DistributedCache:
    """Distributed cache using Redis for shared state"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.cache_prefix = "poise_cache:"
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def set_cached_data(self, key: str, data: Any, ttl_seconds: int = 3600):
        """Set data in distributed cache"""
        
        cache_key = f"{self.cache_prefix}{key}"
        
        try:
            serialized_data = pickle.dumps(data)
            self.redis_client.setex(cache_key, ttl_seconds, serialized_data)
            self.logger.info(f"💾 Cached data with key: {key}")
            
        except Exception as e:
            self.logger.error(f"❌ Error caching data {key}: {e}")
    
    async def get_cached_data(self, key: str) -> Any:
        """Get data from distributed cache"""
        
        cache_key = f"{self.cache_prefix}{key}"
        
        try:
            serialized_data = self.redis_client.get(cache_key)
            
            if serialized_data:
                data = pickle.loads(serialized_data)
                self.logger.info(f"🎯 Cache hit for key: {key}")
                return data
            else:
                self.logger.info(f"❌ Cache miss for key: {key}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting cached data {key}: {e}")
            return None
    
    async def invalidate_cache(self, key: str):
        """Invalidate cached data"""
        
        cache_key = f"{self.cache_prefix}{key}"
        self.redis_client.delete(cache_key)
        self.logger.info(f"🗑️ Invalidated cache key: {key}")
    
    async def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        
        keys = self.redis_client.keys(f"{self.cache_prefix}*")
        total_memory = sum(self.redis_client.memory_usage(key) or 0 for key in keys)
        
        return {
            'total_keys': len(keys),
            'total_memory_bytes': total_memory,
            'cache_prefix': self.cache_prefix
        }

class HealthMonitor:
    """Health monitoring for distributed components"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.components = {}
        self.running = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def register_component(self, component_id: str, health_check_func: Callable[[], bool]):
        """Register component for health monitoring"""
        
        self.components[component_id] = {
            'health_check': health_check_func,
            'status': 'unknown',
            'last_check': None,
            'consecutive_failures': 0
        }
        
        self.logger.info(f"🏥 Registered component for monitoring: {component_id}")
    
    async def start_monitoring(self):
        """Start health monitoring loop"""
        
        self.running = True
        
        while self.running:
            for component_id, component_info in self.components.items():
                try:
                    is_healthy = component_info['health_check']()
                    
                    if is_healthy:
                        component_info['status'] = 'healthy'
                        component_info['consecutive_failures'] = 0
                        self.logger.info(f"✅ Component {component_id} is healthy")
                    else:
                        component_info['status'] = 'unhealthy'
                        component_info['consecutive_failures'] += 1
                        self.logger.warning(f"🚨 Component {component_id} is unhealthy (failures: {component_info['consecutive_failures']})")
                    
                    component_info['last_check'] = datetime.now()
                    
                except Exception as e:
                    component_info['status'] = 'error'
                    component_info['consecutive_failures'] += 1
                    self.logger.error(f"❌ Error checking health of {component_id}: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
    
    def get_health_status(self) -> Dict:
        """Get overall health status"""
        
        healthy_count = sum(1 for comp in self.components.values() if comp['status'] == 'healthy')
        total_count = len(self.components)
        
        overall_health = 'healthy' if healthy_count == total_count else 'degraded' if healthy_count > 0 else 'critical'
        
        return {
            'overall_health': overall_health,
            'healthy_components': healthy_count,
            'total_components': total_count,
            'components': self.components,
            'timestamp': datetime.now().isoformat()
        }

class DistributedOrchestrator:
    """Master orchestrator for distributed trading system"""
    
    def __init__(self, node_id: str, redis_host: str = 'localhost'):
        self.node_id = node_id
        self.message_broker = MessageBroker(redis_host)
        self.task_distributor = TaskDistributor()
        self.load_balancer = LoadBalancer([])
        self.distributed_cache = DistributedCache(redis_host)
        self.health_monitor = HealthMonitor()
        
        # Node registry
        self.known_nodes = set()
        self.node_capabilities = {}
        
        # Service discovery
        self.services = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def start_node(self):
        """Start distributed node"""
        
        self.logger.info(f"🚀 Starting distributed node: {self.node_id}")
        
        # Register health checks
        self.health_monitor.register_component("message_broker", self._check_message_broker_health)
        self.health_monitor.register_component("task_distributor", self._check_task_distributor_health)
        self.health_monitor.register_component("distributed_cache", self._check_cache_health)
        
        # Start health monitoring
        asyncio.create_task(self.health_monitor.start_monitoring())
        
        # Start message processing
        asyncio.create_task(self.message_broker.subscribe_to_messages(self._handle_distributed_message))
        
        # Send heartbeat
        asyncio.create_task(self._send_heartbeat())
        
        # Register node
        await self._register_node()
        
        self.logger.info(f"✅ Distributed node {self.node_id} started successfully")
    
    async def _register_node(self):
        """Register node in distributed system"""
        
        node_info = {
            'node_id': self.node_id,
            'capabilities': ['trading', 'analytics', 'risk_management'],
            'max_load': 100,
            'current_load': 0,
            'services': ['market_data', 'order_execution', 'portfolio_management']
        }
        
        # Store in distributed cache
        await self.distributed_cache.set_cached_data(f"node:{self.node_id}", node_info)
        
        # Announce to other nodes
        announcement = DistributedMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SYSTEM_STATUS,
            sender_id=self.node_id,
            recipient_id=None,  # Broadcast
            timestamp=datetime.now(),
            priority=5,
            payload={'action': 'node_registration', 'node_info': node_info}
        )
        
        await self.message_broker.publish_message(announcement)
    
    async def _send_heartbeat(self):
        """Send periodic heartbeat"""
        
        while True:
            heartbeat = DistributedMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.HEARTBEAT,
                sender_id=self.node_id,
                recipient_id=None,
                timestamp=datetime.now(),
                priority=8,
                payload={'status': 'alive', 'load': self._get_current_load()}
            )
            
            await self.message_broker.publish_message(heartbeat)
            await asyncio.sleep(30)  # Heartbeat every 30 seconds
    
    async def _handle_distributed_message(self, message: DistributedMessage):
        """Handle incoming distributed messages"""
        
        if message.message_type == MessageType.TASK_REQUEST:
            await self._handle_task_request(message)
        elif message.message_type == MessageType.HEARTBEAT:
            await self._handle_heartbeat(message)
        elif message.message_type == MessageType.SYSTEM_STATUS:
            await self._handle_system_status(message)
        else:
            self.logger.info(f"📨 Received message: {message.message_type} from {message.sender_id}")
    
    async def _handle_task_request(self, message: DistributedMessage):
        """Handle task execution request"""
        
        task_data = message.payload.get('task_data', {})
        task_type = task_data.get('type')
        
        if task_type == 'portfolio_optimization':
            task_id = await self.task_distributor.submit_cpu_intensive_task(
                self._execute_portfolio_optimization,
                task_data
            )
        elif task_type == 'market_analysis':
            task_id = await self.task_distributor.submit_io_intensive_task(
                self._execute_market_analysis,
                task_data
            )
        else:
            self.logger.warning(f"⚠️ Unknown task type: {task_type}")
            return
        
        # Send response
        response = DistributedMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_RESPONSE,
            sender_id=self.node_id,
            recipient_id=message.sender_id,
            timestamp=datetime.now(),
            priority=message.priority,
            payload={'task_id': task_id, 'status': 'accepted'}
        )
        
        await self.message_broker.publish_message(response)
    
    async def _handle_heartbeat(self, message: DistributedMessage):
        """Handle heartbeat from other nodes"""
        
        sender_node = message.sender_id
        node_load = message.payload.get('load', 0)
        
        # Update load balancer
        if sender_node not in self.load_balancer.instances:
            self.load_balancer.instances.append(sender_node)
        
        self.load_balancer.update_instance_load(sender_node, node_load)
        self.load_balancer.mark_instance_healthy(sender_node)
    
    async def _handle_system_status(self, message: DistributedMessage):
        """Handle system status messages"""
        
        action = message.payload.get('action')
        
        if action == 'node_registration':
            node_info = message.payload.get('node_info', {})
            self.known_nodes.add(message.sender_id)
            self.node_capabilities[message.sender_id] = node_info.get('capabilities', [])
    
    def _execute_portfolio_optimization(self, task_data: Dict) -> Dict:
        """Execute portfolio optimization task"""
        # Placeholder for actual portfolio optimization
        time.sleep(2)  # Simulate computation
        return {'status': 'completed', 'optimal_weights': [0.4, 0.3, 0.3]}
    
    def _execute_market_analysis(self, task_data: Dict) -> Dict:
        """Execute market analysis task"""
        # Placeholder for actual market analysis
        time.sleep(1)  # Simulate I/O operations
        return {'status': 'completed', 'signals': ['BUY_BTC', 'HOLD_ETH']}
    
    def _get_current_load(self) -> float:
        """Get current node load percentage"""
        # Simplified load calculation
        active_tasks = len([task for task in self.task_distributor.results.values() if task['status'] == 'running'])
        return min(100, (active_tasks / self.task_distributor.max_workers) * 100)
    
    def _check_message_broker_health(self) -> bool:
        """Check message broker health"""
        try:
            return self.message_broker.redis_client.ping()
        except:
            return False
    
    def _check_task_distributor_health(self) -> bool:
        """Check task distributor health"""
        return not self.task_distributor.process_executor._broken
    
    def _check_cache_health(self) -> bool:
        """Check distributed cache health"""
        try:
            return self.distributed_cache.redis_client.ping()
        except:
            return False
    
    async def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        
        health_status = self.health_monitor.get_health_status()
        cache_stats = await self.distributed_cache.get_cache_stats()
        
        return {
            'node_id': self.node_id,
            'health_status': health_status,
            'known_nodes': list(self.known_nodes),
            'load_balancer_status': {
                'instances': self.load_balancer.instances,
                'instance_loads': self.load_balancer.instance_loads,
                'instance_health': self.load_balancer.instance_health
            },
            'cache_stats': cache_stats,
            'active_tasks': len([t for t in self.task_distributor.results.values() if t['status'] == 'running']),
            'current_load': self._get_current_load(),
            'timestamp': datetime.now().isoformat()
        }

# Global distributed orchestrator
distributed_orchestrator = None

def create_distributed_node(node_id: str, redis_host: str = 'localhost') -> DistributedOrchestrator:
    """Create and initialize distributed node"""
    global distributed_orchestrator
    distributed_orchestrator = DistributedOrchestrator(node_id, redis_host)
    return distributed_orchestrator
