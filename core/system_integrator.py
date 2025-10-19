#!/usr/bin/env python3
"""
🔗 SYSTEM INTEGRATOR - UNIFIED OPTIMIZATION LAYER
Integrates all optimization components with the existing Poise Trader architecture
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import threading
import signal
import sys

# Import all optimization components
from .memory_manager import memory_manager, AdvancedMemoryManager
from .async_manager import async_manager, AsyncManager
from .enhanced_ml_system import model_registry, online_learning_engine, feature_engineer
from .config_manager import config_manager, ConfigurationManager
from .advanced_order_system import initialize_advanced_order_system
from .performance_analytics import performance_analyzer, TradeRecord

logger = logging.getLogger(__name__)

class OptimizedTradingSystem:
    """Unified system that integrates all optimizations with existing Poise Trader"""
    
    def __init__(self):
        self.initialized = False
        self.components = {}
        self.config = None
        self.exchange_client = None
        
        # Component states
        self._async_manager: Optional[AsyncManager] = None
        self._memory_manager: Optional[AdvancedMemoryManager] = None
        self._order_system = None
        
        # Integration flags
        self.ai_enhanced = False
        self.async_enabled = False
        self.advanced_orders_enabled = False
        
    async def initialize(self, exchange_client=None):
        """Initialize all optimization components"""
        if self.initialized:
            return
        
        logger.info("🚀 Initializing Optimized Trading System...")
        
        try:
            # 1. Load configuration
            self.config = config_manager
            self.config.start_file_watching()
            
            # 2. Initialize async manager
            self._async_manager = async_manager
            await self._async_manager.initialize()
            self.async_enabled = True
            
            # 3. Initialize memory manager
            self._memory_manager = memory_manager
            self._setup_memory_pools()
            
            # 4. Initialize ML system
            await self._initialize_ml_system()
            
            # 5. Initialize advanced order system
            if exchange_client:
                self.exchange_client = exchange_client
                self._order_system = initialize_advanced_order_system(exchange_client)
                self.advanced_orders_enabled = True
            
            # 6. Start online learning
            online_learning_engine.start_online_learning()
            self.ai_enhanced = True
            
            # 7. Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self.initialized = True
            logger.info("✅ Optimized Trading System initialized successfully")
            
            # Print system status
            self._print_system_status()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize optimized system: {e}")
            raise
    
    def _setup_memory_pools(self):
        """Setup memory pools for common objects"""
        try:
            # Price data pool
            self._memory_manager.register_resource_pool(
                'price_data', 
                lambda: {'prices': [], 'volumes': [], 'timestamps': []},
                max_size=50
            )
            
            # Order pool
            self._memory_manager.register_resource_pool(
                'orders',
                lambda: {'symbol': '', 'side': '', 'quantity': 0, 'price': 0},
                max_size=100
            )
            
            # Analysis pool
            self._memory_manager.register_resource_pool(
                'analysis',
                lambda: {'indicators': {}, 'signals': [], 'confidence': 0},
                max_size=30
            )
            
            logger.info("🧠 Memory pools configured")
            
        except Exception as e:
            logger.error(f"Memory pool setup failed: {e}")
    
    async def _initialize_ml_system(self):
        """Initialize ML components"""
        try:
            # Load existing models
            models = model_registry.list_models()
            if models:
                logger.info(f"📚 Loaded {len(models)} saved models: {models}")
            
            # Configure online learning
            config = self.config.get_ai_config()
            if config.enable_online_learning:
                logger.info("🎯 Online learning enabled")
            
            logger.info("🤖 ML system initialized")
            
        except Exception as e:
            logger.error(f"ML initialization failed: {e}")
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def process_market_data(self, symbol: str, price: float, volume: float = None, 
                                timestamp: datetime = None) -> Dict[str, Any]:
        """Enhanced market data processing with optimizations"""
        if not self.initialized:
            return {}
        
        timestamp = timestamp or datetime.now()
        
        # Use memory pool for price data
        with self._memory_manager.get_from_pool('price_data') as price_data:
            price_data['prices'].append(price)
            if volume:
                price_data['volumes'].append(volume)
            price_data['timestamps'].append(timestamp)
            
            # Add to memory manager cache
            self._memory_manager.add_price_data(symbol, price, timestamp)
            
            # Generate enhanced features
            price_history = self._memory_manager.get_price_history(symbol, max_points=100)
            
            if len(price_history) >= 20:
                # Enhanced feature engineering
                features = feature_engineer.generate_features(
                    price_history, 
                    [volume] * len(price_history) if volume else None
                )
                
                return {
                    'symbol': symbol,
                    'price': price,
                    'volume': volume,
                    'timestamp': timestamp,
                    'features': features,
                    'price_history_length': len(price_history)
                }
        
        return {
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'timestamp': timestamp,
            'features': {},
            'price_history_length': 0
        }
    
    async def execute_enhanced_order(self, order_params: Dict[str, Any]) -> str:
        """Execute order with advanced order management"""
        if not self.advanced_orders_enabled:
            logger.warning("Advanced orders not available, falling back to basic execution")
            return "basic_order"
        
        try:
            from .advanced_order_system import AdvancedOrder, OrderType, OrderSide
            
            # Create advanced order
            order = AdvancedOrder(
                symbol=order_params['symbol'],
                side=OrderSide(order_params['side']),
                order_type=OrderType(order_params.get('order_type', 'smart')),
                quantity=order_params['quantity'],
                price=order_params.get('price'),
                iceberg_visible_qty=order_params.get('iceberg_qty'),
                twap_duration_minutes=order_params.get('twap_duration'),
                max_participation_rate=order_params.get('participation_rate', 0.1)
            )
            
            # Execute with smart routing
            order_id = await self._order_system.submit_smart_order(order)
            
            logger.info(f"🎯 Advanced order executed: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Advanced order execution failed: {e}")
            raise
    
    async def process_trade_completion(self, trade_data: Dict[str, Any]):
        """Process completed trade with all optimizations"""
        try:
            # Create trade record for analytics
            trade_record = TradeRecord(
                trade_id=trade_data.get('trade_id', ''),
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                entry_price=trade_data['entry_price'],
                exit_price=trade_data.get('exit_price'),
                quantity=trade_data['quantity'],
                entry_time=trade_data['entry_time'],
                exit_time=trade_data.get('exit_time'),
                strategy=trade_data.get('strategy', 'unknown'),
                confidence=trade_data.get('confidence', 0.5),
                fees=trade_data.get('fees', 0),
                exit_reason=trade_data.get('exit_reason', '')
            )
            
            # Add to performance analytics
            performance_analyzer.add_trade(trade_record)
            
            # Update ML models with trade outcome
            if trade_record.exit_price and self.ai_enhanced:
                features = trade_data.get('features', [])
                if features:
                    model_name = f"{trade_record.symbol}_{trade_record.strategy}"
                    
                    # Add training sample
                    online_learning_engine.add_training_sample(
                        model_name, 
                        features, 
                        trade_record.pnl
                    )
                    
                    # Add prediction result for performance tracking
                    predicted_pnl = trade_data.get('predicted_pnl', 0)
                    online_learning_engine.add_prediction_result(
                        model_name,
                        predicted_pnl,
                        trade_record.pnl,
                        features
                    )
            
            logger.info(f"📊 Trade processed: {trade_record.symbol} P&L: {trade_record.pnl:+.2f}")
            
        except Exception as e:
            logger.error(f"Trade processing failed: {e}")
    
    async def get_enhanced_signal(self, symbol: str, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get enhanced trading signal using all AI components"""
        try:
            if not self.ai_enhanced:
                return {'action': 'HOLD', 'confidence': 0.3}
            
            # Get price history
            price_history = self._memory_manager.get_price_history(symbol, max_points=100)
            
            if len(price_history) < 20:
                return {'action': 'HOLD', 'confidence': 0.3}
            
            # Get AI brain prediction
            from ai_brain import ai_brain
            prediction = ai_brain.get_enhanced_prediction(symbol, price_history, market_data)
            
            # Add ML features
            features = feature_engineer.generate_features(price_history)
            prediction['features'] = features
            
            # Apply configuration-based adjustments
            confidence_threshold = self.config.get('ai.confidence_threshold', 0.7)
            if prediction['confidence'] < confidence_threshold:
                prediction['action'] = 'HOLD'
                prediction['confidence'] = max(0.1, prediction['confidence'] * 0.8)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Enhanced signal generation failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.3}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'system': {
                'initialized': self.initialized,
                'async_enabled': self.async_enabled,
                'ai_enhanced': self.ai_enhanced,
                'advanced_orders_enabled': self.advanced_orders_enabled
            }
        }
        
        if self.initialized:
            # Memory stats
            if self._memory_manager:
                stats['memory'] = self._memory_manager.get_memory_stats()
            
            # Async stats
            if self._async_manager:
                stats['async'] = self._async_manager.get_comprehensive_stats()
            
            # ML stats
            stats['ml'] = online_learning_engine.get_learning_stats()
            
            # Performance stats
            performance_metrics = performance_analyzer.generate_performance_report()
            stats['performance'] = performance_metrics
            
            # Configuration stats
            stats['config'] = config_manager.get_stats()
        
        return stats
    
    def _print_system_status(self):
        """Print current system status"""
        print("\n" + "="*60)
        print("🏆 POISE TRADER - OPTIMIZED SYSTEM STATUS")
        print("="*60)
        
        print(f"🔧 Async Manager:        {'✅ Active' if self.async_enabled else '❌ Inactive'}")
        print(f"🧠 Memory Manager:       {'✅ Active' if self._memory_manager else '❌ Inactive'}")
        print(f"🤖 AI Enhancement:       {'✅ Active' if self.ai_enhanced else '❌ Inactive'}")
        print(f"🎯 Advanced Orders:      {'✅ Active' if self.advanced_orders_enabled else '❌ Inactive'}")
        print(f"⚙️ Configuration:        {'✅ Loaded' if self.config else '❌ Not Loaded'}")
        
        if self._memory_manager:
            memory_stats = self._memory_manager.get_memory_stats()
            print(f"💾 Memory Usage:         {memory_stats['memory_metrics']['memory_percent']:.1f}%")
            print(f"📊 Active Caches:        {memory_stats['cache_stats']['active_caches']}")
        
        if self.ai_enhanced:
            ml_stats = online_learning_engine.get_learning_stats()
            print(f"🎯 ML Models:            {ml_stats['total_models']}")
            print(f"🔄 Learning Active:      {'✅' if ml_stats['learning_active'] else '❌'}")
        
        print("="*60)
        print("🚀 System Ready for Trading!")
        print("="*60 + "\n")
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("🛑 Initiating system shutdown...")
        
        try:
            # Stop online learning
            if self.ai_enhanced:
                online_learning_engine.stop_online_learning()
            
            # Stop file watching
            if self.config:
                self.config.stop_file_watching()
            
            # Shutdown async manager
            if self._async_manager:
                await self._async_manager.shutdown()
            
            # Memory cleanup
            if self._memory_manager:
                self._memory_manager.cleanup_and_exit()
            
            # Save final state
            if self.ai_enhanced:
                from ai_brain import ai_brain
                ai_brain.save_brain()
            
            logger.info("✅ System shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
        finally:
            sys.exit(0)

# Global optimized system instance
optimized_system = OptimizedTradingSystem()

# Convenience functions for integration
async def initialize_optimizations(exchange_client=None):
    """Initialize all optimizations"""
    await optimized_system.initialize(exchange_client)

def get_optimized_system():
    """Get the global optimized system instance"""
    return optimized_system
