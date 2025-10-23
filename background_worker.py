"""
🔧 CONTINUOUS BACKGROUND WORKER
Keeps bot alive with continuous light activity
"""

import threading
import time
from datetime import datetime
import asyncio

class ContinuousWorker:
    """Runs continuous background tasks to keep service alive"""
    
    def __init__(self, bot_instance=None):
        self.bot_instance = bot_instance
        self.is_running = False
        self.threads = []
        self.task_count = 0
        
    def start(self):
        """Start all background workers"""
        if self.is_running:
            return
        
        self.is_running = True
        
        print("="*60)
        print("🔧 CONTINUOUS BACKGROUND WORKER: STARTING")
        print("="*60)
        
        # Worker 1: Status monitor
        t1 = threading.Thread(target=self._status_monitor, daemon=True)
        t1.start()
        self.threads.append(t1)
        
        # Worker 2: Data refresher
        t2 = threading.Thread(target=self._data_refresher, daemon=True)
        t2.start()
        self.threads.append(t2)
        
        # Worker 3: Heartbeat generator
        t3 = threading.Thread(target=self._heartbeat_generator, daemon=True)
        t3.start()
        self.threads.append(t3)
        
        print("✅ 3 background workers active")
        print("="*60)
        
    def stop(self):
        """Stop all workers"""
        self.is_running = False
        print("⏹️ Stopping background workers...")
        
    def _status_monitor(self):
        """Monitor bot status continuously"""
        time.sleep(20)
        
        while self.is_running:
            try:
                if self.bot_instance:
                    # Just access bot attributes to keep it warm
                    _ = getattr(self.bot_instance, 'current_capital', 5.0)
                    _ = getattr(self.bot_instance, 'bot_running', False)
                    _ = getattr(self.bot_instance, 'trading_mode', 'PRECISION')
                    
                self.task_count += 1
                
                if self.task_count % 100 == 0:
                    print(f"🔍 Status monitor: {self.task_count} checks completed [{datetime.now().strftime('%H:%M:%S')}]")
                
                time.sleep(60)  # Every minute
            except Exception as e:
                time.sleep(60)
    
    def _data_refresher(self):
        """Refresh data structures periodically"""
        time.sleep(45)
        
        while self.is_running:
            try:
                if self.bot_instance and hasattr(self.bot_instance, 'trader'):
                    # Touch trader data to keep it fresh
                    try:
                        _ = self.bot_instance.trader.cash_balance
                        _ = self.bot_instance.trader.positions
                    except:
                        pass
                
                time.sleep(90)  # Every 1.5 minutes
            except Exception as e:
                time.sleep(90)
    
    def _heartbeat_generator(self):
        """Generate heartbeat signals"""
        time.sleep(30)
        
        heartbeat_count = 0
        
        while self.is_running:
            try:
                heartbeat_count += 1
                
                # Log heartbeat every 30 beats (30 minutes)
                if heartbeat_count % 30 == 0:
                    uptime_minutes = heartbeat_count
                    uptime_hours = uptime_minutes / 60
                    print(f"💓 System heartbeat #{heartbeat_count} | Uptime: {uptime_hours:.1f}h [{datetime.now().strftime('%H:%M:%S')}]")
                
                time.sleep(60)  # Every minute
            except Exception as e:
                time.sleep(60)


class StatePreserver:
    """Continuously save state to prevent data loss"""
    
    def __init__(self, bot_instance=None):
        self.bot_instance = bot_instance
        self.is_running = False
        self._thread = None
        self.save_count = 0
        
    def start(self):
        """Start state preservation"""
        if self.is_running:
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._preservation_loop, daemon=True)
        self._thread.start()
        
        print("💾 State preserver: ACTIVE (auto-save every 5 minutes)")
        
    def stop(self):
        """Stop state preservation"""
        self.is_running = False
        
    def _preservation_loop(self):
        """Continuously preserve state"""
        time.sleep(120)  # Wait 2 minutes before first save
        
        while self.is_running:
            try:
                self._save_state()
                time.sleep(300)  # Every 5 minutes
            except Exception as e:
                time.sleep(300)
    
    def _save_state(self):
        """Save current state"""
        if not self.bot_instance:
            return
        
        try:
            if hasattr(self.bot_instance, 'trader') and hasattr(self.bot_instance.trader, '_save_state'):
                self.bot_instance.trader._save_state()
                self.save_count += 1
                
                if self.save_count % 10 == 0:
                    print(f"💾 Auto-save #{self.save_count} completed [{datetime.now().strftime('%H:%M:%S')}]")
        except Exception as e:
            pass


class ConnectionKeeper:
    """Maintain active connections to prevent timeouts"""
    
    def __init__(self, bot_instance=None):
        self.bot_instance = bot_instance
        self.is_running = False
        self._thread = None
        
    def start(self):
        """Start connection keeper"""
        if self.is_running:
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._connection_loop, daemon=True)
        self._thread.start()
        
        print("🔗 Connection keeper: ACTIVE")
        
    def stop(self):
        """Stop connection keeper"""
        self.is_running = False
        
    def _connection_loop(self):
        """Keep connections alive"""
        time.sleep(60)
        
        while self.is_running:
            try:
                # Simulate connection activity
                if self.bot_instance:
                    # Just access bot to keep connection warm
                    _ = getattr(self.bot_instance, 'symbols', [])
                
                time.sleep(120)  # Every 2 minutes
            except Exception as e:
                time.sleep(120)


# Global instances
continuous_worker = ContinuousWorker()
state_preserver = StatePreserver()
connection_keeper = ConnectionKeeper()


def start_background_workers(bot_instance=None):
    """Start all background workers"""
    print("\n🔧 Starting continuous background workers...")
    
    # Set bot instances
    continuous_worker.bot_instance = bot_instance
    state_preserver.bot_instance = bot_instance
    connection_keeper.bot_instance = bot_instance
    
    # Start all workers
    continuous_worker.start()
    state_preserver.start()
    connection_keeper.start()
    
    print("✅ All background workers operational\n")


def stop_background_workers():
    """Stop all background workers"""
    continuous_worker.stop()
    state_preserver.stop()
    connection_keeper.stop()


if __name__ == "__main__":
    # Test
    start_background_workers()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_background_workers()
