"""
🔄 24/7 KEEP-ALIVE SYSTEM FOR RENDER
Prevents service from sleeping and maintains active connections
"""

import threading
import time
import requests
from datetime import datetime
import os

class KeepAliveSystem:
    """Keeps the service alive 24/7 by self-pinging"""
    
    def __init__(self, app_url=None, ping_interval=300):
        """
        Initialize keep-alive system
        
        Args:
            app_url: Your Render app URL (e.g., https://your-app.onrender.com)
            ping_interval: Seconds between pings (default: 300 = 5 minutes)
        """
        self.app_url = app_url or os.environ.get('RENDER_EXTERNAL_URL')
        self.ping_interval = ping_interval
        self.is_running = False
        self.ping_count = 0
        self.last_ping_time = None
        self.failed_pings = 0
        self._thread = None
        
    def start(self):
        """Start the keep-alive system"""
        if self.is_running:
            print("⚠️  Keep-alive already running")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._thread.start()
        
        print("="*60)
        print("🔄 KEEP-ALIVE SYSTEM ACTIVATED")
        print("="*60)
        print(f"📍 Target URL: {self.app_url or 'localhost (dev mode)'}")
        print(f"⏱️  Ping Interval: {self.ping_interval}s ({self.ping_interval//60} minutes)")
        print(f"🎯 Status: Service will stay awake 24/7")
        print("="*60)
        
    def stop(self):
        """Stop the keep-alive system"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("⏹️ Keep-alive system stopped")
        
    def _ping_loop(self):
        """Background loop that pings the service"""
        # Wait a bit before first ping
        time.sleep(30)
        
        while self.is_running:
            try:
                self._do_ping()
                time.sleep(self.ping_interval)
            except Exception as e:
                print(f"❌ Keep-alive error: {e}")
                time.sleep(60)  # Wait a minute on error
    
    def _do_ping(self):
        """Perform a single ping"""
        if not self.app_url:
            # Dev mode - just count
            self.ping_count += 1
            self.last_ping_time = datetime.now()
            if self.ping_count % 10 == 0:
                print(f"💓 Keep-alive heartbeat #{self.ping_count} [{datetime.now().strftime('%H:%M:%S')}]")
            return
        
        try:
            # Ping the health endpoint
            response = requests.get(
                f"{self.app_url}/health",
                timeout=10,
                headers={'User-Agent': 'KeepAlive/1.0'}
            )
            
            if response.status_code == 200:
                self.ping_count += 1
                self.last_ping_time = datetime.now()
                self.failed_pings = 0
                
                # Log every 10th ping
                if self.ping_count % 10 == 0:
                    print(f"💓 Keep-alive ping #{self.ping_count} successful [{datetime.now().strftime('%H:%M:%S')}]")
            else:
                self.failed_pings += 1
                print(f"⚠️  Ping failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.failed_pings += 1
            if self.failed_pings % 5 == 0:
                print(f"⚠️  Connection issue (attempt {self.failed_pings}): {e}")
        except Exception as e:
            print(f"❌ Ping error: {e}")
    
    def get_status(self):
        """Get current status"""
        return {
            'running': self.is_running,
            'ping_count': self.ping_count,
            'last_ping': self.last_ping_time.isoformat() if self.last_ping_time else None,
            'failed_pings': self.failed_pings,
            'app_url': self.app_url
        }


class ConnectionMonitor:
    """Monitors and recovers from connection issues"""
    
    def __init__(self):
        self.is_monitoring = False
        self.connection_checks = 0
        self.reconnection_attempts = 0
        self._thread = None
        
    def start(self, bot_instance=None):
        """Start monitoring connections"""
        if self.is_monitoring:
            return
        
        self.bot_instance = bot_instance
        self.is_monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
        print("🔍 Connection Monitor: ACTIVE")
        
    def stop(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Monitor connection health"""
        while self.is_monitoring:
            try:
                self._check_connections()
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                time.sleep(60)
    
    def _check_connections(self):
        """Check if connections are healthy"""
        self.connection_checks += 1
        
        if self.bot_instance:
            # Check if bot is still responsive
            if hasattr(self.bot_instance, 'trader'):
                try:
                    # Simple health check - access trader data
                    _ = self.bot_instance.trader.cash_balance
                    
                    # Log every 30 minutes
                    if self.connection_checks % 30 == 0:
                        print(f"✅ Connection check #{self.connection_checks}: All systems operational")
                except Exception as e:
                    print(f"⚠️  Connection issue detected: {e}")
                    self._attempt_recovery()
    
    def _attempt_recovery(self):
        """Attempt to recover from connection issues"""
        self.reconnection_attempts += 1
        print(f"🔄 Attempting recovery (attempt {self.reconnection_attempts})...")
        
        # Add recovery logic here if needed
        # For now, just log - the bot should handle its own reconnection


class ActivitySimulator:
    """Simulates user activity to keep service active"""
    
    def __init__(self, endpoints=['/api/status', '/api/metrics', '/api/portfolio']):
        self.endpoints = endpoints
        self.is_running = False
        self.activity_count = 0
        self._thread = None
        
    def start(self, base_url=None):
        """Start simulating activity"""
        if self.is_running:
            return
        
        self.base_url = base_url or os.environ.get('RENDER_EXTERNAL_URL')
        if not self.base_url:
            print("ℹ️  Activity simulator: Dev mode (no URL)")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._activity_loop, daemon=True)
        self._thread.start()
        
        print("🎭 Activity Simulator: ACTIVE")
        
    def stop(self):
        """Stop activity simulation"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _activity_loop(self):
        """Simulate periodic activity"""
        # Wait before starting
        time.sleep(120)
        
        while self.is_running:
            try:
                self._simulate_activity()
                time.sleep(180)  # Every 3 minutes
            except Exception as e:
                print(f"❌ Activity simulator error: {e}")
                time.sleep(180)
    
    def _simulate_activity(self):
        """Simulate user activity"""
        if not self.base_url:
            return
        
        try:
            # Randomly pick an endpoint
            import random
            endpoint = random.choice(self.endpoints)
            
            response = requests.get(
                f"{self.base_url}{endpoint}",
                timeout=10,
                headers={'User-Agent': 'ActivitySimulator/1.0'}
            )
            
            self.activity_count += 1
            
            if self.activity_count % 20 == 0:
                print(f"🎭 Simulated activity #{self.activity_count} [{datetime.now().strftime('%H:%M:%S')}]")
                
        except Exception as e:
            pass  # Silently fail to avoid spam


# Global instances
keep_alive = KeepAliveSystem()
connection_monitor = ConnectionMonitor()
activity_simulator = ActivitySimulator()


def start_all_systems(app_url=None, bot_instance=None):
    """Start all keep-alive systems"""
    print("\n" + "="*60)
    print("🚀 INITIALIZING 24/7 KEEP-ALIVE SYSTEMS")
    print("="*60)
    
    # Start keep-alive pinger
    keep_alive.app_url = app_url or os.environ.get('RENDER_EXTERNAL_URL')
    keep_alive.start()
    
    # Start connection monitor
    connection_monitor.start(bot_instance)
    
    # Start activity simulator
    activity_simulator.start(app_url)
    
    print("="*60)
    print("✅ ALL KEEP-ALIVE SYSTEMS OPERATIONAL")
    print("="*60)
    print()


def stop_all_systems():
    """Stop all keep-alive systems"""
    keep_alive.stop()
    connection_monitor.stop()
    activity_simulator.stop()


if __name__ == "__main__":
    # Test the system
    print("Testing Keep-Alive System...")
    start_all_systems("http://localhost:5000")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_all_systems()
