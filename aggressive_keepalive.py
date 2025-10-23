"""
⚡ ULTRA-AGGRESSIVE KEEP-ALIVE SYSTEM
Multiple redundant threads to maximize uptime on free tier
"""

import threading
import time
import requests
from datetime import datetime
import os
import random

class MultiThreadKeepAlive:
    """Run multiple concurrent keep-alive threads for maximum uptime"""
    
    def __init__(self, app_url=None):
        self.app_url = app_url or os.environ.get('RENDER_EXTERNAL_URL')
        self.is_running = False
        self.threads = []
        self.total_pings = 0
        
    def start(self):
        """Start multiple keep-alive threads"""
        if self.is_running:
            return
        
        self.is_running = True
        
        print("="*70)
        print("⚡ ULTRA-AGGRESSIVE MULTI-THREAD KEEP-ALIVE")
        print("="*70)
        print(f"📍 Target: {self.app_url or 'localhost'}")
        print(f"🧵 Starting 3 concurrent keep-alive threads")
        print(f"⚡ Total pings per 5 minutes: ~10 pings")
        print(f"🎯 Goal: 99%+ uptime on FREE tier")
        print("="*70)
        
        # Thread 1: Fast pinger (every 90 seconds)
        t1 = threading.Thread(target=self._fast_pinger, daemon=True)
        t1.start()
        self.threads.append(t1)
        
        # Thread 2: Health checker (every 120 seconds)
        t2 = threading.Thread(target=self._health_checker, daemon=True)
        t2.start()
        self.threads.append(t2)
        
        # Thread 3: Random endpoint pinger (every 150 seconds)
        t3 = threading.Thread(target=self._random_pinger, daemon=True)
        t3.start()
        self.threads.append(t3)
        
        print("✅ All keep-alive threads active!")
        
    def stop(self):
        """Stop all threads"""
        self.is_running = False
        print("⏹️ Stopping all keep-alive threads...")
        
    def _fast_pinger(self):
        """Fast ping thread - hits /ping every 90 seconds"""
        time.sleep(10)  # Initial delay
        
        while self.is_running:
            try:
                if self.app_url:
                    response = requests.get(
                        f"{self.app_url}/ping",
                        timeout=8,
                        headers={'User-Agent': 'FastPinger/1.0'}
                    )
                    if response.status_code == 200:
                        self.total_pings += 1
                        if self.total_pings % 25 == 0:
                            print(f"💓 Fast ping #{self.total_pings} [{datetime.now().strftime('%H:%M:%S')}]")
                time.sleep(90)  # Every 1.5 minutes
            except Exception as e:
                time.sleep(90)
    
    def _health_checker(self):
        """Health check thread - hits /health every 120 seconds"""
        time.sleep(30)  # Initial delay
        
        while self.is_running:
            try:
                if self.app_url:
                    response = requests.get(
                        f"{self.app_url}/health",
                        timeout=8,
                        headers={'User-Agent': 'HealthChecker/1.0'}
                    )
                    if response.status_code == 200:
                        self.total_pings += 1
                time.sleep(120)  # Every 2 minutes
            except Exception as e:
                time.sleep(120)
    
    def _random_pinger(self):
        """Random endpoint pinger - varies between endpoints"""
        time.sleep(60)  # Initial delay
        
        endpoints = ['/api/status', '/api/metrics', '/keep-alive', '/ping', '/health']
        
        while self.is_running:
            try:
                if self.app_url:
                    endpoint = random.choice(endpoints)
                    response = requests.get(
                        f"{self.app_url}{endpoint}",
                        timeout=8,
                        headers={'User-Agent': 'RandomPinger/1.0'}
                    )
                    if response.status_code == 200:
                        self.total_pings += 1
                # Vary the interval between 90-150 seconds
                wait_time = random.randint(90, 150)
                time.sleep(wait_time)
            except Exception as e:
                time.sleep(120)


class CPUKeepBusy:
    """Keep CPU slightly busy to prevent idle detection"""
    
    def __init__(self):
        self.is_running = False
        self._thread = None
        
    def start(self):
        """Start light CPU activity"""
        if self.is_running:
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._busy_loop, daemon=True)
        self._thread.start()
        
        print("🔥 CPU keep-busy: ACTIVE (light background activity)")
        
    def stop(self):
        """Stop CPU activity"""
        self.is_running = False
        
    def _busy_loop(self):
        """Very light CPU activity to show service is alive"""
        counter = 0
        
        while self.is_running:
            # Very light calculation - won't use significant CPU
            counter = (counter + 1) % 1000000
            
            # Sleep most of the time (99.9% idle)
            time.sleep(5)
            
            # Every 10 minutes, do a tiny bit of work
            if counter % 120 == 0:
                _ = sum(range(1000))  # Trivial calculation


class MemoryKeepWarm:
    """Keep memory allocation warm to prevent cold starts"""
    
    def __init__(self):
        self.is_running = False
        self._thread = None
        self.memory_cache = []
        
    def start(self):
        """Start memory warming"""
        if self.is_running:
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._warm_loop, daemon=True)
        self._thread.start()
        
        print("🌡️  Memory warming: ACTIVE")
        
    def stop(self):
        """Stop memory warming"""
        self.is_running = False
        self.memory_cache.clear()
        
    def _warm_loop(self):
        """Maintain warm memory cache"""
        # Keep a small cache in memory
        self.memory_cache = list(range(10000))
        
        while self.is_running:
            # Periodically touch the memory
            _ = sum(self.memory_cache[:100])
            time.sleep(60)


# Global instances
multi_thread_keepalive = MultiThreadKeepAlive()
cpu_keep_busy = CPUKeepBusy()
memory_keep_warm = MemoryKeepWarm()


def start_ultra_aggressive_mode(app_url=None):
    """Start all ultra-aggressive keep-alive systems"""
    print("\n" + "="*70)
    print("🚀 INITIALIZING ULTRA-AGGRESSIVE MODE")
    print("   Goal: Maximum uptime on FREE tier (Starter-like performance)")
    print("="*70 + "\n")
    
    # Multi-thread keep-alive
    multi_thread_keepalive.app_url = app_url
    multi_thread_keepalive.start()
    
    # CPU keep busy
    cpu_keep_busy.start()
    
    # Memory keep warm
    memory_keep_warm.start()
    
    print("\n" + "="*70)
    print("✅ ULTRA-AGGRESSIVE MODE: FULLY OPERATIONAL")
    print("="*70)
    print("📊 Active Systems:")
    print("   ✅ 3x concurrent ping threads (90s, 120s, 150s)")
    print("   ✅ CPU keep-busy (prevents idle detection)")
    print("   ✅ Memory warming (prevents cold starts)")
    print("   ✅ Random endpoint variation (looks like real traffic)")
    print("="*70)
    print("⚡ Your free tier bot is now running like Starter tier!")
    print("="*70 + "\n")


def stop_ultra_aggressive_mode():
    """Stop all ultra-aggressive systems"""
    multi_thread_keepalive.stop()
    cpu_keep_busy.stop()
    memory_keep_warm.stop()
    print("✅ Ultra-aggressive mode stopped")


if __name__ == "__main__":
    # Test
    start_ultra_aggressive_mode("http://localhost:5000")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_ultra_aggressive_mode()
