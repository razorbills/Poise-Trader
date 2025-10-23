#!/usr/bin/env python3
"""
🚀 RUN TRADING BOT WITH REAL DASHBOARD DATA
This script connects your trading bot to the dashboard for real-time monitoring
"""

import sys
import os
import asyncio
import threading
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import dashboard backend
from dashboard_backend import attach_bot, run_server

# Import the main trading bot
from micro_trading_bot import LegendaryCryptoTitanBot

def start_dashboard_server():
    """Start the dashboard server in a separate thread"""
    print("🎨 Starting Dashboard Server...")
    # Run server in thread so it doesn't block
    server_thread = threading.Thread(target=run_server, kwargs={'host': '0.0.0.0', 'port': 5000})
    server_thread.daemon = True
    server_thread.start()
    time.sleep(3)  # Give server time to start
    print("✅ Dashboard server running on http://localhost:5000")
    print("📊 React Dashboard: http://localhost:5173")

async def run_bot_with_dashboard():
    """Run the trading bot and connect it to dashboard"""
    print("\n" + "="*60)
    print("🤖 STARTING TRADING BOT WITH REAL DASHBOARD")
    print("="*60 + "\n")
    
    # Start dashboard server first
    start_dashboard_server()
    
    print("\n⚡ Initializing Trading Bot...")
    
    # Initialize the bot
    bot = LegendaryCryptoTitanBot()
    
    # Attach bot to dashboard for real data
    attach_bot(bot)
    print("✅ Bot connected to dashboard - Real data enabled!")
    
    # Initialize bot systems
    await bot.initialize()
    
    # Wait a moment for everything to connect
    await asyncio.sleep(2)
    
    print("\n" + "="*60)
    print("🎯 SYSTEM READY!")
    print("="*60)
    print("\n📊 Dashboard: http://localhost:5173")
    print("🤖 Bot Status: Connected and Ready")
    print("💡 Use the dashboard to Start/Stop trading")
    print("⚠️  The bot is NOT running yet - click 'Start Trading' in dashboard\n")
    
    # Run the main bot loop
    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\n\n⏹️ Shutting down...")
        bot.bot_running = False
        await bot.shutdown()
        print("✅ Shutdown complete")

def main():
    """Main entry point"""
    try:
        # Check if dashboard dependencies are installed
        try:
            import flask
            import flask_socketio
        except ImportError:
            print("⚠️ Installing required packages...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "flask", "flask-socketio"])
            print("✅ Dependencies installed")
        
        # Run the bot with dashboard
        asyncio.run(run_bot_with_dashboard())
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
