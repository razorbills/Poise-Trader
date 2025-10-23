#!/usr/bin/env python3
"""
🚀 RUN BOT WITH SIMPLE DASHBOARD
Launches trading bot with the simple control panel
"""

import sys
import os
import asyncio
import threading
import time
import webbrowser
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import simple dashboard server
from simple_dashboard_server import attach_bot, run_server

# Import the main trading bot
from micro_trading_bot import LegendaryCryptoTitanBot

def start_simple_dashboard():
    """Start the simple dashboard server in a separate thread"""
    print("🎯 Starting Simple Dashboard Server...")
    server_thread = threading.Thread(target=run_server, kwargs={'host': '0.0.0.0', 'port': 5000})
    server_thread.daemon = True
    server_thread.start()
    time.sleep(3)  # Give server time to start
    print("✅ Dashboard server running on http://localhost:5000")

def open_browser():
    """Open dashboard in browser after a delay"""
    time.sleep(5)
    webbrowser.open('http://localhost:5000')

async def run_bot_with_simple_dashboard():
    """Run the trading bot with simple dashboard"""
    print("\n" + "="*50)
    print("🤖 STARTING BOT WITH SIMPLE DASHBOARD")
    print("="*50 + "\n")
    
    # Start dashboard server
    start_simple_dashboard()
    
    # Open browser
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("\n⚡ Initializing Trading Bot...")
    
    # Initialize the bot
    bot = LegendaryCryptoTitanBot()
    
    # Set default mode to PRECISION (Normal)
    bot.trading_mode = 'PRECISION'
    bot.bot_running = False  # Start paused
    
    # Attach bot to dashboard
    attach_bot(bot)
    print("✅ Bot connected to dashboard - Controls enabled!")
    
    # Bot is already initialized in __init__, just wait for connection
    await asyncio.sleep(2)
    
    print("\n" + "="*50)
    print("🎯 SIMPLE CONTROL PANEL READY!")
    print("="*50)
    print("\n📊 Dashboard opened in your browser")
    print("🎮 Use the simple controls to:")
    print("   • Select Mode: Aggressive or Normal")
    print("   • Start/Stop trading with one click")
    print("\n⚠️  Bot is PAUSED - Click 'START' to begin trading\n")
    
    # Run the main bot loop
    try:
        await bot.run_micro_trading_cycle(cycles=999999)  # Run indefinitely
    except KeyboardInterrupt:
        print("\n\n⏹️ Shutting down...")
        bot.bot_running = False
        print("✅ Shutdown complete")

def main():
    """Main entry point"""
    try:
        # Check dependencies
        try:
            import flask
            import flask_socketio
        except ImportError:
            print("⚠️ Installing required packages...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "flask", "flask-socketio"])
            print("✅ Dependencies installed")
        
        # Run the bot with simple dashboard
        asyncio.run(run_bot_with_simple_dashboard())
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
