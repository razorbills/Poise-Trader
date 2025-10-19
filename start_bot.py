#!/usr/bin/env python3
"""
🚀 POISE TRADER - AUTONOMOUS STARTUP SCRIPT

This is the simplest way to start your fully autonomous trading bot!

Just run: python start_bot.py

The bot will:
• Install any missing dependencies automatically
• Initialize all systems
• Start trading autonomously
• Run 24/7 until you stop it

YOU LITERALLY JUST RUN THIS FILE AND WALK AWAY!
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def install_dependencies():
    """Install required dependencies automatically"""
    print("🔧 Checking and installing dependencies...")
    
    # Install requirements
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Please upgrade your Python version.")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} - Compatible!")
    return True

def setup_environment():
    """Setup the trading environment"""
    print("🏗️ Setting up trading environment...")
    
    # Create necessary directories
    directories = ["logs", "data", "config", "backups"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("✅ Environment setup complete!")

def main():
    """Main startup function"""
    print("🚀 POISE TRADER - AUTONOMOUS STARTUP")
    print("=" * 50)
    print("🤖 Initializing fully autonomous trading bot...")
    print()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check your internet connection.")
        print("💡 Try running manually: pip install -r requirements.txt")
        return 1
    
    print("\n🎯 LAUNCHING AUTONOMOUS TRADING BOT...")
    print("💰 Target: Maximum profit with zero manual work")
    print("🧠 Method: AI-powered decision making")  
    print("🛡️ Safety: Advanced risk management")
    print("💤 Your job: Relax while the bot trades!")
    print()
    
    try:
        # Import and start the main bot
        from poise_master_bot import main as run_bot
        
        print("🚀 STARTING POISE MASTER BOT...")
        print("=" * 50)
        print("✅ Bot is now running autonomously!")
        print("📊 Check the logs for detailed performance reports")
        print("🛑 Press Ctrl+C to stop the bot when needed")
        print()
        
        # Run the autonomous bot
        return asyncio.run(run_bot())
        
    except ImportError as e:
        print(f"❌ Failed to import bot modules: {e}")
        print("💡 Make sure all files are in the correct directory")
        return 1
        
    except KeyboardInterrupt:
        print("\n🛑 Bot startup cancelled by user")
        return 0
        
    except Exception as e:
        print(f"💥 Critical error during startup: {e}")
        return 1

if __name__ == "__main__":
    """
    🎯 ULTIMATE AUTONOMOUS TRADING SYSTEM
    
    This script does everything for you:
    1. ✅ Checks your Python version
    2. 📁 Creates necessary directories  
    3. 🔧 Installs all dependencies
    4. 🚀 Starts the autonomous trading bot
    5. 💤 Lets you sleep while it makes money!
    
    LITERALLY ZERO MANUAL WORK REQUIRED!
    Just run this script and walk away.
    
    The bot will:
    • Connect to exchanges automatically
    • Analyze markets using AI
    • Execute trades autonomously  
    • Manage risk automatically
    • Report performance continuously
    • Run 24/7 until you stop it
    
    YOUR ONLY JOB: START THIS SCRIPT!
    """
    
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal startup error: {e}")
        sys.exit(1)
