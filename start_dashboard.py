#!/usr/bin/env python3
"""
🚀 QUICK START - Professional Trading Dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install dashboard requirements"""
    print("📦 Installing dashboard dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                              "Flask", "flask-socketio", "python-socketio"])
        print("✅ Dependencies installed!")
        return True
    except Exception as e:
        print(f"⚠️ Could not install dependencies: {e}")
        print("💡 Please run: pip install Flask flask-socketio python-socketio")
        return False

def main():
    print("\n" + "="*60)
    print("🎨 POISE TRADER - PROFESSIONAL DASHBOARD")
    print("="*60)
    
    # Check if dependencies are installed
    try:
        import flask
        import flask_socketio
        print("✅ Dependencies OK")
    except ImportError:
        print("📦 Installing dependencies...")
        if not install_requirements():
            print("\n❌ Please install dependencies manually:")
            print("   pip install Flask flask-socketio python-socketio")
            return
    
    # Start dashboard
    print("\n🚀 Starting Professional Dashboard...")
    print("📊 Dashboard URL: http://localhost:5000")
    print("💡 Open this URL in your browser")
    print("\n⚡ Press Ctrl+C to stop\n")
    
    try:
        import professional_dashboard
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
