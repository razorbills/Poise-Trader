"""
🚀 FULL-FEATURED RENDER LAUNCHER
Runs bot + dashboard with ALL features on Render.com
"""

import asyncio
import os
import sys
from datetime import datetime
from threading import Thread
import time

# Force REST API for cloud
os.environ['USE_WEBSOCKETS'] = 'false'

print("="*70)
print("🚀 POISE TRADER - RENDER.COM DEPLOYMENT")
print("="*70)
print(f"⏰ Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🎯 Full features enabled (Dashboard + Bot)")
print("🌐 Web service mode activated")
print("-"*70)

# Import dashboard server
try:
    from simple_dashboard_server import app, socketio, set_bot_instance
    print("✅ Dashboard server imported")
except ImportError:
    print("⚠️ Dashboard not available, creating basic server...")
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return jsonify({
            'status': 'alive',
            'message': 'Poise Trader Bot Running',
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy'})
    
    set_bot_instance = lambda x: None

# Global bot reference
bot_instance = None

def run_bot():
    """Run the trading bot in background"""
    global bot_instance
    
    async def start_trading_bot():
        global bot_instance
        
        try:
            print("\n🔄 Initializing trading bot...")
            
            from micro_trading_bot import LegendaryCryptoTitanBot
            
            # Create bot instance
            initial_capital = 5.0
            bot_instance = LegendaryCryptoTitanBot(initial_capital=initial_capital)
            
            # Set default mode (can be changed via dashboard)
            default_mode = os.environ.get('TRADING_MODE', 'PRECISION')
            bot_instance.trading_mode = default_mode
            
            # Register bot with dashboard
            set_bot_instance(bot_instance)
            
            print(f"✅ Bot initialized!")
            print(f"   💰 Initial Capital: ${initial_capital:.2f}")
            print(f"   🎯 Default Mode: {default_mode}")
            print(f"   📊 Dashboard: Available at root URL")
            print(f"   🔧 Control Panel: Access via web browser")
            print("-"*70)
            print("🚀 Starting trading loop...\n")
            
            # Start bot in running state (can be controlled via dashboard)
            bot_instance.bot_running = True
            
            # Run bot (this blocks)
            await bot_instance.run_micro_trading_cycle(cycles=999999)
            
        except Exception as e:
            print(f"❌ Bot error: {e}")
            import traceback
            traceback.print_exc()
    
    # Run async bot
    try:
        asyncio.run(start_trading_bot())
    except KeyboardInterrupt:
        print("\n⚠️ Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")

def main():
    """Main entry point"""
    
    # Start bot in background thread
    print("🎬 Launching bot in background thread...")
    bot_thread = Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Give bot a moment to initialize
    time.sleep(2)
    
    # Get port from environment (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\n🌐 Starting web server on port {port}...")
    print(f"📊 Dashboard will be available at: https://YOUR_APP.onrender.com/")
    print(f"🎯 Features available:")
    print(f"   ✅ Start/Stop bot")
    print(f"   ✅ Switch modes (Aggressive/Precision)")
    print(f"   ✅ Monitor trades in real-time")
    print(f"   ✅ View P&L chart")
    print(f"   ✅ Manage positions")
    print(f"   ✅ Update TP/SL")
    print("="*70)
    print("✨ BOT IS NOW RUNNING 24/7 ON RENDER!\n")
    
    # Start Flask app with SocketIO
    try:
        socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
    except:
        # Fallback to regular Flask if socketio not available
        app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
