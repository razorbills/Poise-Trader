"""
🚀 FULL-FEATURED RENDER LAUNCHER
Runs bot + dashboard with ALL features on Render.com
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from threading import Thread
import time

# Force REST API for cloud
os.environ['USE_WEBSOCKETS'] = 'false'

# Ensure logs are not stuck in buffers on Render
try:
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
except Exception:
    pass

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

def _apply_poise_preset():
    try:
        preset = str(os.getenv('POISE_PRESET', '') or '').strip().lower()
    except Exception:
        preset = ''

    if not preset:
        return

    presets = {
        'render_realistic': {
            'PAPER_EXECUTION_MODEL': 'realistic',
            'PAPER_SPREAD_BPS': '1.5',
            'PAPER_SLIPPAGE_BPS': '2.0',
            'PAPER_LATENCY_MS_MIN': '100',
            'PAPER_LATENCY_MS_MAX': '600',
            'PAPER_PARTIAL_FILL_PROB': '0.10',
            'PAPER_PARTIAL_FILL_MIN_PCT': '0.6',
            'PAPER_PARTIAL_FILL_MAX_PCT': '0.95',
            'AI_LEARNING_MIN_SECONDS_BETWEEN_UPDATES': '2',
            'AI_LEARNING_MIN_TRADES_BETWEEN_SAVES': '3',
            'MAX_FEED_STALE_SECONDS': '60',
            'MAX_FEED_CONSECUTIVE_FAILURES': '8',
            'SAFETY_PAUSE_SECONDS': '30',
            'STRATEGY_COOLDOWN_ENABLED': '1',
            'STRATEGY_COOLDOWN_MIN_TRADES': '10',
            'STRATEGY_COOLDOWN_WIN_RATE': '0.40',
            'STRATEGY_COOLDOWN_SECONDS': '900',
        },
        'world_class': {
            'RISK_ENGINE_ENABLED': '1',
            'RISK_ENGINE_PROFILE': 'world_class',
            'PAPER_MARKET_TYPE': 'futures',
            'PAPER_LEVERAGE': '10',
            'ENABLE_PAPER_SHORTING': '1',
            'PAPER_EXECUTION_MODEL': 'realistic',
            'PAPER_SPREAD_BPS': '1.5',
            'PAPER_SLIPPAGE_BPS': '2.0',
            'PAPER_LATENCY_MS_MIN': '100',
            'PAPER_LATENCY_MS_MAX': '600',
            'PAPER_PARTIAL_FILL_PROB': '0.10',
            'PAPER_PARTIAL_FILL_MIN_PCT': '0.6',
            'PAPER_PARTIAL_FILL_MAX_PCT': '0.95',
            'AI_LEARNING_MIN_SECONDS_BETWEEN_UPDATES': '2',
            'AI_LEARNING_MIN_TRADES_BETWEEN_SAVES': '3',
            'MAX_FEED_STALE_SECONDS': '45',
            'MAX_FEED_CONSECUTIVE_FAILURES': '5',
            'SAFETY_PAUSE_SECONDS': '45',
            'STRATEGY_COOLDOWN_ENABLED': '1',
            'STRATEGY_COOLDOWN_MIN_TRADES': '10',
            'STRATEGY_COOLDOWN_WIN_RATE': '0.45',
            'STRATEGY_COOLDOWN_SECONDS': '900',
        },
        'world-class': {
            'RISK_ENGINE_ENABLED': '1',
            'RISK_ENGINE_PROFILE': 'world_class',
            'PAPER_MARKET_TYPE': 'futures',
            'PAPER_LEVERAGE': '10',
            'ENABLE_PAPER_SHORTING': '1',
            'PAPER_EXECUTION_MODEL': 'realistic',
            'PAPER_SPREAD_BPS': '1.5',
            'PAPER_SLIPPAGE_BPS': '2.0',
            'PAPER_LATENCY_MS_MIN': '100',
            'PAPER_LATENCY_MS_MAX': '600',
            'PAPER_PARTIAL_FILL_PROB': '0.10',
            'PAPER_PARTIAL_FILL_MIN_PCT': '0.6',
            'PAPER_PARTIAL_FILL_MAX_PCT': '0.95',
            'AI_LEARNING_MIN_SECONDS_BETWEEN_UPDATES': '2',
            'AI_LEARNING_MIN_TRADES_BETWEEN_SAVES': '3',
            'MAX_FEED_STALE_SECONDS': '45',
            'MAX_FEED_CONSECUTIVE_FAILURES': '5',
            'SAFETY_PAUSE_SECONDS': '45',
            'STRATEGY_COOLDOWN_ENABLED': '1',
            'STRATEGY_COOLDOWN_MIN_TRADES': '10',
            'STRATEGY_COOLDOWN_WIN_RATE': '0.45',
            'STRATEGY_COOLDOWN_SECONDS': '900',
        },
        'render_minimal': {
            'AI_LEARNING_MIN_SECONDS_BETWEEN_UPDATES': '2',
            'AI_LEARNING_MIN_TRADES_BETWEEN_SAVES': '5',
            'MAX_FEED_STALE_SECONDS': '60',
            'MAX_FEED_CONSECUTIVE_FAILURES': '8',
            'SAFETY_PAUSE_SECONDS': '30',
        },
    }

    cfg = presets.get(preset)
    if not cfg:
        return

    for k, v in cfg.items():
        try:
            os.environ.setdefault(str(k), str(v))
        except Exception:
            pass

_apply_poise_preset()

print("="*70)
print("🚀 POISE TRADER - RENDER.COM DEPLOYMENT")
print("="*70)
print(f"⏰ Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🎯 Full features enabled (Dashboard + Bot)")
print("🌐 Web service mode activated")
print("-"*70)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)

try:
    from supabase_state_sync import SupabaseStateSync
except Exception:
    SupabaseStateSync = None

# Import full dashboard server with ALL routes
import simple_dashboard_server

# Import keep-alive system
from keep_alive_system import start_all_systems, stop_all_systems

# Import ultra-aggressive keep-alive for maximum uptime
from aggressive_keepalive import start_ultra_aggressive_mode, stop_ultra_aggressive_mode

# Import background workers for continuous operation
from background_worker import start_background_workers, stop_background_workers

app = simple_dashboard_server.app
socketio = simple_dashboard_server.socketio

# Use the dashboard's attach_bot function
def set_bot_instance(bot):
    simple_dashboard_server.attach_bot(bot)
    print("✅ Bot registered with full dashboard")

print("✅ Full dashboard server loaded with all API endpoints")

# Global bot reference
bot_instance = None

def run_bot():
    """Run the trading bot in background"""
    global bot_instance
    
    async def start_trading_bot():
        global bot_instance

        state_sync = None
        
        try:
            try:
                if hasattr(simple_dashboard_server, 'set_bot_startup'):
                    simple_dashboard_server.set_bot_startup(state='starting', message='Initializing trading bot')
            except Exception:
                pass
            print("\n🔄 Initializing trading bot...")

            if SupabaseStateSync is not None:
                try:
                    state_sync = SupabaseStateSync.from_env()
                except Exception:
                    state_sync = None

            if state_sync:
                logging.getLogger("render_launcher").info("☁️ Restoring state from Supabase...")
                await state_sync.restore_on_startup()
                state_sync.start_background_sync()
            
            from micro_trading_bot import LegendaryCryptoTitanBot
            
            # Create bot instance
            try:
                initial_capital = float(os.environ.get('INITIAL_CAPITAL', '5.0') or 5.0)
            except Exception:
                initial_capital = 5.0
            bot_instance = LegendaryCryptoTitanBot(initial_capital=initial_capital)
            
            # Set default mode (can be changed via dashboard)
            default_mode = os.environ.get('TRADING_MODE', 'PRECISION')
            bot_instance.trading_mode = default_mode
            
            # Register bot with dashboard
            set_bot_instance(bot_instance)

            try:
                if hasattr(simple_dashboard_server, 'set_bot_startup'):
                    simple_dashboard_server.set_bot_startup(state='ready', message='Bot initialized and registered with dashboard')
            except Exception:
                pass
            
            print(f"✅ Bot initialized!")
            print(f"   💰 Initial Capital: ${initial_capital:.2f}")
            print(f"   🎯 Default Mode: {default_mode}")
            print(f"   📊 Dashboard: Available at root URL")
            print(f"   🔧 Control Panel: Access via web browser")
            print("-"*70)
            print("🚀 Starting trading loop...\n")
            
            # Start bot in PAUSED state - wait for dashboard command
            bot_instance.bot_running = False
            print("⏸️  Bot initialized in PAUSED state")
            print("   👉 Use dashboard to START trading")
            
            # Run bot (this blocks)
            await bot_instance.run_micro_trading_cycle(cycles=999999)
            
        except Exception as e:
            try:
                if hasattr(simple_dashboard_server, 'set_bot_startup'):
                    simple_dashboard_server.set_bot_startup(state='error', message=f'Bot startup failed: {str(e)[:200]}')
                if hasattr(simple_dashboard_server, 'set_bot_last_error'):
                    simple_dashboard_server.set_bot_last_error(str(e))
            except Exception:
                pass
            print(f"❌ Bot error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if state_sync:
                try:
                    await state_sync.sync_once()
                    await state_sync.stop()
                except Exception:
                    pass
    
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

    try:
        if hasattr(simple_dashboard_server, 'bot_thread'):
            simple_dashboard_server.bot_thread = bot_thread
        if hasattr(simple_dashboard_server, 'set_bot_startup'):
            simple_dashboard_server.set_bot_startup(state='thread_started', message='Bot thread started')
    except Exception:
        pass
    
    # Give bot a moment to initialize
    time.sleep(2)
    
    # Get port from environment (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    # Get app URL from environment
    app_url = os.environ.get('RENDER_EXTERNAL_URL')
    if not app_url:
        # Try to construct from RENDER_SERVICE_NAME
        service_name = os.environ.get('RENDER_SERVICE_NAME')
        if service_name:
            app_url = f"https://{service_name}.onrender.com"
    
    # Start keep-alive systems
    print("\n🔄 Starting 24/7 keep-alive systems...")
    start_all_systems(app_url=app_url, bot_instance=bot_instance)
    
    # Start ultra-aggressive mode for maximum uptime
    print("\n⚡ Activating ULTRA-AGGRESSIVE MODE...")
    start_ultra_aggressive_mode(app_url=app_url)
    
    # Start background workers for continuous operation
    start_background_workers(bot_instance=bot_instance)
    
    print(f"\n🌐 Starting web server on port {port}...")
    if app_url:
        print(f"📊 Dashboard URL: {app_url}/")
        print(f"❤️  Health Check: {app_url}/health")
    else:
        print(f"📊 Dashboard will be available at: https://YOUR_APP.onrender.com/")
    
    print(f"🎯 Features available:")
    print(f"   ✅ Start/Stop bot")
    print(f"   ✅ Switch modes (Aggressive/Precision)")
    print(f"   ✅ Monitor trades in real-time")
    print(f"   ✅ View P&L chart")
    print(f"   ✅ Manage positions")
    print(f"   ✅ Update TP/SL")
    print(f"   🔄 24/7 Keep-Alive Active")
    print(f"   💓 Health Monitoring Active")
    print(f"   ⚡ Ultra-Aggressive Mode Active")
    print(f"   🔧 3x Background Workers Active")
    print(f"   💾 Auto-Save Every 5 Minutes")
    print(f"   🧵 9x Concurrent Threads Running")
    print("="*70)
    print("✨ BOT RUNNING AT STARTER-TIER PERFORMANCE ON FREE TIER!")
    print("   (99%+ uptime expected with all systems active)")
    print("="*70 + "\n")
    
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
        stop_background_workers()
        stop_ultra_aggressive_mode()
        stop_all_systems()
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        stop_background_workers()
        stop_ultra_aggressive_mode()
        stop_all_systems()
        sys.exit(1)
