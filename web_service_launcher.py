"""
Web Service Launcher for Render.com Free Tier
Runs bot in background while serving a simple web endpoint
"""

import asyncio
import os
from datetime import datetime
from threading import Thread
from flask import Flask, jsonify

# Force REST API only
os.environ['USE_WEBSOCKETS'] = 'false'

app = Flask(__name__)

# Bot status tracking
bot_status = {
    'status': 'starting',
    'start_time': datetime.now().isoformat(),
    'trades': 0,
    'capital': 5.0,
    'last_update': None
}

@app.route('/')
def home():
    """Health check endpoint - keeps Render happy"""
    return jsonify({
        'status': 'alive',
        'bot': bot_status,
        'uptime': (datetime.now() - datetime.fromisoformat(bot_status['start_time'])).total_seconds()
    })

@app.route('/health')
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/status')
def status():
    """Bot status endpoint"""
    return jsonify(bot_status)

def run_bot():
    """Run the trading bot in background"""
    async def start_bot():
        try:
            from micro_trading_bot import MicroTradingBot
            
            bot_status['status'] = 'initializing'
            
            # Initialize bot
            bot = MicroTradingBot(initial_capital=5.0)
            bot.set_trading_mode('PRECISION')
            
            bot_status['status'] = 'running'
            bot_status['last_update'] = datetime.now().isoformat()
            
            print("✅ Bot started successfully!")
            
            # Run bot
            await bot.run()
            
        except Exception as e:
            bot_status['status'] = f'error: {str(e)}'
            print(f"❌ Bot error: {e}")
            raise
    
    # Run async bot
    asyncio.run(start_bot())

if __name__ == '__main__':
    # Start bot in background thread
    print("🚀 Starting Poise Trader Web Service...")
    print("📊 Bot will run in background")
    print("🌐 Web endpoint for health checks")
    
    bot_thread = Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Start Flask web service (required for Render)
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
