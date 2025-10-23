#!/usr/bin/env python3
"""
🎨 DASHBOARD BACKEND SERVER
Real-time data provider for React dashboard with proper bot integration
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("⚠️ flask-cors not installed. Run: pip install flask-cors")
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
import random

app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app)
app.config['SECRET_KEY'] = 'poise_trader_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for dashboard data
dashboard_data = {
    'metrics': {
        'total_pnl': 0.0,
        'win_rate': 0.0,
        'active_positions': 0,
        'daily_volume': 0.0,
        'pnl_history': []
    },
    'portfolio': {
        'total_value': 0.0,
        'cash': 0.0,
        'positions': {}
    },
    'bot_status': {
        'running': False,
        'mode': 'PRECISION',
        'connected': False
    },
    'alerts': []
}

# Bot instance reference
bot_instance = None
data_update_thread = None

def fetch_bot_data():
    """Fetch real data from connected bot instance"""
    global dashboard_data, bot_instance
    
    while True:
        try:
            if bot_instance is not None:
                # Get real data from bot
                if hasattr(bot_instance, 'current_capital'):
                    current_capital = bot_instance.current_capital
                    initial_capital = getattr(bot_instance, 'initial_capital', 5000)
                    pnl = current_capital - initial_capital
                    
                    # Get portfolio data
                    portfolio = {'total_value': current_capital, 'cash': current_capital, 'positions': {}}
                    if hasattr(bot_instance, 'trader') and hasattr(bot_instance.trader, 'get_portfolio_value_sync'):
                        try:
                            portfolio = bot_instance.trader.get_portfolio_value_sync()
                        except:
                            pass
                    
                    # Get active positions
                    active_positions = len([p for p in portfolio.get('positions', {}).values() if p.get('quantity', 0) > 0])
                    
                    # Update metrics
                    dashboard_data['metrics'] = {
                        'total_pnl': pnl,
                        'win_rate': getattr(bot_instance, 'win_rate', 0.0),
                        'active_positions': active_positions,
                        'daily_volume': getattr(bot_instance, 'total_volume_traded', 0.0),
                        'pnl_history': dashboard_data['metrics'].get('pnl_history', [])
                    }
                    
                    # Add to history
                    history_point = {
                        'timestamp': datetime.now().isoformat(),
                        'value': pnl
                    }
                    dashboard_data['metrics']['pnl_history'].append(history_point)
                    if len(dashboard_data['metrics']['pnl_history']) > 50:
                        dashboard_data['metrics']['pnl_history'].pop(0)
                    
                    # Update portfolio
                    dashboard_data['portfolio'] = portfolio
                    
                    # Update status
                    dashboard_data['bot_status'] = {
                        'running': getattr(bot_instance, 'bot_running', False),
                        'mode': getattr(bot_instance, 'trading_mode', 'PRECISION'),
                        'connected': True
                    }
                    
                    # Emit updates
                    socketio.emit('metrics_update', dashboard_data['metrics'])
                else:
                    dashboard_data['bot_status']['connected'] = False
            else:
                # No bot connected - show waiting state
                dashboard_data['bot_status']['connected'] = False
            
            time.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            print(f"Data fetch error: {e}")
            dashboard_data['bot_status']['connected'] = False
            time.sleep(2)

@app.route('/')
def index():
    """Serve the React app"""
    dist_path = os.path.join('dashboard', 'dist')
    if os.path.exists(os.path.join(dist_path, 'index.html')):
        return send_from_directory(dist_path, 'index.html')
    return jsonify({'error': 'Dashboard not built. Run: cd dashboard && npm run build'}), 404

@app.route('/assets/<path:path>')
def serve_assets(path):
    """Serve React assets"""
    dist_path = os.path.join('dashboard', 'dist', 'assets')
    return send_from_directory(dist_path, path)

@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    return jsonify(dashboard_data['metrics'])

@app.route('/api/portfolio')
def get_portfolio():
    """Get portfolio data"""
    return jsonify(dashboard_data['portfolio'])

@app.route('/api/status')
def get_status():
    """Get bot status"""
    status = dashboard_data['bot_status'].copy()
    status.update({
        'capital': dashboard_data['portfolio']['total_value'],
        'positions': len(dashboard_data['portfolio'].get('positions', {})),
        'total_trades': len(dashboard_data['metrics'].get('pnl_history', [])),
        'win_rate': dashboard_data['metrics']['win_rate'] * 100
    })
    return jsonify(status)

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts"""
    return jsonify(dashboard_data['alerts'])

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    global bot_instance
    
    if bot_instance is not None:
        bot_instance.bot_running = True
        dashboard_data['bot_status']['running'] = True
        return jsonify({'success': True, 'message': 'Bot started successfully'})
    
    return jsonify({'success': False, 'message': 'No bot connected. Please run your trading bot first.'})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global bot_instance
    
    if bot_instance is not None:
        bot_instance.bot_running = False
        dashboard_data['bot_status']['running'] = False
        return jsonify({'success': True, 'message': 'Bot stopped'})
    
    return jsonify({'success': False, 'message': 'No bot connected'})

@app.route('/api/mode', methods=['POST'])
def set_mode():
    """Set trading mode"""
    data = request.get_json() or {}
    mode = data.get('mode', 'PRECISION').upper()
    
    if mode in ['AGGRESSIVE', 'PRECISION', 'NORMAL']:
        if mode == 'NORMAL':
            mode = 'PRECISION'
        dashboard_data['bot_status']['mode'] = mode
        return jsonify({'success': True, 'mode': mode})
    
    return jsonify({'success': False, 'message': 'Invalid mode'})

def attach_bot(bot):
    """Attach real bot instance for live data"""
    global bot_instance
    bot_instance = bot
    dashboard_data['bot_status']['connected'] = True
    print("✅ Bot attached to dashboard backend")

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the dashboard server"""
    print(f"""
╔════════════════════════════════════════════╗
║     🚀 DASHBOARD BACKEND SERVER READY      ║
╠════════════════════════════════════════════╣
║  Dashboard URL: http://localhost:{port:<10} ║
║  React Dev: http://localhost:5173          ║
║  Status: Waiting for bot connection...     ║
╚════════════════════════════════════════════╝
    """)
    
    # Start data fetching thread
    global data_update_thread
    data_update_thread = threading.Thread(target=fetch_bot_data)
    data_update_thread.daemon = True
    data_update_thread.start()
    
    if not bot_instance:
        print("⚠️  No bot connected yet. Run your trading bot to see real data.")
        print("💡 To connect: Run micro_trading_bot.py or any other trading bot")
    
    socketio.run(app, host=host, port=port, debug=debug)

if __name__ == "__main__":
    run_server()
