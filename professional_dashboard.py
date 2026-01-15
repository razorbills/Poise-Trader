#!/usr/bin/env python3
"""
🎨 PROFESSIONAL TRADING DASHBOARD
Real-time monitoring for Poise Trader with stunning UI/UX
"""

from flask import Flask, render_template, jsonify, request
import threading
import asyncio
import time
import subprocess
import os
import json
import traceback
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
app.config['SECRET_KEY'] = 'poise_trader_2025'

# Global bot instance
bot_instance = None
bot_thread = None
selected_mode = 'PRECISION'  # Default to PRECISION mode (user can change)

@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon to prevent 404"""
    return '', 204

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current trading status - CHECKS bot_running FLAG"""
    if bot_instance is None:
        return jsonify({'status': 'not_running', 'message': 'Bot not initialized'})
    
    try:
        # Get portfolio value
        if hasattr(bot_instance.trader, 'get_portfolio_value_sync'):
            portfolio = bot_instance.trader.get_portfolio_value_sync()
        else:
            # Fallback to calculating from bot state
            portfolio = {
                'total_value': bot_instance.current_capital,
                'positions': {}
            }
        
        # Get win rate stats
        if hasattr(bot_instance, 'get_win_rate_stats'):
            win_stats = bot_instance.get_win_rate_stats()
        else:
            win_stats = {
                'total_trades': bot_instance.total_completed_trades,
                'current_win_rate': bot_instance.win_rate,
                'current_streak': 0
            }
        
        # CHECK bot_running FLAG - not just if bot exists!
        status = 'running' if bot_instance.bot_running else 'waiting'
        
        # Calculate active positions
        active_positions = 0
        if 'positions' in portfolio:
            active_positions = len([p for p in portfolio.get('positions', {}).values() if p.get('quantity', 0) > 0])

        try:
            real_env = str(os.getenv('REAL_TRADING', '0') or '0').strip().lower()
            real_enabled = real_env in ['1', 'true', 'yes', 'on']
        except Exception:
            real_enabled = False

        try:
            api_key = os.getenv('MEXC_API_KEY', '')
            api_secret = os.getenv('MEXC_API_SECRET', '') or os.getenv('MEXC_SECRET_KEY', '')
            keys_present = bool(api_key and api_secret)
        except Exception:
            keys_present = False

        try:
            market_type = str(getattr(bot_instance.trader, 'market_type', '') or os.getenv('PAPER_MARKET_TYPE', '') or '').strip().lower()
        except Exception:
            market_type = ''

        real_trading_block = None
        try:
            real_trading_block = {
                'enabled': bool(real_enabled),
                'keys_present': bool(keys_present),
                'market_type': market_type or None,
                'leverage': float(getattr(bot_instance.trader, 'leverage', 1.0) or 1.0),
                'ready': bool(getattr(bot_instance.trader, '_real_trading_ready', lambda: False)()),
                'last_error': getattr(bot_instance.trader, 'last_real_order_error', None),
                'last_order': getattr(bot_instance.trader, 'last_real_order', None)
            }
        except Exception:
            real_trading_block = {
                'enabled': bool(real_enabled),
                'keys_present': bool(keys_present),
                'market_type': market_type or None,
                'ready': False,
                'last_error': None,
                'last_order': None
            }
        
        return jsonify({
            'status': status,  # 'running' or 'waiting'
            'capital': portfolio.get('total_value', bot_instance.current_capital),
            'positions': active_positions,
            'total_trades': win_stats.get('total_trades', bot_instance.total_completed_trades),
            'win_rate': win_stats.get('current_win_rate', bot_instance.win_rate) * 100,
            'current_streak': win_stats.get('current_streak', 0),
            'trading_mode': bot_instance.trading_mode,
            'bot_running': bot_instance.bot_running,  # Explicit flag
            'real_trading': real_trading_block
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the trading bot - ONLY endpoint that should set bot_running=True"""
    global bot_instance, selected_mode
    
    print("\n" + "="*70)
    print("🎮 DASHBOARD: /api/start endpoint called by USER!")
    print(f"   Bot instance exists: {bot_instance is not None}")
    if bot_instance:
        print(f"   Bot instance ID: {id(bot_instance)}")
        print(f"   Bot type: {type(bot_instance).__name__}")
        print(f"   BEFORE: bot_running = {bot_instance.bot_running}")
    print(f"   Selected mode: {selected_mode}")
    print("="*70)
    
    # Bot instance should already be set by main script
    if bot_instance is None:
        print("❌ ERROR: Bot instance is None!")
        print("💡 TIP: Make sure the bot finished initializing before clicking Start")
        return jsonify({'success': False, 'message': 'Bot not initialized yet. Refresh page and wait 5 seconds, then try again.'})
    
    try:
        # Apply selected mode
        mode = selected_mode
        bot_instance.trading_mode = mode
        cfg = bot_instance.mode_config[mode]
        bot_instance.target_accuracy = cfg['target_accuracy']
        bot_instance.min_confidence_for_trade = cfg['min_confidence']
        bot_instance.ensemble_threshold = cfg['ensemble_threshold']
        bot_instance.confidence_threshold = cfg['min_confidence']
        bot_instance.base_confidence_threshold = cfg['min_confidence']
        
        if mode == 'AGGRESSIVE':
            bot_instance.fast_mode_enabled = True
            bot_instance.precision_mode_enabled = False
            bot_instance.min_price_history = 5  # VERY LOW - start trading FAST!
            bot_instance.confidence_adjustment_factor = 0.05
            bot_instance.aggressive_trade_guarantee = True
            bot_instance.aggressive_trade_interval = 60.0
            bot_instance.cycle_sleep_override = 10.0
            bot_instance.win_rate_optimizer_enabled = False  # CRITICAL: Disable optimizer!
            bot_instance.min_trade_quality_score = 10.0  # VERY LOW - take almost all trades!
            bot_instance.min_confidence_for_trade = 0.10  # 10% minimum
            print(f"\n{'='*70}")
            print(f"⚡ DASHBOARD: AGGRESSIVE MODE ACTIVATED!")
            print(f"   • Win rate optimizer: DISABLED")
            print(f"   • Confidence threshold: {cfg['min_confidence']:.0%}")
            print(f"   • Min quality score: 10/100")
            print(f"   • Trade guarantee: ACTIVE (≥1/min)")
            print(f"{'='*70}\n")
        else:
            bot_instance.fast_mode_enabled = False
            bot_instance.precision_mode_enabled = True
            bot_instance.min_price_history = 10  # LOWERED from 50 - faster trading!
            bot_instance.confidence_adjustment_factor = 0.01
            bot_instance.aggressive_trade_guarantee = False
            bot_instance.cycle_sleep_override = None
            bot_instance.win_rate_optimizer_enabled = False  # DISABLED - no over-filtering!
            bot_instance.min_trade_quality_score = 25.0  # Lower threshold
            bot_instance.min_confidence_for_trade = 0.30  # 30% minimum
            print(f"\n{'='*70}")
            print(f"🎯 DASHBOARD: NORMAL MODE ACTIVATED!")
            print(f"   • Win rate optimizer: DISABLED (less filtering)")
            print(f"   • Confidence threshold: {cfg['min_confidence']:.0%}")
            print(f"   • Min quality score: 25/100")
            print(f"   • Quality-focused but will trade!")
            print(f"{'='*70}\n")
        
        # 🔥 THIS IS THE CRITICAL LINE - SET bot_running = True!
        print(f"🔥 BEFORE SET: bot_running = {bot_instance.bot_running}")
        bot_instance.bot_running = True
        print(f"🔥 AFTER SET: bot_running = {bot_instance.bot_running}")
        print(f"▶️▶️▶️ DASHBOARD: TRADING STARTED IN {mode} MODE! ◀️◀️◀️")
        print(f"🔥 Bot will now execute trades! Watch the logs below...\n")
        
        mode_msg = '⚡ AGGRESSIVE (≥1 trade/min)' if mode == 'AGGRESSIVE' else '🎯 NORMAL (best signals)'
        return jsonify({'success': True, 'message': f'Bot started in {mode_msg} mode! Check console for trades.'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

def set_bot_instance(bot):
    """Set the bot instance from external source"""
    global bot_instance
    bot_instance = bot
    print(f"🎛️ Dashboard connected to bot instance")

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stop the trading bot - KEEPS bot alive, just pauses it"""
    global bot_instance
    
    if bot_instance is None:
        return jsonify({'success': False, 'message': 'Bot not running'})
    
    print("\n" + "="*70)
    print("⏸️ DASHBOARD: Stop button clicked!")
    print(f"   BEFORE: bot_running = {bot_instance.bot_running}")
    
    bot_instance.bot_running = False
    # DON'T destroy bot instance! Just pause it
    # bot_instance = None  ← REMOVED! Keep bot alive
    
    print(f"   AFTER: bot_running = {bot_instance.bot_running}")
    print("⏸️ Bot paused (ready to restart)")
    print("="*70 + "\n")
    
    return jsonify({'success': True, 'message': 'Bot stopped (ready to restart)'})

@app.route('/api/ai-brain')
def get_ai_brain_status():
    """Get AI brain learning status"""
    import json
    import os
    
    try:
        brain_file = 'ai_brain.json'
        if not os.path.exists(brain_file):
            return jsonify({'error': 'AI brain not found', 'total_trades': 0})
        
        with open(brain_file, 'r') as f:
            brain = json.load(f)
        
        # Extract key metrics
        status = {
            'total_trades': brain.get('total_trades', 0),
            'total_pnl': brain.get('total_profit_loss', 0),
            'win_rate': brain.get('win_rate', 0.5) * 100,
            'learning_sessions': brain.get('learning_sessions', 0),
            'last_updated': brain.get('last_updated', 'Never'),
            'strategies': brain.get('strategy_performance', {}),
            'recent_trades': brain.get('recent_trades', [])[-5:],  # Last 5 trades
            'symbol_knowledge': {}
        }
        
        # Add top symbols
        symbols = brain.get('symbol_knowledge', {})
        for symbol, data in list(symbols.items())[:5]:
            status['symbol_knowledge'][symbol] = {
                'trades': data.get('total_trades', 0),
                'win_rate': data.get('win_rate', 0.5) * 100
            }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e), 'total_trades': 0})

@app.route('/api/market-filter', methods=['POST'])
def set_market_filter():
    """Update the bot's active markets filter"""
    global bot_instance
    
    try:
        data = request.get_json() or {}
        markets = data.get('markets', [])
        
        if bot_instance:
            # Update the bot's active_symbols list
            bot_instance.active_symbols = markets
            print(f"🎯 Market filter updated: {len(markets)} markets selected")
            print(f"   Selected: {', '.join(markets[:5])}{'...' if len(markets) > 5 else ''}")
            return jsonify({'success': True, 'message': f'Filter applied: {len(markets)} markets'})
        else:
            return jsonify({'success': False, 'message': 'Bot not initialized'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/reset', methods=['POST'])
def reset_trading():
    """Reset trading state to fresh start"""
    import os
    import json
    
    try:
        try:
            start_capital = float(os.getenv('INITIAL_CAPITAL', '20.0') or 20.0)
        except Exception:
            start_capital = 20.0

        # Create fresh state file
        fresh_state = {
            "cash_balance": start_capital,
            "initial_capital": start_capital,
            "positions": {},
            "trade_history": [],
            "total_trades": 0,
            "winning_trades": 0,
            "last_save_time": datetime.now().isoformat()
        }
        
        with open('trading_state.json', 'w') as f:
            json.dump(fresh_state, f, indent=2)
        
        # If bot is running, reinitialize the trader
        global bot_instance
        if bot_instance and hasattr(bot_instance, 'trader'):
            from live_paper_trading_test import LivePaperTradingManager
            bot_instance.trader = LivePaperTradingManager(start_capital)
            print(f"♻️ Trading state reset - starting fresh with ${start_capital:.2f}")

            try:
                if hasattr(bot_instance, 'risk_engine') and bot_instance.risk_engine is not None:
                    try:
                        from risk_engine_v2 import RiskEngineV2
                        bot_instance.risk_engine = RiskEngineV2(initial_equity=float(start_capital or 0.0))
                    except Exception:
                        re = bot_instance.risk_engine
                        try:
                            re.initial_equity = float(start_capital or 0.0)
                            re.daily_start_equity = float(start_capital or 0.0)
                            re.weekly_start_equity = float(start_capital or 0.0)
                            re.peak_equity = float(start_capital or 0.0)
                            re.last_equity = float(start_capital or 0.0)
                        except Exception:
                            pass
                        try:
                            re.consecutive_losses = 0
                            re.pause_until_ts = 0.0
                            re.pause_reason = None
                        except Exception:
                            pass
            except Exception:
                pass
        
        return jsonify({'success': True, 'message': f'Trading state reset to ${start_capital:.2f}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/mode', methods=['POST'])
def set_mode():
    """Set trading mode (Aggressive or Normal/Precision) - DOES NOT START BOT"""
    global bot_instance, selected_mode
    
    try:
        data = request.get_json() or {}
        mode = str(data.get('mode', 'PRECISION')).upper()
        if mode == 'NORMAL':
            mode = 'PRECISION'
        if mode not in ('AGGRESSIVE', 'PRECISION'):
            return jsonify({'success': False, 'message': f'Invalid mode {mode}'})
        
        # Store selected mode for use when bot starts
        selected_mode = mode
        print(f"🎯 Mode selected: {mode} (bot not started yet)")
        
        # If bot is running, apply mode immediately (but don't start it!)
        if bot_instance is not None:
            was_running = bot_instance.bot_running  # Save current state
            bot_instance.trading_mode = mode
            cfg = bot_instance.mode_config[mode]
            bot_instance.target_accuracy = cfg['target_accuracy']
            bot_instance.min_confidence_for_trade = cfg['min_confidence']
            bot_instance.ensemble_threshold = cfg['ensemble_threshold']
            bot_instance.confidence_threshold = cfg['min_confidence']
            bot_instance.base_confidence_threshold = cfg['min_confidence']
            
            if mode == 'AGGRESSIVE':
                bot_instance.fast_mode_enabled = True
                bot_instance.precision_mode_enabled = False
                bot_instance.min_price_history = 20
                bot_instance.confidence_adjustment_factor = 0.05
                bot_instance.aggressive_trade_guarantee = True
                bot_instance.aggressive_trade_interval = 60.0
                bot_instance.cycle_sleep_override = 10.0
                bot_instance.win_rate_optimizer_enabled = False  # DISABLE for aggressive
                print(f"⚡ AGGRESSIVE MODE configured (bot_running={was_running})")
            else:
                bot_instance.fast_mode_enabled = False
                bot_instance.precision_mode_enabled = True
                bot_instance.min_price_history = 50
                bot_instance.confidence_adjustment_factor = 0.01
                bot_instance.aggressive_trade_guarantee = False
                bot_instance.cycle_sleep_override = None
                print(f"🎯 PRECISION MODE configured (bot_running={was_running})")
            
            # CRITICAL: Restore original bot_running state (don't auto-start!)
            bot_instance.bot_running = was_running
            print(f"✅ Mode set to {mode}, bot_running={bot_instance.bot_running}")
        
        return jsonify({'success': True, 'mode': mode})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

def create_dashboard_template():
    """Create the HTML template"""
    template_dir = Path('templates')
    template_dir.mkdir(exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Poise Trader - Professional Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255,255,255,0.15);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 10px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #ffd700;
        }
        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 30px;
        }
        .btn {
            padding: 15px 40px;
            font-size: 1.1em;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .btn-start {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-stop {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .btn-mode {
            background: rgba(255,255,255,0.2);
            color: white;
            border: 2px solid rgba(255,255,255,0.3);
        }
        .btn-mode.active {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            border-color: #38ef7d;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(0,0,0,0.3);
        }
        .btn:active {
            transform: translateY(0);
        }
        .chart-container {
            background: rgba(255,255,255,0.15);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            margin-bottom: 30px;
        }
        .positions-table {
            background: rgba(255,255,255,0.15);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        th {
            font-weight: bold;
            color: #ffd700;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-running { background-color: #00ff00; box-shadow: 0 0 10px #00ff00; }
        .status-stopped { background-color: #ff0000; box-shadow: 0 0 10px #ff0000; }
        .profit { color: #00ff00; }
        .loss { color: #ff6b6b; }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>🏆 POISE TRADER</h1>
            <p>Professional Trading Dashboard | 90% Win Rate System</p>
            <p><span class="status-indicator status-stopped" id="statusIndicator"></span><span id="statusText">Not Running</span></p>
        </div>

        <div class="controls">
            <button class="btn btn-mode" id="aggressiveBtn" onclick="setMode('AGGRESSIVE')">⚡ Aggressive</button>
            <button class="btn btn-mode" id="normalBtn" onclick="setMode('PRECISION')">🎯 Normal</button>
            <button class="btn btn-start" onclick="startBot()">▶️ Start Trading</button>
            <button class="btn btn-stop" onclick="stopBot()">⏹️ Stop Trading</button>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">💰 Portfolio Value</div>
                <div class="stat-value" id="portfolioValue">$0.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">🏆 Win Rate</div>
                <div class="stat-value" id="winRate">0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">📊 Total Trades</div>
                <div class="stat-value" id="totalTrades">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">🔥 Current Streak</div>
                <div class="stat-value" id="currentStreak">0</div>
            </div>
        </div>

        <div class="chart-container">
            <h3>📈 Portfolio Performance</h3>
            <canvas id="performanceChart"></canvas>
        </div>

        <div class="positions-table">
            <h3>📊 Active Positions</h3>
            <table id="positionsTable">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Size</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                    </tr>
                </thead>
                <tbody id="positionsBody">
                    <tr><td colspan="6" style="text-align:center">No active positions</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let currentMode = 'AGGRESSIVE';
        
        function setMode(mode) {
            fetch('/api/mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: mode})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentMode = data.mode;
                    document.getElementById('aggressiveBtn').classList.toggle('active', mode === 'AGGRESSIVE');
                    document.getElementById('normalBtn').classList.toggle('active', mode === 'PRECISION');
                    console.log('✅ Mode set to:', mode);
                } else {
                    alert('Error: ' + (data.message || 'Unknown error'));
                }
            })
            .catch(err => console.error('Error setting mode:', err));
        }
        
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Portfolio Value',
                    data: [],
                    borderColor: '#ffd700',
                    backgroundColor: 'rgba(255, 215, 0, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { labels: { color: '#fff' } } },
                scales: {
                    y: { ticks: { color: '#fff' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                    x: { ticks: { color: '#fff' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                }
            }
        });

        function updateDashboard() {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    // Update status indicator based on actual bot_running flag
                    if (data.status === 'running') {
                        document.getElementById('statusIndicator').className = 'status-indicator status-running';
                        document.getElementById('statusText').textContent = 'Running';
                    } else if (data.status === 'waiting') {
                        document.getElementById('statusIndicator').className = 'status-indicator status-stopped';
                        document.getElementById('statusText').textContent = 'Waiting';
                    } else {
                        document.getElementById('statusIndicator').className = 'status-indicator status-stopped';
                        document.getElementById('statusText').textContent = 'Stopped';
                    }
                    
                    // Always update stats (whether running or not)
                    if (data.capital !== undefined) {
                        document.getElementById('portfolioValue').textContent = '$' + data.capital.toFixed(2);
                        document.getElementById('winRate').textContent = data.win_rate.toFixed(1) + '%';
                        document.getElementById('totalTrades').textContent = data.total_trades;
                        document.getElementById('currentStreak').textContent = data.current_streak;
                        
                        // Only update chart if bot is actually running
                        if (data.status === 'running') {
                            chart.data.labels.push(new Date().toLocaleTimeString());
                            chart.data.datasets[0].data.push(data.capital);
                            if (chart.data.labels.length > 20) {
                                chart.data.labels.shift();
                                chart.data.datasets[0].data.shift();
                            }
                            chart.update();
                        }
                    }
                });
        }

        function startBot() {
            fetch('/api/start', { method: 'POST' })
                .then(res => res.json())
                .then(data => alert(data.message));
        }

        function stopBot() {
            fetch('/api/stop', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    alert(data.message);
                    document.getElementById('statusIndicator').className = 'status-indicator status-stopped';
                    document.getElementById('statusText').textContent = 'Stopped';
                });
        }

        // DON'T auto-select any mode - let user choose
        // document.getElementById('aggressiveBtn').classList.add('active');  ← REMOVED
        
        // Update dashboard status every 2 seconds
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>'''
    
    with open(template_dir / 'dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    print("🎨 Creating Professional Trading Dashboard...")
    create_dashboard_template()
    print("✅ Dashboard ready!")
    print("\n🚀 Starting dashboard server...")
    print("📊 Open your browser to: http://localhost:5000")
    print("\n💡 Features:")
    print("   • Real-time portfolio tracking")
    print("   • Live win rate monitoring")
    print("   • Beautiful charts and graphs")
    print("   • Start/Stop bot controls")
    print("   • Professional UI/UX")
    print("\n⚡ Press Ctrl+C to stop\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
