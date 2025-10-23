#!/usr/bin/env python3
"""
📊 REAL-TIME MONITORING & ALERTING SYSTEM
Grafana-style Dashboard with ML-based Anomaly Detection
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

try:
    from flask import Flask, render_template, jsonify, send_from_directory, request
    from flask_socketio import SocketIO, emit
    import plotly.graph_objs as go
    import plotly.utils
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

@dataclass
class Alert:
    alert_id: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str
    message: str
    details: Dict
    timestamp: datetime
    acknowledged: bool = False

class AnomalyDetector:
    """ML-based anomaly detection system"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.5  # Standard deviations
        
    async def detect_trading_anomalies(self, metrics: Dict) -> List[Alert]:
        """Detect anomalies in trading metrics"""
        alerts = []
        
        # PnL anomaly detection
        if 'pnl' in metrics:
            pnl_anomaly = self._detect_pnl_anomaly(metrics['pnl'])
            if pnl_anomaly:
                alerts.append(Alert(
                    alert_id=f"pnl_anomaly_{datetime.now().strftime('%H%M%S')}",
                    severity='high',
                    category='performance',
                    message=f"Unusual P&L pattern detected: {pnl_anomaly['description']}",
                    details=pnl_anomaly,
                    timestamp=datetime.now()
                ))
        
        # Volume anomaly detection
        if 'volume' in metrics:
            vol_anomaly = self._detect_volume_anomaly(metrics['volume'])
            if vol_anomaly:
                alerts.append(Alert(
                    alert_id=f"vol_anomaly_{datetime.now().strftime('%H%M%S')}",
                    severity='medium',
                    category='market_data',
                    message=f"Unusual volume pattern: {vol_anomaly['description']}",
                    details=vol_anomaly,
                    timestamp=datetime.now()
                ))
        
        return alerts
    
    def _detect_pnl_anomaly(self, pnl_data: List[float]) -> Optional[Dict]:
        """Detect P&L anomalies using statistical analysis"""
        if len(pnl_data) < 10:
            return None
        
        recent_pnl = pnl_data[-5:]
        historical_pnl = pnl_data[:-5]
        
        if len(historical_pnl) == 0:
            return None
        
        hist_mean = np.mean(historical_pnl)
        hist_std = np.std(historical_pnl)
        
        if hist_std == 0:
            return None
        
        recent_mean = np.mean(recent_pnl)
        z_score = (recent_mean - hist_mean) / hist_std
        
        if abs(z_score) > self.anomaly_threshold:
            return {
                'type': 'pnl_deviation',
                'z_score': z_score,
                'recent_avg': recent_mean,
                'historical_avg': hist_mean,
                'description': f"Recent P&L deviates {z_score:.1f} standard deviations from historical average"
            }
        
        return None
    
    def _detect_volume_anomaly(self, volume_data: List[float]) -> Optional[Dict]:
        """Detect volume anomalies"""
        if len(volume_data) < 5:
            return None
        
        current_volume = volume_data[-1]
        avg_volume = np.mean(volume_data[:-1])
        
        if avg_volume == 0:
            return None
        
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > 3.0:  # Volume spike
            return {
                'type': 'volume_spike',
                'ratio': volume_ratio,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'description': f"Volume spike: {volume_ratio:.1f}x normal volume"
            }
        elif volume_ratio < 0.3:  # Volume drop
            return {
                'type': 'volume_drop',
                'ratio': volume_ratio,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'description': f"Volume drop: {volume_ratio:.1f}x normal volume"
            }
        
        return None

class RealTimeMonitor:
    """Real-time monitoring system"""
    
    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = deque(maxlen=100)
        self.anomaly_detector = AnomalyDetector()
        self.alert_callbacks = []
        self.bot = None
        self.bot_loop = None
        self.status_provider = None
        
        if WEB_AVAILABLE:
            self.app = Flask(__name__)
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")
            self.setup_routes()
    
    def setup_routes(self):
        """Setup web dashboard routes"""
        @self.app.route('/')
        def dashboard():
            # If a React production build exists, serve it; otherwise fallback to inline
            dist_path = os.path.join('dashboard', 'dist')
            index_path = os.path.join(dist_path, 'index.html')
            if os.path.exists(index_path):
                return send_from_directory(dist_path, 'index.html')
            return self.render_dashboard()
        
        # Favicon (avoid 404 noise)
        @self.app.route('/favicon.ico')
        def favicon():
            return ('', 204)
        
        # Support /index.html directly
        @self.app.route('/index.html')
        def index_html():
            return dashboard()
        
        # Serve React static assets when present
        @self.app.route('/assets/<path:filename>')
        def serve_assets(filename):
            dist_path = os.path.join('dashboard', 'dist')
            if os.path.exists(os.path.join(dist_path, 'assets', filename)):
                return send_from_directory(os.path.join(dist_path, 'assets'), filename)
            return ('Not Found', 404)
        
        @self.app.route('/api/metrics')
        def get_metrics():
            return jsonify(self.get_current_metrics())
        
        @self.app.route('/api/status')
        def get_status():
            try:
                status = {
                    'dashboard': True,
                    'bot_attached': self.bot is not None,
                }
                if callable(self.status_provider):
                    status.update(self.status_provider())
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alerts')
        def get_alerts():
            return jsonify([{
                'alert_id': alert.alert_id,
                'severity': alert.severity,
                'category': alert.category,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'acknowledged': alert.acknowledged
            } for alert in self.alerts])
        
        # Simple REST controls (start/stop/mode) for convenience
        @self.app.route('/api/start', methods=['POST', 'GET'])
        def api_start():
            try:
                if not self.bot:
                    return jsonify({'error': 'Bot not attached'}), 400
                self.bot.bot_running = True
                return jsonify({'ok': True, 'bot_running': True})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stop', methods=['POST', 'GET'])
        def api_stop():
            try:
                if not self.bot:
                    return jsonify({'error': 'Bot not attached'}), 400
                self.bot.bot_running = False
                return jsonify({'ok': True, 'bot_running': False})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/mode', methods=['POST', 'GET'])
        def api_mode():
            try:
                if not self.bot:
                    return jsonify({'error': 'Bot not attached'}), 400
                mode = (request.args.get('mode') or (request.json or {}).get('mode') or 'PRECISION').upper()
                if mode == 'NORMAL':
                    mode = 'PRECISION'
                if mode not in ('AGGRESSIVE','PRECISION'):
                    return jsonify({'error': f'Invalid mode {mode}'}), 400
                # Apply config (same as socket handler)
                self.bot.trading_mode = mode
                cfg = self.bot.mode_config[mode]
                self.bot.target_accuracy = cfg['target_accuracy']
                self.bot.min_confidence_for_trade = cfg['min_confidence']
                self.bot.ensemble_threshold = cfg['ensemble_threshold']
                self.bot.confidence_threshold = cfg['min_confidence']
                self.bot.base_confidence_threshold = cfg['min_confidence']
                if mode == 'AGGRESSIVE':
                    self.bot.fast_mode_enabled = True
                    self.bot.precision_mode_enabled = False
                    self.bot.min_price_history = 20
                    self.bot.confidence_adjustment_factor = 0.05
                    self.bot.aggressive_trade_guarantee = True
                    self.bot.aggressive_trade_interval = 60.0
                    self.bot.cycle_sleep_override = 10.0
                else:
                    self.bot.fast_mode_enabled = False
                    self.bot.precision_mode_enabled = True
                    self.bot.min_price_history = 50
                    self.bot.confidence_adjustment_factor = 0.01
                    self.bot.aggressive_trade_guarantee = False
                    self.bot.cycle_sleep_override = None
                return jsonify({'ok': True, 'mode': mode})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        # Graceful fallback for 404: show dashboard for non-API root paths
        @self.app.errorhandler(404)
        def handle_404(error):
            try:
                path = request.path or '/'
                # For root-level or unknown non-API paths, show dashboard
                if path in ('/', '/index.html') or not (path.startswith('/api/') or path.startswith('/assets/')):
                    return self.render_dashboard(), 200
            except Exception:
                pass
            return ('Not Found', 404)
        
        @self.app.route('/api/portfolio')
        def get_portfolio():
            try:
                if not self.bot:
                    return jsonify({'error': 'Bot not attached'}), 400
                # If we have the bot loop, call async trader.get_portfolio_value()
                if self.bot_loop and hasattr(self.bot, 'trader') and hasattr(self.bot.trader, 'get_portfolio_value'):
                    import asyncio as _aio
                    fut = _aio.run_coroutine_threadsafe(self.bot.trader.get_portfolio_value(), self.bot_loop)
                    result = fut.result(timeout=5)
                    return jsonify(result)
                return jsonify({'error': 'Portfolio unavailable'}), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        # Socket.IO control channel for dashboard -> bot commands
        @self.socketio.on('control')
        def handle_control(data):
            action = (data or {}).get('action')
            payload = (data or {}).get('payload', {})
            try:
                if not self.bot or not self.bot_loop:
                    raise RuntimeError('Bot not attached')

                # Utility to schedule a coroutine on the bot loop
                def schedule_coro(coro):
                    import asyncio as _aio
                    _aio.run_coroutine_threadsafe(coro, self.bot_loop)

                if action == 'set_mode':
                    mode = str(payload.get('mode', 'PRECISION')).upper()
                    if mode not in ('AGGRESSIVE', 'PRECISION', 'NORMAL'):
                        emit('control_error', {'action': action, 'error': f'Invalid mode {mode}'})
                        return
                    # Map NORMAL -> PRECISION
                    if mode == 'NORMAL':
                        mode = 'PRECISION'
                    setattr(self.bot, 'trading_mode', mode)
                    # Apply full mode config immediately (mirror select_trading_mode behavior)
                    try:
                        cfg = self.bot.mode_config[mode]
                        self.bot.target_accuracy = cfg['target_accuracy']
                        self.bot.min_confidence_for_trade = cfg['min_confidence']
                        self.bot.ensemble_threshold = cfg['ensemble_threshold']
                        self.bot.confidence_threshold = cfg['min_confidence']
                        self.bot.base_confidence_threshold = cfg['min_confidence']
                        if mode == 'AGGRESSIVE':
                            self.bot.fast_mode_enabled = True
                            self.bot.precision_mode_enabled = False
                            self.bot.min_price_history = 20
                            self.bot.confidence_adjustment_factor = 0.05
                            # Enable guarantee and faster cycles
                            self.bot.aggressive_trade_guarantee = True
                            self.bot.aggressive_trade_interval = 60.0
                            self.bot.cycle_sleep_override = 10.0
                        else:
                            self.bot.fast_mode_enabled = False
                            self.bot.precision_mode_enabled = True
                            self.bot.min_price_history = 50
                            self.bot.confidence_adjustment_factor = 0.01
                            self.bot.aggressive_trade_guarantee = False
                            self.bot.cycle_sleep_override = None
                    except Exception as _e:
                        emit('control_error', {'action': action, 'error': f'mode apply failed: {str(_e)}'})
                        return
                elif action == 'toggle_trading':
                    running = bool(payload.get('running', True))
                    setattr(self.bot, 'bot_running', running)
                elif action == 'set_symbols':
                    symbols = payload.get('symbols') or []
                    if isinstance(symbols, list) and symbols:
                        setattr(self.bot, 'symbols', symbols)
                elif action == 'set_risk':
                    if 'min_confidence' in payload:
                        setattr(self.bot, 'min_confidence_for_trade', float(payload['min_confidence']))
                    if 'risk_multiplier' in payload:
                        setattr(self.bot, 'risk_multiplier', float(payload['risk_multiplier']))
                    if 'min_trade_size' in payload:
                        setattr(self.bot, 'min_trade_size', float(payload['min_trade_size']))
                elif action == 'set_cycle_interval':
                    seconds = payload.get('seconds')
                    try:
                        setattr(self.bot, 'cycle_sleep_override', float(seconds) if seconds is not None else None)
                    except Exception:
                        setattr(self.bot, 'cycle_sleep_override', None)
                elif action == 'place_order':
                    symbol = payload.get('symbol')
                    side = payload.get('side', 'buy').lower()
                    amount = float(payload.get('amount', 0))
                    tp = payload.get('take_profit')
                    sl = payload.get('stop_loss')
                    if symbol and amount > 0:
                        schedule_coro(self.bot.submit_manual_order(symbol, side, amount, tp, sl))
                elif action == 'close_position':
                    symbol = payload.get('symbol')
                    amount_usd = payload.get('amount_usd')
                    if symbol:
                        schedule_coro(self.bot.request_close_position(symbol, amount_usd))
                else:
                    emit('control_error', {'action': action, 'error': 'Unknown action'})
                    return
                emit('control_ack', {'action': action})
            except Exception as e:
                emit('control_error', {'action': action, 'error': str(e)})

    def attach_bot(self, bot, status_provider=None):
        """Attach bot instance for control and status reporting"""
        try:
            self.bot = bot
            # Capture current running loop to schedule async work from Flask thread
            self.bot_loop = asyncio.get_running_loop()
            if callable(status_provider):
                self.status_provider = status_provider
            return True
        except RuntimeError:
            # Not in async loop context; status provider still set
            self.bot = bot
            self.bot_loop = None
            self.status_provider = status_provider
            return True
    
    def render_dashboard(self):
        """Render real-time dashboard"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Poise Trader - Real-Time Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #0a0e27; color: #fff; }
        .header { text-align: center; margin-bottom: 30px; }
        .control-panel { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin: 10px 0 20px; }
        .control-panel button { background: #2d3561; color: #fff; border: 1px solid #4a5699; padding: 10px 14px; border-radius: 6px; cursor: pointer; }
        .control-panel button.active { background: #4CAF50; border-color: #4CAF50; }
        .control-panel .status { margin-left: 10px; font-weight: bold; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: #1a1d3a; padding: 20px; border-radius: 10px; border: 1px solid #2d3561; }
        .metric-value { font-size: 2em; font-weight: bold; color: #4CAF50; }
        .alerts-panel { margin-top: 30px; background: #1a1d3a; padding: 20px; border-radius: 10px; }
        .alert { padding: 10px; margin: 5px 0; border-radius: 5px; }
        .alert.critical { background: #f44336; }
        .alert.high { background: #ff9800; }
        .alert.medium { background: #2196F3; }
        .alert.low { background: #4CAF50; }
        .chart-container { height: 400px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏆 POISE TRADER - REAL-TIME MONITORING</h1>
        <p>Professional-Grade Trading Dashboard</p>
    </div>
    <div class="control-panel">
        <button id="mode-aggressive">Aggressive</button>
        <button id="mode-normal">Normal</button>
        <button id="start-trading">Start Trading</button>
        <button id="stop-trading">Stop Trading</button>
        <span class="status">Mode: <strong id="mode-text">-</strong> | Trading: <strong id="trading-text">-</strong></span>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Total P&L</h3>
            <div class="metric-value" id="total-pnl">$0.00</div>
        </div>
        <div class="metric-card">
            <h3>Win Rate</h3>
            <div class="metric-value" id="win-rate">0%</div>
        </div>
        <div class="metric-card">
            <h3>Active Positions</h3>
            <div class="metric-value" id="active-positions">0</div>
        </div>
        <div class="metric-card">
            <h3>Daily Volume</h3>
            <div class="metric-value" id="daily-volume">$0</div>
        </div>
    </div>
    
    <div class="chart-container">
        <div id="pnl-chart"></div>
    </div>
    
    <div class="alerts-panel">
        <h3>🚨 Real-Time Alerts</h3>
        <div id="alerts-container"></div>
    </div>
    
    <script>
        const socket = io();
        
        // Update metrics in real-time
        socket.on('metrics_update', function(data) {
            document.getElementById('total-pnl').textContent = '$' + data.total_pnl.toFixed(2);
            document.getElementById('win-rate').textContent = (data.win_rate * 100).toFixed(1) + '%';
            document.getElementById('active-positions').textContent = data.active_positions;
            document.getElementById('daily-volume').textContent = '$' + data.daily_volume.toLocaleString();
            
            // Update P&L chart
            updatePnLChart(data.pnl_history);
        });
        
        socket.on('control_ack', function(evt) {
            refreshStatus();
        });
        socket.on('control_error', function(evt) {
            alert('Control error: ' + (evt && evt.error ? evt.error : 'unknown'));
        });
        
        // Handle alerts
        socket.on('new_alert', function(alert) {
            addAlert(alert);
        });
        
        function updatePnLChart(pnlHistory) {
            const trace = {
                x: pnlHistory.map(p => p.timestamp),
                y: pnlHistory.map(p => p.value),
                type: 'scatter',
                mode: 'lines',
                line: { color: '#4CAF50', width: 2 },
                name: 'P&L'
            };
            
            const layout = {
                title: 'Real-Time P&L Performance',
                paper_bgcolor: '#1a1d3a',
                plot_bgcolor: '#0a0e27',
                font: { color: '#fff' },
                xaxis: { gridcolor: '#2d3561' },
                yaxis: { gridcolor: '#2d3561' }
            };
            
            Plotly.newPlot('pnl-chart', [trace], layout);
        }
        
        function addAlert(alert) {
            const alertsContainer = document.getElementById('alerts-container');
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert ${alert.severity}`;
            alertDiv.innerHTML = `
                <strong>[${alert.timestamp}] ${alert.category.toUpperCase()}</strong><br>
                ${alert.message}
            `;
            alertsContainer.insertBefore(alertDiv, alertsContainer.firstChild);
            
            // Keep only last 10 alerts visible
            while (alertsContainer.children.length > 10) {
                alertsContainer.removeChild(alertsContainer.lastChild);
            }
        }
        
        // Fetch initial data
        fetch('/api/metrics')
            .then(response => response.json())
            .then(data => socket.emit('metrics_update', data));
        
        function setButtons(mode, running) {
            const aggr = document.getElementById('mode-aggressive');
            const normal = document.getElementById('mode-normal');
            aggr.classList.toggle('active', mode === 'AGGRESSIVE');
            normal.classList.toggle('active', mode === 'PRECISION' || mode === 'NORMAL');
            document.getElementById('mode-text').textContent = mode || '-';
            document.getElementById('trading-text').textContent = running ? 'RUNNING' : 'PAUSED';
        }
        function refreshStatus() {
            fetch('/api/status').then(r => r.json()).then(s => {
                setButtons((s && s.trading_mode) ? s.trading_mode : '-', !!(s && s.bot_running));
            }).catch(()=>{});
        }
        
        document.getElementById('mode-aggressive').addEventListener('click', () => {
            socket.emit('control', { action: 'set_mode', payload: { mode: 'AGGRESSIVE' } });
        });
        document.getElementById('mode-normal').addEventListener('click', () => {
            socket.emit('control', { action: 'set_mode', payload: { mode: 'PRECISION' } });
        });
        document.getElementById('start-trading').addEventListener('click', () => {
            socket.emit('control', { action: 'toggle_trading', payload: { running: true } });
        });
        document.getElementById('stop-trading').addEventListener('click', () => {
            socket.emit('control', { action: 'toggle_trading', payload: { running: false } });
        });
        
        // Initial
        refreshStatus();
    </script>
</body>
</html>
        '''
    
    async def update_metrics(self, metrics: Dict):
        """Update metrics and check for anomalies (robust to floats or series)"""
        timestamp = datetime.now()

        # Normalize incoming metrics for storage
        store_metrics = {}
        pnl_in = metrics.get('pnl', 0)
        vol_in = metrics.get('volume', 0)

        # Last points for storage/visualization
        last_pnl = (pnl_in[-1] if isinstance(pnl_in, (list, tuple)) and pnl_in else float(pnl_in) or 0.0)
        last_vol = (vol_in[-1] if isinstance(vol_in, (list, tuple)) and vol_in else float(vol_in) or 0.0)

        store_metrics['pnl'] = float(last_pnl)
        store_metrics['volume'] = float(last_vol)
        # Pass-through other simple metrics
        for k in ('win_rate', 'active_positions', 'daily_volume', 'sharpe_ratio', 'max_drawdown'):
            if k in metrics:
                store_metrics[k] = metrics[k]

        # Store normalized metrics
        for key, value in store_metrics.items():
            self.metrics_history[key].append({
                'timestamp': timestamp,
                'value': value
            })

        # Build anomaly series (prefer incoming series; else reconstruct from history)
        pnl_series = pnl_in if isinstance(pnl_in, (list, tuple)) else [pt['value'] for pt in self.metrics_history.get('pnl', [])]
        vol_series = vol_in if isinstance(vol_in, (list, tuple)) else [pt['value'] for pt in self.metrics_history.get('volume', [])]

        anomaly_metrics = {
            'pnl': list(pnl_series),
            'volume': list(vol_series)
        }

        # Detect anomalies
        alerts = await self.anomaly_detector.detect_trading_anomalies(anomaly_metrics)

        # Process new alerts
        for alert in alerts:
            self.alerts.append(alert)
            await self.send_alert(alert)

        # Broadcast metrics update with UI-friendly keys
        if WEB_AVAILABLE:
            self.socketio.emit('metrics_update', {
                'total_pnl': float(last_pnl),
                'win_rate': float(store_metrics.get('win_rate', 0)),
                'active_positions': int(store_metrics.get('active_positions', 0)),
                'daily_volume': float(store_metrics.get('daily_volume', 0)),
                'pnl_history': list(self.metrics_history.get('pnl', [])),
                'timestamp': timestamp.isoformat()
            })
    
    async def send_alert(self, alert: Alert):
        """Send alert through all channels"""
        # Log alert
        logging.warning(f"ALERT [{alert.severity.upper()}] {alert.category}: {alert.message}")
        
        # Broadcast to web dashboard
        if WEB_AVAILABLE:
            self.socketio.emit('new_alert', {
                'alert_id': alert.alert_id,
                'severity': alert.severity,
                'category': alert.category,
                'message': alert.message,
                'timestamp': alert.timestamp.strftime('%H:%M:%S')
            })
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logging.error(f"Alert callback error: {e}")
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics summary"""
        current_metrics = {}
        
        for key, history in self.metrics_history.items():
            if history:
                current_metrics[key] = history[-1]['value']
        
        return {
            'total_pnl': current_metrics.get('pnl', 0),
            'win_rate': current_metrics.get('win_rate', 0),
            'active_positions': current_metrics.get('active_positions', 0),
            'daily_volume': current_metrics.get('daily_volume', 0),
            'sharpe_ratio': current_metrics.get('sharpe_ratio', 0),
            'max_drawdown': current_metrics.get('max_drawdown', 0),
            'total_trades': int(current_metrics.get('total_trades', 0)),
            'current_streak': int(current_metrics.get('current_streak', 0)),
            'portfolio_value': current_metrics.get('portfolio_value', 0),
            'pnl_history': list(self.metrics_history.get('pnl', [])),
            'timestamp': datetime.now().isoformat()
        }
    
    def register_alert_callback(self, callback):
        """Register callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self, host='localhost', port=5000):
        """Start the monitoring system"""
        if WEB_AVAILABLE:
            print(f"🚀 REAL-TIME DASHBOARD STARTING...")
            print(f"📊 Dashboard URL: http://{host}:{port}")
            print(f"🔔 Real-time alerts enabled")
            # Run Flask-SocketIO server in a background thread to avoid blocking event loop
            await asyncio.to_thread(self.socketio.run, self.app, host=host, port=port, debug=False)
        else:
            print("⚠️ Web dashboard not available - install Flask and plotly")
            # Continue with basic monitoring
            while True:
                await asyncio.sleep(1)

# Global instance
real_time_monitor = RealTimeMonitor()
