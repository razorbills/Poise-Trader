#!/usr/bin/env python3
"""
🎯 SIMPLE DASHBOARD SERVER
Minimal backend for the simple control panel
"""

from flask import Flask, jsonify, request, send_file
from flask_socketio import SocketIO
import os
import threading
import time
from datetime import datetime

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'simple_poise_2025'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global bot reference and data
bot_instance = None
current_mode = 'PRECISION'
pnl_history = []
max_history_points = 50
start_time = time.time()

try:
    DEFAULT_STARTING_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '5.0') or 5.0)
except Exception:
    DEFAULT_STARTING_CAPITAL = 5.0

@app.route('/')
def index():
    """Serve the enhanced dashboard"""
    # Check if enhanced dashboard exists, otherwise use simple
    import os
    if os.path.exists('enhanced_simple_dashboard.html'):
        return send_file('enhanced_simple_dashboard.html')
    return send_file('simple_dashboard.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring services"""
    global bot_instance
    
    health = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'bot_connected': bot_instance is not None,
        'bot_running': False,
        'uptime': time.time() - start_time if 'start_time' in globals() else 0
    }
    
    if bot_instance:
        health['bot_running'] = getattr(bot_instance, 'bot_running', False)
        health['current_capital'] = getattr(bot_instance, 'current_capital', 0)
    
    return jsonify(health)

@app.route('/ping')
def ping():
    """Simple ping endpoint"""
    return jsonify({'status': 'alive', 'timestamp': datetime.now().isoformat()})

@app.route('/keep-alive')
def keep_alive_endpoint():
    """Keep-alive endpoint"""
    return jsonify({
        'status': 'active',
        'message': 'Service is running 24/7',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/status')
def get_status():
    """Get current bot status"""
    global bot_instance, current_mode
    
    status = {
        'connected': bot_instance is not None,
        'running': False,
        'mode': current_mode
    }
    
    if bot_instance:
        status['running'] = getattr(bot_instance, 'bot_running', False)
        status['mode'] = getattr(bot_instance, 'trading_mode', current_mode)
        status['capital'] = getattr(bot_instance, 'current_capital', 0)
        status['win_rate'] = getattr(bot_instance, 'win_rate', 0) * 100

        try:
            if hasattr(bot_instance, 'trader') and bot_instance.trader is not None:
                status['paper_shorting'] = bool(getattr(bot_instance.trader, 'enable_shorting', False))
                status['paper_leverage'] = float(getattr(bot_instance.trader, 'leverage', 1.0) or 1.0)
                status['paper_fee_rate'] = float(getattr(bot_instance.trader, 'fee_rate', 0.0002) or 0.0002)

                if hasattr(bot_instance.trader, 'data_feed') and bot_instance.trader.data_feed is not None:
                    df = bot_instance.trader.data_feed
                    if hasattr(df, 'get_health'):
                        status['mexc'] = df.get_health()

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

                status['real_trading'] = {
                    'enabled': bool(real_enabled),
                    'keys_present': bool(keys_present),
                    'market_type': market_type or None,
                    'leverage': float(getattr(bot_instance.trader, 'leverage', 1.0) or 1.0),
                    'ready': bool(getattr(bot_instance.trader, '_real_trading_ready', lambda: False)()),
                    'last_error': getattr(bot_instance.trader, 'last_real_order_error', None),
                    'last_order': getattr(bot_instance.trader, 'last_real_order', None)
                }
        except Exception as e:
            status['mexc'] = {'connected': False, 'last_error': str(e)[:120]}
    
    return jsonify(status)

@app.route('/api/metrics')
def get_metrics():
    """Get trading metrics"""
    global bot_instance, pnl_history
    
    metrics = {
        'total_pnl': 0.0,
        'win_rate': 0.0,
        'active_positions': 0,
        'daily_volume': 0.0,
        'total_trades': 0,
        'pnl_history': pnl_history
    }
    
    if bot_instance:
        # Calculate P&L
        current_capital = getattr(bot_instance, 'current_capital', 5.0)
        initial_capital = getattr(bot_instance, 'initial_capital', 5.0)
        pnl = current_capital - initial_capital
        
        # Update metrics
        metrics['total_pnl'] = pnl
        metrics['win_rate'] = getattr(bot_instance, 'win_rate', 0.0)
        metrics['daily_volume'] = getattr(bot_instance, 'total_volume_traded', 0.0)
        
        # Get actual trade count from bot
        if hasattr(bot_instance, 'trader') and hasattr(bot_instance.trader, 'trade_history'):
            metrics['total_trades'] = len(bot_instance.trader.trade_history)
        else:
            metrics['total_trades'] = getattr(bot_instance, 'trade_count', 0)
        
        # Get active positions count
        if hasattr(bot_instance, 'trader') and hasattr(bot_instance.trader, 'positions'):
            positions = bot_instance.trader.positions
            metrics['active_positions'] = len([p for p in positions.values() if p.get('quantity', 0) > 0])
        
        # Update P&L history with REAL values only
        from datetime import datetime
        
        # Only use real P&L, no fake variations
        history_point = {
            'timestamp': datetime.now().isoformat(),
            'value': pnl  # Real P&L only
        }
        pnl_history.append(history_point)
        if len(pnl_history) > max_history_points:
            pnl_history.pop(0)
        metrics['pnl_history'] = pnl_history

        try:
            recent = []
            if hasattr(bot_instance, 'trade_history') and isinstance(getattr(bot_instance, 'trade_history', None), list):
                recent = bot_instance.trade_history[-10:]
            elif hasattr(bot_instance, 'trader') and hasattr(bot_instance.trader, 'trade_history'):
                recent = bot_instance.trader.trade_history[-10:]
            metrics['recent_trades'] = recent
        except Exception:
            metrics['recent_trades'] = []
    
    return jsonify(metrics)


@app.route('/api/trades')
def get_recent_trades():
    """Get recent trades in a simple normalized format for the dashboard"""
    global bot_instance

    trades = []
    try:
        bot_trades = []
        trader_trades = []

        if bot_instance and hasattr(bot_instance, 'trade_history') and isinstance(getattr(bot_instance, 'trade_history', None), list):
            bot_trades = bot_instance.trade_history

        if bot_instance and hasattr(bot_instance, 'trader') and hasattr(bot_instance.trader, 'trade_history') and isinstance(getattr(bot_instance.trader, 'trade_history', None), list):
            trader_trades = bot_instance.trader.trade_history

        # Prefer the source that has more records (usually trader.trade_history)
        trades = trader_trades if len(trader_trades) >= len(bot_trades) else bot_trades
    except Exception:
        trades = []

    normalized = []
    for t in (trades or [])[-50:]:
        try:
            if not isinstance(t, dict):
                continue
            normalized.append({
                'timestamp': t.get('timestamp') or t.get('time') or datetime.now().isoformat(),
                'symbol': t.get('symbol', 'UNKNOWN'),
                'side': (t.get('side') or t.get('action') or '').lower() or 'buy',
                'pnl': float(t.get('pnl', t.get('profit_loss', 0.0)) or 0.0),
                'event': t.get('event') or None,
                'notional_usd': float(t.get('notional_usd', 0.0) or 0.0),
                'commission': float(t.get('commission', 0.0) or 0.0)
            })
        except Exception:
            continue

    try:
        limit = int(str(request.args.get('limit', '50') or '50').strip() or 50)
    except Exception:
        limit = 50
    limit = max(1, min(limit, 500))

    try:
        order = str(request.args.get('order', 'desc') or 'desc').strip().lower()
    except Exception:
        order = 'desc'

    if order == 'asc':
        trades_out = normalized[-limit:]
    else:
        trades_out = list(reversed(normalized))[:limit]

    return jsonify({'trades': trades_out, 'total': len(trades or []), 'returned': len(trades_out)})

@app.route('/api/portfolio')
def get_portfolio():
    """Get portfolio positions"""
    portfolio = {
        'total_value': 0.0,
        'cash': 0.0,
        'positions': {},
        'daily_volume': 0,
        'pnl': 0.0
    }
    
    # Debug logging
    print(f"\n DASHBOARD: Getting portfolio data...")
    print(f"   Bot instance exists: {bot_instance is not None}")
    
    if bot_instance:
        # Get current capital first
        portfolio['total_value'] = getattr(bot_instance, 'current_capital', 5.0)
        portfolio['cash'] = portfolio['total_value']  # Default values

        # Prefer the trader's sync portfolio if available (supports BUY/SELL positions)
        if hasattr(bot_instance, 'trader') and hasattr(bot_instance.trader, 'get_portfolio_value_sync'):
            try:
                trader_portfolio = bot_instance.trader.get_portfolio_value_sync()
                portfolio['total_value'] = trader_portfolio.get('total_value', portfolio['total_value'])
                portfolio['cash'] = trader_portfolio.get('cash', portfolio['cash'])

                positions_out = {}
                for symbol, pos_data in (trader_portfolio.get('positions') or {}).items():
                    if not isinstance(pos_data, dict):
                        continue
                    merged = dict(pos_data)
                    if 'entry_price' not in merged:
                        merged['entry_price'] = merged.get('avg_price')
                    positions_out[symbol] = merged

                portfolio['positions'] = positions_out
                if portfolio['positions']:
                    portfolio['pnl'] = sum(float(p.get('unrealized_pnl', 0) or 0) for p in portfolio['positions'].values())
                print(f"   Synced with trader: Total=${portfolio['total_value']:.2f}, Positions={len(portfolio['positions'])}")
                print(f"   Returning {len(portfolio['positions'])} positions to dashboard")
                return jsonify(portfolio)
            except Exception as e:
                print(f"   Could not sync with trader: {e}")
        
        # Get positions if available
        if hasattr(bot_instance, 'trader') and hasattr(bot_instance.trader, 'positions'):
            positions = bot_instance.trader.positions
            
            print(f"   Found {len(positions)} positions in trader")
            
            # Fetch REAL current prices for all positions
            for symbol, pos in positions.items():
                quantity = pos.get('quantity', 0)
                print(f"   {symbol}: quantity={quantity}")
                
                if quantity > 0:
                    avg_price = pos.get('avg_price', 0) or pos.get('entry_price', 0)
                    
                    # Get EXACT current price from the data feed or price history
                    current_price = pos.get('current_price', avg_price)  # Use stored current price first
                    
                    # Try to get from bot's price history (most recent real price)
                    if hasattr(bot_instance, 'price_history') and symbol in bot_instance.price_history:
                        price_history = bot_instance.price_history[symbol]
                        if price_history and len(price_history) > 0:
                            current_price = list(price_history)[-1]  # Get most recent REAL price
                    
                    # Update position with exact current price
                    pos['current_price'] = current_price
                    
                    # Calculate P&L
                    side = str(pos.get('action', 'BUY') or 'BUY').upper()
                    margin = float(pos.get('total_cost', 0) or 0)
                    cost_basis = margin if margin > 0 else (avg_price * quantity)
                    current_value = current_price * quantity
                    unrealized_pnl = (current_price - avg_price) * quantity if side == 'BUY' else (avg_price - current_price) * quantity
                    
                    portfolio['positions'][symbol] = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'cost_basis': cost_basis,
                        'current_value': current_value,
                        'current_price': current_price,
                        'avg_price': avg_price,
                        'entry_price': avg_price,  # For compatibility
                        'unrealized_pnl': unrealized_pnl,
                        'action': side,
                        # IMPORTANT: Include custom TP/SL values if they exist
                        'take_profit': pos.get('take_profit'),
                        'stop_loss': pos.get('stop_loss')
                    }
                    
                    # Log if custom TP/SL exists
                    if pos.get('take_profit') or pos.get('stop_loss'):
                        print(f"   Added {symbol} to portfolio: P&L=${unrealized_pnl:.2f} | TP=${pos.get('take_profit', 'None')} SL=${pos.get('stop_loss', 'None')}")
                    else:
                        print(f"   Added {symbol} to portfolio: P&L=${unrealized_pnl:.2f}")
        
        # Calculate total P&L from positions
        if portfolio['positions']:
            portfolio['pnl'] = sum(pos['unrealized_pnl'] for pos in portfolio['positions'].values())
            portfolio['total_value'] = portfolio['cash'] + sum(pos['current_value'] for pos in portfolio['positions'].values())
            print(f"   Total P&L: ${portfolio['pnl']:.2f}, Total Value: ${portfolio['total_value']:.2f}")
        
    print(f"   Returning {len(portfolio['positions'])} positions to dashboard")
    return jsonify(portfolio)

@app.route('/api/mode', methods=['POST'])
def set_mode():
    """Set trading mode"""
    global bot_instance, current_mode
    
    data = request.get_json() or {}
    mode = data.get('mode', 'PRECISION').upper()
    
    if mode not in ['AGGRESSIVE', 'PRECISION', 'NORMAL']:
        return jsonify({'success': False, 'message': 'Invalid mode'}), 400
    
    # Map NORMAL to PRECISION
    if mode == 'NORMAL':
        mode = 'PRECISION'
    
    current_mode = mode
    
    if bot_instance:
        # Apply mode to bot
        bot_instance.trading_mode = mode
        
        # Get mode config
        if hasattr(bot_instance, 'mode_config'):
            cfg = bot_instance.mode_config.get(mode, {})
            bot_instance.target_accuracy = cfg.get('target_accuracy', 0.6)
            bot_instance.min_confidence_for_trade = cfg.get('min_confidence', 0.3)
            bot_instance.ensemble_threshold = cfg.get('ensemble_threshold', 2)
            bot_instance.confidence_threshold = cfg.get('min_confidence', 0.3)
            bot_instance.base_confidence_threshold = cfg.get('min_confidence', 0.3)
        
        # Set mode-specific parameters
        if mode == 'AGGRESSIVE':
            bot_instance.fast_mode_enabled = True
            bot_instance.precision_mode_enabled = False
            bot_instance.min_price_history = 5
            bot_instance.confidence_adjustment_factor = 0.05
            bot_instance.aggressive_trade_guarantee = True
            bot_instance.aggressive_trade_interval = 60.0
            bot_instance.cycle_sleep_override = 10.0
            bot_instance.win_rate_optimizer_enabled = False
            bot_instance.min_trade_quality_score = 10.0
            bot_instance.min_confidence_for_trade = 0.10
            print(f"⚡ AGGRESSIVE MODE ACTIVATED")
        else:
            bot_instance.fast_mode_enabled = False
            bot_instance.precision_mode_enabled = True
            bot_instance.min_price_history = 10
            bot_instance.confidence_adjustment_factor = 0.01
            bot_instance.aggressive_trade_guarantee = False
            bot_instance.cycle_sleep_override = None
            bot_instance.win_rate_optimizer_enabled = False
            bot_instance.min_trade_quality_score = 25.0
            bot_instance.min_confidence_for_trade = 0.30
            print(f"🎯 NORMAL MODE ACTIVATED")
    
    return jsonify({'success': True, 'mode': mode})

@app.route('/api/start', methods=['POST'])
def start_trading():
    """Start trading"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'success': False, 'message': 'No bot connected'}), 400
    
    bot_instance.bot_running = True
    print(f"▶️ Trading STARTED in {bot_instance.trading_mode} mode - REAL TRADING ACTIVE!")
    
    return jsonify({'success': True, 'message': 'Trading started'})

@app.route('/api/stop', methods=['POST'])
def stop_trading():
    """Stop trading"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'success': False, 'message': 'No bot connected'}), 400
    
    bot_instance.bot_running = False
    print(f"⏹️ Trading STOPPED")
    
    return jsonify({'success': True, 'message': 'Trading stopped'})

@app.route('/api/update_position', methods=['POST'])
def update_position():
    """Update position TP/SL"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'success': False, 'message': 'No bot connected'}), 400
    
    data = request.get_json() or {}
    symbol = data.get('symbol')
    take_profit = data.get('take_profit')
    stop_loss = data.get('stop_loss')
    
    if not symbol:
        return jsonify({'success': False, 'message': 'Symbol required'}), 400
    
    try:
        print(f"\n🔧 UPDATE_POSITION REQUEST:")
        print(f"   Symbol: {symbol}")
        print(f"   Take Profit: {take_profit}")
        print(f"   Stop Loss: {stop_loss}")
        
        # Update position TP/SL in bot's trader
        if hasattr(bot_instance, 'trader') and hasattr(bot_instance.trader, 'positions'):
            if symbol in bot_instance.trader.positions:
                position = bot_instance.trader.positions[symbol]
                
                print(f"   📊 Position BEFORE update: TP=${position.get('take_profit', 'None')}, SL=${position.get('stop_loss', 'None')}")
                
                # Store TP/SL values
                if take_profit:
                    position['take_profit'] = float(take_profit)
                    print(f"   ✅ Set custom TP for {symbol}: ${float(take_profit):.2f}")
                if stop_loss:
                    position['stop_loss'] = float(stop_loss)
                    print(f"   ✅ Set custom SL for {symbol}: ${float(stop_loss):.2f}")
                
                # Log the full position data for debugging
                print(f"   📊 Position AFTER update: TP=${position.get('take_profit', 'None')}, SL=${position.get('stop_loss', 'None')}")
                
                # Save to trading state for persistence
                if hasattr(bot_instance.trader, '_save_state'):
                    bot_instance.trader._save_state()
                else:
                    # Manual save if _save_state not available
                    import json
                    from datetime import datetime
                    state = {
                        "cash_balance": bot_instance.trader.cash_balance,
                        "positions": bot_instance.trader.positions,
                        "trade_history": bot_instance.trader.trade_history,
                        "total_trades": bot_instance.trader.total_trades,
                        "winning_trades": bot_instance.trader.winning_trades,
                        "initial_capital": bot_instance.trader.initial_capital,
                        "last_save_time": datetime.now().isoformat()
                    }
                    with open('trading_state.json', 'w') as f:
                        json.dump(state, f, indent=2, default=str)
                
                print(f"📝 ✅ UPDATE SUCCESSFUL for {symbol}")
                print(f"   Final TP: ${position.get('take_profit', 'None')}")
                print(f"   Final SL: ${position.get('stop_loss', 'None')}\n")
                
                return jsonify({
                    'success': True, 
                    'message': f'Updated {symbol} targets',
                    'take_profit': position.get('take_profit'),
                    'stop_loss': position.get('stop_loss')
                })
            else:
                return jsonify({'success': False, 'message': 'Position not found'}), 404
        
        return jsonify({'success': False, 'message': 'Trading system not available'}), 500
        
    except Exception as e:
        print(f"❌ Error updating position: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/markets')
def get_markets():
    """Get available markets"""
    global bot_instance
    
    markets = []
    if bot_instance and hasattr(bot_instance, 'symbols'):
        markets = bot_instance.symbols
    else:
        # Default markets if bot not available
        markets = [
            # Crypto
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT', 'DOT/USDT', 'AVAX/USDT',
            'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'APT/USDT',
            'ARB/USDT', 'OP/USDT', 'SUI/USDT', 'TIA/USDT', 'SEI/USDT',
            # Metals
            'XAU/USDT', 'XAG/USDT',
            # Commodities
            'WTI/USDT'
        ]
    
    return jsonify({'markets': markets})

@app.route('/api/update_markets', methods=['POST'])
def update_markets():
    """Update selected markets for trading"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'success': False, 'message': 'No bot connected'}), 400
    
    data = request.get_json() or {}
    selected_markets = data.get('markets', [])
    
    try:
        # Store selected markets in bot instance
        if selected_markets:  # If specific markets selected
            bot_instance.active_symbols = selected_markets
            print(f"🎯 Market filter updated: Trading {len(selected_markets)} markets")
            print(f"   Selected: {', '.join(selected_markets[:5])}{'...' if len(selected_markets) > 5 else ''}")
        else:  # If no markets selected, use all
            bot_instance.active_symbols = bot_instance.symbols.copy()
            print(f"🌍 Market filter cleared: Trading all {len(bot_instance.symbols)} markets")
        
        return jsonify({
            'success': True,
            'message': f'Updated to {len(bot_instance.active_symbols)} markets',
            'active_count': len(bot_instance.active_symbols)
        })
        
    except Exception as e:
        print(f"❌ Error updating markets: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_trading():
    """Reset trading to fresh $5.00 start"""
    global bot_instance
    
    try:
        start_capital = DEFAULT_STARTING_CAPITAL
        # Create fresh trading state
        fresh_state = {
            "cash_balance": start_capital,
            "positions": {},
            "trade_history": [],
            "total_trades": 0,
            "winning_trades": 0,
            "initial_capital": start_capital,
            "last_save_time": datetime.now().isoformat()
        }
        
        # Save fresh state
        import json
        with open('trading_state.json', 'w') as f:
            json.dump(fresh_state, f, indent=2)
        
        # Reset bot if connected
        if bot_instance and hasattr(bot_instance, 'trader'):
            try:
                bot_instance.trader.cash_balance = start_capital
            except Exception:
                pass
            bot_instance.trader.positions = {}
            bot_instance.trader.trade_history = []
            try:
                bot_instance.trader.initial_capital = start_capital
            except Exception:
                pass
            bot_instance.total_completed_trades = 0
            bot_instance.winning_trades = 0
            bot_instance.current_capital = start_capital
            try:
                bot_instance.initial_capital = start_capital
            except Exception:
                pass
            
        print(f"✅ Trading reset to fresh ${start_capital:.2f}")
        return jsonify({'success': True, 'message': f'Reset to ${start_capital:.2f} starting capital'})
        
    except Exception as e:
        print(f"❌ Error resetting: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/close_position', methods=['POST'])
def close_position():
    """Close a position"""
    global bot_instance
    
    if not bot_instance:
        return jsonify({'success': False, 'message': 'No bot connected'}), 400
    
    data = request.get_json() or {}
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({'success': False, 'message': 'Symbol required'}), 400
    
    try:
        # Check if position exists
        if hasattr(bot_instance, 'trader') and hasattr(bot_instance.trader, 'positions'):
            if symbol in bot_instance.trader.positions:
                position = bot_instance.trader.positions[symbol]
                
                # Get position details
                quantity = position.get('quantity', 0)
                if quantity <= 0:
                    return jsonify({'success': False, 'message': 'No position to close'}), 400
                
                # Calculate position value at current price
                current_price = position.get('current_price', position.get('avg_price', 0))
                side = str(position.get('action', 'BUY') or 'BUY').upper()
                avg_price = float(position.get('avg_price', current_price) or current_price)
                margin = float(position.get('total_cost', 0) or 0)
                notional = float(quantity) * float(current_price)
                pnl = (current_price - avg_price) * quantity if side == 'BUY' else (avg_price - current_price) * quantity
                fee_rate = float(getattr(bot_instance.trader, 'fee_rate', 0.0002) or 0.0002)
                fee = notional * fee_rate
                realized_value = margin + pnl - fee

                # Prefer bot close path so it records trades and triggers learning
                if hasattr(bot_instance, '_close_micro_position'):
                    import asyncio
                    pos_for_close = dict(position)
                    pos_for_close['current_value'] = notional
                    pos_for_close['unrealized_pnl'] = pnl
                    pos_for_close['cost_basis'] = margin
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(bot_instance._close_micro_position(symbol, pos_for_close, 'MANUAL_DASHBOARD_CLOSE'))
                    finally:
                        try:
                            loop.close()
                        except Exception:
                            pass

                    try:
                        updated = bot_instance.trader.positions.get(symbol)
                        still_open = bool(updated and float(updated.get('quantity', 0) or 0) > 0)
                    except Exception:
                        still_open = False

                    if still_open:
                        return jsonify({'success': False, 'message': f'Close submitted but {symbol} still open'}), 500

                    print(f"✅ Closed {symbol} position (bot close) - Value: ${realized_value:.4f}")
                    return jsonify({'success': True, 'message': f'Closed {symbol} position', 'value': realized_value})

                # Fallback: simplified close if bot method not available
                if hasattr(bot_instance.trader, 'cash_balance'):
                    bot_instance.trader.cash_balance += realized_value
                    del bot_instance.trader.positions[symbol]
                    print(f"✅ Closed {symbol} position - Value: ${realized_value:.4f}")
                    return jsonify({'success': True, 'message': f'Closed {symbol} position', 'value': realized_value})

                return jsonify({'success': False, 'message': 'Cannot close position'}), 500
                        
                return jsonify({'success': False, 'message': 'Position not found'}), 404
        
        return jsonify({'success': False, 'message': 'Trading system not available'}), 500
        
    except Exception as e:
        print(f"❌ Error closing position: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

def attach_bot(bot):
    """Attach bot instance"""
    global bot_instance
    bot_instance = bot
    print("✅ Bot attached to simple dashboard")

def run_server(host='0.0.0.0', port=5000):
    """Run the server"""
    print(f"""
╔════════════════════════════════════════╗
║   🎯 SIMPLE DASHBOARD SERVER READY     ║
╠════════════════════════════════════════╣
║  Dashboard URL: http://localhost:{port:<6}  ║
║  Controls: Mode + Start/Stop           ║
╚════════════════════════════════════════╝
    """)
    
    if not bot_instance:
        print("⚠️  No bot connected. Run your trading bot to enable controls.")
    
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    run_server()
