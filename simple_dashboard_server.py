#!/usr/bin/env python3
"""
🎯 SIMPLE DASHBOARD SERVER
Minimal backend for the simple control panel
"""

from flask import Flask, jsonify, request, send_file, Response
from flask_socketio import SocketIO
import os
import json
import re
import uuid
import threading
import time
from datetime import datetime
import csv
import io
import urllib.request
import urllib.error

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

bot_thread = None
bot_startup_state = 'unknown'
bot_startup_message = None
bot_last_error = None
bot_last_error_ts = None

assistant_pending_actions = {}

ASSISTANT_MEMORY_PATH = os.path.join('data', 'assistant_memory.jsonl')
ASSISTANT_MEMORY_MAX_BYTES = 900_000
ASSISTANT_MEMORY_KEEP_LINES = 300

ASSISTANT_STATE_PATH = os.path.join('data', 'assistant_state.json')
ASSISTANT_STATE_MAX_BYTES = 200_000

try:
    DEFAULT_STARTING_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '20.0') or 20.0)
except Exception:
    DEFAULT_STARTING_CAPITAL = 20.0

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
        'bot_thread_alive': False,
        'bot_startup_state': bot_startup_state,
        'bot_startup_message': bot_startup_message,
        'bot_last_error': (str(bot_last_error)[:800] if bot_last_error else None),
        'bot_last_error_ts': bot_last_error_ts,
        'uptime': time.time() - start_time if 'start_time' in globals() else 0
    }

    try:
        if bot_thread is not None and hasattr(bot_thread, 'is_alive'):
            health['bot_thread_alive'] = bool(bot_thread.is_alive())
    except Exception:
        pass

    try:
        if (bot_instance is None) and (bot_thread is None):
            health['hint'] = 'Bot not started. Ensure Render Start Command is: python render_launcher.py'
    except Exception:
        pass
    
    if bot_instance:
        health['bot_running'] = getattr(bot_instance, 'bot_running', False)
        health['current_capital'] = getattr(bot_instance, 'current_capital', 0)
    
    return jsonify(health)

@app.route('/ping')
def ping():
    """Simple ping endpoint"""
    return jsonify({'status': 'alive', 'timestamp': datetime.now().isoformat()})


def _assistant_ensure_storage():
    try:
        os.makedirs(os.path.dirname(ASSISTANT_MEMORY_PATH), exist_ok=True)
    except Exception:
        pass


def _assistant_load_state():
    _assistant_ensure_storage()
    try:
        if not os.path.exists(ASSISTANT_STATE_PATH):
            return {'sessions': {}}
        with open(ASSISTANT_STATE_PATH, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {'sessions': {}}
        if 'sessions' not in obj or not isinstance(obj.get('sessions'), dict):
            obj['sessions'] = {}
        return obj
    except Exception:
        return {'sessions': {}}


def _assistant_save_state(state: dict):
    try:
        _assistant_ensure_storage()
        tmp = ASSISTANT_STATE_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(state or {'sessions': {}}, f, ensure_ascii=False)
        os.replace(tmp, ASSISTANT_STATE_PATH)
    except Exception:
        pass


def _assistant_compact_state_if_needed():
    try:
        if not os.path.exists(ASSISTANT_STATE_PATH):
            return
        if os.path.getsize(ASSISTANT_STATE_PATH) <= int(ASSISTANT_STATE_MAX_BYTES):
            return
    except Exception:
        return

    try:
        st = _assistant_load_state()
        sessions = st.get('sessions') or {}
        if not isinstance(sessions, dict):
            return
        items = []
        for sid, s in sessions.items():
            try:
                ts = float((s or {}).get('updated_ts', 0) or 0)
            except Exception:
                ts = 0
            items.append((ts, sid))
        items.sort(reverse=True)
        keep = {}
        for _, sid in items[:10]:
            obj = sessions.get(sid)
            if isinstance(obj, dict):
                try:
                    if 'summary' in obj:
                        obj['summary'] = str(obj.get('summary', '') or '')[-6000:]
                except Exception:
                    pass
            keep[sid] = obj
        st['sessions'] = keep
        _assistant_save_state(st)
    except Exception:
        pass


def _assistant_merge_summary(prev: str, new: str):
    p = str(prev or '').strip()
    n = str(new or '').strip()
    if not n:
        return p
    if not p:
        return n[:6000]
    merged = (p + "\n\n" + n).strip()
    return merged[-6000:]


def _assistant_summarize_session_update(session_id: str, new_text: str):
    sid = str(session_id or '').strip()
    nt = str(new_text or '').strip()
    if not sid or not nt:
        return nt[-6000:]

    try:
        st = _assistant_load_state()
        existing = ''
        sess = (st.get('sessions') or {}).get(sid)
        if isinstance(sess, dict):
            existing = str(sess.get('summary', '') or '').strip()
    except Exception:
        existing = ''

    # Try to use OpenAI to compress into a stable long-term memory summary (if configured)
    try:
        if str(os.getenv('OPENAI_API_KEY', '') or '').strip():
            prompt = [
                {
                    'role': 'system',
                    'content': (
                        'You are a memory compressor for a trading dashboard assistant. '
                        'Update the long-term memory summary with new conversation excerpts. '
                        'Keep it concise and information-dense. Preserve: user preferences, constraints, '
                        'important decisions, and any persistent facts. Avoid quoting long logs.'
                    )
                },
                {
                    'role': 'user',
                    'content': (
                        'Existing summary (may be empty):\n'
                        + (existing or '')
                        + '\n\nNew excerpts to incorporate:\n'
                        + nt[-8000:]
                        + '\n\nReturn an updated summary as bullet points.'
                    )
                }
            ]
            out = _assistant_openai_chat(prompt)
            if out:
                return str(out).strip()[-6000:]
    except Exception:
        pass

    merged = _assistant_merge_summary(existing, nt)
    return str(merged or '')[-6000:]


def _assistant_extract_facts_from_text(text: str):
    t = str(text or '')
    facts = {}
    try:
        if re.search(r'\brender\b', t, flags=re.IGNORECASE):
            facts['hosting'] = 'render'
        if re.search(r'\b512\s*mb\b', t, flags=re.IGNORECASE):
            facts['storage_limit'] = '512MB'
        if re.search(r'\bopenai\b', t, flags=re.IGNORECASE):
            facts['uses_openai'] = True
    except Exception:
        return {}
    return facts


def _assistant_update_session_state(session_id: str, summary_append: str = None, facts_update: dict = None):
    sid = str(session_id or '').strip()
    if not sid:
        return
    st = _assistant_load_state()
    sessions = st.get('sessions')
    if not isinstance(sessions, dict):
        st['sessions'] = {}
        sessions = st['sessions']

    cur = sessions.get(sid) if isinstance(sessions.get(sid), dict) else {}
    if summary_append:
        cur['summary'] = _assistant_merge_summary(cur.get('summary', ''), summary_append)
    if isinstance(facts_update, dict) and facts_update:
        cur_f = cur.get('facts') if isinstance(cur.get('facts'), dict) else {}
        cur_f.update(facts_update)
        cur['facts'] = cur_f
    cur['updated_ts'] = time.time()
    sessions[sid] = cur
    st['sessions'] = sessions
    _assistant_save_state(st)
    _assistant_compact_state_if_needed()


def _assistant_compact_memory_if_needed():
    try:
        _assistant_ensure_storage()
        if not os.path.exists(ASSISTANT_MEMORY_PATH):
            return
        if os.path.getsize(ASSISTANT_MEMORY_PATH) <= int(ASSISTANT_MEMORY_MAX_BYTES):
            return
    except Exception:
        return

    try:
        with open(ASSISTANT_MEMORY_PATH, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except Exception:
        return

    keep = lines[-int(ASSISTANT_MEMORY_KEEP_LINES):] if len(lines) > int(ASSISTANT_MEMORY_KEEP_LINES) else lines

    summaries_by_session = {}
    try:
        dropped = lines[:-int(ASSISTANT_MEMORY_KEEP_LINES)] if len(lines) > int(ASSISTANT_MEMORY_KEEP_LINES) else []
        if dropped:
            for ln in dropped[-2000:]:
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                sid = str(obj.get('session_id', '') or '')
                role = str(obj.get('role', '') or '')
                content = str(obj.get('content', '') or '')
                if not sid or not content:
                    continue
                summaries_by_session.setdefault(sid, []).append(f"{role}: {content}")
    except Exception:
        summaries_by_session = {}

    out_lines = []
    try:
        for sid, items in summaries_by_session.items():
            joined = "\n".join(items)
            summary_text = _assistant_summarize_session_update(sid, joined)
            if summary_text:
                facts = _assistant_extract_facts_from_text(summary_text)
                _assistant_update_session_state(sid, summary_append=summary_text, facts_update=facts)
                out_lines.append(json.dumps({
                    'ts': datetime.now().isoformat(),
                    'session_id': sid,
                    'role': 'system',
                    'type': 'memory_summary',
                    'content': summary_text,
                }, ensure_ascii=False))
    except Exception:
        pass
    out_lines.extend(keep)

    try:
        tmp = ASSISTANT_MEMORY_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        os.replace(tmp, ASSISTANT_MEMORY_PATH)
    except Exception:
        pass


def _assistant_append_memory(session_id: str, role: str, content: str, meta: dict = None):
    _assistant_ensure_storage()
    rec = {
        'ts': datetime.now().isoformat(),
        'session_id': str(session_id or ''),
        'role': str(role or ''),
        'content': str(content or '')[:8000],
    }
    if isinstance(meta, dict) and meta:
        try:
            rec['meta'] = meta
        except Exception:
            pass
    try:
        with open(ASSISTANT_MEMORY_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    _assistant_compact_memory_if_needed()
    try:
        if str(role or '') == 'user':
            facts = _assistant_extract_facts_from_text(content)
            if facts:
                _assistant_update_session_state(session_id, facts_update=facts)
    except Exception:
        pass


def _assistant_read_recent_memory(session_id: str, limit: int = 18):
    try:
        if not os.path.exists(ASSISTANT_MEMORY_PATH):
            return []
        with open(ASSISTANT_MEMORY_PATH, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except Exception:
        return []

    out = []
    sid = str(session_id or '')

    try:
        st = _assistant_load_state()
        sess = (st.get('sessions') or {}).get(sid) if sid else None
        if isinstance(sess, dict):
            summary = str(sess.get('summary', '') or '').strip()
            facts = sess.get('facts') if isinstance(sess.get('facts'), dict) else {}
            header_parts = []
            if summary:
                header_parts.append("Long-term memory summary:\n" + summary)
            if facts:
                header_parts.append("Facts:" + json.dumps(facts, ensure_ascii=False))
            if header_parts:
                out.append({'role': 'system', 'content': "\n\n".join(header_parts)[:6000]})
    except Exception:
        pass

    for ln in reversed(lines):
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if obj.get('type') == 'memory_summary':
            if sid and str(obj.get('session_id', '') or '') not in ['', sid]:
                continue
            out.append({'role': 'system', 'content': str(obj.get('content', '') or '')[:6000]})
            if len(out) >= int(limit):
                break
            continue
        if sid and obj.get('session_id') != sid:
            continue
        role = str(obj.get('role', '') or '')
        content = str(obj.get('content', '') or '')
        if role and content:
            out.append({'role': role, 'content': content})
        if len(out) >= int(limit):
            break
    return list(reversed(out))


def _assistant_get_bot_snapshot():
    global bot_instance
    snap = {
        'connected': bool(bot_instance is not None),
    }
    if not bot_instance:
        return snap

    try:
        snap['bot_running'] = bool(getattr(bot_instance, 'bot_running', False))
        snap['trading_mode'] = str(getattr(bot_instance, 'trading_mode', 'UNKNOWN') or 'UNKNOWN')
        snap['current_capital'] = float(getattr(bot_instance, 'current_capital', 0.0) or 0.0)
        snap['initial_capital'] = float(getattr(bot_instance, 'initial_capital', 0.0) or 0.0)
        snap['win_rate'] = float(getattr(bot_instance, 'win_rate', 0.0) or 0.0)
        snap['total_completed_trades'] = int(getattr(bot_instance, 'total_completed_trades', 0) or 0)
        snap['winning_trades'] = int(getattr(bot_instance, 'winning_trades', 0) or 0)
        snap['current_market_regime'] = str(getattr(bot_instance, 'current_market_regime', 'UNKNOWN') or 'UNKNOWN')
        snap['volatility_regime'] = str(getattr(bot_instance, 'volatility_regime', 'NORMAL') or 'NORMAL')
    except Exception:
        pass

    try:
        tr = getattr(bot_instance, 'trader', None)
        if tr is not None:
            snap['leverage'] = float(getattr(tr, 'leverage', 1.0) or 1.0)
            snap['fee_rate'] = float(getattr(tr, 'fee_rate', 0.0) or 0.0)
            snap['paper_spread_bps'] = getattr(tr, 'paper_spread_bps', None)
            snap['paper_slippage_bps'] = getattr(tr, 'paper_slippage_bps', None)

            positions = {}
            if hasattr(tr, 'get_portfolio_value_sync'):
                try:
                    pf = tr.get_portfolio_value_sync() or {}
                    for sym, p in (pf.get('positions') or {}).items():
                        if isinstance(p, dict) and float(p.get('quantity', 0) or 0) > 0:
                            positions[str(sym)] = dict(p)
                except Exception:
                    positions = {}

            if not positions and hasattr(tr, 'positions') and isinstance(getattr(tr, 'positions', None), dict):
                try:
                    for sym, p in (tr.positions or {}).items():
                        if isinstance(p, dict) and float(p.get('quantity', 0) or 0) > 0:
                            positions[str(sym)] = dict(p)
                except Exception:
                    positions = {}

            snap['positions'] = positions
    except Exception:
        pass

    return snap


def _assistant_symbol_from_text(text: str):
    t = str(text or '')
    m = re.search(r'\b([A-Za-z0-9]{2,12})\s*/\s*([A-Za-z0-9]{2,12})\b', t)
    if m:
        return f"{m.group(1).upper()}/{m.group(2).upper()}"
    m = re.search(r'\b([A-Za-z0-9]{2,12})\b\s*(?:usdt|usd)\b', t, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}/USDT"
    return None


def _assistant_format_positions(snap: dict):
    positions = (snap or {}).get('positions') or {}
    if not positions:
        return "No open positions right now."

    lines = []
    for sym, p in positions.items():
        try:
            qty = float(p.get('quantity', 0) or 0)
            ep = float(p.get('entry_price', p.get('avg_price', 0)) or 0)
            cp = float(p.get('current_price', 0) or 0)
            side = str(p.get('action', p.get('side', 'BUY')) or 'BUY').upper()
            tp = p.get('take_profit', None)
            sl = p.get('stop_loss', None)
            upnl = p.get('unrealized_pnl', None)
            if upnl is None and cp > 0 and ep > 0 and qty > 0:
                upnl = (cp - ep) * qty if side == 'BUY' else (ep - cp) * qty
            upnl = float(upnl or 0.0)
            lines.append(f"{sym} {side} qty={qty:.6g} entry={ep:.6g} current={cp:.6g} uPnL={upnl:.4f} TP={tp} SL={sl}")
        except Exception:
            continue
    return "\n".join(lines) if lines else "No open positions right now."


def _assistant_openai_chat(messages):
    out, _ = _assistant_openai_chat_ex(messages)
    return out


def _assistant_openai_chat_ex(messages):
    key = (
        str(os.getenv('OPENAI_API_KEY', '') or '').strip()
        or str(os.getenv('OPENAI_SECRET_KEY', '') or '').strip()
        or str(os.getenv('OPENAI_KEY', '') or '').strip()
    )
    if not key:
        return None, 'OpenAI key not found in env. Set OPENAI_API_KEY on Render.'

    last_err = None
    for model in ['gpt-4o-mini', 'gpt-3.5-turbo']:
        payload = {
            'model': model,
            'messages': messages,
            'temperature': 0.2,
            'max_tokens': 450,
        }
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                'https://api.openai.com/v1/chat/completions',
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f"Bearer {key}",
                },
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                raw = resp.read().decode('utf-8', errors='ignore')
            obj = json.loads(raw)
            choices = obj.get('choices') or []
            first = (choices[0] if isinstance(choices, list) and len(choices) > 0 else {}) or {}
            msg = first.get('message') or {}
            content = str((msg.get('content') or '')).strip()
            if content:
                return content, None
            last_err = f'OpenAI returned empty response for model {model}.'
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode('utf-8', errors='ignore')
            except Exception:
                body = ''
            msg = None
            try:
                j = json.loads(body)
                msg = ((j.get('error') or {}) if isinstance(j, dict) else {}).get('message')
            except Exception:
                msg = None
            last_err = f'OpenAI HTTP {getattr(e, "code", "")} for model {model}: {str(msg or body or e)[:300]}'
        except Exception as e:
            last_err = f'OpenAI error for model {model}: {str(e)[:300]}'

    return None, (last_err or 'OpenAI request failed.')


def _assistant_execute_close(symbol: str):
    global bot_instance

    sym = str(symbol or '').strip()
    if not sym:
        return False, 'Symbol required to close a position.'
    if not bot_instance:
        return False, 'No bot connected.'

    try:
        if not hasattr(bot_instance, 'trader') or bot_instance.trader is None:
            return False, 'Trading system not available.'
        tr = bot_instance.trader
        if not hasattr(tr, 'positions') or not isinstance(getattr(tr, 'positions', None), dict):
            return False, 'No positions store available.'
        if sym not in tr.positions:
            return False, f'Position not found: {sym}'
        position = tr.positions.get(sym) or {}
        qty = float(position.get('quantity', 0) or 0)
        if qty <= 0:
            return False, f'No open quantity for {sym}.'

        current_price = float(position.get('current_price', position.get('avg_price', 0)) or 0)
        side = str(position.get('action', 'BUY') or 'BUY').upper()
        avg_price = float(position.get('avg_price', current_price) or current_price)
        margin = float(position.get('total_cost', 0) or 0)
        notional = float(qty) * float(current_price or 0)
        pnl = (current_price - avg_price) * qty if side == 'BUY' else (avg_price - current_price) * qty
        fee_rate = float(getattr(tr, 'fee_rate', 0.0002) or 0.0002)
        fee = notional * fee_rate
        realized_value = margin + pnl - fee

        if hasattr(bot_instance, '_close_micro_position'):
            import asyncio
            pos_for_close = dict(position)
            pos_for_close['current_value'] = notional
            pos_for_close['unrealized_pnl'] = pnl
            pos_for_close['cost_basis'] = margin
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(bot_instance._close_micro_position(sym, pos_for_close, 'MANUAL_DASHBOARD_ASSISTANT_CLOSE'))
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

            try:
                updated = tr.positions.get(sym)
                still_open = bool(updated and float(updated.get('quantity', 0) or 0) > 0)
            except Exception:
                still_open = False

            if still_open:
                return False, f'Close submitted but {sym} still open.'

            return True, f'Closed {sym}. Realized value≈${realized_value:.4f} (fees included).'

        try:
            if hasattr(tr, 'cash_balance'):
                tr.cash_balance = float(getattr(tr, 'cash_balance', 0.0) or 0.0) + float(realized_value or 0.0)
            tr.positions.pop(sym, None)
            try:
                if hasattr(tr, '_save_state'):
                    tr._save_state()
            except Exception:
                pass
            return True, f'Closed {sym}. Realized value≈${realized_value:.4f} (fees included).'
        except Exception as e:
            return False, f'Failed to close {sym}: {str(e)[:200]}'
    except Exception as e:
        return False, f'Close error: {str(e)[:200]}'


def _assistant_execute_set_running(should_run: bool):
    global bot_instance
    if not bot_instance:
        return False, 'No bot connected.'
    try:
        bot_instance.bot_running = bool(should_run)
        if bool(should_run):
            return True, 'Trading started.'
        return True, 'Trading stopped.'
    except Exception as e:
        return False, f'Failed to update trading state: {str(e)[:200]}'


def _assistant_execute_update_position(symbol: str, take_profit, stop_loss):
    global bot_instance
    sym = str(symbol or '').strip()
    if not sym:
        return False, 'Symbol required.'
    if not bot_instance:
        return False, 'No bot connected.'
    try:
        if not hasattr(bot_instance, 'trader') or bot_instance.trader is None:
            return False, 'Trading system not available.'
        tr = bot_instance.trader
        if not hasattr(tr, 'positions') or not isinstance(getattr(tr, 'positions', None), dict):
            return False, 'No positions store available.'
        if sym not in tr.positions:
            return False, f'Position not found: {sym}'
        position = tr.positions.get(sym) or {}

        tp_val = None
        sl_val = None
        if take_profit is not None and str(take_profit) != '':
            tp_val = float(take_profit)
            position['take_profit'] = tp_val
        if stop_loss is not None and str(stop_loss) != '':
            sl_val = float(stop_loss)
            position['stop_loss'] = sl_val

        try:
            if hasattr(tr, '_save_state'):
                tr._save_state()
        except Exception:
            pass

        parts = [f'Updated {sym}.']
        if tp_val is not None:
            parts.append(f'TP=${tp_val:.6g}')
        if sl_val is not None:
            parts.append(f'SL=${sl_val:.6g}')
        return True, ' '.join(parts)
    except Exception as e:
        return False, f'Failed to update position: {str(e)[:200]}'


def _assistant_execute_set_mode(mode: str):
    global bot_instance, current_mode
    m = str(mode or '').upper().strip()
    if m == 'NORMAL':
        m = 'PRECISION'
    if m not in ['AGGRESSIVE', 'PRECISION']:
        return False, 'Invalid mode. Use AGGRESSIVE or PRECISION.'
    current_mode = m
    if not bot_instance:
        return True, f'Mode set to {m} (bot not connected yet).'
    try:
        bot_instance.trading_mode = m

        if hasattr(bot_instance, 'mode_config'):
            cfg = (bot_instance.mode_config or {}).get(m, {})
            bot_instance.target_accuracy = cfg.get('target_accuracy', 0.6)
            bot_instance.min_confidence_for_trade = cfg.get('min_confidence', 0.3)
            bot_instance.ensemble_threshold = cfg.get('ensemble_threshold', 2)
            bot_instance.confidence_threshold = cfg.get('min_confidence', 0.3)
            bot_instance.base_confidence_threshold = cfg.get('min_confidence', 0.3)

        if m == 'AGGRESSIVE':
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
        return True, f'Mode switched to {m}.'
    except Exception as e:
        return False, f'Failed to switch mode: {str(e)[:200]}'


def _assistant_analyze_symbol(symbol: str, snap: dict):
    sym = str(symbol or '').strip()
    if not sym:
        return 'Symbol required.'
    if not (snap or {}).get('connected'):
        return 'Bot is not connected right now.'
    positions = (snap.get('positions') or {})
    if sym not in positions:
        return f'No open position found for {sym}.'
    p = positions.get(sym) or {}

    try:
        qty = float(p.get('quantity', 0) or 0)
    except Exception:
        qty = 0.0
    try:
        ep = float(p.get('entry_price', p.get('avg_price', 0)) or 0)
    except Exception:
        ep = 0.0
    try:
        cp = float(p.get('current_price', 0) or 0)
    except Exception:
        cp = 0.0
    side = str(p.get('action', p.get('side', 'BUY')) or 'BUY').upper()
    tp = p.get('take_profit', None)
    sl = p.get('stop_loss', None)

    try:
        upnl = p.get('unrealized_pnl', None)
        if upnl is None and cp > 0 and ep > 0 and qty > 0:
            upnl = (cp - ep) * qty if side == 'BUY' else (ep - cp) * qty
        upnl = float(upnl or 0.0)
    except Exception:
        upnl = 0.0

    try:
        notional_entry = float(ep) * float(qty)
        upnl_pct = (upnl / notional_entry * 100.0) if notional_entry > 0 else 0.0
    except Exception:
        upnl_pct = 0.0

    lines = []
    lines.append(f'{sym} {side} qty={qty:.6g}')
    lines.append(f'Entry=${ep:.6g} Current=${cp:.6g}')
    lines.append(f'uPnL=${upnl:.4f} ({upnl_pct:.2f}%)')

    try:
        if tp is not None and str(tp) != '' and cp > 0:
            tpv = float(tp)
            dist = (tpv - cp) if side == 'BUY' else (cp - tpv)
            dist_pct = (dist / cp * 100.0) if cp > 0 else 0.0
            lines.append(f'TP=${tpv:.6g} distance={dist:.6g} ({dist_pct:.2f}%)')
        else:
            lines.append('TP=None')
    except Exception:
        lines.append('TP=None')

    try:
        if sl is not None and str(sl) != '' and cp > 0:
            slv = float(sl)
            dist = (cp - slv) if side == 'BUY' else (slv - cp)
            dist_pct = (dist / cp * 100.0) if cp > 0 else 0.0
            lines.append(f'SL=${slv:.6g} buffer={dist:.6g} ({dist_pct:.2f}%)')
        else:
            lines.append('SL=None')
    except Exception:
        lines.append('SL=None')

    regime = str((snap or {}).get('current_market_regime', 'UNKNOWN') or 'UNKNOWN')
    vreg = str((snap or {}).get('volatility_regime', 'NORMAL') or 'NORMAL')
    lines.append(f'Market regime={regime} Volatility={vreg}')

    risk_notes = []
    try:
        if str(vreg).upper() == 'HIGH':
            risk_notes.append('High volatility: consider wider SL or smaller size.')
    except Exception:
        pass
    try:
        if any(k in str(regime).upper() for k in ['SIDEWAYS', 'RANGING', 'CONSOLIDATION']):
            risk_notes.append('Sideways regime: breakouts can fake-out; be conservative.')
    except Exception:
        pass
    try:
        sb = (snap or {}).get('paper_spread_bps', None)
        if sb is not None and float(sb or 0.0) > 50:
            risk_notes.append('Wide spread: execution quality risk is high.')
    except Exception:
        pass

    if risk_notes:
        lines.append('Risk notes:')
        lines.extend([f'- {r}' for r in risk_notes])

    return "\n".join(lines)


def _assistant_handle_user_message(session_id: str, message: str):
    snap = _assistant_get_bot_snapshot()
    text = str(message or '').strip()
    lower = text.lower()

    if (
        ('why' in lower or 'reason' in lower)
        and any(k in lower for k in ['no trades', 'not placing', 'not taking', 'not trading', 'not opening', 'not placing any trades'])
    ):
        if not snap.get('connected'):
            return {'reply': 'Bot is not connected right now, so it cannot place trades.', 'pending_action': None}

        running = bool(snap.get('bot_running', False))
        mode = str(snap.get('trading_mode', 'UNKNOWN') or 'UNKNOWN')
        positions = (snap.get('positions') or {})
        active = len(list(positions.keys()))

        if not running:
            action_id = str(uuid.uuid4())
            assistant_pending_actions[str(session_id)] = {
                'id': action_id,
                'type': 'set_bot_running',
                'should_run': True,
                'ts': time.time(),
            }
            return {
                'reply': f'Trading is currently STOPPED (mode={mode}). Do you want me to start it?',
                'pending_action': {
                    'id': action_id,
                    'type': 'set_bot_running',
                    'should_run': True,
                }
            }

        try:
            max_pos = None
            if bot_instance is not None:
                max_pos = getattr(bot_instance, 'max_concurrent_positions', None)
                if max_pos is None:
                    max_pos = getattr(bot_instance, 'max_positions', None)
            max_pos_i = int(max_pos) if max_pos is not None else None
        except Exception:
            max_pos_i = None

        if active > 0 and (max_pos_i is not None) and active >= max_pos_i:
            return {
                'reply': (
                    f'Bot is RUNNING (mode={mode}) but it is already at max positions ({active}/{max_pos_i}). '
                    'It will wait for exits (TP/SL/close) before opening new trades.'
                ),
                'pending_action': None,
            }

        regime = str(snap.get('current_market_regime', 'UNKNOWN') or 'UNKNOWN')
        vreg = str(snap.get('volatility_regime', 'NORMAL') or 'NORMAL')
        return {
            'reply': (
                f'Bot is RUNNING (mode={mode}). Right now it has {active} open position(s). '\
                f'It is scanning markets and only places trades when signals pass its filters. '\
                f'Current regime={regime}, volatility={vreg}. '\
                'If you want more trades, you can ask: "switch mode to AGGRESSIVE" (I will ask for confirmation).'
            ),
            'pending_action': None,
        }

    if lower in ['cancel', 'no', 'stop']:
        try:
            if str(session_id) in assistant_pending_actions:
                assistant_pending_actions.pop(str(session_id), None)
                return {'reply': 'Cancelled.', 'pending_action': None}
        except Exception:
            pass

    if lower in ['confirm', 'yes', 'do it']:
        try:
            pending = assistant_pending_actions.get(str(session_id))
            if isinstance(pending, dict) and str(pending.get('type', '') or '') == 'close_position':
                symbol = str(pending.get('symbol', '') or '')
                ok, msg = _assistant_execute_close(symbol)
                assistant_pending_actions.pop(str(session_id), None)
                return {'reply': msg, 'pending_action': None}
            if isinstance(pending, dict) and str(pending.get('type', '') or '') == 'set_bot_running':
                should_run = bool(pending.get('should_run', False))
                ok, msg = _assistant_execute_set_running(should_run)
                assistant_pending_actions.pop(str(session_id), None)
                return {'reply': msg, 'pending_action': None}
            if isinstance(pending, dict) and str(pending.get('type', '') or '') == 'update_position_targets':
                symbol = str(pending.get('symbol', '') or '')
                tp = pending.get('take_profit', None)
                sl = pending.get('stop_loss', None)
                ok, msg = _assistant_execute_update_position(symbol, tp, sl)
                assistant_pending_actions.pop(str(session_id), None)
                return {'reply': msg, 'pending_action': None}
            if isinstance(pending, dict) and str(pending.get('type', '') or '') == 'set_trading_mode':
                mode = str(pending.get('mode', '') or '')
                ok, msg = _assistant_execute_set_mode(mode)
                assistant_pending_actions.pop(str(session_id), None)
                return {'reply': msg, 'pending_action': None}
        except Exception:
            pass

    if any(k in lower for k in ['current entries', 'current entry', 'open positions', 'open trades', 'positions', 'entries']):
        return {
            'reply': _assistant_format_positions(snap),
            'pending_action': None,
        }

    if any(k in lower for k in ['switch mode', 'set mode', 'change mode']):
        m = None
        try:
            if re.search(r'\baggressive\b', lower):
                m = 'AGGRESSIVE'
            elif re.search(r'\bprecision\b', lower) or re.search(r'\bnormal\b', lower):
                m = 'PRECISION'
        except Exception:
            m = None
        if not m:
            return {'reply': 'Which mode? Say: switch mode to AGGRESSIVE or PRECISION.', 'pending_action': None}
        action_id = str(uuid.uuid4())
        assistant_pending_actions[str(session_id)] = {
            'id': action_id,
            'type': 'set_trading_mode',
            'mode': m,
            'ts': time.time(),
        }
        return {
            'reply': f'I can switch mode to {m}. Reply with CONFIRM to execute, or CANCEL to ignore.',
            'pending_action': {
                'id': action_id,
                'type': 'set_trading_mode',
                'mode': m,
            }
        }

    if any(k in lower for k in ['set tp', 'set sl', 'update tp', 'update sl', 'tp ', 'sl '] ) and any(k in lower for k in ['tp', 'sl']):
        symbol = _assistant_symbol_from_text(text)
        if not symbol:
            return {'reply': 'Which symbol? Example: set tp 67000 sl 64500 for BTC/USDT', 'pending_action': None}
        tp = None
        sl = None
        try:
            m = re.search(r'\btp\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)', lower)
            if m:
                tp = float(m.group(1))
        except Exception:
            tp = None
        try:
            m = re.search(r'\bsl\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)', lower)
            if m:
                sl = float(m.group(1))
        except Exception:
            sl = None
        if tp is None and sl is None:
            return {'reply': 'I could not parse TP/SL. Example: set tp 67000 sl 64500 for BTC/USDT', 'pending_action': None}

        action_id = str(uuid.uuid4())
        assistant_pending_actions[str(session_id)] = {
            'id': action_id,
            'type': 'update_position_targets',
            'symbol': symbol,
            'take_profit': tp,
            'stop_loss': sl,
            'ts': time.time(),
        }
        parts = [f'I can update {symbol}.']
        if tp is not None:
            parts.append(f'TP=${tp:.6g}.')
        if sl is not None:
            parts.append(f'SL=${sl:.6g}.')
        parts.append('Reply with CONFIRM to execute, or CANCEL to ignore.')
        return {
            'reply': ' '.join(parts),
            'pending_action': {
                'id': action_id,
                'type': 'update_position_targets',
                'symbol': symbol,
                'take_profit': tp,
                'stop_loss': sl,
            }
        }

    if any(k in lower for k in ['analyze', 'analysis', 'distance to tp', 'distance to sl', 'risk notes']):
        symbol = _assistant_symbol_from_text(text)
        if not symbol:
            return {'reply': 'Which symbol? Example: analyze BTC/USDT', 'pending_action': None}
        return {'reply': _assistant_analyze_symbol(symbol, snap), 'pending_action': None}

    if any(k in lower for k in ['leverage', 'fee', 'fees', 'slippage', 'spread']):
        parts = []
        if not snap.get('connected'):
            parts.append('Bot is not connected right now.')
        else:
            parts.append(f"Mode: {snap.get('trading_mode')}")
            parts.append(f"Leverage: {snap.get('leverage', 1.0)}")
            parts.append(f"Fee rate: {snap.get('fee_rate', None)}")
            parts.append(f"Paper spread bps: {snap.get('paper_spread_bps', None)}")
            parts.append(f"Paper slippage bps: {snap.get('paper_slippage_bps', None)}")
        return {'reply': "\n".join(parts), 'pending_action': None}

    if any(k in lower for k in ['win rate', 'total trades', 'profitable trades', 'winning trades', 'pnl', 'profit']):
        if not snap.get('connected'):
            return {'reply': 'Bot is not connected right now.', 'pending_action': None}
        pnl = float(snap.get('current_capital', 0.0) or 0.0) - float(snap.get('initial_capital', 0.0) or 0.0)
        total = int(snap.get('total_completed_trades', 0) or 0)
        wins = int(snap.get('winning_trades', 0) or 0)
        wr = float(snap.get('win_rate', 0.0) or 0.0)
        return {
            'reply': "\n".join([
                f"PnL: {pnl:.4f}",
                f"Total completed trades: {total}",
                f"Winning trades: {wins}",
                f"Win rate: {wr:.2%}",
                f"Market regime: {snap.get('current_market_regime')}",
                f"Volatility regime: {snap.get('volatility_regime')}",
            ]),
            'pending_action': None,
        }

    if any(k in lower for k in ['close', 'exit', 'sell now', 'close trade', 'close position']):
        symbol = _assistant_symbol_from_text(text)
        if not symbol:
            return {
                'reply': 'Tell me which symbol to close (example: "close BTC/USDT").',
                'pending_action': None,
            }
        action_id = str(uuid.uuid4())
        assistant_pending_actions[str(session_id)] = {
            'id': action_id,
            'type': 'close_position',
            'symbol': symbol,
            'ts': time.time(),
        }
        return {
            'reply': f"I can close {symbol}. Reply with CONFIRM to execute, or CANCEL to ignore.",
            'pending_action': {
                'id': action_id,
                'type': 'close_position',
                'symbol': symbol,
            }
        }

    if any(k in lower for k in ['start trading', 'resume trading', 'start bot', 'resume bot', 'start the bot', 'resume the bot']):
        action_id = str(uuid.uuid4())
        assistant_pending_actions[str(session_id)] = {
            'id': action_id,
            'type': 'set_bot_running',
            'should_run': True,
            'ts': time.time(),
        }
        return {
            'reply': 'I can START trading. Reply with CONFIRM to execute, or CANCEL to ignore.',
            'pending_action': {
                'id': action_id,
                'type': 'set_bot_running',
                'should_run': True,
            }
        }

    if any(k in lower for k in ['stop trading', 'pause trading', 'stop bot', 'pause bot', 'stop the bot', 'pause the bot']):
        action_id = str(uuid.uuid4())
        assistant_pending_actions[str(session_id)] = {
            'id': action_id,
            'type': 'set_bot_running',
            'should_run': False,
            'ts': time.time(),
        }
        return {
            'reply': 'I can STOP trading. Reply with CONFIRM to execute, or CANCEL to ignore.',
            'pending_action': {
                'id': action_id,
                'type': 'set_bot_running',
                'should_run': False,
            }
        }

    if any(k in lower for k in ['will it be profitable', 'will it be profit', 'is it profitable', 'should i hold', 'should we hold']):
        if not snap.get('connected'):
            return {'reply': 'Bot is not connected right now.', 'pending_action': None}
        positions = (snap.get('positions') or {})
        if not positions:
            return {'reply': 'There are no open positions right now.', 'pending_action': None}
        lines = []
        for sym, p in positions.items():
            try:
                qty = float(p.get('quantity', 0) or 0)
                ep = float(p.get('entry_price', p.get('avg_price', 0)) or 0)
                cp = float(p.get('current_price', 0) or 0)
                side = str(p.get('action', p.get('side', 'BUY')) or 'BUY').upper()
                tp = p.get('take_profit', None)
                sl = p.get('stop_loss', None)
                upnl = p.get('unrealized_pnl', None)
                if upnl is None and cp > 0 and ep > 0 and qty > 0:
                    upnl = (cp - ep) * qty if side == 'BUY' else (ep - cp) * qty
                upnl = float(upnl or 0.0)
                hint = 'profitable now' if upnl > 0 else 'not profitable yet'
                lines.append(f"{sym}: {hint}, uPnL={upnl:.4f}, entry={ep:.6g}, current={cp:.6g}, TP={tp}, SL={sl}")
            except Exception:
                continue
        lines.append('I cannot guarantee future profit, but I can tell you current uPnL and distance to TP/SL. If you want, ask: "analyze BTC/USDT".')
        return {'reply': "\n".join(lines), 'pending_action': None}

    recent = _assistant_read_recent_memory(session_id, limit=18)
    sys_prompt = "You are the Poise Trader dashboard assistant. Answer using the provided bot snapshot when relevant. If you do not know, say so. Do not claim certainty about future price. Keep answers concise."
    messages = [{'role': 'system', 'content': sys_prompt}]
    try:
        messages.append({'role': 'system', 'content': f"BOT_SNAPSHOT: {json.dumps(snap, ensure_ascii=False)[:6000]}"})
    except Exception:
        pass
    for m in recent:
        if isinstance(m, dict) and m.get('role') in ['user', 'assistant', 'system']:
            messages.append({'role': m['role'], 'content': str(m.get('content', '') or '')[:2000]})
    messages.append({'role': 'user', 'content': text})

    llm, llm_err = _assistant_openai_chat_ex(messages)
    if llm:
        return {'reply': llm, 'pending_action': None}

    key_present = bool(
        str(os.getenv('OPENAI_API_KEY', '') or '').strip()
        or str(os.getenv('OPENAI_SECRET_KEY', '') or '').strip()
        or str(os.getenv('OPENAI_KEY', '') or '').strip()
    )
    if key_present:
        return {
            'reply': (
                'OpenAI is configured but the request failed. Most common causes: '
                'Render service not restarted after setting env vars, invalid key, or model access issue. '
                f'Error: {str(llm_err or "unknown")[:350]}'
            ),
            'pending_action': None,
        }

    return {
        'reply': (
            'OpenAI key is not detected by the server process. Set OPENAI_API_KEY in Render env vars and restart the service. '
            'Meanwhile I can still answer bot-state questions (positions, PnL, leverage, TP/SL) and perform safe actions.'
        ),
        'pending_action': None,
    }

@app.route('/keep-alive')
def keep_alive_endpoint():
    """Keep-alive endpoint"""
    return jsonify({
        'status': 'active',
        'message': 'Service is running 24/7',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/assistant/chat', methods=['POST'])
def assistant_chat():
    global assistant_pending_actions

    data = request.get_json() or {}
    session_id = str(data.get('session_id', '') or '').strip()
    if not session_id:
        session_id = 'default'

    confirm_id = str(data.get('confirm_action_id', '') or '').strip()
    message = str(data.get('message', '') or '').strip()

    if confirm_id:
        pending = assistant_pending_actions.get(session_id)
        if isinstance(pending, dict) and str(pending.get('id', '')) == confirm_id:
            if str(pending.get('type', '') or '') == 'close_position':
                symbol = str(pending.get('symbol', '') or '')
                ok, msg = _assistant_execute_close(symbol)
                assistant_pending_actions.pop(session_id, None)
                _assistant_append_memory(session_id, 'user', f"CONFIRM action={confirm_id}")
                _assistant_append_memory(session_id, 'assistant', msg)
                return jsonify({'reply': msg, 'executed': ok, 'pending_action': None})

            if str(pending.get('type', '') or '') == 'set_bot_running':
                should_run = bool(pending.get('should_run', False))
                ok, msg = _assistant_execute_set_running(should_run)
                assistant_pending_actions.pop(session_id, None)
                _assistant_append_memory(session_id, 'user', f"CONFIRM action={confirm_id}")
                _assistant_append_memory(session_id, 'assistant', msg)
                return jsonify({'reply': msg, 'executed': ok, 'pending_action': None})

            if str(pending.get('type', '') or '') == 'update_position_targets':
                symbol = str(pending.get('symbol', '') or '')
                tp = pending.get('take_profit', None)
                sl = pending.get('stop_loss', None)
                ok, msg = _assistant_execute_update_position(symbol, tp, sl)
                assistant_pending_actions.pop(session_id, None)
                _assistant_append_memory(session_id, 'user', f"CONFIRM action={confirm_id}")
                _assistant_append_memory(session_id, 'assistant', msg)
                return jsonify({'reply': msg, 'executed': ok, 'pending_action': None})

            if str(pending.get('type', '') or '') == 'set_trading_mode':
                mode = str(pending.get('mode', '') or '')
                ok, msg = _assistant_execute_set_mode(mode)
                assistant_pending_actions.pop(session_id, None)
                _assistant_append_memory(session_id, 'user', f"CONFIRM action={confirm_id}")
                _assistant_append_memory(session_id, 'assistant', msg)
                return jsonify({'reply': msg, 'executed': ok, 'pending_action': None})

        return jsonify({'reply': 'No pending action to confirm.', 'executed': False, 'pending_action': None})

    if message:
        _assistant_append_memory(session_id, 'user', message)
        out = _assistant_handle_user_message(session_id, message)
        reply = str((out or {}).get('reply', '') or '')
        pending_action = (out or {}).get('pending_action', None)
        _assistant_append_memory(session_id, 'assistant', reply, meta={'pending_action': pending_action} if pending_action else None)
        return jsonify({'reply': reply, 'pending_action': pending_action})

    return jsonify({'reply': 'Send a message.', 'pending_action': None})

@app.route('/api/status')
def get_status():
    """Get current bot status"""
    global bot_instance, current_mode
    
    status = {
        'connected': bot_instance is not None,
        'running': False,
        'mode': current_mode,
        'bot_thread_alive': False,
        'bot_startup_state': bot_startup_state,
        'bot_startup_message': bot_startup_message,
        'bot_last_error': (str(bot_last_error)[:800] if bot_last_error else None),
        'bot_last_error_ts': bot_last_error_ts,
    }

    try:
        if bot_thread is not None and hasattr(bot_thread, 'is_alive'):
            status['bot_thread_alive'] = bool(bot_thread.is_alive())
    except Exception:
        pass

    try:
        if (bot_instance is None) and (bot_thread is None):
            status['hint'] = 'Bot not started. Ensure Render Start Command is: python render_launcher.py'
    except Exception:
        pass
    
    if bot_instance:
        status['running'] = getattr(bot_instance, 'bot_running', False)
        status['mode'] = getattr(bot_instance, 'trading_mode', current_mode)
        status['capital'] = getattr(bot_instance, 'current_capital', 0)
        status['win_rate'] = getattr(bot_instance, 'win_rate', 0) * 100

        try:
            status['market_regime'] = getattr(bot_instance, 'current_market_regime', None)
            status['volatility_regime'] = getattr(bot_instance, 'volatility_regime', None)
        except Exception:
            pass

        try:
            status['safety'] = {
                'pause_until_ts': getattr(bot_instance, 'safety_pause_until_ts', 0.0),
                'pause_reason': getattr(bot_instance, 'safety_pause_reason', None),
                'last_feed_health': getattr(bot_instance, 'last_feed_health', None),
            }
        except Exception:
            pass

        try:
            cds = getattr(bot_instance, 'strategy_cooldowns', None)
            if isinstance(cds, dict):
                status['strategy_cooldowns'] = cds
        except Exception:
            pass

        try:
            if hasattr(bot_instance, 'trader') and bot_instance.trader is not None:
                status['paper_shorting'] = bool(getattr(bot_instance.trader, 'enable_shorting', False))
                status['paper_leverage'] = float(getattr(bot_instance.trader, 'leverage', 1.0) or 1.0)
                status['paper_fee_rate'] = float(getattr(bot_instance.trader, 'fee_rate', 0.0002) or 0.0002)

                try:
                    status['paper_execution'] = {
                        'model': getattr(bot_instance.trader, 'paper_execution_model', None),
                        'spread_bps': getattr(bot_instance.trader, 'paper_spread_bps', None),
                        'slippage_bps': getattr(bot_instance.trader, 'paper_slippage_bps', None),
                        'latency_ms_min': getattr(bot_instance.trader, 'paper_latency_ms_min', None),
                        'latency_ms_max': getattr(bot_instance.trader, 'paper_latency_ms_max', None),
                        'partial_fill_prob': getattr(bot_instance.trader, 'paper_partial_fill_prob', None),
                    }
                except Exception:
                    pass

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

        try:
            metrics['market_regime'] = getattr(bot_instance, 'current_market_regime', None)
            metrics['volatility_regime'] = getattr(bot_instance, 'volatility_regime', None)
        except Exception:
            pass
        
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

@app.route('/api/export_trades')
def export_trades():
    global bot_instance

    fmt = str(request.args.get('format', 'json') or 'json').strip().lower()

    trades = []
    try:
        bot_trades = []
        trader_trades = []
        if bot_instance and hasattr(bot_instance, 'trade_history') and isinstance(getattr(bot_instance, 'trade_history', None), list):
            bot_trades = bot_instance.trade_history
        if bot_instance and hasattr(bot_instance, 'trader') and hasattr(bot_instance.trader, 'trade_history') and isinstance(getattr(bot_instance.trader, 'trade_history', None), list):
            trader_trades = bot_instance.trader.trade_history
        trades = trader_trades if len(trader_trades) >= len(bot_trades) else bot_trades
    except Exception:
        trades = []

    try:
        limit = int(str(request.args.get('limit', '0') or '0').strip() or 0)
    except Exception:
        limit = 0
    limit = max(0, min(limit, 5000))

    try:
        order = str(request.args.get('order', 'desc') or 'desc').strip().lower()
    except Exception:
        order = 'desc'

    out = list(trades or [])
    if order == 'desc':
        out = list(reversed(out))
    if limit and limit > 0:
        out = out[:limit]

    if fmt in ['csv']:
        rows = [t for t in out if isinstance(t, dict)]
        fieldnames = []
        try:
            keys = set()
            for r in rows:
                keys.update((r or {}).keys())
            fieldnames = sorted(list(keys))
        except Exception:
            fieldnames = []

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
        try:
            writer.writeheader()
        except Exception:
            pass
        for r in rows:
            try:
                writer.writerow(r)
            except Exception:
                continue

        data = buf.getvalue()
        buf.close()
        return Response(
            data,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=trades.csv'}
        )

    return jsonify({'trades': out, 'total': len(trades or []), 'returned': len(out)})

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
                    state_file = os.path.join('data', 'trading_state.json')
                    try:
                        sf = getattr(bot_instance.trader, 'state_file', None)
                        if sf:
                            state_file = str(sf)
                    except Exception:
                        pass
                    try:
                        os.makedirs(os.path.dirname(state_file), exist_ok=True)
                    except Exception:
                        pass
                    with open(state_file, 'w') as f:
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
    """Reset trading to fresh $20.00 start"""
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
        state_file = 'trading_state.json'
        try:
            if bot_instance and hasattr(bot_instance, 'trader'):
                sf = getattr(bot_instance.trader, 'state_file', None)
                if sf:
                    state_file = str(sf)
        except Exception:
            pass
        try:
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
        except Exception:
            pass
        with open(state_file, 'w') as f:
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


def set_bot_startup(state: str = None, message: str = None):
    global bot_startup_state, bot_startup_message
    try:
        if state is not None:
            bot_startup_state = str(state)
        if message is not None:
            bot_startup_message = str(message)[:500]
    except Exception:
        pass


def set_bot_last_error(err: str = None):
    global bot_last_error, bot_last_error_ts
    try:
        bot_last_error = (str(err) if err is not None else None)
        bot_last_error_ts = datetime.now().isoformat()
    except Exception:
        pass

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
