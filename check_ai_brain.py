#!/usr/bin/env python3
"""
Check what the AI has learned so far
"""

import json
import os
from datetime import datetime

def _resolve_brain_path():
    base = str(os.getenv('AI_STATE_DIR', '') or '').strip()
    if base:
        p = os.path.join(base, 'ai_brain.json')
        if os.path.exists(p):
            return p
    try:
        p = os.path.join('data', 'ai_brain.json')
        if os.path.exists(p):
            return p
    except Exception:
        pass
    if os.path.exists('ai_brain.json'):
        return 'ai_brain.json'
    return None


def check_ai_brain():
    """Display current AI brain learning status"""
    brain_file = _resolve_brain_path()
    if not brain_file:
        print("❌ AI brain file not found!")
        return
    
    try:
        with open(brain_file, 'r') as f:
            brain = json.load(f)
        
        print("\n" + "="*80)
        print("🧠 AI BRAIN STATUS REPORT")
        print("="*80)
        print(f"\nSource: {brain_file}")
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   • Total Trades Learned: {brain.get('total_trades', 0)}")
        print(f"   • Total P&L: ${brain.get('total_profit_loss', 0):.2f}")
        print(f"   • Win Rate: {brain.get('win_rate', 0.5)*100:.1f}%")
        print(f"   • Learning Sessions: {brain.get('learning_sessions', 0)}")
        print(f"   • Last Updated: {brain.get('last_updated', 'Never')}")
        
        print(f"\n📈 STRATEGY PERFORMANCE:")
        strategies = brain.get('strategy_performance', {})
        for name, data in strategies.items():
            wins = data.get('wins', 0)
            losses = data.get('losses', 0)
            total = wins + losses
            win_rate = (wins / total * 100) if total > 0 else 0
            returns = data.get('total_return', 0)
            print(f"   • {name.title():15} - W:{wins:3} L:{losses:3} WR:{win_rate:5.1f}% Returns:${returns:+.2f}")
        
        print(f"\n🎯 MARKET PATTERNS:")
        patterns = brain.get('market_patterns', {})
        for pattern, data in patterns.items():
            trades = data.get('trades', 0)
            success = data.get('success_rate', 0.5) * 100
            print(f"   • {pattern.replace('_', ' ').title():20} - Trades:{trades:4} Success:{success:.1f}%")
        
        print(f"\n💡 CONFIDENCE CALIBRATION:")
        confidence = brain.get('confidence_adjustments', {})
        print(f"   • High Confidence:   {confidence.get('high_confidence_accuracy', 0.7)*100:.1f}% accuracy")
        print(f"   • Medium Confidence: {confidence.get('medium_confidence_accuracy', 0.6)*100:.1f}% accuracy")
        print(f"   • Low Confidence:    {confidence.get('low_confidence_accuracy', 0.5)*100:.1f}% accuracy")
        
        print(f"\n📊 SYMBOL KNOWLEDGE:")
        symbols = brain.get('symbol_knowledge', {})
        if symbols:
            for symbol, data in list(symbols.items())[:10]:  # Show top 10
                trades = data.get('total_trades', 0)
                win_rate = data.get('win_rate', 0.5) * 100
                best_strategy = data.get('best_strategy', 'unknown')
                print(f"   • {symbol:10} - Trades:{trades:3} WR:{win_rate:5.1f}% Best:{best_strategy}")
        else:
            print("   • No symbol-specific data yet")
        
        print(f"\n📚 RECENT TRADES:")
        recent = brain.get('recent_trades', [])
        if recent:
            for trade in recent[-5:]:  # Show last 5
                symbol = trade.get('symbol', 'UNKNOWN')
                pnl = trade.get('profit', 0)
                confidence = trade.get('confidence', 0) * 100
                result = "WIN" if pnl > 0 else "LOSS"
                print(f"   • {symbol:10} {result} ${pnl:+6.2f} (Conf:{confidence:.0f}%)")
        else:
            print("   • No recent trades recorded")
        
        # Check position tracking
        positions = brain.get('position_tracking', {})
        if positions:
            print(f"\n📍 POSITION TRACKING:")
            for symbol, tracking in positions.items():
                if isinstance(tracking, list) and tracking:
                    last = tracking[-1]
                    pnl = last.get('unrealized_pnl', 0)
                    cycles = last.get('cycles_held', 0)
                    print(f"   • {symbol}: ${pnl:+.2f} after {cycles} cycles")
        
        # Machine learning performance
        ml_perf = brain.get('ml_performance', {})
        if any(ml_perf.values()):
            print(f"\n🤖 MACHINE LEARNING:")
            print(f"   • Neural Network: {ml_perf.get('neural_accuracy', 0)*100:.1f}% accuracy")
            print(f"   • Pattern Recognition: {ml_perf.get('pattern_success', 0)*100:.1f}% success")
            print(f"   • RL Reward: {ml_perf.get('rl_reward', 0):.2f}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"❌ Error reading AI brain: {e}")

def check_trading_state():
    """Also check trading state for comparison"""
    
    state_file = "trading_state.json"
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        print(f"\n💼 TRADING STATE:")
        print(f"   • Cash Balance: ${state.get('cash_balance', 0):.2f}")
        print(f"   • Total Trades: {state.get('total_trades', 0)}")
        print(f"   • Winning Trades: {state.get('winning_trades', 0)}")
        
        positions = state.get('positions', {})
        if positions:
            print(f"   • Active Positions: {len(positions)}")
            for symbol, pos in positions.items():
                qty = pos.get('quantity', 0)
                if qty > 0:
                    print(f"      - {symbol}: {qty:.6f} units")

if __name__ == "__main__":
    check_ai_brain()
    check_trading_state()
    
    print("\n💡 TIP: If AI isn't learning, make sure:")
    print("   1. Trades are actually executing (check trading_state.json)")
    print("   2. Positions are closing (not stuck in grace period)")
    print("   3. The bot is running with START button clicked")
    print("   4. Wait for trades to complete their full cycle")
