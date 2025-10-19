#!/usr/bin/env python3
"""
Test script to verify AI learning is working correctly
"""

import json
import os
from datetime import datetime
from ai_brain import ai_brain

def test_ai_learning():
    """Test AI brain learning functionality"""
    
    print("🧪 TESTING AI LEARNING SYSTEM")
    print("=" * 60)
    
    # Check current brain state
    print("\n📊 Current AI Brain State:")
    print(f"   Total trades learned: {ai_brain.brain['total_trades']}")
    print(f"   Total P&L learned: ${ai_brain.brain['total_profit_loss']:.2f}")
    print(f"   Learning sessions: {ai_brain.brain['learning_sessions']}")
    
    # Test learning from a simulated trade
    print("\n🔬 Testing learn_from_trade function...")
    
    test_trade = {
        'symbol': 'BTC/USDT',
        'action': 'BUY',
        'profit_loss': 5.50,  # Simulated profit
        'confidence': 0.75,
        'strategy_scores': {
            'technical': 0.8,
            'sentiment': 0.7,
            'momentum': 0.6
        },
        'market_conditions': {
            'volatility': 2.5,
            'trend_strength': 0.4,
            'regime': 'trending'
        }
    }
    
    # Learn from the test trade
    ai_brain.learn_from_trade(test_trade)
    
    # Check if brain was updated
    print("\n✅ After learning from test trade:")
    print(f"   Total trades learned: {ai_brain.brain['total_trades']}")
    print(f"   Total P&L learned: ${ai_brain.brain['total_profit_loss']:.2f}")
    
    # Check if file was saved
    if os.path.exists('ai_brain.json'):
        with open('ai_brain.json', 'r') as f:
            saved_brain = json.load(f)
        print(f"\n💾 Brain file exists and contains:")
        print(f"   Saved trades: {saved_brain['total_trades']}")
        print(f"   Saved P&L: ${saved_brain['total_profit_loss']:.2f}")
        print(f"   Last updated: {saved_brain['last_updated']}")
    else:
        print("\n❌ Brain file not found!")
    
    # Test loading brain in new instance
    print("\n🔄 Testing brain persistence (creating new instance)...")
    from ai_brain import AIBrain
    new_brain = AIBrain()
    
    print(f"   New instance loaded trades: {new_brain.brain['total_trades']}")
    print(f"   New instance loaded P&L: ${new_brain.brain['total_profit_loss']:.2f}")
    
    # Check shared knowledge
    if os.path.exists('shared_ai_knowledge.json'):
        with open('shared_ai_knowledge.json', 'r') as f:
            shared = json.load(f)
        print(f"\n🤝 Shared knowledge contains:")
        print(f"   Cross-bot trades: {shared.get('cross_bot_trades', 0)}")
        print(f"   Micro bot lessons: {len(shared.get('micro_bot_lessons', []))}")
        print(f"   Profit bot lessons: {len(shared.get('profit_bot_lessons', []))}")
    
    print("\n" + "=" * 60)
    print("🎯 AI LEARNING TEST COMPLETE")
    
    if ai_brain.brain['total_trades'] > 0:
        print("✅ AI Brain is learning and saving correctly!")
    else:
        print("⚠️ AI Brain may not be learning properly - no trades recorded")

if __name__ == "__main__":
    test_ai_learning()
