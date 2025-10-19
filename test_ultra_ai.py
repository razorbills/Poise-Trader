#!/usr/bin/env python3
"""Test Ultra AI Optimizer Integration"""

import json
import time
import random
from datetime import datetime
from ai_brain import AIBrain
from ultra_ai_optimizer import UltraAIOptimizer

def test_ultra_ai_learning():
    """Test the Ultra AI learning and prediction capabilities"""
    print("=" * 80)
    print("🚀 ULTRA AI OPTIMIZER TEST - 90% WIN RATE TARGET")
    print("=" * 80)
    
    # Initialize AI Brain (which includes Ultra AI Optimizer)
    brain = AIBrain()
    
    # Simulate trading data
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    print("\n📊 Simulating 20 trades to test learning...")
    
    for i in range(20):
        symbol = random.choice(symbols)
        
        # Generate random trade outcome (70% win rate for testing)
        is_winner = random.random() < 0.7
        
        # Create trade data
        entry_price = 50000 + random.uniform(-1000, 1000)
        if is_winner:
            exit_price = entry_price * (1 + random.uniform(0.01, 0.03))  # 1-3% profit
            profit = (exit_price - entry_price) / entry_price * 100
        else:
            exit_price = entry_price * (1 - random.uniform(0.005, 0.015))  # 0.5-1.5% loss
            profit = (exit_price - entry_price) / entry_price * 100
        
        trade_data = {
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit': profit,
            'strategy': random.choice(['momentum', 'mean_reversion', 'breakout']),
            'timestamp': datetime.now().isoformat(),
            'position_size': 0.1,
            'side': random.choice(['BUY', 'SELL']),
            'duration': random.randint(60, 3600),
            'market_conditions': {
                'volatility': random.uniform(0.5, 2.0),
                'trend': random.choice(['bullish', 'bearish', 'neutral']),
                'volume': random.uniform(0.8, 1.5)
            }
        }
        
        print(f"\nTrade #{i+1}: {symbol}")
        print(f"  Result: {'✅ WIN' if is_winner else '❌ LOSS'} ({profit:.2f}%)")
        
        # Let AI learn from the trade
        brain.learn_from_trade(trade_data)
        
        # Test prediction after learning
        if i >= 5:  # After some learning
            prices = [entry_price + random.uniform(-100, 100) for _ in range(20)]
            market_data = {
                'strategy_scores': {
                    'technical': random.uniform(0.3, 0.8),
                    'momentum': random.uniform(0.3, 0.8),
                    'sentiment': 0.5,
                    'mean_reversion': random.uniform(0.3, 0.8),
                    'breakout': random.uniform(0.3, 0.8)
                },
                'market_conditions': {
                    'volatility': random.uniform(0.5, 2.0),
                    'trend_strength': random.uniform(-1, 1),
                    'volume': random.uniform(0.8, 1.5)
                }
            }
            
            # Get enhanced prediction
            prediction = brain.get_enhanced_prediction(symbol, prices, market_data)
            
            print(f"  🎯 Ultra AI Prediction:")
            print(f"     • Action: {prediction['action']}")
            print(f"     • Confidence: {prediction['confidence']:.1%}")
            print(f"     • Win Probability: {prediction['win_probability']:.1%}")
            print(f"     • Pattern Strength: {prediction['pattern_strength']:.2f}")
            print(f"     • Risk Score: {prediction['risk_score']:.2f}")
    
    print("\n" + "=" * 80)
    print("✅ ULTRA AI OPTIMIZER TEST COMPLETE")
    print("=" * 80)
    
    # Show final statistics
    if hasattr(brain, 'ultra_optimizer'):
        print("\n📈 Final Ultra AI Statistics:")
        print(f"  • Total patterns learned: {len(brain.ultra_optimizer.winning_patterns)}")
        print(f"  • Dynamic confidence: {brain.ultra_optimizer.dynamic_confidence:.2f}")
        print(f"  • Learning rate: {brain.ultra_optimizer.learning_rate:.4f}")
        
        # Get final recommendations
        recommendations = brain.ultra_optimizer.get_recommendations()
        if recommendations:
            print("\n🎯 Top AI Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")
    
    print("\n💾 Saving enhanced AI brain...")
    brain.save_brain()
    print("✅ AI brain saved successfully!")

if __name__ == "__main__":
    test_ultra_ai_learning()
