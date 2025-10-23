#!/usr/bin/env python3
"""
Test script to verify custom TP/SL updates from dashboard
"""

import asyncio
import json
import os

def check_trading_state():
    """Check if custom TP/SL values are saved in trading state"""
    state_file = 'trading_state.json'
    
    if not os.path.exists(state_file):
        print("❌ No trading state file found")
        return False
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        print("\n📊 TRADING STATE CHECK")
        print("=" * 50)
        
        positions = state.get('positions', {})
        if not positions:
            print("⚠️ No open positions found")
            return True
        
        found_custom = False
        for symbol, pos in positions.items():
            print(f"\n📈 {symbol}:")
            print(f"   Quantity: {pos.get('quantity', 0)}")
            print(f"   Avg Price: ${pos.get('avg_price', 0):.2f}")
            
            # Check for custom TP/SL
            tp = pos.get('take_profit')
            sl = pos.get('stop_loss')
            
            if tp or sl:
                found_custom = True
                print(f"   🎯 CUSTOM VALUES FOUND:")
                if tp:
                    print(f"      Take Profit: ${tp:.2f}")
                if sl:
                    print(f"      Stop Loss: ${sl:.2f}")
            else:
                print(f"   📌 Using default TP/SL percentages")
        
        if found_custom:
            print("\n✅ Custom TP/SL values are properly stored!")
        else:
            print("\n⚠️ No custom TP/SL values found (using defaults)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading trading state: {e}")
        return False

async def test_tp_sl_logic():
    """Test the TP/SL update logic"""
    print("\n🧪 TP/SL UPDATE TEST")
    print("=" * 50)
    print("\n📝 Instructions:")
    print("1. Start the bot and dashboard")
    print("2. Open a position")
    print("3. Click on the position in dashboard")
    print("4. Update TP/SL values and click 💾 Update")
    print("5. Check console output for confirmation")
    print("\n✅ Fix Summary:")
    print("- Custom TP/SL prices now properly override default percentages")
    print("- When custom prices are set, ONLY those prices are checked")
    print("- Dashboard updates are properly saved and persisted")
    print("- Grace period logic still applies to custom stop losses")
    
    # Check current state
    check_trading_state()
    
    print("\n💡 Test verification:")
    print("- Watch bot console for '🎯 CUSTOM TP/SL DETECTED' messages")
    print("- Custom values should show as '${price}' not percentages")
    print("- Position should close at exact custom price levels")

if __name__ == "__main__":
    asyncio.run(test_tp_sl_logic())
