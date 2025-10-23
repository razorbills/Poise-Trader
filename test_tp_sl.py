#!/usr/bin/env python3
"""
Test TP/SL update functionality
"""

import json
import requests

API_URL = 'http://localhost:5000'

def test_tp_sl_update():
    """Test updating TP/SL values via dashboard API"""
    
    print("Testing TP/SL Update Functionality")
    print("="*50)
    
    # First, check current positions
    try:
        response = requests.get(f"{API_URL}/api/portfolio")
        if response.ok:
            portfolio = response.json()
            positions = portfolio.get('positions', {})
            
            if not positions:
                print("❌ No positions found. Create a position first!")
                return
            
            # Pick the first position
            symbol = list(positions.keys())[0]
            pos = positions[symbol]
            
            print(f"\n📊 Testing with position: {symbol}")
            print(f"   Entry Price: ${pos.get('avg_price', 0):.2f}")
            print(f"   Current Price: ${pos.get('current_price', 0):.2f}")
            
            # Calculate new TP/SL based on entry price
            entry = pos.get('avg_price', 100)
            new_tp = entry * 1.05  # 5% profit
            new_sl = entry * 0.97  # 3% loss
            
            print(f"\n🎯 Setting new TP/SL:")
            print(f"   Take Profit: ${new_tp:.2f} (+5%)")
            print(f"   Stop Loss: ${new_sl:.2f} (-3%)")
            
            # Send update request
            update_data = {
                'symbol': symbol,
                'take_profit': new_tp,
                'stop_loss': new_sl
            }
            
            response = requests.post(
                f"{API_URL}/api/update_position",
                json=update_data
            )
            
            if response.ok:
                print(f"\n✅ Successfully updated {symbol} TP/SL!")
                
                # Verify it was saved
                with open('trading_state.json', 'r') as f:
                    state = json.load(f)
                    
                saved_pos = state['positions'].get(symbol, {})
                if 'take_profit' in saved_pos and 'stop_loss' in saved_pos:
                    print(f"\n💾 Verified in trading_state.json:")
                    print(f"   TP: ${saved_pos['take_profit']:.2f}")
                    print(f"   SL: ${saved_pos['stop_loss']:.2f}")
                else:
                    print(f"\n⚠️ TP/SL not found in saved state!")
            else:
                print(f"\n❌ Failed to update: {response.json()}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Dashboard server not running! Start it first.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_tp_sl_update()
