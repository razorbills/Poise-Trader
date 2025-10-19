#!/usr/bin/env python3
"""
Quick fix: Add select_trading_mode method to LegendaryCryptoTitanBot
"""

import re

# Read the file
with open('micro_trading_bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the method definition that needs to be added
method_code = '''
    def select_trading_mode(self):
        """Interactive trading mode selection"""
        print("\\n" + "="*70)
        print("🎯 SELECT TRADING MODE")
        print("="*70)
        print("\\n📊 Available Modes:")
        print("   1️⃣  AGGRESSIVE MODE")
        print("      ⚡ Guarantees ≥1 trade per minute")
        print("      📊 Lower confidence threshold (25%)")
        print("      🔥 High-frequency trading")
        print("      🎯 Target: 12+ trades/hour")
        print("")
        print("   2️⃣  NORMAL MODE (PRECISION)")
        print("      🎯 Best-of-the-best signals only")
        print("      📊 Higher confidence threshold (75%)")
        print("      💎 Patient, quality-focused")
        print("      🎯 Target: 4+ trades/hour")
        print("")
        
        while True:
            try:
                choice = input("\\nSelect mode (1 for AGGRESSIVE, 2 for NORMAL): ").strip()
                
                if choice == '1':
                    self.trading_mode = 'AGGRESSIVE'
                    config = self.mode_config['AGGRESSIVE']
                    self.target_accuracy = config['target_accuracy']
                    self.min_confidence_for_trade = config['min_confidence']
                    self.ensemble_threshold = config['ensemble_threshold']
                    self.confidence_threshold = config['min_confidence']
                    self.base_confidence_threshold = config['min_confidence']
                    self.fast_mode_enabled = True
                    self.precision_mode_enabled = False
                    self.min_price_history = 20
                    self.confidence_adjustment_factor = 0.05
                    self.aggressive_trade_guarantee = True
                    self.aggressive_trade_interval = 60.0
                    self.cycle_sleep_override = 10.0
                    self.win_rate_optimizer_enabled = False
                    print("\\n⚡ AGGRESSIVE MODE SELECTED!")
                    print("   🔥 Win rate optimizer: DISABLED")
                    print("   🎯 Trade guarantee: ACTIVE (≥1 trade/min)")
                    print(f"   📊 Min confidence: {config['min_confidence']:.0%}")
                    break
                    
                elif choice == '2':
                    self.trading_mode = 'PRECISION'
                    config = self.mode_config['PRECISION']
                    self.target_accuracy = config['target_accuracy']
                    self.min_confidence_for_trade = config['min_confidence']
                    self.ensemble_threshold = config['ensemble_threshold']
                    self.confidence_threshold = config['min_confidence']
                    self.base_confidence_threshold = config['min_confidence']
                    self.fast_mode_enabled = False
                    self.precision_mode_enabled = True
                    self.min_price_history = 50
                    self.confidence_adjustment_factor = 0.01
                    self.aggressive_trade_guarantee = False
                    self.cycle_sleep_override = None
                    self.win_rate_optimizer_enabled = True
                    print("\\n🎯 NORMAL (PRECISION) MODE SELECTED!")
                    print("   💎 Win rate optimizer: ENABLED")
                    print("   🎯 Quality-focused trading")
                    print(f"   📊 Min confidence: {config['min_confidence']:.0%}")
                    break
                    
                else:
                    print("❌ Invalid choice. Please enter 1 or 2.")
                    
            except (EOFError, KeyboardInterrupt):
                print("\\n⚠️ Defaulting to NORMAL mode")
                self.trading_mode = 'PRECISION'
                config = self.mode_config['PRECISION']
                break
        
        print(f"\\n✅ {self.trading_mode} MODE ACTIVATED")
        print("=" * 70)
'''

# Check if method already exists
if 'def select_trading_mode(self):' in content:
    print("✅ select_trading_mode method already exists!")
else:
    # Find a good place to insert - after run_micro_trading_cycle
    pattern = r'(    async def run_micro_trading_cycle\(self.*?\n        self\.bot_running = False\n)'
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        # Insert after this method
        insert_pos = match.end()
        new_content = content[:insert_pos] + method_code + content[insert_pos:]
        
        # Write back
        with open('micro_trading_bot.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Added select_trading_mode method!")
        print(f"   Inserted at position: {insert_pos}")
    else:
        print("❌ Could not find insertion point")
        print("   Looking for run_micro_trading_cycle method...")
        
print("\n🎯 Done! Try running the bot again.")
