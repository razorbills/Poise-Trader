"""
🆓 FREE CLOUD LAUNCHER FOR POISE TRADER
Optimized for PythonAnywhere and other free hosting platforms
"""

import asyncio
import os
import sys
from datetime import datetime

# Force REST API only (no websockets for free hosting)
os.environ['USE_WEBSOCKETS'] = 'false'

try:
    from micro_trading_bot import MicroTradingBot
except ImportError as e:
    print(f"❌ Error importing bot: {e}")
    print("💡 Make sure all required files are uploaded!")
    sys.exit(1)

# Telegram notifications (optional but recommended)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
TELEGRAM_ENABLED = TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

class SimpleTelegramNotifier:
    """Lightweight Telegram notifier for free hosting"""
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send(self, message):
        """Send message to Telegram"""
        try:
            import requests
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id, 
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️ Telegram failed: {e}")
            return False
    
    def startup(self, capital, mode):
        """Send startup notification"""
        message = f"""
🚀 <b>POISE TRADER STARTED</b>

💰 Capital: ${capital:.2f}
🎯 Mode: {mode}
🆓 Platform: Free Cloud Hosting
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Bot is now trading 24/7! 🔥
        """
        return self.send(message.strip())
    
    def trade_alert(self, symbol, action, price, pnl=None):
        """Send trade notification"""
        if pnl is not None:
            emoji = "✅" if pnl > 0 else "❌"
            message = f"""
{emoji} <b>TRADE CLOSED</b>

📊 {symbol}
💵 Price: ${price:.2f}
💰 P&L: ${pnl:+.2f}
            """
        else:
            message = f"""
🚀 <b>NEW TRADE</b>

📊 {symbol}
📈 {action}
💵 Entry: ${price:.2f}
            """
        return self.send(message.strip())
    
    def status(self, capital, win_rate, trades):
        """Send status update"""
        pnl = capital - 5.0
        pnl_pct = (pnl / 5.0) * 100
        
        message = f"""
📊 <b>BOT STATUS</b>

💰 Capital: ${capital:.2f}
📈 P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)
🎯 Win Rate: {win_rate:.1f}%
📊 Trades: {trades}
⏰ {datetime.now().strftime('%H:%M:%S')}
        """
        return self.send(message.strip())

async def main():
    """Main function for free cloud hosting"""
    
    print("="*60)
    print("🆓 POISE TRADER - FREE HOSTING MODE")
    print("="*60)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔄 REST API Mode (WebSocket-free)")
    print("💰 Initial Capital: $5.00")
    
    # Initialize Telegram
    notifier = None
    if TELEGRAM_ENABLED:
        notifier = SimpleTelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        print("📱 Telegram: ENABLED ✅")
        print(f"   Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...")
        print(f"   Chat ID: {TELEGRAM_CHAT_ID}")
    else:
        print("📱 Telegram: DISABLED ⚠️")
        print("   To enable: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        print("   in environment variables or edit this file")
    
    print("-"*60)
    
    # Initialize bot
    try:
        initial_capital = 5.0
        bot = MicroTradingBot(initial_capital=initial_capital)
        
        # Set PRECISION mode (safer for unattended operation)
        mode = os.environ.get('TRADING_MODE', 'PRECISION')
        bot.set_trading_mode(mode)
        
        print(f"✅ Bot initialized successfully!")
        print(f"🎯 Trading Mode: {mode}")
        print(f"💵 Initial Capital: ${initial_capital:.2f}")
        print(f"🔐 Paper Trading: ON (Safe testing)")
        
        # Send startup notification
        if notifier:
            success = notifier.startup(initial_capital, mode)
            if success:
                print("📱 Telegram startup notification sent!")
            else:
                print("⚠️ Telegram notification failed")
        
        print("-"*60)
        print("🚀 Starting trading loop...")
        print("💡 Bot will run continuously until stopped")
        print("📊 Check logs to monitor activity")
        print("="*60)
        print()
        
        # Periodic status updates (every 6 hours)
        if notifier:
            async def send_status_updates():
                """Send periodic status to Telegram"""
                while True:
                    await asyncio.sleep(21600)  # 6 hours
                    try:
                        portfolio = await bot.trader.get_portfolio_value()
                        total_capital = portfolio.get('total', 0)
                        win_rate = bot.win_rate * 100
                        trades = bot.total_completed_trades
                        notifier.status(total_capital, win_rate, trades)
                    except Exception as e:
                        print(f"⚠️ Status update failed: {e}")
            
            # Start status task in background
            asyncio.create_task(send_status_updates())
        
        # Run bot
        await bot.run()
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Run: pip install -r requirements.txt")
        if notifier:
            notifier.send(f"❌ Bot failed to start: Missing dependencies\n{str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        print(f"📋 Error details: {type(e).__name__}")
        if notifier:
            notifier.send(f"❌ Bot error:\n{str(e)}")
        raise

if __name__ == "__main__":
    print()
    print("🏆 POISE TRADER - ULTRA AI SYSTEM V2.0")
    print("🆓 FREE CLOUD HOSTING EDITION")
    print("⏰ Timestamp:", datetime.now().isoformat())
    print()
    
    try:
        # Run the bot
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n⚠️ Received stop signal (Ctrl+C)")
        print("💾 Shutting down gracefully...")
        print("✅ Bot stopped")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        print(f"📋 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
