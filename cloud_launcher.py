"""
☁️ CLOUD LAUNCHER FOR POISE TRADER
Runs the bot 24/7 in cloud environment with notifications
"""

import asyncio
import os
import sys
from datetime import datetime
from micro_trading_bot import MicroTradingBot

# Optional: Telegram notifications
# Set these as environment variables in Railway/Cloud or edit directly
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID_HERE')
TELEGRAM_ENABLED = TELEGRAM_BOT_TOKEN != 'YOUR_BOT_TOKEN_HERE'

class TelegramNotifier:
    """Simple Telegram notifier for trade alerts"""
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.enabled = bot_token != "YOUR_BOT_TOKEN_HERE"
    
    def send_message(self, message):
        if not self.enabled:
            return
        try:
            import requests
            url = f"{self.base_url}/sendMessage"
            data = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            print(f"⚠️ Telegram notification failed: {e}")
    
    def notify_startup(self, capital, mode):
        message = f"""
🚀 <b>POISE TRADER STARTED</b>

💰 Initial Capital: ${capital:.2f}
🎯 Mode: {mode}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Bot is now running 24/7! 🔥
        """
        self.send_message(message.strip())
    
    def notify_trade_opened(self, symbol, action, price, position_size, confidence):
        message = f"""
🚀 <b>NEW TRADE OPENED</b>

📊 Symbol: {symbol}
📈 Action: {action}
💵 Entry: ${price:.2f}
💰 Size: ${position_size:.2f}
🎯 Confidence: {confidence*100:.0f}%
        """
        self.send_message(message.strip())
    
    def notify_trade_closed(self, symbol, action, entry_price, exit_price, pnl, win_rate, total_trades):
        emoji = "✅" if pnl > 0 else "❌"
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if action == 'BUY' else ((entry_price - exit_price) / entry_price * 100)
        
        message = f"""
{emoji} <b>TRADE CLOSED</b>

📊 Symbol: {symbol}
📈 Action: {action}
💵 Entry: ${entry_price:.2f}
💵 Exit: ${exit_price:.2f}
💰 P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)

📊 <b>Bot Stats:</b>
🎯 Win Rate: {win_rate:.1f}%
📈 Total Trades: {total_trades}
        """
        self.send_message(message.strip())
    
    def notify_status(self, capital, win_rate, trades, winning_trades):
        change = capital - 5.0
        change_pct = (change / 5.0) * 100
        
        message = f"""
📊 <b>BOT STATUS UPDATE</b>

💰 Current Capital: ${capital:.2f}
📈 P&L: ${change:+.2f} ({change_pct:+.1f}%)
🎯 Win Rate: {win_rate:.1f}%
✅ Wins: {winning_trades}
❌ Losses: {trades - winning_trades}
📊 Total Trades: {trades}
⏰ Time: {datetime.now().strftime('%H:%M:%S')}
        """
        self.send_message(message.strip())

async def main():
    """Main cloud launcher function"""
    print("="*80)
    print("☁️ POISE TRADER - CLOUD MODE")
    print("🚀 Starting 24/7 automated trading...")
    print("="*80)
    
    # Initialize Telegram notifier
    notifier = None
    if TELEGRAM_ENABLED:
        notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        print("📱 Telegram notifications: ENABLED")
    else:
        print("📱 Telegram notifications: DISABLED")
        print("   💡 To enable: Set TELEGRAM_ENABLED = True and add your bot token")
    
    # Initialize bot
    initial_capital = 5.0
    bot = MicroTradingBot(initial_capital=initial_capital)
    
    # Auto-select mode based on environment
    # Cloud default: PRECISION (safer for unattended operation)
    mode = os.environ.get('TRADING_MODE', 'PRECISION')
    bot.set_trading_mode(mode)
    
    print(f"\n🎯 Trading Mode: {mode}")
    print(f"💰 Initial Capital: ${initial_capital:.2f}")
    print(f"🔄 Auto-restart: ENABLED")
    print(f"📊 Logs: Visible in cloud console")
    
    # Send startup notification
    if notifier and notifier.enabled:
        notifier.notify_startup(initial_capital, mode)
    
    # Hook into bot for notifications
    if notifier and notifier.enabled:
        original_execute = bot.trader.execute_live_trade
        
        async def wrapped_execute(symbol, action, amount, *args, **kwargs):
            result = await original_execute(symbol, action, amount, *args, **kwargs)
            if result.get('success'):
                notifier.notify_trade_opened(
                    symbol, action, 
                    result.get('execution_price', 0),
                    amount,
                    kwargs.get('confidence', 0.5)
                )
            return result
        
        bot.trader.execute_live_trade = wrapped_execute
        
        # Status updates every hour
        async def send_periodic_status():
            while True:
                await asyncio.sleep(3600)  # Every hour
                portfolio = await bot.trader.get_portfolio_value()
                notifier.notify_status(
                    portfolio['total'],
                    bot.win_rate * 100,
                    bot.total_completed_trades,
                    bot.winning_trades
                )
        
        # Start status task
        asyncio.create_task(send_periodic_status())
    
    print("\n✅ Bot ready! Starting main trading loop...")
    print("💡 Press Ctrl+C to stop (will auto-restart in cloud)\n")
    
    # Run bot
    try:
        await bot.run()
    except KeyboardInterrupt:
        print("\n⚠️ Received stop signal...")
        print("💾 Saving state...")
        if notifier and notifier.enabled:
            portfolio = await bot.trader.get_portfolio_value()
            notifier.send_message(
                f"⚠️ <b>BOT STOPPED</b>\n\n"
                f"💰 Final Capital: ${portfolio['total']:.2f}\n"
                f"🎯 Final Win Rate: {bot.win_rate*100:.1f}%\n"
                f"📊 Total Trades: {bot.total_completed_trades}"
            )
        print("✅ Shutdown complete")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if notifier and notifier.enabled:
            notifier.send_message(f"❌ <b>BOT ERROR</b>\n\n{str(e)}")
        raise

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🏆 POISE TRADER - ULTRA-ADVANCED AI SYSTEM V2.0")
    print("☁️ CLOUD DEPLOYMENT MODE")
    print("="*80 + "\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)
