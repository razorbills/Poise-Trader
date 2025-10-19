# 🎛️ Poise Trader - Web Dashboard Control Guide

## 🚀 Quick Start

### **Option 1: One-Click Launcher (Windows)**
```bash
START_DASHBOARD.bat
```

### **Option 2: Manual Start**
```bash
python micro_trading_bot.py
```

## 🌐 Dashboard Interface

Once started, the bot will:
1. ✅ Start the web dashboard at **http://localhost:5000**
2. 🌐 Automatically open your browser
3. ⏸️ Wait for your commands (not trading yet)

## 🎮 How To Control The Bot

### **From The Dashboard:**

1. **Select Trading Mode:**
   - Click **⚡ Aggressive** for high-frequency trading (12+ trades/hour)
   - Click **🎯 Normal** for quality-focused trading (4+ trades/hour)

2. **Start Trading:**
   - Click **▶️ Start Trading** button
   - Bot will begin generating signals and placing trades

3. **Monitor Performance:**
   - View real-time P&L
   - Track win rate
   - See active positions
   - Monitor trade history

4. **Stop Trading:**
   - Click **⏸️ Stop Trading** button
   - Bot will pause after current cycle

## 📊 Terminal Window

The terminal shows **LOGS ONLY**:
- ✅ Trade executions
- 📡 Market data collection
- 🔮 Signal generation
- 💎 Position updates
- ⚠️ Errors and warnings

**DO NOT** interact with the terminal window.
**USE ONLY** the web dashboard for controls.

## ⚡ Aggressive Mode Features

When you select Aggressive mode:
- ✅ Win rate optimizer: **DISABLED**
- ✅ Minimum confidence: **25%** (vs 75% in Normal)
- ✅ Trade guarantee: **Active** (forces ≥1 trade/minute)
- ✅ Filters: **Bypassed**
- 🎯 Expected: **12+ trades per hour**

## 🎯 Normal Mode Features

When you select Normal mode:
- ✅ Win rate optimizer: **ENABLED**
- ✅ Minimum confidence: **75%**
- ✅ Quality filtering: **Active**
- ✅ Target win rate: **90%+**
- 🎯 Expected: **4+ trades per hour**

## 🔥 What You'll See When Trading

### In Dashboard:
```
💰 Portfolio Value: $5.23
📈 Total P&L: +$0.23 (+4.6%)
🏆 Win Rate: 100%
📊 Total Trades: 3
⚡ Mode: AGGRESSIVE
▶️ Status: RUNNING
```

### In Terminal:
```
============================================================
✅ TRADE EXECUTED SUCCESSFULLY!
   Symbol: BTC/USDT
   Action: BUY
   Amount: $1.00
   Price: $106,881.50
   Strategy: AGGRESSIVE_FORCED
   💰 New Cash Balance: $4.23
   📊 Total Trades: 3
============================================================
```

## 🛑 Stopping The Bot

1. **Pause Trading:** Click "Stop Trading" in dashboard
2. **Shutdown:** Press `Ctrl+C` in terminal window
3. **Emergency Stop:** Close terminal window

## 🔧 Environment Variables (Optional)

Force a mode without dashboard:
```bash
# Aggressive mode
set POISE_MODE=aggressive
python micro_trading_bot.py

# Normal mode  
set POISE_MODE=normal
python micro_trading_bot.py

# Auto-start trading
set POISE_AUTOSTART=1
python micro_trading_bot.py
```

## 📞 Troubleshooting

### **Dashboard not opening?**
- Manually go to: http://localhost:5000

### **Port 5000 already in use?**
- Close other Flask apps
- Or edit `professional_dashboard.py` to change port

### **No trades in Aggressive mode?**
- Check terminal for errors
- Verify signals are being generated
- Check portfolio has funds ($5+)

### **Terminal asking for input?**
- You're using an old version
- Run: `git pull` to update
- Or re-download `micro_trading_bot.py`

## 🎯 Pro Tips

1. **Start with Aggressive mode** to see trades immediately
2. **Monitor terminal logs** while using dashboard
3. **Check win rate** after 10+ trades before judging performance
4. **Normal mode** is better for long-term results
5. **Dashboard updates every 2 seconds** automatically

---

## 🏆 Architecture

```
┌─────────────────────────────────────────┐
│     WEB DASHBOARD (PRIMARY CONTROL)     │
│         http://localhost:5000           │
│                                         │
│  [⚡ Aggressive] [🎯 Normal]            │
│  [▶️ Start]     [⏸️ Stop]               │
│                                         │
│  📊 Real-time Stats                     │
│  💰 Portfolio View                      │
│  📈 Trade History                       │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       TRADING BOT (BACKGROUND)          │
│       micro_trading_bot.py              │
│                                         │
│  • Waits for dashboard commands         │
│  • Generates signals                    │
│  • Executes trades                      │
│  • Sends updates to dashboard           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      TERMINAL (LOGS ONLY)               │
│                                         │
│  ✅ Trade executions                    │
│  📡 Market data                         │
│  🔮 Signal generation                   │
│  💎 Position updates                    │
└─────────────────────────────────────────┘
```

**Remember: Dashboard controls, Terminal observes!** 🎮📊
