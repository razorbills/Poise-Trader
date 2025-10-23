# 🚀 24/7 RENDER KEEP-ALIVE SETUP GUIDE

This guide ensures your trading bot and dashboard stay alive 24/7 on Render.com without going to sleep.

---

## ✅ What's Already Implemented

Your bot now includes:

1. **✅ Self-Ping System** - Automatically pings itself every 5 minutes
2. **✅ Health Check Endpoints** - `/health`, `/ping`, `/keep-alive`
3. **✅ Connection Monitor** - Detects and recovers from connection issues
4. **✅ Activity Simulator** - Simulates user activity to keep service active

---

## 🎯 STEP 1: Deploy to Render

### Option A: Web Service (Recommended)

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Create New Web Service**
3. **Connect your GitHub repository**
4. **Configure:**
   - **Name**: `poise-trader-bot` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python render_launcher.py`
   - **Instance Type**: Free or Starter ($7/month recommended for 24/7)

5. **Add Environment Variables** (if needed):
   - `BINANCE_API_KEY` - Your Binance API Key
   - `BINANCE_API_SECRET` - Your Binance API Secret
   - Any other keys your bot needs

6. **Deploy!**

### Important: Render Free Tier Limitations

⚠️ **Free tier services spin down after 15 minutes of inactivity**
- Our keep-alive system helps, but Render Free still has limits
- **For TRUE 24/7 operation, upgrade to Starter plan ($7/month)**
- Starter plan = Always on, no spin-down

---

## 🔄 STEP 2: Setup External Monitoring (HIGHLY RECOMMENDED)

External monitoring services will ping your bot every few minutes to keep it awake.

### Option 1: UptimeRobot (FREE - BEST OPTION)

1. **Go to**: https://uptimerobot.com/
2. **Sign up for free** (monitors up to 50 URLs)
3. **Add New Monitor**:
   - **Monitor Type**: HTTP(s)
   - **Friendly Name**: Poise Trader Bot
   - **URL**: `https://YOUR-APP-NAME.onrender.com/health`
   - **Monitoring Interval**: 5 minutes (free tier)
   - Click **Create Monitor**

✅ **UptimeRobot will now ping your bot every 5 minutes, keeping it awake!**

### Option 2: Cron-job.org (FREE)

1. **Go to**: https://cron-job.org/
2. **Sign up for free**
3. **Create Cronjob**:
   - **Title**: Poise Trader Keepalive
   - **Address**: `https://YOUR-APP-NAME.onrender.com/health`
   - **Schedule**: Every 5 minutes (`*/5 * * * *`)
   - **Enable**: ✅
   - Save

### Option 3: Pingdom (Paid, but very reliable)

- Similar setup to UptimeRobot
- Better uptime tracking and alerts

---

## 📊 STEP 3: Monitor Your Bot

### Access Your Dashboard

```
https://YOUR-APP-NAME.onrender.com/
```

### Check Health Status

```
https://YOUR-APP-NAME.onrender.com/health
```

Returns:
```json
{
  "status": "healthy",
  "timestamp": "2024-10-23T18:30:00",
  "bot_connected": true,
  "bot_running": true,
  "uptime": 86400,
  "current_capital": 5.23
}
```

### Simple Ping Test

```
https://YOUR-APP-NAME.onrender.com/ping
```

---

## 🔧 STEP 4: Configure Render for Best Performance

### In Render Dashboard:

1. **Go to your service settings**
2. **Environment Variables** (Optional but recommended):
   ```
   RENDER_EXTERNAL_URL = https://your-app-name.onrender.com
   PYTHON_VERSION = 3.11
   ```

3. **Health Check Path** (in Advanced settings):
   ```
   /health
   ```

4. **Auto-Deploy**: Enable if you want automatic updates from GitHub

---

## 💡 STEP 5: Verify Everything Works

### Check Logs

In Render Dashboard:
1. Go to your service
2. Click **Logs** tab
3. You should see:
   ```
   ✅ Full dashboard server loaded with all API endpoints
   🔄 KEEP-ALIVE SYSTEM ACTIVATED
   💓 Keep-alive heartbeat #1
   ✅ ALL KEEP-ALIVE SYSTEMS OPERATIONAL
   ```

### Test Endpoints

Test these URLs in your browser:

1. **Dashboard**: `https://YOUR-APP.onrender.com/`
2. **Health**: `https://YOUR-APP.onrender.com/health`
3. **Status**: `https://YOUR-APP.onrender.com/api/status`
4. **Metrics**: `https://YOUR-APP.onrender.com/api/metrics`

---

## 🎯 Best Practices for 24/7 Operation

### 1. Use Starter Plan ($7/month)
- Free tier WILL still spin down after inactivity
- Starter plan = True 24/7, no interruptions
- **This is the BEST solution for reliable trading**

### 2. Setup Multiple Monitors
- Use UptimeRobot + Cron-job.org together
- Redundancy ensures your bot stays awake

### 3. Check Logs Daily
- Monitor for any errors or issues
- Render provides 7 days of logs on free tier

### 4. Set Up Alerts
- UptimeRobot can email/SMS you if bot goes down
- Configure alerts in UptimeRobot dashboard

### 5. Monitor Capital
- Check dashboard daily
- Ensure trades are executing properly

---

## 🚨 Troubleshooting

### Bot Goes to Sleep

**Solution 1**: Upgrade to Starter plan
**Solution 2**: Add more external monitors (UptimeRobot + Cron-job.org)
**Solution 3**: Reduce ping interval in `keep_alive_system.py` (edit line 15):
```python
ping_interval=180  # 3 minutes instead of 5
```

### Connection Lost

- Check Render logs for errors
- Verify API keys are correct
- Check if exchange API is responding
- The connection monitor will auto-retry

### No Trades Executing

1. Check if bot is running: `/api/status`
2. Make sure you clicked "Start Trading" in dashboard
3. Verify trading mode is set correctly
4. Check if bot has enough capital

---

## 📈 Monitoring Dashboard

Your dashboard shows:
- ✅ Bot Status (Running/Stopped)
- 📊 Current Capital & P&L
- 💰 Active Positions
- 📈 P&L Chart
- 🎯 Trading Mode
- ⚙️ Controls (Start/Stop, Mode Switch)

---

## 🔐 Security Tips

1. **Never commit API keys** to GitHub
2. **Use Render Environment Variables** for secrets
3. **Enable 2FA** on Render account
4. **Use IP whitelist** on Binance API (if possible)
5. **Set API restrictions**:
   - ✅ Enable Spot Trading
   - ❌ Disable Withdrawals
   - ❌ Disable Transfers

---

## 💰 Cost Comparison

### Free Tier (Render + UptimeRobot)
- **Cost**: $0/month
- **Uptime**: ~95-98% (will still spin down occasionally)
- **Best for**: Testing, Demo

### Starter Tier (Render + UptimeRobot)
- **Cost**: $7/month
- **Uptime**: 99.9%+
- **Best for**: Real Trading, Production

### Recommended Setup
- **Render Starter**: $7/month
- **UptimeRobot Free**: $0/month
- **Total**: $7/month for TRUE 24/7 operation

---

## ✅ Quick Setup Checklist

- [ ] Deploy bot to Render
- [ ] Add Environment Variables (API keys)
- [ ] Verify deployment successful (check logs)
- [ ] Test dashboard: `https://YOUR-APP.onrender.com/`
- [ ] Test health endpoint: `/health`
- [ ] Setup UptimeRobot monitoring
- [ ] (Optional) Setup Cron-job.org as backup
- [ ] Configure Render health check path
- [ ] Start trading from dashboard
- [ ] Monitor first few trades
- [ ] Setup UptimeRobot alerts
- [ ] Check daily for first week

---

## 🎉 You're All Set!

Your bot is now running 24/7 with:
- ✅ Automatic keep-alive
- ✅ Health monitoring
- ✅ Connection recovery
- ✅ Web dashboard
- ✅ Full trading capabilities

**For best results**: Upgrade to Render Starter plan ($7/month) for true 24/7 uptime!

---

## 📞 Support

If you encounter issues:
1. Check Render logs first
2. Test health endpoint
3. Verify API keys
4. Check external monitor status

---

**Happy Trading! 🚀📈💰**
