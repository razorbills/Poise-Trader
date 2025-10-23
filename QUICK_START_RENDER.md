# ⚡ QUICK START - Deploy to Render in 5 Minutes

## 🚀 Deploy Your Bot

### 1. Push to GitHub (if not already)

```bash
git add .
git commit -m "Added 24/7 keep-alive system"
git push origin main
```

### 2. Create Render Web Service

1. Go to: https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo
4. Use these settings:

```
Name: poise-trader-bot
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: python render_launcher.py
```

5. Click **"Create Web Service"**

### 3. Add Environment Variables (Optional)

If you use API keys:
```
BINANCE_API_KEY = your_key_here
BINANCE_API_SECRET = your_secret_here
```

### 4. Wait for Deployment (2-3 minutes)

### 5. Open Your Dashboard

```
https://YOUR-APP-NAME.onrender.com/
```

---

## 🔄 Setup Keep-Alive (Takes 2 minutes)

### Free External Monitor - UptimeRobot

1. Go to: https://uptimerobot.com/
2. Sign up (FREE)
3. Add Monitor:
   - Type: **HTTP(s)**
   - URL: `https://YOUR-APP-NAME.onrender.com/health`
   - Interval: **5 minutes**
4. Save

**Done! Your bot will now stay awake 24/7** ✅

---

## ✅ Verify It's Working

Test these URLs:

1. **Dashboard**: `https://YOUR-APP.onrender.com/`
2. **Health Check**: `https://YOUR-APP.onrender.com/health`
3. **Bot Status**: `https://YOUR-APP.onrender.com/api/status`

If all load successfully, you're live! 🎉

---

## 🎯 Start Trading

1. Open dashboard
2. Click **"Start Trading"**
3. Select mode (Aggressive/Precision)
4. Monitor your trades!

---

## 💡 Important Notes

### Free Tier
- ⚠️ Will still spin down after 15 min inactivity (even with keep-alive)
- ✅ Good for testing
- ⚠️ NOT recommended for real trading

### Starter Tier ($7/month)
- ✅ TRUE 24/7 - Never spins down
- ✅ Recommended for real trading
- ✅ Better performance

**Upgrade here**: Render Dashboard → Your Service → Settings → Instance Type

---

## 🚨 Troubleshooting

**Bot not starting?**
- Check logs in Render dashboard
- Verify requirements.txt installed correctly

**Connection issues?**
- Check API keys in Environment Variables
- Verify exchange API is working

**Dashboard won't load?**
- Wait 2-3 minutes after deployment
- Check if service is "Live" in Render

---

## 📊 What You Get

- ✅ Live trading bot
- ✅ Web dashboard with real-time data
- ✅ Start/Stop controls
- ✅ Mode switching (Aggressive/Precision)
- ✅ Position management
- ✅ P&L tracking
- ✅ Health monitoring
- ✅ Auto keep-alive system

---

**Total Setup Time: 5-7 minutes** ⚡

**Questions?** Check `RENDER_24_7_SETUP_GUIDE.md` for detailed instructions.
