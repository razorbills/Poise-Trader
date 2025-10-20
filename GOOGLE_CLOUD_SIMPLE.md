# ☁️ GOOGLE CLOUD - 12 MONTHS FREE! (Better than Railway)

## 🎉 **100% FREE FOR 12 MONTHS!**

**Railway:** $5 credit = ~10-20 days free  
**Google Cloud:** $300 credit = **12 MONTHS FREE!** 🏆

---

## ⚡ **30-MINUTE SETUP FOR 12 MONTHS FREE**

### **Step 1: Sign Up (5 minutes)**

1. Go to: https://console.cloud.google.com
2. Click "Get started for free"
3. Sign in with Google account
4. Add credit card (NOT CHARGED! Just for verification)
5. Get **$300 free credit** ✅

**No charges until:**
- You manually upgrade to paid
- Or 12 months pass
- Or you use $300 (impossible for a trading bot!)

---

### **Step 2: Create VM (5 minutes)**

**2a. Enable Compute Engine:**
1. Search bar: "Compute Engine"
2. Click "Enable" (takes 2 minutes)

**2b. Create Instance:**
1. Click "Create Instance"
2. **Name:** poise-trader
3. **Region:** us-central1 (or closest to you)
4. **Machine type:** e2-micro (FREE tier!)
5. **Boot disk:** Ubuntu 22.04 LTS
6. **Firewall:** ✅ Allow HTTP, ✅ Allow HTTPS
7. Click "CREATE"

✅ **Done! VM is running!**

---

### **Step 3: Connect & Setup (10 minutes)**

**3a. Connect via SSH:**
1. Find your VM in Compute Engine
2. Click "SSH" button (opens terminal in browser)

**3b. Install Python:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip git -y
```

**3c. Upload Your Bot:**

**Option A: GitHub (Recommended)**
```bash
git clone https://github.com/YOUR_USERNAME/poise-trader.git
cd poise-trader
```

**Option B: Direct Upload**
1. Click "⚙️" in SSH window → "Upload file"
2. Upload your entire "Poise Trader" folder
3. ```bash
   cd poise-trader
   ```

**3d. Install Dependencies:**
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

### **Step 4: Run Bot 24/7 (5 minutes)**

**4a. Install Screen (keeps bot running):**
```bash
sudo apt install screen -y
```

**4b. Start Bot:**
```bash
# Start screen session
screen -S poise-trader

# Run bot
python cloud_launcher.py

# Press Ctrl+A, then D to detach
# Bot keeps running even after you close browser!
```

**4c. Verify It's Running:**
```bash
# Reconnect to screen anytime
screen -r poise-trader

# Or check if running
ps aux | grep cloud_launcher
```

✅ **DONE! Bot runs 24/7 for FREE!**

---

### **Step 5: Setup Telegram (Optional - 2 minutes)**

**5a. Get Credentials:**
1. Telegram → @BotFather → `/newbot` → Get TOKEN
2. Telegram → @userinfobot → Get CHAT_ID

**5b. Set Environment Variables:**
```bash
# Edit cloud_launcher.py
nano cloud_launcher.py

# Change these lines:
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# Save: Ctrl+X, Y, Enter
```

**5c. Restart Bot:**
```bash
screen -r poise-trader
# Press Ctrl+C
python cloud_launcher.py
# Press Ctrl+A, then D
```

✅ **Now get phone notifications!**

---

## 🔄 **MANAGING YOUR BOT**

### **View Bot (Anytime):**
```bash
# SSH into VM (click SSH button)
screen -r poise-trader
# See live bot output!
```

### **Stop Bot:**
```bash
screen -r poise-trader
# Press Ctrl+C
```

### **Restart Bot:**
```bash
screen -r poise-trader
python cloud_launcher.py
# Press Ctrl+A, then D
```

### **View Logs:**
```bash
screen -r poise-trader
# Scroll up to see history
```

---

## 💰 **COST BREAKDOWN**

### **Google Cloud Free Tier:**

**What You Get:**
- $300 free credit
- 12 months duration
- e2-micro VM: ~$7/month
- **Result: 12+ months FREE!**

**Math:**
```
$300 credit ÷ $7/month = 42 months FREE!
But free tier ends after 12 months.
Then you can upgrade or stop.
```

**After 12 Months:**
```
Continue paying: ~$7/month
Or stop and lose nothing!
Or switch to Railway/elsewhere
```

---

## 📱 **PHONE ACCESS**

### **SSH from Phone:**

**Download App:**
- iOS: Termius
- Android: Termius or JuiceSSH

**Connect:**
1. Add server: your-vm-ip (find in Console)
2. Username: your-google-username
3. SSH key: Generate in app
4. Connect!

Now manage bot from phone! 📱

### **Telegram Notifications:**
Once setup, you get all trade alerts on phone automatically!

---

## 🆚 **GOOGLE CLOUD vs RAILWAY**

| Feature | Railway | Google Cloud |
|---------|---------|--------------|
| **Free Duration** | ~10-20 days | **12 months** 🏆 |
| **Free Credit** | $5 | **$300** 🏆 |
| **Setup Time** | 5 min | 30 min |
| **Difficulty** | ⭐ Easy | ⭐⭐ Medium |
| **After Free** | $5/month | $7/month |
| **Auto-restart** | ✅ Built-in | ⚙️ Need to setup |
| **Web Dashboard** | ✅ Nice | ⭐ Basic |

**Winner:** Google Cloud (12 months FREE!) 🏆

---

## 🚀 **AUTO-RESTART SETUP (Optional - 5 minutes)**

Make bot restart automatically if it crashes:

**Create systemd service:**

```bash
sudo nano /etc/systemd/system/poise-trader.service
```

**Paste this:**
```ini
[Unit]
Description=Poise Trader Bot
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/poise-trader
ExecStart=/home/YOUR_USERNAME/poise-trader/venv/bin/python cloud_launcher.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Replace YOUR_USERNAME with your actual username!**

**Enable service:**
```bash
sudo systemctl enable poise-trader
sudo systemctl start poise-trader

# Check status
sudo systemctl status poise-trader

# View logs
sudo journalctl -u poise-trader -f
```

✅ **Now bot auto-starts on boot and auto-restarts on crash!**

---

## 🎯 **VERIFICATION CHECKLIST**

### **✅ VM Created:**
- [ ] Compute Engine → VM Instances shows "poise-trader"
- [ ] Status: Green dot (running)

### **✅ Bot Running:**
- [ ] SSH into VM
- [ ] Run: `screen -r poise-trader`
- [ ] See: "Bot ready! Starting main trading loop..."

### **✅ Telegram Working:**
- [ ] Received startup message
- [ ] Shows capital and mode
- [ ] Bot token and chat ID correct

### **✅ Can Close PC:**
- [ ] Close laptop
- [ ] Bot still runs (check Telegram)
- [ ] **SUCCESS!** ✅

---

## 💡 **PRO TIPS**

### **Save Even More Money:**

**1. Use Smallest VM (e2-micro)**
Already selected! ✅ This is the free tier.

**2. Stop VM When Not Trading**
```bash
# In Google Console
Compute Engine → Stop
# Restart when needed
```

**3. Set Budget Alerts**
```
Google Console → Billing → Budgets
Set alert: $10
Get email if approaching limit
```

**4. Monitor Usage**
```
Console → Billing → Reports
Track daily spending
```

### **Optimize Performance:**

**1. Choose Region Wisely**
- `us-central1`: Good for Americas
- `europe-west1`: Good for Europe
- `asia-southeast1`: Good for Asia

**2. Keep VM Updated**
```bash
sudo apt update && sudo apt upgrade -y
```

**3. Monitor Resources**
```bash
# CPU/Memory usage
htop

# Disk space
df -h
```

---

## 🆘 **TROUBLESHOOTING**

### **Can't connect to VM?**
- Check VM is running (green dot)
- Try different browser
- Check firewall rules

### **Bot not starting?**
```bash
# Check Python version
python3.11 --version

# Check dependencies
pip list

# Check for errors
python cloud_launcher.py
```

### **Out of disk space?**
```bash
# Check space
df -h

# Clean apt cache
sudo apt clean
```

### **Want to upgrade VM?**
```
Compute Engine → Edit
Change machine type → Save
Restart VM
```

---

## 📊 **USAGE CALCULATOR**

**How long will $300 last?**

```
e2-micro: $7.11/month
$300 ÷ $7.11 = 42 months

But free tier only 12 months.
After that: $7/month or stop.
```

**Typical Bot Usage:**
```
CPU: 5-10% (minimal)
RAM: 200-400 MB
Network: ~1 GB/month
Disk: 2-5 GB

e2-micro specs:
CPU: 2 cores (shared)
RAM: 1 GB
Disk: 10 GB
More than enough! ✅
```

---

## 🎉 **SUMMARY**

**Railway Issues:**
- ❌ Only $5 free (~10-20 days)
- ❌ Build errors with requirements
- ❌ Need GitHub setup

**Google Cloud:**
- ✅ $300 free (12 months!)
- ✅ No build errors
- ✅ More control
- ✅ Professional solution

**Setup:**
- ⏱️ 30 minutes one-time
- 💰 12 months FREE
- 🚀 Bot runs 24/7
- 📱 Telegram notifications
- 💻 Can close PC!

---

## 🏁 **START NOW!**

**Step-by-Step:**

```
1. Go to: console.cloud.google.com
2. Sign up (add card, not charged)
3. Create e2-micro VM
4. SSH → Install Python
5. Upload bot files
6. Run in screen session
7. Close PC and relax! ✅
```

**Time:** 30 minutes  
**Cost:** $0 for 12 months  
**Result:** Professional 24/7 trading! 🚀

---

## 📚 **ADDITIONAL HELP**

**Full Guide:** `CLOUD_DEPLOYMENT_GUIDE.md`  
**Railway Fix:** `RAILWAY_FIX.md`  
**Quick Reference:** `DEPLOYMENT_QUICK_REFERENCE.md`

---

**Google Cloud = Better Value = 12 Months FREE! 🏆**

**Questions? Just ask!** 😊
