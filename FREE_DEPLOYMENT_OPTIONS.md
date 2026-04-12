# SALEVORA - COMPLETELY FREE DEPLOYMENT OPTIONS

## 🆓 Top Free Deployment Platforms

### **Option 1: Google Cloud Run (RECOMMENDED - Most Free)**

**Why it's best:** Generous free tier, no credit card required for free tier.

**Free Tier Includes:**
- 2 million requests per month
- 360,000 GB-seconds of compute time per month
- 1 GB of outbound data per month
- Automatic scaling (0 to N instances)
- No cost when not in use (scales to zero)

**Step-by-step Deployment:**

```bash
# 1. Create Google Cloud account (free)
# https://cloud.google.com/free

# 2. Install Google Cloud CLI
# Download from: https://cloud.google.com/sdk/docs/install

# 3. Initialize and authenticate
gcloud init
gcloud auth login

# 4. Create project
gcloud projects create salevora-free --name="Salevora Project"
gcloud config set project salevora-free

# 5. Enable required APIs
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com

# 6. Build container image
gcloud builds submit --tag gcr.io/salevora-free/salevora:latest

# 7. Deploy to Cloud Run
gcloud run deploy salevora \
  --image gcr.io/salevora-free/salevora:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --set-env-vars "PYTHONUNBUFFERED=1"

# 8. Get your public URL
# Output shows: Service [salevora] revision [salevora-1] has been deployed
# Your URL: https://salevora-xxxxx-uc.a.run.app
```

**Cost:** 
- Completely FREE for first 2M requests/month
- After: $0.40 per million requests (very affordable)

---

### **Option 2: Oracle Cloud Always Free (FREE Forever)**

**Why it's good:** Truly free forever, not a trial, with compute resources.

**Free Forever Includes:**
- 2 AMD-based compute instances (1/8 OCPU, 1GB RAM each)
- 2 ARM-based compute instances (1 OCPU, 6GB RAM each) - Better!
- 200GB storage
- 10M API calls per month
- Database, Load Balancer, etc.

**Deployment Steps:**

```bash
# 1. Sign up for Oracle Cloud Free Tier
# https://www.oracle.com/cloud/free/

# 2. Create Compute Instance
# - Choose ARM-based Ampere instance
# - Select Ubuntu 20.04 LTS
# - SSH key pair: Generate or provide your own
# - Add storage

# 3. SSH into instance
ssh ubuntu@your-instance-ip

# 4. Install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3-venv python3-pip nginx

# 5. Clone/copy your project
git clone your-repo url
# or scp files over SSH

# 6. Setup application
cd salevora
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn

# 7. Create systemd service (see template below)
sudo nano /etc/systemd/system/salevora.service

# 8. Start service
sudo systemctl enable salevora
sudo systemctl start salevora

# 9. Configure Nginx
sudo nano /etc/nginx/sites-available/salevora
sudo ln -s /etc/nginx/sites-available/salevora /etc/nginx/sites-enabled/
sudo systemctl restart nginx

# 10. Access your application
# http://your-instance-public-ip
```

**Cost:** Completely FREE Forever (no credit card downgrade)

---

### **Option 3: Render (FREE with Limitations)**

**Why it's good:** Simple deployment, generous free tier, no credit card needed.

**Free Tier Includes:**
- 750 hours/month of compute (covers ~1 instance continuously)
- 100GB/month bandwidth
- Automatic deploys from GitHub
- Free SSL/TLS
- Auto-sleep after 15 mins inactivity

**Deployment Steps:**

```bash
# 1. Push code to GitHub
git push origin main

# 2. Sign up on Render
# https://render.com

# 3. Create new Web Service
# - Connect GitHub account
# - Select your repository
# - Set runtime: Python
# - Set build command: pip install -r requirements.txt
# - Set start command: gunicorn --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT api:app
# - Select Free tier when creating

# 4. Deploy
# Render auto-deploys on git push

# 5. Get your URL
# https://salevora-xxxxx.onrender.com
```

**Cost:** FREE (limited by 750 compute hours/month, enough for 1 instance)

---

### **Option 4: Railway (FREE with $5 Credit)**

**Why it's good:** Fast deployment, free tier with $5 credit.

**Free Tier Includes:**
- $5/month free credit (pays for basic deployment)
- Unlimited projects
- Auto-scaling
- Free SSL
- Environment variables

**Deployment Steps:**

```bash
# 1. Sign up on Railway
# https://railway.app

# 2. Create new project
# - Import from GitHub or
# - Connect Git repository

# 3. Add environment variables
# - Set PYTHONUNBUFFERED=1

# 4. Deploy
# Railway auto-builds and deploys

# 5. Get your URL
# https://salevora-production.up.railway.app
```

**Cost:** FREE (covered by $5 free credit monthly)

---

### **Option 5: Fly.io (FREE Tier)**

**Why it's good:** Lightweight, free tier, global deployment.

**Free Tier Includes:**
- 3 shared-cpu-1x 256MB VMs
- 3GB persistent volume storage
- Data Transfer: 100GB/month
- 160 Anycast IP addresses

**Deployment Steps:**

```bash
# 1. Install Fly CLI
# Download from: https://fly.io/docs/hands-on/install/

# 2. Create free account
fly auth signup

# 3. Create fly.toml
fly launch --no-deploy

# 4. Deploy
fly deploy

# 5. Get your URL
# https://salevora.fly.dev
```

**Cost:** FREE for small apps

---

### **Option 6: Your Own Linux Server (COMPLETELY FREE)**

**Best for:** If you own a computer/server at home or have access to one.

**What you need:**
- Old laptop/desktop with Linux
- Internet connection
- Basic networking knowledge

**Setup:**

```bash
# 1. Install Ubuntu Server 20.04 LTS (free)
# Download from: https://ubuntu.com/download/server

# 2. Install dependencies
sudo apt update
sudo apt install -y python3.11 python3-venv python3-pip nginx

# 3. Deploy application (see Oracle Cloud steps above)

# 4. Use DynamicDNS for free domain
# FreeDNS (freedns.afraid.org) - Free subdomain
# No-IP (noip.com) - Free dynamic DNS
```

**Cost:** Completely FREE (just electricity + internet)

**Limitations:**
- Your home internet must allow port 80/443
- No SSL without proxy
- Downtime if your internet goes down

---

## 📊 Comparison of FREE Options

| Platform | Best For | Free Tier | Setup Difficulty | Cost |
|----------|----------|-----------|------------------|------|
| **Google Cloud Run** | Scale-to-zero, pay-per-use | 2M req/mo | Medium | FREE → $0.40/M req |
| **Oracle Cloud Always Free** | Always-on server | ARM instance 24/7 | Hard | FREE Forever |
| **Render** | GitHub integration | 750 hrs/mo | Easy | FREE |
| **Railway** | Quick deploy | $5 credit/mo | Easy | FREE (credit covers) |
| **Fly.io** | Global reach | 3 shared VMs | Medium | FREE |
| **Home Server** | Full control | Your hardware | Hard | FREE ($electricity) |

---

## 🏆 RECOMMENDED FOR YOU: Google Cloud Run

### Why it's the BEST FREE option:

1. **Startup cost:** $0 (no credit card needed for free tier)
2. **Scaling:** Auto-scales from 0 to thousands
3. **Usage-based:** You only pay when requests come in
4. **Generous free tier:** 2M requests covers most small apps
5. **Simple deployment:** Docker-based
6. **Quick setup:** ~5 minutes

### Estimated Monthly Cost:
- **0-2M requests:** $0.00
- **10M requests:** ~$4.00
- **100M requests:** ~$35.00

---

## 🚀 FASTEST DEPLOYMENT: Render or Railway

If you want the **easiest and fastest free deployment:**

```bash
# 1. Push to GitHub
git push origin main

# 2. Sign up on Render/Railway

# 3. Click "New" → "Web Service"

# 4. Select your repository

# 5. Done! (auto-deploys when you push)
```

**Time to live:** 5 minutes

---

## 💡 MY RECOMMENDATION

### For Small Testing/Demo:
```
Render or Railway
- Simplest setup
- Free tier sufficient
- No configuration needed
```

### For Production Small App:
```
Google Cloud Run
- Most cost-efficient
- Auto-scales to zero
- Free tier generous
```

### For Always-On 24/7:
```
Oracle Cloud Always Free
- Truly free forever
- Decent compute power (ARM instance)
- No time limits
```

### For Maximum Control:
```
Home Linux Server
- Completely free
- Full customization
- Must manage yourself
```

---

## ⚡ IMMEDIATE ACTION: Google Cloud Run (Recommended)

### Setup in 10 Minutes:

```bash
# 1. Create account (free, no credit card)
# https://cloud.google.com/free

# 2. Install Google Cloud CLI
# https://cloud.google.com/sdk/docs/install

# 3. Build and deploy
gcloud auth login
gcloud projects create salevora-free --name="Salevora"
gcloud config set project salevora-free
gcloud services enable containerregistry.googleapis.com run.googleapis.com
gcloud builds submit --tag gcr.io/salevora-free/salevora:latest
gcloud run deploy salevora \
  --image gcr.io/salevora-free/salevora:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi

# 4. You're live!
# URL provided in output
```

---

## 🎁 Free Domain Names (to go with free hosting)

1. **Freenom** - Free .tk, .ml, .ga, .cf domains
   - https://www.freenom.com/

2. **Sub.domains** - Free subdomains
   - https://freedns.afraid.org/

3. **GitHub Pages** - Free docs hosting
   - https://pages.github.com/

4. **Routes free domain** - Free .routes.run
   - Included with Cloud Run

---

## 📋 FREE Deployment Checklist

- [ ] Choose platform (I recommend Google Cloud Run)
- [ ] Create account (free, no credit card)
- [ ] Install required CLI tools
- [ ] Build Docker image
- [ ] Deploy application
- [ ] Test endpoints
- [ ] Get public URL
- [ ] (Optional) Connect domain

---

## ✅ Getting Started Now

### Option A: Google Cloud Run (Recommended)
```bash
# See section above - takes 10 minutes
# Most cost-effective after free tier
```

### Option B: Render (Simplest)
```bash
# Push to GitHub → Sign up → Click deploy
# Most hands-off approach
```

### Option C: Oracle Free Tier (Best Forever)
```bash
# Create VM → SSH → Install → Deploy
# Truly free forever, but more setup
```

---

## 🆘 Need Help?

All platforms have free tiers with:
- 24/7 uptime (except Render/Railway sleep after 15 mins inactivity)
- Free SSL/HTTPS
- Automatic scaling
- Free monitoring

**Choose one and start deploying!** 🚀

---

**Created:** April 12, 2026  
**Last Updated:** April 12, 2026  
**Version:** 1.0.0
