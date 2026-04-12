# RENDER DEPLOYMENT - COMPLETE GUIDE

## 🚀 Why Render? (Best for Beginners)

✅ **No CLI installation needed** - Deploy via web browser  
✅ **Connected to GitHub** - Auto-deploy on every push  
✅ **Free tier** - 750 hours/month (covers 1 instance 24/7)  
✅ **Takes 5 minutes** - Literally faster than reading this guide  
✅ **$0 setup cost** - No credit card needed for free tier  
✅ **Easy updates** - Just push to GitHub, auto-deploys  

---

## 📋 PREREQUISITES

You need:
1. A GitHub account (free at https://github.com)
2. Your Salevora project code in a GitHub repository
3. A Render account (free at https://render.com)

**Time needed:** 15 minutes total

---

## PHASE 1: PREPARE GITHUB REPOSITORY

### Step 1: Create GitHub Repository

**If you already have a GitHub account:**

1. Go to: https://github.com/new
2. Repository name: `salevora` (or anyname you like)
3. Description: "Real-time Sales Forecasting & Inventory Management"
4. Choose "Public" or "Private" (doesn't matter)
5. Click "Create repository"

**If you're new to GitHub:**

1. Sign up: https://github.com/signup
2. Create repository (see steps above)

---

### Step 2: Push Your Project to GitHub

**From PowerShell in your project directory:**

```powershell
cd "C:\Users\ankit\OneDrive\Desktop\data science project"

# Initialize Git if not already done
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Salevora application"

# Add GitHub as remote (replace YOUR_USERNAME and repo name)
git remote add origin https://github.com/YOUR_USERNAME/salevora.git

# Push to GitHub
git branch -M main
git push -u origin main

# Verify: Go to https://github.com/YOUR_USERNAME/salevora
# You should see all your files there!
```

**✓ Code is now on GitHub**

---

## PHASE 2: CREATE RENDER ACCOUNT

### Step 3: Sign Up for Render

1. Go to: https://render.com
2. Click "Get started" or "Sign up"
3. Choose "Sign up with GitHub" (easiest)
4. Authorize Render to access your GitHub
5. Done!

**✓ Render account created**

---

## PHASE 3: DEPLOY TO RENDER

### Step 4: Create New Web Service on Render

**In Render Dashboard:**

1. Click **"New+"** button (top right)
2. Select **"Web Service"**
3. Search for your repository: `salevora`
4. Click **"Connect"**

---

### Step 5: Configure Render Deployment

**Fill in the deployment settings:**

| Setting | Value |
|---------|-------|
| **Name** | `salevora` |
| **Region** | `Oregon (US West)` |
| **Branch** | `main` |
| **Runtime** | `Docker` |
| **Plan** | `Free` ← Select this! |

**Under "Build & Deploy":**

- **Build Command:** Leave empty (uses Dockerfile)
- **Start Command:** Leave empty (uses Dockerfile)

**Click "Create Web Service"**

**Render will now:**
1. ✓ Build your Docker image (2-3 minutes)
2. ✓ Deploy the container
3. ✓ Assign a public URL
4. ✓ Start your application

---

## 🎉 DEPLOYMENT COMPLETE!

### Step 6: Get Your Live URL

**Your application is now LIVE!**

After deployment finishes (3-5 minutes), you'll see:

```
✓ Your service is live at:
https://salevora-xxxxx.onrender.com
```

---

## 🧪 TEST YOUR LIVE APPLICATION

**Test these URLs (replace xxxxx with your actual URL):**

```
Health check:   https://salevora-xxxxx.onrender.com/
API Docs:       https://salevora-xxxxx.onrender.com/docs
Data Info:      https://salevora-xxxxx.onrender.com/data/info
API Reference:  https://salevora-xxxxx.onrender.com/redoc
```

---

## 🔄 AUTO-DEPLOYMENT WITH GITHUB

### The Magic of Render + GitHub

Every time you push code to GitHub, Render **automatically redeploys**:

```powershell
# Make changes to your code
# ... edit files ...

# Commit and push to GitHub
git add .
git commit -m "Updated API endpoint"
git push origin main

# Render automatically:
# 1. Detects the push
# 2. Rebuilds the Docker image
# 3. Deploys new version
# 4. No downtime!
```

---

## 📊 RENDERING FEATURES

### View Logs

**In Render Dashboard:**

1. Click your service: `salevora`
2. Click **"Logs"** tab
3. See real-time application logs

---

### Monitor Performance

**In Render Dashboard:**

1. Click **"Metrics"** tab
2. See:
   - Request count
   - Response time
   - Error rate
   - CPU usage
   - Memory usage

---

### Check Deployment History

**In Render Dashboard:**

1. Click **"Deploys"** tab
2. See all deployments
3. Rollback to previous version if needed

---

## 💰 COSTS

### Free Tier (Perfect for You)

- **750 compute hours per month** = Covers 1 instance running 24/7
- **Unlimited bandwidth** within free tier
- **Auto-sleeping** (after 15 mins of inactivity)
  - Note: Service goes to sleep if no requests
  - First request wakes it up (5-10 seconds delay)

### If You Need Always-On (No Sleep):

```
Just add a credit card and upgrade to Starter plan:
- $7/month for always-on
- No auto-sleep
- Still very affordable
```

---

## 🔐 ENVIRONMENT VARIABLES

### Add Environment Variables (Optional)

**In Render Dashboard:**

1. Click your service: `salevora`
2. Click **"Environment"** tab
3. Click **"Add Environment Variable"**
4. Add variables:

```
LOG_LEVEL = INFO
API_WORKERS = 4
PYTHONUNBUFFERED = 1
```

---

## 🚀 ADVANCED FEATURES

### Connect Custom Domain (Optional)

**If you have a domain:**

1. In Render, click your service
2. Click **"Settings"** tab
3. Scroll to "Custom Domain"
4. Enter your domain: `api.yourdomain.com`
5. Add CNAME record in your domain registrar:
   ```
   Name: api.yourdomain.com
   Value: salevora-xxxxx.onrender.com
   ```

---

### Scale Resources (Optional)

**For better performance:**

1. Click your service
2. Click **"Settings"** tab
3. Upgrade **Plan** to Starter or higher
4. Increase **Memory** if needed

---

## 🔄 UPDATING YOUR APPLICATION

### When You Make Changes

**Method 1: Automatic (GitHub)**
```powershell
# Edit files locally
# ... make changes ...

# Push to GitHub
git add .
git commit -m "Updated feature"
git push origin main

# Render automatically redeploys!
```

**Method 2: Manual Redeploy**

1. In Render Dashboard
2. Click your service
3. Click **"Manual Deploy"** button
4. Select branch: `main`
5. Click **"Deploy latest commit"**

---

## 🆘 TROUBLESHOOTING

### Application won't start

1. Check logs: Click **"Logs"** tab
2. Look for errors in the output
3. Common fixes:
   - Missing dependency (add to requirements.txt)
   - Wrong port (should be auto-detected as 8000)
   - Environment variable needed

### Service is too slow

1. Upgrade to Starter plan ($7/month)
2. Disable auto-sleep
3. Increase memory

### Want to delete the service

1. Click **"Settings"** tab
2. Scroll to bottom
3. Click **"Delete Web Service"**

---

## 📱 FILE REQUIREMENTS

Render automatically looks for:

```
✓ Dockerfile          - We created this ✓
✓ requirements.txt    - We already have this ✓
✓ api.py              - Your main app ✓
✓ website/            - Frontend files ✓
✓ data/               - Your data files ✓
```

**Everything is ready!**

---

## ✅ DEPLOYMENT CHECKLIST

- [ ] GitHub account created
- [ ] Code pushed to GitHub
- [ ] Render account created
- [ ] New Web Service created
- [ ] Docker selected as Runtime
- [ ] Free plan selected
- [ ] Service deployed
- [ ] Live URL received
- [ ] Health check working
- [ ] API endpoints responding
- [ ] GitHub auto-deploy tested

---

## 🎯 YOUR LIVE URLS

After deployment, you'll have:

```
Main URL:           https://salevora-xxxxx.onrender.com
API Docs (Swagger): https://salevora-xxxxx.onrender.com/docs
API Docs (ReDoc):   https://salevora-xxxxx.onrender.com/redoc
Health Check:       https://salevora-xxxxx.onrender.com/
Data Info:          https://salevora-xxxxx.onrender.com/data/info
```

Share these with your team!

---

## 📚 QUICK REFERENCE

| Task | How To |
|------|--------|
| **View logs** | Dashboard → Logs tab |
| **Check metrics** | Dashboard → Metrics tab |
| **Update app** | Push to GitHub (auto-deploys) |
| **Manual deploy** | Dashboard → Manual Deploy button |
| **Scale up** | Dashboard → Settings → Upgrade plan |
| **Add domain** | Dashboard → Settings → Custom Domain |
| **Delete service** | Dashboard → Settings → Delete button |

---

## 🎉 CONGRATULATIONS!

Your Salevora application is now **live on Render**!

**Share your URL:**
- Give to clients: `https://salevora-xxxxx.onrender.com/docs`
- For API testing: Use the Swagger UI
- Monitor performance in Render Dashboard

---

## 📞 SUPPORT & RESOURCES

- **Render Docs:** https://render.com/docs
- **GitHub Help:** https://docs.github.com
- **Docker Reference:** https://docs.docker.com
- **FastAPI Docs:** https://fastapi.tiangolo.com

---

## 🔄 NEXT STEPS

1. **Push code to GitHub** (if not done)
2. **Create Render account**
3. **Connect GitHub service**
4. **Wait for deployment** (3-5 minutes)
5. **Test your live URL**
6. **Share with your team!**

---

**Render makes deployment this easy!** 🚀

---

**Deployment Date:** April 12, 2026  
**Platform:** Render  
**Status:** Ready to Deploy  
**Estimated Time:** 15 minutes total
