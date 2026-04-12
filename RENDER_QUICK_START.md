# RENDER DEPLOYMENT - QUICK START (5 MINUTES)

## 🚀 Your Exact Steps (Copy & Paste)

### STEP 1: Push Code to GitHub (2 minutes)

**Open PowerShell and run these commands:**

```powershell
# Navigate to project
cd "C:\Users\ankit\OneDrive\Desktop\data science project"

# Check if Git is installed
git --version

# If error, download from: https://git-scm.com/download/win
```

**If you DON'T have a GitHub repository yet, create one:**

1. Go to: https://github.com/new
2. Repository name: `salevora`
3. Click "Create repository"
4. Copy the URL (looks like: `https://github.com/YOUR_USERNAME/salevora.git`)

**Then run (replace YOUR_USERNAME):**

```powershell
# Initialize Git
git init

# Add files
git add .

# Commit
git commit -m "Salevora application - ready to deploy"

# Add GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/salevora.git

# Push to GitHub
git branch -M main
git push -u origin main

# Done! Check: https://github.com/YOUR_USERNAME/salevora
```

**✓ Your code is now on GitHub**

---

### STEP 2: Sign Up for Render (1 minute)

1. Go to: https://render.com
2. Click "Sign up"
3. Choose "Continue with GitHub" (easiest!)
4. Click "Authorize"

**✓ Render account ready**

---

### STEP 3: Deploy on Render (2 minutes)

**In Render Dashboard:**

1. Click **"New +"** (top right)
2. Select **"Web Service"**
3. Search for your repo: `salevora`
4. Click **"Connect"**

**On deployment page:**

```
Name:        salevora
Region:      Oregon (US West)
Branch:      main
Runtime:     Docker
Plan:        Free ← Click this!
```

5. Click **"Create Web Service"**
6. **Wait 3-5 minutes** for deployment

**That's it!** 🎉

---

## ✅ AFTER DEPLOYMENT

You'll get a live URL like:
```
https://salevora-xxxxx.onrender.com
```

**Test it:**
```
https://salevora-xxxxx.onrender.com/docs
```

Should show API documentation!

---

## 🎯 WHAT YOU GET

✓ Live API endpoint  
✓ Auto-HTTPS/SSL  
✓ Auto-scaling  
✓ FREE tier (750 hours/month)  
✓ Auto-deploy on git push  

---

## 💰 COST

**FREE** (forever, as long as under 750 hours/month)

After: $7/month if you want always-on

---

## 🔄 AUTO-DEPLOY MAGIC

Once deployed, whenever you push to GitHub:

```powershell
git add .
git commit -m "Your changes"
git push origin main

# Render automatically redeploys! ✨
```

---

**That's literally all you need to do!**

Your app will be live in 5 minutes! 🚀
