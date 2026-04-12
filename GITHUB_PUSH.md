# PUSH TO GITHUB - SIMPLE SCRIPT

## ✅ Your System is Ready!
Git is installed: ✓

---

## 📋 BEFORE RUNNING SCRIPT

### Do This First (1 minute):

1. **Create GitHub Account (if you don't have one):**
   - Go to: https://github.com/signup
   - Fill in: username, email, password
   - Verify email
   - Done!

2. **Create Repository on GitHub:**
   - Go to: https://github.com/new
   - Repository name: `salevora`
   - Keep everything else default
   - Click "Create repository"
   - **Copy the URL** you see (should look like: `https://github.com/YOUR_USERNAME/salevora.git`)

3. **Note your GitHub username for the script below**

---

## 🚀 NOW RUN THIS SCRIPT

**Copy everything below and paste into PowerShell:**

```powershell
# ============================================================
# SALEVORA - PUSH TO GITHUB
# ============================================================

cd "C:\Users\ankit\OneDrive\Desktop\data science project"

echo "Initializing Git repository..."
git init

echo "Adding files..."
git add .

echo "Committing..."
git commit -m "Salevora - Sales Forecasting & Inventory Management System"

echo ""
echo "============================================================"
echo "PASTE YOUR GITHUB URL HERE:"
echo "Go to: https://github.com/new"
echo "Create repo 'salevora' and copy the URL"
echo "Then run this command (replace URL):"
echo ""
echo "git remote add origin https://github.com/YOUR_USERNAME/salevora.git"
echo ""
echo "Then run:"
echo "git branch -M main"
echo "git push -u origin main"
echo "============================================================"
```

---

## 📝 DETAILED STEPS

### Option 1: If you already created a GitHub repo

Replace `YOUR_USERNAME` with your GitHub username:

```powershell
cd "C:\Users\ankit\OneDrive\Desktop\data science project"

git init
git add .
git commit -m "Salevora deployment"

git remote add origin https://github.com/YOUR_USERNAME/salevora.git
git branch -M main
git push -u origin main
```

---

### Option 2: Step-by-step

```powershell
# Step 1: Navigate to project
cd "C:\Users\ankit\OneDrive\Desktop\data science project"

# Step 2: Initialize git
git init

# Step 3: Add all files
git add .

# Step 4: Create first commit
git commit -m "Initial commit - Salevora application ready"

# Step 5: Set GitHub URL (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/salevora.git

# Step 6: Set branch to main
git branch -M main

# Step 7: Push to GitHub
git push -u origin main

# That's it!
```

---

## ✅ VERIFY IT WORKED

After pushing, go to:
```
https://github.com/YOUR_USERNAME/salevora
```

You should see all your project files there!

---

## 🎯 NEXT: DEPLOY ON RENDER

Once files are on GitHub:

1. Go to: https://render.com
2. Click "Sign up" (use GitHub)
3. Click "New Web Service"
4. Connect your `salevora` repository
5. Select Free plan
6. Click "Create Web Service"
7. Wait 3-5 minutes
8. Done! You'll get a live URL! 🚀

---

## 💡 QUICK TIP

If you get an error about GitHub authentication:

```powershell
# Configure Git with your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@gmail.com"

# Then try pushing again
git push -u origin main
```

---

## 🎉 AFTER GITHUB & RENDER

Your app will be live at:
```
https://salevora-xxxxx.onrender.com
```

Share this URL! It has your full API! 🚀

---

**Ready? Let me know your GitHub username and I'll help!**
