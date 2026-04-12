# GOOGLE CLOUD RUN - QUICK DEPLOYMENT SCRIPT

## 🚀 Copy and Paste These Commands (Windows PowerShell)

Run these commands one at a time in PowerShell:

### ✅ STEP 1: Check if gcloud is installed
```powershell
gcloud --version
```

**If it says "gcloud command not found", download from:**
https://cloud.google.com/sdk/docs/install

---

### ✅ STEP 2: Authenticate with Google
```powershell
gcloud auth login
```
This opens your browser. Select your Google account and click "Allow".

---

### ✅ STEP 3: Create a new Google Cloud project
```powershell
gcloud projects create salevora-free --name="Salevora"
gcloud config set project salevora-free
```

---

### ✅ STEP 4: Enable the APIs we need
```powershell
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

Wait for each to complete (shows "done" in green).

---

### ✅ STEP 5: Navigate to your project
```powershell
cd "C:\Users\ankit\OneDrive\Desktop\data science project"
```

---

### ✅ STEP 6: Build the Docker image
```powershell
gcloud builds submit --tag gcr.io/salevora-free/salevora:latest
```

**This will take 5-10 minutes. Wait for it to show:**
```
BUILD SUCCESS
```

---

### ✅ STEP 7: Deploy to Cloud Run (FINALLY!)
```powershell
gcloud run deploy salevora `
  --image gcr.io/salevora-free/salevora:latest `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 512Mi
```

---

## 🎉 WHEN IT'S DONE:

You'll see something like:

```
Service [salevora] revision [salevora-1] has been deployed and is serving 100 percent of traffic.
Service URL: https://salevora-xxxxx-uc.a.run.app
```

---

## 🧪 Test Your Live Application:

```powershell
# Replace xxxxx with your actual URL hash

# Get your URL:
gcloud run services list

# Test health check:
curl https://salevora-xxxxx-uc.a.run.app/

# Test API:
curl https://salevora-xxxxx-uc.a.run.app/data/info

# Open in browser:
# https://salevora-xxxxx-uc.a.run.app/docs
```

---

## 💰 FREE TIER REMINDER:

✅ 2 million requests per month = FREE
✅ No credit card charged if you stay within free tier
✅ After: $0.40 per million requests

---

**That's it! Your app is live!** 🚀
