# 🔧 GOOGLE CLOUD CLI INSTALLATION FOR WINDOWS

## ⚠️ Current Status:
Google Cloud CLI is not installed on your system.

---

## 🚀 INSTALLATION STEPS (Windows)

### MANUAL INSTALLATION (Recommended):

**Step 1: Download the installer**
- Go to: https://cloud.google.com/sdk/docs/install-gcloud
- Click "Download and Install" for Windows
- File name: `GoogleCloudSDKInstaller.exe`

**Step 2: Run the installer**
- Double-click `GoogleCloudSDKInstaller.exe`
- Click "Next" through the wizard
- Accept all defaults
- Installer will:
  - Extract Python (included)
  - Install gcloud CLI
  - Install additional components
- Takes about 2-3 minutes

**Step 3: Restart your terminal/PowerShell**
- Close all PowerShell windows
- Open new PowerShell window
- Run:
```powershell
gcloud --version
```
Should show version number if installed correctly

---

## ⚡ QUICK INSTALL (Alternative - PowerShell)

**Run this in PowerShell as Administrator:**

```powershell
# Download and run installer
(New-Object Net.WebClient).DownloadFile(
  "https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe",
  "$env:Temp\GoogleCloudSDKInstaller.exe"
)

# Run installer
& "$env:Temp\GoogleCloudSDKInstaller.exe"
```

Then follow the installer prompts.

---

## ✅ AFTER INSTALLATION:

**1. Verify installation:**
```powershell
gcloud --version
```

**2. Authenticate:**
```powershell
gcloud auth login
```
Browser opens → Select Google account → Click "Allow" → Done!

**3. Check you logged in:**
```powershell
gcloud auth list
```
Should show your email with ✓ checkmark

---

## THEN: DEPLOY TO CLOUD RUN

Once gcloud is installed and you're authenticated, run these commands:

```powershell
# 1. Create project
gcloud projects create salevora-free --name="Salevora"
gcloud config set project salevora-free

# 2. Enable APIs
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# 3. Navigate to project
cd "C:\Users\ankit\OneDrive\Desktop\data science project"

# 4. Build Docker image
gcloud builds submit --tag gcr.io/salevora-free/salevora:latest

# 5. Deploy to Cloud Run
gcloud run deploy salevora `
  --image gcr.io/salevora-free/salevora:latest `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 512Mi

# 6. Get your live URL from output!
```

---

## 📋 WHAT TO EXPECT

**After Step 4 (Build image):**
- Takes 5-10 minutes
- Shows building progress
- Ends with: ✓ BUILD SUCCESS

**After Step 5 (Deploy):**
- Takes 2-3 minutes
- Shows deployment progress
- Ends with: ✓ Service deployed
- Shows: `Service URL: https://salevora-xxxxx.run.app`

**TADA! 🎉 You're live on Google Cloud Run!**

---

## 🧪 IMMEDIATELY AFTER DEPLOYMENT:

Test your live application:
```powershell
# Copy your actual URL from the deployment output

# Test health check:
curl https://salevora-xxxxx-uc.a.run.app/

# Test API:
curl https://salevora-xxxxx-uc.a.run.app/data/info

# View API docs in browser:
# https://salevora-xxxxx-uc.a.run.app/docs
```

---

## 🆘 If installation fails:

**Download directly from Google:**
https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe

Or download using Python:
```powershell
python -m pip install google-cloud-client-library
```

---

## 📞 NEXT STEPS:

1. **Install Google Cloud CLI** (follow steps above)
2. **Restart PowerShell**
3. **Create Google Cloud free account** (if not already done)
   - Go to: https://cloud.google.com/free
   - Click "Get started for free"
4. **Run deployment commands** from DEPLOY_NOW.md
5. **Test your live URL**
6. **Share your API URL!**

---

**Let me know once Google Cloud CLI is installed and I'll help you deploy!**
