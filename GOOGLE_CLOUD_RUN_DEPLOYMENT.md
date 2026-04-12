# GOOGLE CLOUD RUN DEPLOYMENT - COMPLETE GUIDE

## 🚀 Step-by-Step Google Cloud Run Deployment for Salevora

### What is Google Cloud Run?
- **Serverless platform** - Run containers without managing servers
- **Scale to zero** - Pay only when requests come in
- **Free tier** - 2 million requests per month FREE
- **Auto-scaling** - Handles traffic spikes automatically
- **HTTPS included** - Free SSL/TLS certificates

---

## ✅ PHASE 1: SETUP & PREREQUISITES

### Step 1: Create Google Cloud Account

**Time:** 2 minutes

1. Go to: https://cloud.google.com/free
2. Click "Start free" or "Get started for free"
3. Sign in with Google account (or create new one)
4. Accept terms and conditions
5. You get:
   - $300 free credit (3 months)
   - Always free tier (2M requests/month after $300 expires)
   - No credit card charged if you stay within free tier

**✓ Account created**

---

### Step 2: Install Google Cloud CLI

**Time:** 5 minutes

#### For Windows:

**Option A: Using PowerShell (Recommended)**
```powershell
# Open PowerShell as Administrator and run:
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& "$env:Temp\GoogleCloudSDKInstaller.exe"
```

**Option B: Manual Download**
1. Download installer: https://cloud.google.com/sdk/docs/install-gcloud
2. Run: `GoogleCloudSDKInstaller.exe`
3. Follow installer prompts
4. Accept all defaults

**Verify installation:**
```powershell
gcloud --version
# Output should show: Google Cloud SDK 123.0.0
```

#### For Linux/Mac:
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud --version
```

**✓ Google Cloud CLI installed**

---

### Step 3: Authenticate with Google Cloud

**Time:** 3 minutes

```powershell
# Run authentication
gcloud auth login

# A browser window will open
# 1. Select your Google account
# 2. Click "Allow"
# 3. Copy the verification code (if prompted)
# 4. Paste in terminal
# 5. You're authenticated!

# Verify:
gcloud auth list
# Should show your email with ✓ checkmark
```

**✓ Authenticated with Google Cloud**

---

### Step 4: Create or Select a Google Cloud Project

**Time:** 2 minutes

```powershell
# Option A: Create new project
gcloud projects create salevora-free --name="Salevora Sales Forecasting"

# Option B: List existing projects
gcloud projects list

# Set the project as active (use your project ID)
gcloud config set project salevora-free

# Verify:
gcloud config get-value project
# Output: salevora-free
```

**✓ Project created and selected**

---

### Step 5: Enable Required Google Cloud APIs

**Time:** 2 minutes

```powershell
# Enable Container Registry (for storing Docker images)
gcloud services enable containerregistry.googleapis.com

# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Verify:
gcloud services list --enabled
# Should show:
# - containerregistry.googleapis.com
# - run.googleapis.com
```

**✓ APIs enabled**

---

## 🐳 PHASE 2: BUILD DOCKER IMAGE

### Step 6: Build Docker Image

**Time:** 5-10 minutes

```powershell
# Navigate to project directory
cd "C:\Users\ankit\OneDrive\Desktop\data science project"

# Build Docker image (this uses the Dockerfile we created)
gcloud builds submit --tag gcr.io/salevora-free/salevora:latest

# What it does:
# 1. Uploads project to Google Cloud
# 2. Builds Docker image using Dockerfile
# 3. Stores in Google Container Registry
# 4. Shows progress in terminal

# Output will show:
# BUILD SUCCESSFUL
# Image pushed to: gcr.io/salevora-free/salevora:latest
```

**✓ Docker image built and stored**

---

## 🚀 PHASE 3: DEPLOY TO CLOUD RUN

### Step 7: Deploy Application

**Time:** 2-3 minutes

```powershell
# Deploy to Cloud Run
gcloud run deploy salevora `
  --image gcr.io/salevora-free/salevora:latest `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 512Mi `
  --cpu 1 `
  --timeout 3600 `
  --max-instances 100

# What each flag means:
# --image: Docker image to deploy
# --platform: Type of Cloud Run (managed is easiest)
# --region: Google Cloud region (us-central1 is free tier eligible)
# --allow-unauthenticated: Anyone can access (needed for API)
# --memory: RAM per instance (512MB is enough, can reduce to 256MB)
# --cpu: CPU allocation per instance
# --timeout: Request timeout in seconds (1 hour)
# --max-instances: Maximum concurrent instances (controls costs)
```

**Expected output:**
```
Deploying container to Cloud Run service [salevora]...
✓ Deploying...
✓ Creating Revision...
✓ Setting Traffic...
✓ Setting IAM Policy...
✓ Creating Service Account...
Done.

Service [salevora] revision [salevora-1] has been deployed and is serving 100 percent of traffic.
Service URL: https://salevora-xxxxx-uc.a.run.app
```

**✓ Application deployed!**

---

## ✅ PHASE 4: VERIFICATION

### Step 8: Verify Deployment

**Time:** 2 minutes

```powershell
# Check service status
gcloud run services list

# Output shows:
# NAME     STATUS   LAST DEPLOYED BY  LAST DEPLOYED AT  URL
# salevora ACTIVE   your@email.com    Today, 3:45 PM    https://salevora-xxxxx-uc.a.run.app

# Get detailed service info
gcloud run services describe salevora --region us-central1

# Test the API endpoint
curl https://salevora-xxxxx-uc.a.run.app/

# Expected response:
# {"status":"ok","service":"Salevora Data API"}

# Test with detailed output
curl -v https://salevora-xxxxx-uc.a.run.app/data/info
```

**✓ Deployment verified**

---

## 🌐 ACCESS YOUR APPLICATION

Your application is now live! Here are your URLs:

```
MAIN URL (Website & API):
https://salevora-xxxxx-uc.a.run.app

API ENDPOINTS:
- Health check:     https://salevora-xxxxx-uc.a.run.app/
- API Docs:         https://salevora-xxxxx-uc.a.run.app/docs
- Redoc Docs:       https://salevora-xxxxx-uc.a.run.app/redoc
- Data Info:        https://salevora-xxxxx-uc.a.run.app/data/info
- Data Download:    https://salevora-xxxxx-uc.a.run.app/data/download
- Data Sample:      https://salevora-xxxxx-uc.a.run.app/data/sample?n=10
```

**Replace xxxxx with your actual service hash from the URL**

---

## 📊 PHASE 5: MONITORING & MANAGEMENT

### Step 9: Monitor Your Application

**View logs:**
```powershell
# View recent logs
gcloud run services logs read salevora

# View in real-time
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=salevora" --limit 50 --follow
```

**View metrics:**
```powershell
# View in Google Cloud Console
# https://console.cloud.google.com/run/detail/us-central1/salevora/metrics
```

**View requests and errors:**
```powershell
# Open Google Cloud Console
gcloud compute config-ssh --project=salevora-free
```

---

### Step 10: List All Deployments

```powershell
# List all Cloud Run services
gcloud run services list --region us-central1

# Describe specific service (see all details)
gcloud run services describe salevora --region us-central1
```

---

## 🔧 MANAGING YOUR DEPLOYMENT

### Update Application (Deploy New Version)

**After you make code changes:**

```powershell
# 1. Rebuild Docker image
gcloud builds submit --tag gcr.io/salevora-free/salevora:latest

# 2. Deploy new version
gcloud run deploy salevora `
  --image gcr.io/salevora-free/salevora:latest `
  --region us-central1

# 3. View deployment progress
gcloud run services describe salevora --region us-central1
```

---

### Scale the Deployment

```powershell
# Reduce memory to save costs
gcloud run deploy salevora `
  --image gcr.io/salevora-free/salevora:latest `
  --memory 256Mi `
  --region us-central1

# Increase max instances for traffic spikes
gcloud run deploy salevora `
  --max-instances 200 `
  --region us-central1

# Set minimum instances (to reduce cold starts)
gcloud run deploy salevora `
  --min-instances 1 `
  --region us-central1
```

---

### View Usage and Costs

```powershell
# Open Google Cloud Console to see usage and billing:
# https://console.cloud.google.com/billing

# Or use gcloud to estimate:
gcloud billing accounts list
```

---

## 🎯 CLOUD RUN CONSOLE (Web UI)

Access your deployment dashboard:
https://console.cloud.google.com/run

From there you can:
- ✓ View service details
- ✓ Monitor metrics and logs
- ✓ View traffic and requests
- ✓ Scale resources
- ✓ Manage environment variables
- ✓ Set up alerts
- ✓ View costs/billing

---

## 💰 COSTS & PRICING

### Free Tier (Every Month):
- **2 million requests** - FREE
- **360,000 GB-seconds compute time** - FREE
- **1 GB egress data** - FREE

### After Free Tier Exceeds:
- **Per request:** $0.00003334 (after 2M requests)
- **Per GB-second:** $0.00000024 (after 360k GB-seconds)

### Estimated Monthly Costs:
| Monthly Requests | Estimated Cost |
|-----------------|----------------|
| 0 - 2M | $0 |
| 5M | ~$1.00 |
| 10M | ~$2.00 |
| 50M | ~$10.00 |
| 100M | ~$20.00 |

**Tip:** Set `--max-instances` to control costs (prevents runaway scaling)

---

## ❌ TROUBLESHOOTING

### Error: "Cloud Build API not enabled"
```powershell
gcloud services enable cloudbuild.googleapis.com
# Then retry deployment
```

### Error: "Permission denied"
```powershell
# Make sure you're authenticated
gcloud auth login
gcloud auth application-default login
```

### Error: "No space left on device"
```powershell
# Docker storage issue - clean up
docker system prune -a
```

### Application is slow/timing out
```powershell
# Increase timeout
gcloud run deploy salevora `
  --timeout 3600 `
  --region us-central1

# Increase memory
gcloud run deploy salevora `
  --memory 1Gi `
  --region us-central1
```

### Want to delete the service
```powershell
gcloud run services delete salevora --region us-central1
```

---

## 🔐 SECURITY & BEST PRACTICES

### Restrict Access (if needed)
```powershell
# Make service private (requires authentication)
gcloud run services add-iam-policy-binding salevora `
  --region us-central1 `
  --member=user:yourteam@company.com `
  --role=roles/run.invoker
```

### Set Environment Variables
```powershell
gcloud run deploy salevora `
  --set-env-vars="LOG_LEVEL=INFO,API_PORT=8000" `
  --region us-central1
```

### Enable VPC Connector (for database)
```powershell
# If connecting to private database
gcloud run deploy salevora `
  --vpc-connector=projects/salevora-free/locations/us-central1/connectors/connector-name `
  --region us-central1
```

---

## 📱 CUSTOM DOMAIN (Optional)

**Connect your own domain to Cloud Run:**

```powershell
# From Google Cloud Console:
# 1. Go to Cloud Run → Services → salevora
# 2. Click "Manage Custom Domains"
# 3. Add mapping:
#    Domain: salevora.yourdomain.com
#    Service: salevora
#    Region: us-central1
# 4. Copy DNS records provided
# 5. Add CNAME record in your domain registrar
# 6. Wait for verification (5-10 minutes)
```

---

## 📞 SUPPORT & RESOURCES

- **Google Cloud Run Docs:** https://cloud.google.com/run/docs
- **Pricing Calculator:** https://cloud.google.com/products/calculator
- **Cloud Console:** https://console.cloud.google.com
- **Google Cloud SDK Reference:** https://cloud.google.com/sdk/gcloud/reference

---

## ✅ DEPLOYMENT CHECKLIST

- [ ] Google Cloud account created
- [ ] Google Cloud CLI installed
- [ ] Authenticated with `gcloud auth login`
- [ ] Project created and selected
- [ ] APIs enabled (Cloud Run, Container Registry)
- [ ] Docker image built with `gcloud builds submit`
- [ ] Service deployed with `gcloud run deploy`
- [ ] Deployment verified (HTTP 200 response)
- [ ] URL tested in browser
- [ ] API endpoints responding correctly
- [ ] Logs visible in Cloud Console
- [ ] Costs monitored

---

## 🎉 YOU'RE LIVE!

Your Salevora application is now running on Google Cloud Run! 

**Your live URL is:** `https://salevora-xxxxx-uc.a.run.app`

**Next steps:**
1. Share your API URL with your team
2. Monitor costs in Cloud Console
3. Set up custom domain (optional)
4. Configure alerts for errors
5. Set up continuous deployment from GitHub

---

**Deployment Date:** April 12, 2026  
**Version:** 1.0.0  
**Status:** ✅ LIVE ON GOOGLE CLOUD RUN
