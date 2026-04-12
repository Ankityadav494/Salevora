# SALEVORA DEPLOYMENT GUIDE
## Complete Instructions for Production Deployment

---

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Local Server Deployment](#local-server-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Platform Deployment](#cloud-platform-deployment)
5. [Production Configuration](#production-configuration)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## PRE-DEPLOYMENT CHECKLIST

Before deploying, ensure you have:

- [ ] Python 3.9+ installed
- [ ] All dependencies in requirements.txt
- [ ] API source code (api.py)
- [ ] Website files (website/*)
- [ ] Data files (data/raw/sales_data.csv)
- [ ] Environment variables configured
- [ ] SSL/TLS certificates (for HTTPS)
- [ ] Domain name (if deploying to internet)
- [ ] Backup of database
- [ ] Logging configured
- [ ] Error monitoring setup (Sentry, etc.)

---

## OPTION 1: LOCAL SERVER DEPLOYMENT

### 1.1 Windows Server

**Requirements:**
- Windows Server 2016 or later
- Python 3.10+
- IIS (Internet Information Services) optional

**Step 1: Prepare the server**
```powershell
# Create project directory
mkdir "C:\Salevora"
cd "C:\Salevora"

# Copy project files
Copy-Item -Path "C:\Users\ankit\OneDrive\Desktop\data science project\*" -Destination "C:\Salevora" -Recurse
```

**Step 2: Create virtual environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**Step 3: Create startup script**
```powershell
# salevora_start.bat
@echo off
cd C:\Salevora
.\venv\Scripts\activate.bat
python api.py --host 0.0.0.0 --port 8000
```

**Step 4: Run with Windows Task Scheduler**
```
- Create Basic Task: "Salevora API"
- Trigger: At startup
- Action: Run "salevora_start.bat"
- Set to run with highest privileges
```

**Step 5: Set up reverse proxy (IIS)**
```
- Install Application Request Routing (ARR)
- Create proxy rule to forward traffic to http://localhost:8000
- Configure SSL certificate binding
```

### 1.2 Linux Server (Ubuntu/Debian)

**Requirements:**
- Ubuntu 20.04 LTS or later
- Python 3.9+
- systemd (for service management)

**Step 1: Create application user**
```bash
sudo useradd -m -s /bin/bash salevora
sudo su - salevora
```

**Step 2: Clone/copy project**
```bash
mkdir ~/salevora
cd ~/salevora
# Copy project files here
```

**Step 3: Set up virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn  # Production WSGI server
```

**Step 4: Create systemd service file**
```bash
sudo nano /etc/systemd/system/salevora.service
```

```ini
[Unit]
Description=Salevora Sales Forecasting API
After=network.target

[Service]
Type=notify
User=salevora
WorkingDirectory=/home/salevora/salevora
Environment="PATH=/home/salevora/salevora/venv/bin"
ExecStart=/home/salevora/salevora/venv/bin/gunicorn \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    api:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Step 5: Enable and start service**
```bash
sudo systemctl daemon-reload
sudo systemctl enable salevora
sudo systemctl start salevora
sudo systemctl status salevora
```

**Step 6: Set up Nginx reverse proxy**
```bash
sudo apt install nginx
sudo nano /etc/nginx/sites-available/salevora
```

```nginx
upstream salevora {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name yourdomain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://salevora;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static files (website)
    location /website/ {
        alias /home/salevora/salevora/website/;
        expires 1h;
    }
}
```

**Step 7: Enable site and restart Nginx**
```bash
sudo ln -s /etc/nginx/sites-available/salevora /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## OPTION 2: DOCKER DEPLOYMENT

### 2.1 Create Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn uvicorn[standard]

# Copy project files
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/external

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run application
CMD ["gunicorn", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "api:app"]
```

### 2.2 Create docker-compose.yml

```yaml
version: '3.8'

services:
  salevora-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: salevora-api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
      - API_HOST=0.0.0.0
      - API_PORT=8000
    restart: unless-stopped
    networks:
      - salevora-network

  nginx:
    image: nginx:alpine
    container_name: salevora-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./website:/usr/share/nginx/html:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - salevora-api
    restart: unless-stopped
    networks:
      - salevora-network

networks:
  salevora-network:
    driver: bridge
```

### 2.3 Build and run Docker

```bash
# Build image
docker build -t salevora:1.0 .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f salevora-api

# Stop containers
docker-compose down
```

### 2.4 Deploy to Docker Hub (optional)

```bash
# Tag image
docker tag salevora:1.0 yourusername/salevora:1.0

# Login to Docker Hub
docker login

# Push image
docker push yourusername/salevora:1.0
```

---

## OPTION 3: CLOUD PLATFORM DEPLOYMENT

### 3.1 Azure App Service Deployment

**Step 1: Install Azure CLI**
```bash
# Download from https://learn.microsoft.com/cli/azure/install-azure-cli
```

**Step 2: Login and create resource group**
```bash
az login
az group create --name salevora-rg --location eastus
```

**Step 3: Create App Service Plan**
```bash
az appservice plan create \
  --name salevora-plan \
  --resource-group salevora-rg \
  --sku B1 \
  --is-linux
```

**Step 4: Create Web App**
```bash
az webapp create \
  --resource-group salevora-rg \
  --plan salevora-plan \
  --name salevora-app \
  --runtime "PYTHON:3.11"
```

**Step 5: Configure deployment**
```bash
# Create deployment credentials
az webapp deployment user set \
  --user-name <deployment_username> \
  --password <deployment_password>

# Get Git clone URL
az webapp deployment source config-local-git \
  --resource-group salevora-rg \
  --name salevora-app
```

**Step 6: Deploy via Git**
```bash
cd ~/salevora
git remote add azure <git_clone_url>
git push azure main
```

**Step 7: Add startup command**
```bash
az webapp config set \
  --resource-group salevora-rg \
  --name salevora-app \
  --startup-file "gunicorn --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 api:app"
```

### 3.2 AWS Elastic Beanstalk

**Step 1: Install EB CLI**
```bash
pip install awsebcli --upgrade --user
```

**Step 2: Initialize EB Application**
```bash
cd ~/salevora
eb init -p python-3.11 salevora --region us-east-1
```

**Step 3: Create .ebextensions/python.config**
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: api:app
  aws:elasticbeanstalk:application:environment:
    PYTHONUNBUFFERED: "1"
```

**Step 4: Create environment and deploy**
```bash
eb create salevora-env --instance-type t3.micro
eb deploy
eb open  # Open in browser
```

**Step 5: Monitor application**
```bash
eb logs
eb status
```

### 3.3 Google Cloud Run

**Step 1: Create Google Cloud Project**
```bash
gcloud projects create salevora-project
gcloud config set project salevora-project
```

**Step 2: Build and push container**
```bash
# Enable required APIs
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com

# Configure Docker
gcloud auth configure-docker

# Build image
docker build -t gcr.io/salevora-project/salevora:latest .

# Push to Google Container Registry
docker push gcr.io/salevora-project/salevora:latest
```

**Step 3: Deploy to Cloud Run**
```bash
gcloud run deploy salevora \
  --image gcr.io/salevora-project/salevora:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 120 \
  --set-env-vars "API_HOST=0.0.0.0,API_PORT=8000"
```

### 3.4 Heroku Deployment

**Step 1: Install Heroku CLI**
```bash
# Download from https://devcenter.heroku.com/articles/heroku-cli
```

**Step 2: Create Procfile**
```
web: gunicorn --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT api:app
```

**Step 3: Create runtime.txt**
```
python-3.11.0
```

**Step 4: Deploy**
```bash
heroku login
heroku create salevora
git push heroku main
```

**Step 5: View logs**
```bash
heroku logs --tail
```

---

## OPTION 4: PRODUCTION CONFIGURATION

### 4.1 Environment Variables

Create `.env` file:
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false

# Database/Data
DATA_PATH=/data
BACKUP_PATH=/data/backup

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_HOURS=24

# CORS
CORS_ORIGINS=["https://yourdomain.com", "https://www.yourdomain.com"]
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=["GET", "POST", "PUT", "DELETE"]
CORS_ALLOW_HEADERS=["*"]

# Logging
LOG_LEVEL=INFO
LOG_FILE=/logs/salevora.log
LOG_MAX_BYTES=10485760  # 10MB
LOG_BACKUP_COUNT=5

# Monitoring
SENTRY_DSN=your-sentry-dsn
ENABLE_METRICS=true

# Email Notifications (for alerts)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=admin@yourdomain.com
```

### 4.2 SSL/TLS Configuration

**For Nginx:**
```bash
# Generate SSL certificate with Let's Encrypt
sudo apt install certbot python3-certbot-nginx

sudo certbot certonly --nginx -d yourdomain.com -d www.yourdomain.com

# Update Nginx config
sudo nano /etc/nginx/sites-available/salevora
```

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Rest of config...
}
```

**Auto-renew certificate:**
```bash
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

### 4.3 Logging Setup

Create `logging_config.py`:
```python
import logging
import logging.handlers
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logging():
    logger = logging.getLogger("salevora")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / "salevora.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
```

---

## OPTION 5: MONITORING & MAINTENANCE

### 5.1 Health Checks

```bash
# Check API status
curl https://yourdomain.com/

# Check data availability
curl https://yourdomain.com/data/info
```

### 5.2 Backup Strategy

```bash
#!/bin/bash
# backup.sh - Daily backup script

BACKUP_DIR="/backups/salevora"
DATA_DIR="/app/data"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup data files
tar -czf $BACKUP_DIR/data_$DATE.tar.gz $DATA_DIR

# Keep only last 7 days
find $BACKUP_DIR -name "data_*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/data_$DATE.tar.gz"
```

Schedule with cron:
```bash
0 2 * * * /usr/local/bin/backup.sh
```

### 5.3 Monitoring with Prometheus

Create `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'salevora'
    static_configs:
      - targets: ['localhost:8000']
```

### 5.4 Error Tracking with Sentry

```python
# In api.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration()]
)
```

### 5.5 Performance Monitoring

```python
# Add to api.py
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

---

## DEPLOYMENT CHECKLIST

- [ ] All dependencies installed
- [ ] Environment variables configured
- [ ] SSL/TLS certificates installed
- [ ] Database backups configured
- [ ] Logging system setup
- [ ] Monitoring configured
- [ ] CORS properly configured
- [ ] API documentation accessible
- [ ] Health checks working
- [ ] Performance tested
- [ ] Load testing completed
- [ ] Security audit passed
- [ ] Documentation updated
- [ ] Runbooks created
- [ ] On-call procedures documented

---

## QUICK START DEPLOYMENT

### For Testing (Development):
```bash
python api.py
# Open http://localhost:8000
```

### For Production (Linux):
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install gunicorn

# 3. Create systemd service (see Option 1.2)

# 4. Start service
sudo systemctl start salevora

# 5. Monitor
sudo systemctl status salevora
```

### For Production (Docker):
```bash
# 1. Build image
docker build -t salevora:1.0 .

# 2. Run
docker run -d -p 80:8000 --name salevora salevora:1.0

# 3. Monitor
docker logs -f salevora
```

---

## SUPPORT & TROUBLESHOOTING

**API won't start:**
```bash
# Check port availability
netstat -tuln | grep 8000

# Check Python installation
python --version

# Check dependencies
pip list
```

**Database errors:**
```bash
# Verify data files exist
ls -la data/

# Check file permissions
chmod 755 data/processed/
```

**Performance issues:**
```bash
# Increase workers
gunicorn --workers 8 api:app

# Monitor system resources
top
free -h
df -h
```

---

**Created:** April 12, 2026  
**Last Updated:** April 12, 2026  
**Version:** 1.0.0
