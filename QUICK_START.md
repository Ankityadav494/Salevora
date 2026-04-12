# SALEVORA DEPLOYMENT QUICK START

## 📋 Files Created

| File | Purpose |
|------|---------|
| `DEPLOYMENT_GUIDE.md` | Comprehensive deployment guide for all platforms |
| `Dockerfile` | Docker container definition |
| `docker-compose.yml` | Docker Compose orchestration |
| `nginx.conf` | Nginx reverse proxy configuration |
| `.env.example` | Environment variables template |
| `deploy.sh` | Linux automated deployment script |
| `deploy.bat` | Windows automated deployment script |

---

## 🚀 QUICK START OPTIONS

### Option A: Docker (Recommended for Production)

```bash
# 1. Build and run with Docker Compose
docker-compose up -d

# 2. Check status
docker-compose ps
docker-compose logs -f salevora-api

# 3. Access application
# API: http://localhost:8000
# Website: http://localhost
# Docs: http://localhost/docs

# 4. Stop services
docker-compose down
```

### Option B: Linux (Ubuntu/Debian)

```bash
# 1. Run deployment script
sudo bash deploy.sh

# 2. Check service status
sudo systemctl status salevora

# 3. View logs
sudo journalctl -u salevora -f

# 4. Access application
# http://your-server-ip/

# 5. Setup SSL (Let's Encrypt)
sudo certbot --nginx -d yourdomain.com
```

### Option C: Windows

```bash
# 1. Run as Administrator
deploy.bat

# 2. Start manually (or Task Scheduler will auto-start)
C:\Salevora\start.bat

# 3. Access application
# http://localhost:8000
```

### Option D: Manual Linux Deployment

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install gunicorn

# 3. Run with Gunicorn
gunicorn \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  api:app

# 4. In another terminal, setup Nginx reverse proxy
# (See nginx.conf for configuration)
```

---

## ☁️ CLOUD DEPLOYMENT QUICK LINKS

### Azure App Service
```bash
az webapp create --name salevora --resource-group rg --plan plan
```
→ See DEPLOYMENT_GUIDE.md section 3.1

### AWS Elastic Beanstalk
```bash
eb init salevora -p python-3.11
eb create salevora-env
eb deploy
```
→ See DEPLOYMENT_GUIDE.md section 3.2

### Google Cloud Run
```bash
gcloud run deploy salevora --image gcr.io/project/salevora:latest
```
→ See DEPLOYMENT_GUIDE.md section 3.3

### Heroku
```bash
heroku create salevora
git push heroku main
```
→ See DEPLOYMENT_GUIDE.md section 3.4

---

## 🔧 ENVIRONMENT SETUP

1. **Copy environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your settings:**
   ```bash
   nano .env
   ```

3. **Key variables to update:**
   - `SECRET_KEY` - Change to a secure random string
   - `CORS_ORIGINS` - Add your domain
   - `SMTP_*` - Configure if using email alerts
   - `SENTRY_DSN` - Add if using error tracking

---

## 🔒 SECURITY SETUP

### SSL/TLS Certificate
```bash
# Linux with Nginx
sudo certbot --nginx -d yourdomain.com

# Docker
# Copy certificates to ./certs/ directory

# Azure/AWS/GCP
# Configure through provider's console
```

### Firewall Rules
```bash
# Allow HTTP/HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Only on specific IP (optional)
sudo ufw allow from 203.0.113.0 to any port 22
```

### HTTPS Configuration (Nginx)
Already included in `nginx.conf` - uncomment SSL section and add certificate paths

---

## 📊 MONITORING & LOGS

### Docker
```bash
# View logs
docker-compose logs -f salevora-api

# Check container stats
docker stats salevora-api

# Access shell
docker exec -it salevora-api /bin/bash
```

### Linux Service
```bash
# View logs
sudo journalctl -u salevora -f

# Check status
sudo systemctl status salevora

# Enable on boot
sudo systemctl enable salevora

# Restart service
sudo systemctl restart salevora
```

### Windows Task Scheduler
```bash
tasklist | findstr python
taskkill /F /IM python.exe

# Check scheduled tasks
schtasks /query /tn "Salevora API"

# Delete task if needed
schtasks /delete /tn "Salevora API" /f
```

---

## 🧪 HEALTH CHECKS

```bash
# Check API is running
curl http://localhost:8000/

# Get dataset info
curl http://localhost:8000/data/info

# Check API documentation
# Browser: http://localhost:8000/docs

# WebSocket connection test
# Browser console: new WebSocket('ws://localhost:8000/ws/inventory')
```

---

## 📈 PERFORMANCE TUNING

### Docker
```yaml
# In docker-compose.yml
services:
  salevora-api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### Gunicorn Workers
```bash
# Calculate: (2 × CPU cores) + 1
# Example: 4 cores = (2 × 4) + 1 = 9 workers
gunicorn --workers 9 api:app
```

### Nginx Optimization
```nginx
# In nginx.conf
worker_processes auto;
worker_connections 2048;
```

---

## 🐛 TROUBLESHOOTING

### Port Already in Use
```bash
# Linux/Mac
sudo lsof -i :8000
kill -9 <PID>

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Database Connection Errors
```bash
# Check data files exist
ls -la data/processed/

# Fix permissions
chmod 755 data/processed/
```

### SSL Certificate Issues
```bash
# Check certificate validity
openssl x509 -in /path/to/cert.pem -text -noout

# Check expiration
certbot certificates

# Renew (for Let's Encrypt)
sudo certbot renew --dry-run
```

### API Connection Timeout
```bash
# Increase Gunicorn timeout
gunicorn --timeout 120 api:app

# Check Nginx timeout (in nginx.conf)
proxy_connect_timeout 60s;
proxy_read_timeout 60s;
```

---

## 📚 ADDITIONAL RESOURCES

- **API Documentation:** http://yourdomain.com/docs
- **Nginx Docs:** https://nginx.org/en/docs/
- **Docker Docs:** https://docs.docker.com/
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Let's Encrypt:** https://letsencrypt.org/getting-started/

---

## 🎯 DEPLOYMENT CHECKLIST

- [ ] Environment variables configured (`.env`)
- [ ] SSL/TLS certificate obtained
- [ ] Firewall rules configured
- [ ] Health checks passing
- [ ] Logs being collected
- [ ] Backups configured
- [ ] Monitoring setup
- [ ] Load testing completed
- [ ] Failover tested
- [ ] Documentation updated
- [ ] Team trained
- [ ] On-call rotation established

---

## ✅ AFTER DEPLOYMENT

1. **Monitor the application:**
   ```bash
   # Check logs regularly
   docker-compose logs -f
   # or
   journalctl -u salevora -f
   ```

2. **Set up automated backups:**
   ```bash
   # See DEPLOYMENT_GUIDE.md for backup scripts
   ```

3. **Configure monitoring:**
   - Set up alerts (Sentry, Datadog, CloudWatch)
   - Monitor CPU, memory, disk usage
   - Track API response times

4. **Update documentation:**
   - Document API changes
   - Update runbooks
   - Document custom configurations

5. **Communicate status:**
   - Update team
   - Set up status page
   - Document known issues

---

## 🆘 SUPPORT

For detailed deployment instructions for your platform:
- Read `DEPLOYMENT_GUIDE.md` (comprehensive guide)
- Check platform-specific documentation
- Review logs in case of errors
- Test health endpoints regularly

**Last Updated:** April 12, 2026  
**Version:** 1.0.0  
**Status:** Ready for Production
