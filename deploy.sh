#!/bin/bash
# Linux deployment script for Salevora
# Usage: sudo bash deploy.sh

set -e

echo "=================================================="
echo "SALEVORA DEPLOYMENT SCRIPT"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_USER="salevora"
APP_HOME="/home/$APP_USER/salevora"
APP_SERVICE="salevora"
PYTHON_VERSION="python3.11"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

echo -e "${GREEN}[1/8]${NC} Creating application user..."
if ! id "$APP_USER" &>/dev/null; then
    useradd -m -s /bin/bash "$APP_USER"
    echo -e "${GREEN}✓ User created${NC}"
else
    echo -e "${YELLOW}✓ User already exists${NC}"
fi

echo -e "${GREEN}[2/8]${NC} Installing system dependencies..."
apt-get update
apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    nginx \
    certbot \
    python3-certbot-nginx \
    supervisor \
    || true
echo -e "${GREEN}✓ Dependencies installed${NC}"

echo -e "${GREEN}[3/8]${NC} Setting up application directory..."
mkdir -p "$APP_HOME"
chown "$APP_USER:$APP_USER" "$APP_HOME"
echo -e "${GREEN}✓ Directory created${NC}"

echo -e "${GREEN}[4/8]${NC} Creating Python virtual environment..."
sudo -u "$APP_USER" "$PYTHON_VERSION" -m venv "$APP_HOME/venv"
echo -e "${GREEN}✓ Virtual environment created${NC}"

echo -e "${GREEN}[5/8]${NC} Installing Python dependencies..."
sudo -u "$APP_USER" "$APP_HOME/venv/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$APP_HOME/venv/bin/pip" install -r "$APP_HOME/requirements.txt"
sudo -u "$APP_USER" "$APP_HOME/venv/bin/pip" install gunicorn
echo -e "${GREEN}✓ Dependencies installed${NC}"

echo -e "${GREEN}[6/8]${NC} Creating systemd service..."
cat > "/etc/systemd/system/${APP_SERVICE}.service" << EOF
[Unit]
Description=Salevora Sales Forecasting API
After=network.target

[Service]
Type=notify
User=$APP_USER
WorkingDirectory=$APP_HOME
Environment="PATH=$APP_HOME/venv/bin"
ExecStart=$APP_HOME/venv/bin/gunicorn \\
    --workers 4 \\
    --worker-class uvicorn.workers.UvicornWorker \\
    --bind 0.0.0.0:8000 \\
    --timeout 120 \\
    --access-logfile - \\
    --error-logfile - \\
    api:app
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable "$APP_SERVICE"
echo -e "${GREEN}✓ Service created${NC}"

echo -e "${GREEN}[7/8]${NC} Configuring Nginx..."
cat > "/etc/nginx/sites-available/salevora" << 'NGINX_EOF'
upstream salevora {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name _;

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

    location /website/ {
        alias /home/salevora/salevora/website/;
        expires 1h;
    }
}
NGINX_EOF

ls -s /etc/nginx/sites-available/salevora /etc/nginx/sites-enabled/salevora
nginx -t
systemctl reload nginx
echo -e "${GREEN}✓ Nginx configured${NC}"

echo -e "${GREEN}[8/8]${NC} Starting services..."
systemctl start "$APP_SERVICE"
systemctl status "$APP_SERVICE" --no-pager
echo -e "${GREEN}✓ Services started${NC}"

echo ""
echo -e "${GREEN}=================================================="
echo "DEPLOYMENT SUCCESSFUL!"
echo "=================================================="
echo -e "API URL: http://$(hostname -I | awk '{print $1}')"
echo -e "API Docs: http://$(hostname -I | awk '{print $1}')/docs"
echo -e "Logs: journalctl -u ${APP_SERVICE} -f"
echo -e "Config: ${APP_HOME}/config.yaml"
echo -e "${NC}"

# Optional: SSL setup
read -p "Setup SSL with Let's Encrypt? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Enter your domain (e.g., salevora.example.com):"
    read DOMAIN
    certbot --nginx -d "$DOMAIN"
fi
