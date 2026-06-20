#!/usr/bin/env bash
# Configure custom domain + free HTTPS (Let's Encrypt) on Ubuntu EC2.
#
# Usage (on the server, from repo root):
#   sudo DOMAIN=salevora.com EMAIL=you@example.com bash scripts/setup-domain.sh
#
# Before running:
#   1. Buy a domain (Route 53, Namecheap, GoDaddy, etc.)
#   2. Create DNS A records → your EC2 public IP (see scripts/domain-dns.md)
#   3. Wait 5–30 min for DNS to propagate (check: dig +short YOUR_DOMAIN)

set -euo pipefail

DOMAIN="${DOMAIN:-}"
EMAIL="${EMAIL:-}"

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo DOMAIN=... EMAIL=... bash $0"
  exit 1
fi

if [[ -z "$DOMAIN" || -z "$EMAIL" ]]; then
  echo "Usage: sudo DOMAIN=yourdomain.com EMAIL=you@example.com bash $0"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$REPO_ROOT/deploy/nginx/salevora.conf"
TARGET="/etc/nginx/sites-available/salevora"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Missing template: $TEMPLATE"
  exit 1
fi

echo "==> Checking DNS for $DOMAIN ..."
RESOLVED="$(dig +short "$DOMAIN" A | tail -n1 || true)"
PUBLIC_IP="$(curl -s --max-time 5 ifconfig.me || curl -s --max-time 5 icanhazip.com || true)"
echo "    Domain resolves to: ${RESOLVED:-<none>}"
echo "    This server IP:     ${PUBLIC_IP:-<unknown>}"

if [[ -n "$RESOLVED" && -n "$PUBLIC_IP" && "$RESOLVED" != "$PUBLIC_IP" ]]; then
  echo ""
  echo "WARNING: DNS does not match this server's IP yet."
  echo "Fix your A record, wait for propagation, then re-run."
  if [[ "${SKIP_DNS_CHECK:-}" != "1" ]]; then
    echo "To force anyway: SKIP_DNS_CHECK=1 sudo DOMAIN=... EMAIL=... bash $0"
    exit 1
  fi
fi

echo "==> Writing nginx config ..."
sed "s/DOMAIN_PLACEHOLDER/$DOMAIN/g" "$TEMPLATE" > "$TARGET"
ln -sf "$TARGET" /etc/nginx/sites-enabled/salevora
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
nginx -t
systemctl reload nginx

echo "==> Installing certbot (if needed) ..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq certbot python3-certbot-nginx

echo "==> Requesting SSL certificate ..."
certbot --nginx \
  -d "$DOMAIN" \
  -d "www.$DOMAIN" \
  --non-interactive \
  --agree-tos \
  -m "$EMAIL" \
  --redirect

echo "==> Enabling cert auto-renewal ..."
systemctl enable certbot.timer 2>/dev/null || true
systemctl start certbot.timer 2>/dev/null || true

echo ""
echo "Done! Open https://$DOMAIN"
echo "Renewal test: certbot renew --dry-run"
