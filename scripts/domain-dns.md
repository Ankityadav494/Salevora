# Custom domain for Salevora (EC2)

## 1. Buy a domain

Pick any registrar, for example:

| Provider | Notes |
|----------|--------|
| [AWS Route 53](https://console.aws.amazon.com/route53/) | Same AWS account as EC2 |
| [Namecheap](https://www.namecheap.com/) | Often cheap for `.com` / `.in` |
| [GoDaddy](https://www.godaddy.com/) | Common, easy DNS UI |

Good names to try: `salevora.com`, `salevora.in`, `getsalevora.com`, `salevora.app`

## 2. (Recommended) Elastic IP

Without an Elastic IP, your EC2 **public IP changes** when you stop/start the instance, and DNS breaks.

1. AWS Console → **EC2** → **Elastic IPs** → **Allocate**
2. **Actions** → **Associate** → select your `salevora` instance
3. Use this **Elastic IP** in DNS (not the old temporary IP)

Update **MongoDB Atlas** → Network Access → add the new Elastic IP if it changed.

## 3. DNS records

In your domain’s DNS panel, add:

| Type | Name / Host | Value | TTL |
|------|-------------|-------|-----|
| **A** | `@` (root) | `13.232.171.152` (or your Elastic IP) | 300 |
| **A** | `www` | same IP | 300 |

**Route 53 example:** Hosted zone → Create record → A → alias to EC2 or paste IP.

Wait 5–30 minutes. Test from your PC:

```bash
nslookup yourdomain.com
```

The IP should match your EC2 public IP.

## 4. Run setup on the server

SSH in, then:

```bash
cd ~/Salevora
git pull
sudo DOMAIN=yourdomain.com EMAIL=you@example.com bash scripts/setup-domain.sh
```

This updates nginx, installs Certbot, and enables **HTTPS** with auto-renewal.

## 5. Security group

Ensure inbound rules allow:

- **HTTP** 80 — anywhere (needed for Certbot + redirect)
- **HTTPS** 443 — anywhere

## 6. Optional — email on your domain

In **Brevo**, verify `alerts@yourdomain.com` and update on EC2:

```bash
nano ~/Salevora/.env
# BREVO_SENDER_EMAIL=alerts@yourdomain.com
sudo systemctl restart salevora
```

## 7. Test

- `https://yourdomain.com`
- Sign in, upload, Stock, Email alerts
