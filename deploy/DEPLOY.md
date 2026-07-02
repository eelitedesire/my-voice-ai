# Deploying Sanuvia AI to buyafraction.com

This app is a **FastAPI service** (WebSocket + REST + ML models), not a static
site. So the setup is: **nginx (TLS reverse proxy) → uvicorn (systemd service)**.

> **HTTPS is mandatory.** Browsers only allow microphone access on a *secure
> context*, so the live page will not work over plain `http://`. We use Let's
> Encrypt below.

Facts for this deployment:

| Thing | Value |
|---|---|
| Domain | `buyafraction.com` (+ `www`) |
| Public IP | `YOUR_PUBLIC_IP` |
| nginx server (internal) | `YOUR_NGINX_LAN_IP` |
| App upstream | `127.0.0.1:8000` (uvicorn) |
| Repo | `github.com/eelitedesire/my-voice-ai` |

Server sizing: **Ubuntu 22.04+**, 1–2 vCPU. The service uses **~0.6 GB RAM
resident** (measured, torch + Sherpa loaded), so **2 GB RAM is comfortable** and
even ~1.9 GB works with a little swap. The tighter constraint is **disk**: the
venv (~1.5 GB) + models (~0.4 GB) + pip cache need roughly **4 GB free** — check
with `df -h` before installing.

---

## 1. DNS + network (do this first)

1. At your DNS registrar, create **A records** pointing the domain at the public IP:
   - `buyafraction.com` → `YOUR_PUBLIC_IP`
   - `www.buyafraction.com` → `YOUR_PUBLIC_IP`
2. On your router/firewall, **port-forward** to the internal nginx box:
   - TCP **80** → `YOUR_NGINX_LAN_IP:80`
   - TCP **443** → `YOUR_NGINX_LAN_IP:443`
3. Verify DNS has propagated before requesting a certificate:
   ```bash
   dig +short buyafraction.com     # should print YOUR_PUBLIC_IP
   ```

---

## 2. Server prerequisites (on YOUR_NGINX_LAN_IP)

```bash
sudo apt update
# Ubuntu's system Python (3.10) is fine — all deps have 3.10 Linux wheels.
sudo apt install -y python3 python3-venv python3-dev \
    build-essential git ffmpeg nginx certbot python3-certbot-nginx
```

Create a dedicated service user and app dir:

```bash
sudo useradd --system --create-home --home-dir /opt/sanuvia --shell /usr/sbin/nologin sanuvia
sudo mkdir -p /opt/sanuvia && sudo chown sanuvia:sanuvia /opt/sanuvia
```

---

## 3. Get the code + install (as the `sanuvia` user)

```bash
sudo -u sanuvia -H bash
cd /opt/sanuvia
git clone https://github.com/eelitedesire/my-voice-ai.git
cd my-voice-ai

# create venv + install deps (downloads torch etc. — takes a while)
PYTHON=python3 ./setup.sh

# fetch the Sherpa streaming Zipformer model
./scripts/download_sherpa.sh
exit   # back to your sudo user
```

> The `torch==2.2.2` / `sherpa-onnx==1.10.46` pins in `requirements.txt` were set
> for the macOS dev box; they also have Linux wheels, so they install as-is. On
> Linux you *may* relax them to newer versions later if you wish.

Quick smoke test (optional):
```bash
sudo -u sanuvia /opt/sanuvia/my-voice-ai/.venv/bin/uvicorn backend.main:app \
     --host 127.0.0.1 --port 8000    # Ctrl-C after "Application startup complete"
```

---

## 4. systemd service

```bash
sudo cp /opt/sanuvia/my-voice-ai/deploy/systemd/sanuvia.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sanuvia
sudo systemctl status sanuvia          # should be "active (running)"
journalctl -u sanuvia -f               # watch model loading on first boot
```

Confirm it answers locally:
```bash
curl -s http://127.0.0.1:8000/api/config | head -c 200
```

---

## 5. nginx site

```bash
sudo cp /opt/sanuvia/my-voice-ai/deploy/nginx/buyafraction.com.conf \
        /etc/nginx/sites-available/buyafraction.com
sudo ln -s /etc/nginx/sites-available/buyafraction.com /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default    # optional
sudo mkdir -p /var/www/certbot
```

The config references a cert that doesn't exist yet, so **get the certificate
first** (certbot will also wire it into the config):

```bash
sudo certbot --nginx -d buyafraction.com -d www.buyafraction.com \
     --redirect --agree-tos -m you@example.com
sudo nginx -t && sudo systemctl reload nginx
```

If `certbot --nginx` complains about the pre-written SSL lines, instead run
`sudo certbot certonly --webroot -w /var/www/certbot -d buyafraction.com -d www.buyafraction.com`
and keep the cert paths already in the conf.

---

## 6. Verify

- Open **https://buyafraction.com** — you should see the Sanuvia home page (padlock).
- Go to **Enroll**, add 2+ speakers.
- Go to **Live Session**, press **Record**, allow the mic — you should get live
  diarized transcription. (Live mic only works over HTTPS.)
- Upload a file on the same page — it should transcribe with speaker segments.

---

## 7. Updating (GitHub-based)

Push changes to GitHub, then on the server:

```bash
sudo bash /opt/sanuvia/my-voice-ai/deploy/deploy.sh
```

(That script does `git pull`, `pip install -r requirements.txt`, ensures the
Sherpa model, and `systemctl restart sanuvia`.)

---

## Notes & recommendations

- **No app-level auth.** Anyone who reaches the site can enroll voices and
  transcribe. Since it stores voiceprints, strongly consider protecting it with
  HTTP Basic Auth (commented block in the nginx conf) or an allow-list, or put it
  behind a VPN.
- **Enrollment data** lives in `data/speakers/` on the server (git-ignored — not
  in the repo). Back it up if you care about it.
- **CPU only.** Latency matches the dev box (partials ~0.5 s). A GPU host would be
  faster but needs CUDA torch/onnxruntime builds.
- **First request is slow** while models load; systemd `TimeoutStartSec=600`
  allows for that.
- **Low-RAM host (~2 GB):** it fits (~0.6 GB resident). Keep 1–2 GB swap as a
  cushion and `MALLOC_ARENA_MAX=2` (set in the service). Watch the first boot with
  `free -h` and `journalctl -u sanuvia -f`. Concurrency adds a little per live
  session, so a ~2 GB box is best for a handful of simultaneous users.
- **Firewall:** if using `ufw`, `sudo ufw allow 'Nginx Full'`.
