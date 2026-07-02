#!/usr/bin/env bash
# Pull latest code and restart the service. Run on the server after `git push`.
#   ssh youruser@server 'sudo -u sanuvia /opt/sanuvia/my-voice-ai/deploy/deploy.sh'
set -euo pipefail
APP=/opt/sanuvia/my-voice-ai
cd "$APP"

echo "==> git pull"
git pull --ff-only

echo "==> install/update deps"
./.venv/bin/pip install -q -r requirements.txt

echo "==> ensure sherpa model present"
[ -f models/sherpa-streaming-zipformer-en/tokens.txt ] || ./scripts/download_sherpa.sh

echo "==> restart service"
sudo systemctl restart sanuvia
sleep 2
sudo systemctl --no-pager --lines=15 status sanuvia || true
echo "==> done"
