#!/usr/bin/env bash
# Update the deployed app after `git push`.
#
# Run as a user WITH sudo (e.g. localadmin) — NOT as the `sanuvia` service user:
#   /opt/sanuvia/my-voice-ai/deploy/deploy.sh
#
# It pulls the code + installs deps AS the `sanuvia` user (which owns the files),
# then restarts the service via sudo. You'll be asked for YOUR sudo password once.
set -euo pipefail
APP=/opt/sanuvia/my-voice-ai
SVC=sanuvia

echo "==> git pull (as $SVC)"
sudo -u "$SVC" git -C "$APP" pull --ff-only

echo "==> install/update deps (as $SVC)"
sudo -u "$SVC" "$APP/.venv/bin/pip" install -q -r "$APP/requirements.txt"

echo "==> ensure sherpa model present"
[ -f "$APP/models/sherpa-streaming-zipformer-en/tokens.txt" ] \
  || sudo -u "$SVC" bash "$APP/scripts/download_sherpa.sh"

echo "==> restart service"
sudo systemctl restart "$SVC"
sleep 2
sudo systemctl --no-pager --lines=15 status "$SVC" || true
echo "==> done — hard-refresh the browser (Ctrl-Shift-R)"
