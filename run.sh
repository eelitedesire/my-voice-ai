#!/usr/bin/env bash
# Launch the server. Models load on startup (first run downloads them).
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

# Avoid Hugging Face's flaky xet CDN for any first-run model fetches.
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

echo "==> http://$HOST:$PORT   (Enroll: /enroll   Live: /live)"
exec uvicorn backend.main:app --host "$HOST" --port "$PORT" "$@"
