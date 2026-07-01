#!/usr/bin/env bash
# One-time setup: create a Python 3.11 venv and install dependencies.
# (torch/speechbrain/faster-whisper have no wheels for Python 3.14.)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3.11}
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: $PY not found. Install it (brew install python@3.11) or set PYTHON=..." >&2
  exit 1
fi

echo "==> Creating virtualenv (.venv) with $PY"
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "==> Installing requirements (this downloads torch etc.; may take a while)"
pip install -r requirements.txt

echo "==> Done. Start the server with:  ./run.sh"
