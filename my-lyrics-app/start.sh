#!/usr/bin/env bash
# ============================================================
#  Apple Motion Lyrics — one-click launcher (Linux / macOS)
#  Usage:  chmod +x start.sh && ./start.sh
# ============================================================
set -e

cd "$(dirname "$0")"
APP_DIR="$(pwd)"
VENV_DIR="$APP_DIR/.venv"
PORT="${PORT:-4770}"
HOST="${HOST:-127.0.0.1}"
URL="http://$HOST:$PORT/static/index.html"
ARENA_DIR="$APP_DIR/../arenaaispofandpars"

echo ""
echo "============================================================"
echo "  Apple Motion Lyrics"
echo "============================================================"
echo ""

# ---------- 1. Find Python ----------
PYTHON=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" 2>/dev/null; then
      PYTHON="$cand"; break
    fi
  fi
done
if [ -z "$PYTHON" ]; then
  echo "[x] Python 3.10+ is required. Install it (apt install python3-venv / brew install python)."
  exit 1
fi
echo "[i] Using $PYTHON: $($PYTHON --version)"

# ---------- 2. Create venv if missing ----------
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[i] Creating virtual environment in $VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip --quiet

# ---------- 3. Install base deps ----------
echo "[i] Installing base dependencies ..."
pip install --quiet \
  "fastapi>=0.110" \
  "uvicorn[standard]>=0.27" \
  "httpx[http2]>=0.27" \
  "python-multipart>=0.0.9"

# ---------- 4. faster-whisper ----------
echo "[i] Installing faster-whisper ..."
pip install --quiet "faster-whisper>=1.0.0"

# ---------- 5. arena-api from sibling folder ----------
if [ -f "$ARENA_DIR/pyproject.toml" ]; then
  echo "[i] Installing arena-api from $ARENA_DIR ..."
  pip install --quiet -e "$ARENA_DIR[auto]"
  echo "[i] Installing Playwright Chromium ..."
  python -m playwright install chromium
else
  echo "[!] arenaaispofandpars/ not found — Arena calls will be unavailable."
fi

# ---------- 6. Folders ----------
mkdir -p "$APP_DIR/static/uploads" "$APP_DIR/static/state"

# ---------- 7. Open browser + launch ----------
echo ""
echo "============================================================"
echo "  Opening $URL"
echo "  (keep this terminal open; press Ctrl+C to stop)"
echo "============================================================"
echo ""

(
  sleep 3
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  elif command -v open >/dev/null 2>&1; then open "$URL"
  fi
) &

export HOST PORT RELOAD=0
exec python server.py
