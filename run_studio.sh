#!/bin/bash
# NEUROFORGE studio: builds the React app if needed, then serves API + UI on localhost.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PORT="${NEUROFORGE_PORT:-8000}"
PYTHON="${PYTHON:-python3}"
[ -x .venv/bin/python ] && PYTHON=.venv/bin/python

if lsof -ti ":$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[!] Port $PORT is already in use (PID $(lsof -ti ":$PORT" -sTCP:LISTEN | tr '\n' ' '))."
    echo "    Stop that process or run: NEUROFORGE_PORT=8010 ./run_studio.sh"
    exit 1
fi

if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src -newer frontend/dist/index.html -print -quit)" ]; then
    echo "[*] Building the studio frontend..."
    (cd frontend && { [ -d node_modules ] || npm install; } && npm run build)
fi

echo "[*] NEUROFORGE studio on http://127.0.0.1:$PORT"
exec "$PYTHON" -m uvicorn server:app --host 127.0.0.1 --port "$PORT"
