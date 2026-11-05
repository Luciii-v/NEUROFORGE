#!/bin/bash
# NEUROFORGE Unified Studio Runner

echo "=========================================================="
echo "      NEUROFORGE // THE ALCHEMIST'S ATELIER (v2.0)        "
echo "=========================================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Check if port 8000 is occupied and clear it
PID_8000=$(lsof -ti :8000 2>/dev/null)
if [ -n "$PID_8000" ]; then
    echo "[!] Freeing port 8000 (PID(s) $PID_8000)..."
    while read -r pid; do
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done <<< "$PID_8000"
fi

echo "[*] Launching NEUROFORGE Studio Server on http://127.0.0.1:8000..."
python3 -m uvicorn server:app --port 8000 --host 0.0.0.0
