#!/usr/bin/env bash
# Start the SHL Recommender API server.
# Usage:  bash start.sh [--port 8000]

set -euo pipefail

PORT="${1:-8000}"

# Load .env if present (robust macOS-compatible parser)
if [ -f ".env" ]; then
  while IFS='=' read -r key value; do
    # Skip comments and blank lines
    [[ "$key" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$key" ]] && continue
    # Trim whitespace
    key="${key// /}"
    value="${value%%#*}"   # strip inline comments
    value="${value%"${value##*[![:space:]]}"}"   # trim trailing space
    export "$key=$value"
  done < .env
fi

if [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo "ERROR: GOOGLE_API_KEY is not set. Add it to .env or export it."
  exit 1
fi

source venv/bin/activate 2>/dev/null || true

echo "Starting SHL Recommender on http://0.0.0.0:${PORT}"
uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
