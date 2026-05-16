#!/usr/bin/env bash
# One-time setup: install deps, scrape catalog, build FAISS index.
# Usage:  bash setup.sh

set -euo pipefail

echo "=== SHL Recommender Setup ==="

# 1. Python environment
if [ ! -d "venv" ]; then
  echo "[1/5] Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate
echo "[2/5] Installing dependencies..."
pip install --quiet -r requirements.txt

# 2. Validate environment
if [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo ""
  echo "WARNING: GOOGLE_API_KEY is not set."
  echo "  Copy .env.example to .env and add your Gemini API key."
  echo "  Then run:  source .env && bash setup.sh"
  echo ""
fi

# 3. Scrape catalog (if catalog.json does not exist or is empty)
mkdir -p data
if [ ! -s "data/catalog.json" ]; then
  echo "[3/5] Scraping SHL catalog (this takes ~5-10 minutes)..."
  python3 scripts/fetch_all_catalog.py
else
  COUNT=$(python3 -c "import json; d=json.load(open('data/catalog.json')); print(len(d))" 2>/dev/null || echo "?")
  echo "[3/5] Catalog already exists with $COUNT assessments. Skipping scrape."
  echo "      Delete data/catalog.json and re-run to refresh."
fi

# 4. Build FAISS index
if [ ! -f "data/faiss.index" ] || [ "data/catalog.json" -nt "data/faiss.index" ]; then
  echo "[4/5] Building FAISS vector index..."
  python3 scripts/build_index.py
else
  echo "[4/5] FAISS index is up-to-date. Skipping."
fi

echo "[5/5] Setup complete!"
echo ""
echo "Run the server with:  bash start.sh"
echo "Or directly:          uvicorn app.main:app --reload"
