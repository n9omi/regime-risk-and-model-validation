#!/usr/bin/env bash
# setup.sh | One-time setup: a virtual environment with every package, plus a headless
# browser for the PDF reports and dashboard screenshots (skipped if Chrome or Edge is installed).
# Needs Python 3.11+. To pick a specific one: PYTHON=python3.12 bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON:-python3}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip -q
.venv/bin/python -m pip install -r requirements.txt -q
if [[ ! -d "/Applications/Google Chrome.app" && ! -d "/Applications/Microsoft Edge.app" ]] \
   && ! command -v google-chrome >/dev/null 2>&1; then
  echo "No Chrome/Edge found: installing Playwright's Chromium (~150 MB, once)"
  .venv/bin/python -m playwright install chromium
fi
echo "Setup complete. Next: bash run.sh"
