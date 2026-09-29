#!/usr/bin/env bash
# run.sh | Run both workflows end to end, then build the dashboard, screenshots and tests.
# Usage:  bash run.sh            (everything, ~5 minutes)
#         bash run.sh risk       (regime-risk workflow only)
#         bash run.sh factors    (factor-validation workflow only)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ -x .venv/bin/python ] && PY=".venv/bin/python"
WHAT="${1:-all}"

step() { echo; echo "=== $1"; "$PY" "$1"; }

if [[ "$WHAT" == "all" || "$WHAT" == "risk" ]]; then
  for s in regime-risk/0[1-5]_*.py; do step "$s"; done
fi
if [[ "$WHAT" == "all" || "$WHAT" == "factors" ]]; then
  for s in factor-validation/0[1-5]_*.py; do step "$s"; done
fi
if [[ "$WHAT" == "all" ]]; then
  step make_dashboard.py
  echo; echo "=== tests"; "$PY" -m pytest -q tests
fi
echo; echo "Done. Reports: reports/*.pdf · Dashboard: docs/index.html"
