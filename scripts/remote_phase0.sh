#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-root@e-ai-server}"
REMOTE_DIR="${2:-~/Mini-LLM}"

ssh "$REMOTE" "cd $REMOTE_DIR && python3 scripts/environment_diagnostic.py --output artifacts/environment_report.json && python3 scripts/xpu_smoke_test.py --output artifacts/xpu_smoke_test.json"
