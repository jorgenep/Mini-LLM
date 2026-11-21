#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-root@e-ai-server}"
REMOTE_DIR="${2:-~/Mini-LLM}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

rsync -avz --progress \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='node_modules/' \
  --exclude='__pycache__/' \
  --exclude='*.py[cod]' \
  --exclude='artifacts/' \
  "$PROJECT_DIR/" "$REMOTE:$REMOTE_DIR/"

printf 'Synced to %s:%s\n' "$REMOTE" "$REMOTE_DIR"
