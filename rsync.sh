#!/usr/bin/env bash
set -euo pipefail

rsync -av --progress \
  --exclude='.git' \
  --exclude='__pycache__/' \
  --exclude='.venv/' \
  --exclude='*.py[cod]' \
  --exclude='data/' \
  dlisuser@20.57.218.73:/home/dlisuser/quanwen/SHADOW-250M-A55/ \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"
