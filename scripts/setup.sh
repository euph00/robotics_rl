#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tmp .cache/pip
export TMPDIR="$PWD/tmp"
export PIP_CACHE_DIR="$PWD/.cache/pip"
PNR_PYTHON_BIN="${PNR_PYTHON_BIN:-python3.10}"
if [ ! -x .venv/bin/python ]; then
  "$PNR_PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip freeze > requirements.lock.txt
