#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/eval-harness-pycache
python3 -m compileall -q eval_harness tests
python3 -m unittest discover -v
