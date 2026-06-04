#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PHYSICAL_AI_PYTHON:-python3}"
fi

PYTHONPATH=src "$PYTHON_BIN" -B -m physical_ai_agent.sim.so101_live_viewer "$@"
