#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

PYTHON_BIN="${PHYSICAL_AI_PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

PYTHONPATH=src "$PYTHON_BIN" -B -m physical_ai_agent.checkpoints.checkpoint_22 "$@"
