#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

PYTHONPATH=src "$PYTHON_BIN" -B -m physical_ai_agent.checkpoints.checkpoint_25 "$@"
