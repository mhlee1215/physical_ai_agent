#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

BUNDLED_PYTHON="/Users/minhaeng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
PYTHON_BIN="${PHYSICAL_AI_PYTHON:-}"

if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$BUNDLED_PYTHON" ]; then
    PYTHON_BIN="$BUNDLED_PYTHON"
  else
    PYTHON_BIN="python3"
  fi
fi

if [ ! -x ".venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install "lerobot[smolvla]>=0.5.1,<0.6"
