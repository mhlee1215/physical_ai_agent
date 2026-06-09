#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."

BUNDLED_PYTHON="/Users/minhaeng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
PYTHON_BIN="${PHYSICAL_AI_PYTHON:-}"

if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  elif [ -x "$BUNDLED_PYTHON" ]; then
    PYTHON_BIN="$BUNDLED_PYTHON"
  else
    PYTHON_BIN="python3"
  fi
fi

MODE="smoke"
for arg in "$@"; do
  if [ "$arg" = "--strict-local-sim" ]; then
    MODE="local_sim"
  fi
  if [ "$arg" = "--strict-sim-deps" ]; then
    MODE="libero_strict"
  fi
done

EVIDENCE_PATH="${CHECKPOINT_01_EVIDENCE:-_workspace/checkpoints/checkpoint_01_${MODE}.json}"

PYTHONPATH=src "$PYTHON_BIN" -B -m physical_ai_agent.checkpoints.checkpoint_01 \
  --output "$EVIDENCE_PATH" \
  "$@"
