#!/bin/sh
set -eu

# Write LIBERO config.yaml before any Python code calls libero.libero.get_libero_path().
# Without this, LIBERO can prompt for dataset paths and crash non-interactive
# RunPod jobs with EOFError.

WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
PROJECT_DIR="${PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}"
PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
PYTHON_BIN="${PYTHON_BIN:-$PY312_VENV/bin/python}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_CONFIG_DIR:-$HOME/.libero}}"
LIBERO_ASSETS_DIR="${LIBERO_ASSETS_DIR:-$WORK_ROOT/libero_assets}"
LIBERO_PACKAGE_DIR="${LIBERO_PACKAGE_DIR:-}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "venv python is missing or not executable: $PYTHON_BIN" >&2
  echo "BLOCKER_CATEGORY=volume_path_mismatch" >&2
  exit 1
fi

PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
LIBERO_CONFIG_DIR="$LIBERO_CONFIG_PATH" \
LIBERO_ASSETS_DIR="$LIBERO_ASSETS_DIR" \
LIBERO_PACKAGE_DIR="$LIBERO_PACKAGE_DIR" \
"$PYTHON_BIN" -m physical_ai_agent.imagine_then_act.libero_config \
  --config-dir "$LIBERO_CONFIG_PATH" \
  --assets-dir "$LIBERO_ASSETS_DIR" \
  --libero-package-dir "$LIBERO_PACKAGE_DIR" \
  --json
