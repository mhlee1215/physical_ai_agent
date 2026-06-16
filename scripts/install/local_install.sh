#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)}"
BUNDLED_PYTHON="/Users/minhaeng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
PYTHON_BIN="${PHYSICAL_AI_PYTHON:-}"
CHECKPOINT="all"

usage() {
  cat <<'EOF'
Usage: sh scripts/install/local_install.sh [--checkpoint 01|05-06|07-13|14-15|24|all]

Unified local dependency bootstrap entrypoint. Checkpoint-specific bootstrap
script names are compatibility shims only.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="${2:?missing value for --checkpoint}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$BUNDLED_PYTHON" ]; then
    PYTHON_BIN="$BUNDLED_PYTHON"
  else
    PYTHON_BIN="python3"
  fi
fi

cd "$PROJECT_DIR"

ensure_venv() {
  if [ ! -x ".venv/bin/python" ]; then
    "$PYTHON_BIN" -m venv .venv
  fi
}

install_checkpoint() {
  checkpoint="$1"
  case "$checkpoint" in
    01)
      ensure_venv
      .venv/bin/python -m pip install mujoco
      ;;
    05-06)
      ensure_venv
      .venv/bin/python -m pip install "lerobot[smolvla]>=0.5.1,<0.6"
      ;;
    07-13)
      ensure_venv
      .venv/bin/python -m pip install "so101-nexus-mujoco>=0.3.12,<0.4"
      ;;
    14-15)
      install_checkpoint 07-13
      .venv/bin/python -m pip install "lerobot[smolvla]>=0.5.1,<0.6"
      ;;
    24)
      ensure_venv
      .venv/bin/python -m pip install -e ".[maniskill,smolvla]"
      ;;
    *)
      echo "unknown checkpoint: $checkpoint" >&2
      exit 2
      ;;
  esac
}

if [ "$CHECKPOINT" = "all" ]; then
  for checkpoint in 01 05-06 07-13 14-15 24; do
    install_checkpoint "$checkpoint"
  done
else
  install_checkpoint "$CHECKPOINT"
fi
