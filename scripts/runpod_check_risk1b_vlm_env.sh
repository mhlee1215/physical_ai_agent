#!/bin/sh
set -eu

# Hard preflight gate for the separate Risk1-B Qwen/Gemma JSON-generation env.
# This gate intentionally does not validate LIBERO/LeRobot; those stay in the
# canonical env checked by runpod_check_libero_env.sh.

PROJECT_DIR="${PROJECT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
VLM_VENV="${VLM_VENV:-$WORK_ROOT/envs/risk1b_vlm_py312}"
PYTHON_BIN="${VLM_PYTHON_BIN:-$VLM_VENV/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"

log() {
  printf '[risk1b-vlm-env-gate] %s\n' "$*"
}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "VLM env python is missing or not executable: $PYTHON_BIN" >&2
  echo "BLOCKER_CATEGORY=risk1b_vlm_env_missing" >&2
  exit 1
fi

if [ ! -f "$PROJECT_DIR/scripts/generate_risk1b_vlm_subgoals.py" ]; then
  echo "missing generator script under PROJECT_DIR=$PROJECT_DIR" >&2
  echo "BLOCKER_CATEGORY=risk1b_generator_missing" >&2
  exit 1
fi

log "python=$PYTHON_BIN"
log "model_id=$MODEL_ID"
PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" -B "$PROJECT_DIR/scripts/generate_risk1b_vlm_subgoals.py" \
  --backend transformers \
  --dependency-check-only \
  --model-id "$MODEL_ID" \
  --json

log "Risk1-B VLM env gate OK"

