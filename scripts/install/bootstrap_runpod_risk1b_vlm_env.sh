#!/bin/sh
set -eu

# Build a separate Risk1-B external-VLM environment.
#
# Do not mutate the canonical LeRobot/LIBERO/SmolVLA env. That env is pinned to
# torch==2.5.1+cu124 for benchmark execution. Risk1-B VLM generation only needs
# to write JSON, so it can run in its own venv with a Transformers version known
# to support Qwen2.5-VL without the torch/Transformers 5.10 float8 mismatch.

WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
PROJECT_DIR="${PROJECT_DIR:-$WORK_ROOT/physical_ai_agent}"
VLM_VENV="${VLM_VENV:-$WORK_ROOT/envs/risk1b_vlm_py312}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1+cu124}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.20.1+cu124}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.49.0}"
ACCELERATE_SPEC="${ACCELERATE_SPEC:-accelerate>=1.0,<2}"
HUGGINGFACE_HUB_SPEC="${HUGGINGFACE_HUB_SPEC:-huggingface_hub>=0.26,<1.0}"
PILLOW_SPEC="${PILLOW_SPEC:-Pillow>=10,<13}"
QWEN_VL_UTILS_SPEC="${QWEN_VL_UTILS_SPEC:-qwen-vl-utils[decord]>=0.0.8,<0.1}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_ROOT/pip_cache}"
RISK1B_VLM_HF_HOME="${RISK1B_VLM_HF_HOME:-/tmp/risk1b_vlm_hf_home}"
HF_HOME="${HF_HOME:-$RISK1B_VLM_HF_HOME}"

export PIP_CACHE_DIR
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

log() {
  printf '[risk1b-vlm-bootstrap] %s\n' "$*"
}

if [ ! -d "$VLM_VENV" ]; then
  log "creating VLM venv at $VLM_VENV"
  "$PYTHON_BIN" -m venv "$VLM_VENV"
fi

PY="$VLM_VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "VLM venv python is missing or not executable: $PY" >&2
  exit 1
fi

log "python=$PY"
log "installing base packaging tools"
"$PY" -m pip install --upgrade pip setuptools wheel

log "installing torch CUDA 12.4 stack for VLM generation"
"$PY" -m pip install --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC" "$TORCHVISION_SPEC"

log "installing Qwen/Gemma VLM runtime without touching canonical LeRobot env"
"$PY" -m pip install \
  "$TRANSFORMERS_SPEC" \
  "$ACCELERATE_SPEC" \
  "$HUGGINGFACE_HUB_SPEC" \
  "$PILLOW_SPEC" \
  "$QWEN_VL_UTILS_SPEC"

log "running Risk1-B VLM env gate"
PROJECT_DIR="$PROJECT_DIR" VLM_PYTHON_BIN="$PY" sh "$PROJECT_DIR/scripts/install/runpod_check_risk1b_vlm_env.sh"
