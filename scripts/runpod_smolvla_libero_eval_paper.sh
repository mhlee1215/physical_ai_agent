#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

PROJECT_DIR="${PROJECT_DIR:-/workspace/physical-ai/physical_ai_agent}" \
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}" \
PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}" \
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/physical-ai/pip_cache}" \
SMOLVLA_MODEL_ID="${SMOLVLA_MODEL_ID:-HuggingFaceVLA/smolvla_libero}" \
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-0}" \
LIBERO_BATCH_SIZE="${LIBERO_BATCH_SIZE:-10}" \
LIBERO_MAX_PARALLEL_TASKS="${LIBERO_MAX_PARALLEL_TASKS:-1}" \
LIBERO_USE_ASYNC_ENVS="${LIBERO_USE_ASYNC_ENVS:-false}" \
LIBERO_EXTRA_ARGS="${LIBERO_EXTRA_ARGS:---policy.num_steps=10 --policy.n_action_steps=1 --policy.device=cuda --policy.empty_cameras=0}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_libero_paper_$(date -u +%Y%m%dT%H%M%SZ)}" \
  "$SCRIPT_DIR/eval_smolvla_libero_linux.sh"
