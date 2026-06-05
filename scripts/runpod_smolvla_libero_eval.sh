#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

PROJECT_DIR="${PROJECT_DIR:-/workspace/physical-ai/physical_ai_agent}" \
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_libero_$(date -u +%Y%m%dT%H%M%SZ)}" \
  "$SCRIPT_DIR/eval_smolvla_libero_linux.sh"
