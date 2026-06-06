#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

PROJECT_DIR="${PROJECT_DIR:-/workspace/physical-ai/physical_ai_agent}"
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/physical-ai/pip_cache}"
SMOLVLA_MODEL_ID="${SMOLVLA_MODEL_ID:-lerobot/smolvla_libero}"
LIBERO_TASKS="${LIBERO_TASKS:-libero_10}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-[0,1,6,7,8]}"
LIBERO_N_EPISODES="${LIBERO_N_EPISODES:-5}"
LIBERO_BATCH_SIZE="${LIBERO_BATCH_SIZE:-1}"
LIBERO_USE_ASYNC_ENVS="${LIBERO_USE_ASYNC_ENVS:-false}"
LIBERO_MAX_PARALLEL_TASKS="${LIBERO_MAX_PARALLEL_TASKS:-1}"
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-0}"
if [ -z "${LIBERO_CAMERA_NAME_MAPPING+x}" ]; then
  LIBERO_CAMERA_NAME_MAPPING='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/_workspace/runpod_results/smolvla_protocol_sweep_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$OUTPUT_ROOT"

SUMMARY="$OUTPUT_ROOT/protocol_sweep_summary.tsv"
REPORT="$OUTPUT_ROOT/protocol_sweep_report.md"

cat > "$SUMMARY" <<'EOF'
variant	seed	n_action_steps	exit_code	elapsed_sec	overall_success	video_count	output_root
EOF

cat > "$REPORT" <<EOF
# SmolVLA LIBERO Protocol Sweep

- status: running
- model_id: \`$SMOLVLA_MODEL_ID\`
- tasks: \`$LIBERO_TASKS\`
- task_ids: \`$LIBERO_TASK_IDS\`
- episodes_per_task: \`$LIBERO_N_EPISODES\`
- batch_size: \`$LIBERO_BATCH_SIZE\`
- use_async_envs: \`$LIBERO_USE_ASYNC_ENVS\`
- max_parallel_tasks: \`$LIBERO_MAX_PARALLEL_TASKS\`
- camera_name_mapping: \`$LIBERO_CAMERA_NAME_MAPPING\`
- policy_empty_cameras: \`$POLICY_EMPTY_CAMERAS\`
- output_root: \`$OUTPUT_ROOT\`

This is a protocol-debug subset, not a paper-comparable full result.

EOF

run_one() {
  variant="$1"
  seed="$2"
  n_action_steps="$3"
  variant_root="$OUTPUT_ROOT/$variant"
  mkdir -p "$variant_root"

  start_epoch="$(date +%s)"
  set +e
  PROJECT_DIR="$PROJECT_DIR" \
  WORK_ROOT="$WORK_ROOT" \
  PY312_VENV="$PY312_VENV" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  SKIP_BOOTSTRAP=1 \
  SMOLVLA_MODEL_ID="$SMOLVLA_MODEL_ID" \
  LIBERO_TASKS="$LIBERO_TASKS" \
  LIBERO_TASK_IDS="$LIBERO_TASK_IDS" \
  LIBERO_N_EPISODES="$LIBERO_N_EPISODES" \
  LIBERO_BATCH_SIZE="$LIBERO_BATCH_SIZE" \
  LIBERO_USE_ASYNC_ENVS="$LIBERO_USE_ASYNC_ENVS" \
  LIBERO_MAX_PARALLEL_TASKS="$LIBERO_MAX_PARALLEL_TASKS" \
  POLICY_EMPTY_CAMERAS="$POLICY_EMPTY_CAMERAS" \
  LIBERO_CAMERA_NAME_MAPPING="$LIBERO_CAMERA_NAME_MAPPING" \
  LIBERO_EXTRA_ARGS="--policy.num_steps=10 --policy.n_action_steps=$n_action_steps --policy.device=cuda --seed=$seed" \
  OUTPUT_ROOT="$variant_root" \
    "$SCRIPT_DIR/eval_smolvla_libero_linux.sh" > "$variant_root.driver.log" 2>&1
  exit_code="$?"
  set -e
  end_epoch="$(date +%s)"
  elapsed_sec="$((end_epoch - start_epoch))"

  overall_success="nan"
  if [ -f "$variant_root/eval_logs/eval_info.json" ]; then
    overall_success="$("$PY312_VENV/bin/python" - "$variant_root/eval_logs/eval_info.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
print(data.get("overall", {}).get("pc_success", "nan"))
PY
)"
  fi
  video_count="$(find "$variant_root" -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$seed" "$n_action_steps" "$exit_code" "$elapsed_sec" \
    "$overall_success" "$video_count" "$variant_root" >> "$SUMMARY"

  {
    echo "## $variant"
    echo
    echo "- seed: \`$seed\`"
    echo "- n_action_steps: \`$n_action_steps\`"
    echo "- exit_code: \`$exit_code\`"
    echo "- elapsed_sec: \`$elapsed_sec\`"
    echo "- overall_success: \`$overall_success\`"
    echo "- video_count: \`$video_count\`"
    echo "- output_root: \`$variant_root\`"
    echo
  } >> "$REPORT"
}

run_one seed1000_steps1 1000 1
run_one seed0_steps1 0 1
run_one seed10000_steps1 10000 1
run_one seed1000_steps10 1000 10
run_one seed1000_steps50 1000 50

{
  echo
  echo "## Summary"
  echo
  echo "\`\`\`tsv"
  cat "$SUMMARY"
  echo "\`\`\`"
  echo
  echo "- status: completed"
} >> "$REPORT"

echo "summary=$SUMMARY"
echo "report=$REPORT"
