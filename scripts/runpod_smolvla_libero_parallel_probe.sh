#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

PROJECT_DIR="${PROJECT_DIR:-/workspace/physical-ai/physical_ai_agent}"
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/physical-ai/pip_cache}"
SMOLVLA_MODEL_ID="${SMOLVLA_MODEL_ID:-HuggingFaceVLA/smolvla_libero}"
LIBERO_TASKS="${LIBERO_TASKS:-libero_spatial}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-[0,1]}"
LIBERO_N_EPISODES="${LIBERO_N_EPISODES:-2}"
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-0}"
LIBERO_CAMERA_NAME_MAPPING="${LIBERO_CAMERA_NAME_MAPPING:-{\"agentview_image\": \"image\", \"robot0_eye_in_hand_image\": \"image2\"}}"
LIBERO_EXTRA_ARGS="${LIBERO_EXTRA_ARGS:---policy.num_steps=10 --policy.n_action_steps=1 --policy.device=cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/_workspace/runpod_results/smolvla_parallel_probe_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$OUTPUT_ROOT"

SUMMARY="$OUTPUT_ROOT/parallel_probe_summary.tsv"
REPORT="$OUTPUT_ROOT/parallel_probe_report.md"

cat > "$SUMMARY" <<'EOF'
variant	batch_size	use_async_envs	max_parallel_tasks	exit_code	elapsed_sec	overall_success	video_count	output_root
EOF

cat > "$REPORT" <<EOF
# SmolVLA LIBERO Parallel Probe

- status: running
- model_id: \`$SMOLVLA_MODEL_ID\`
- tasks: \`$LIBERO_TASKS\`
- task_ids: \`$LIBERO_TASK_IDS\`
- episodes_per_task: \`$LIBERO_N_EPISODES\`
- camera_name_mapping: \`$LIBERO_CAMERA_NAME_MAPPING\`
- policy_empty_cameras: \`$POLICY_EMPTY_CAMERAS\`
- output_root: \`$OUTPUT_ROOT\`

This probe is not a benchmark number. It compares throughput-oriented settings
on a small fixed subset before using any parallel setting for a full run.

EOF

run_variant() {
  variant="$1"
  batch_size="$2"
  use_async_envs="$3"
  max_parallel_tasks="$4"

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
  LIBERO_BATCH_SIZE="$batch_size" \
  LIBERO_USE_ASYNC_ENVS="$use_async_envs" \
  LIBERO_MAX_PARALLEL_TASKS="$max_parallel_tasks" \
  POLICY_EMPTY_CAMERAS="$POLICY_EMPTY_CAMERAS" \
  LIBERO_CAMERA_NAME_MAPPING="$LIBERO_CAMERA_NAME_MAPPING" \
  LIBERO_EXTRA_ARGS="$LIBERO_EXTRA_ARGS" \
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

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$batch_size" "$use_async_envs" "$max_parallel_tasks" \
    "$exit_code" "$elapsed_sec" "$overall_success" "$video_count" "$variant_root" \
    >> "$SUMMARY"

  {
    echo "## $variant"
    echo
    echo "- batch_size: \`$batch_size\`"
    echo "- use_async_envs: \`$use_async_envs\`"
    echo "- max_parallel_tasks: \`$max_parallel_tasks\`"
    echo "- exit_code: \`$exit_code\`"
    echo "- elapsed_sec: \`$elapsed_sec\`"
    echo "- overall_success: \`$overall_success\`"
    echo "- video_count: \`$video_count\`"
    echo "- output_root: \`$variant_root\`"
    echo
  } >> "$REPORT"
}

run_variant b1_sync_t1 1 false 1
run_variant b4_async_t1 4 true 1
run_variant b8_async_t1 8 true 1
run_variant b4_async_t2 4 true 2

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
