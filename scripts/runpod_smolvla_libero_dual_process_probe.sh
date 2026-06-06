#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

PROJECT_DIR="${PROJECT_DIR:-/workspace/physical-ai/physical_ai_agent}"
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/physical-ai/pip_cache}"
SMOLVLA_MODEL_ID="${SMOLVLA_MODEL_ID:-lerobot/smolvla_libero}"
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-0}"
if [ -z "${LIBERO_CAMERA_NAME_MAPPING+x}" ]; then
  LIBERO_CAMERA_NAME_MAPPING='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
fi
LIBERO_N_EPISODES="${LIBERO_N_EPISODES:-2}"
LIBERO_EXTRA_ARGS="${LIBERO_EXTRA_ARGS:---policy.num_steps=10 --policy.n_action_steps=10 --policy.device=cuda --seed=1000}"
LANE_A_TASKS="${LANE_A_TASKS:-libero_spatial}"
LANE_A_TASK_IDS="${LANE_A_TASK_IDS:-[0,1]}"
LANE_B_TASKS="${LANE_B_TASKS:-libero_object}"
LANE_B_TASK_IDS="${LANE_B_TASK_IDS:-[0,1]}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/_workspace/runpod_results/smolvla_dual_process_probe_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$OUTPUT_ROOT"

SUMMARY="$OUTPUT_ROOT/dual_process_probe_summary.tsv"
REPORT="$OUTPUT_ROOT/dual_process_probe_report.md"

cat > "$SUMMARY" <<'EOF'
phase	lane	tasks	task_ids	exit_code	elapsed_sec	overall_success	video_count	output_root
EOF

cat > "$REPORT" <<EOF
# SmolVLA LIBERO Dual-Process Probe

- status: running
- model_id: \`$SMOLVLA_MODEL_ID\`
- episodes_per_task: \`$LIBERO_N_EPISODES\`
- lane_a: \`$LANE_A_TASKS $LANE_A_TASK_IDS\`
- lane_b: \`$LANE_B_TASKS $LANE_B_TASK_IDS\`
- policy_args: \`$LIBERO_EXTRA_ARGS\`
- camera_name_mapping: \`$LIBERO_CAMERA_NAME_MAPPING\`
- policy_empty_cameras: \`$POLICY_EMPTY_CAMERAS\`
- output_root: \`$OUTPUT_ROOT\`

This probe keeps each eval at batch_size=1, sync envs, and one task worker. It
only tests whether two independent eval processes can run concurrently without
changing success semantics.

EOF

run_lane() {
  phase="$1"
  lane="$2"
  tasks="$3"
  task_ids="$4"
  out="$OUTPUT_ROOT/$phase/$lane"
  mkdir -p "$out"

  start_epoch="$(date +%s)"
  set +e
  PROJECT_DIR="$PROJECT_DIR" \
  WORK_ROOT="$WORK_ROOT" \
  PY312_VENV="$PY312_VENV" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  SKIP_BOOTSTRAP=1 \
  SMOLVLA_MODEL_ID="$SMOLVLA_MODEL_ID" \
  LIBERO_TASKS="$tasks" \
  LIBERO_TASK_IDS="$task_ids" \
  LIBERO_N_EPISODES="$LIBERO_N_EPISODES" \
  LIBERO_BATCH_SIZE=1 \
  LIBERO_USE_ASYNC_ENVS=false \
  LIBERO_MAX_PARALLEL_TASKS=1 \
  POLICY_EMPTY_CAMERAS="$POLICY_EMPTY_CAMERAS" \
  LIBERO_CAMERA_NAME_MAPPING="$LIBERO_CAMERA_NAME_MAPPING" \
  LIBERO_EXTRA_ARGS="$LIBERO_EXTRA_ARGS" \
  OUTPUT_ROOT="$out" \
    "$SCRIPT_DIR/eval_smolvla_libero_linux.sh" > "$out.driver.log" 2>&1
  exit_code="$?"
  set -e
  end_epoch="$(date +%s)"
  elapsed_sec="$((end_epoch - start_epoch))"

  overall_success="nan"
  if [ -f "$out/eval_logs/eval_info.json" ]; then
    overall_success="$("$PY312_VENV/bin/python" - "$out/eval_logs/eval_info.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle)
print(data.get("overall", {}).get("pc_success", "nan"))
PY
)"
  fi
  video_count="$(find "$out" -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$phase" "$lane" "$tasks" "$task_ids" "$exit_code" "$elapsed_sec" \
    "$overall_success" "$video_count" "$out" >> "$SUMMARY"
}

append_phase() {
  phase="$1"
  start="$2"
  end="$3"
  {
    echo "## $phase"
    echo
    echo "- wall_clock_sec: \`$((end - start))\`"
    echo
  } >> "$REPORT"
}

seq_start="$(date +%s)"
run_lane sequential lane_a "$LANE_A_TASKS" "$LANE_A_TASK_IDS"
run_lane sequential lane_b "$LANE_B_TASKS" "$LANE_B_TASK_IDS"
seq_end="$(date +%s)"
append_phase sequential "$seq_start" "$seq_end"

con_start="$(date +%s)"
run_lane concurrent lane_a "$LANE_A_TASKS" "$LANE_A_TASK_IDS" &
pid_a="$!"
run_lane concurrent lane_b "$LANE_B_TASKS" "$LANE_B_TASK_IDS" &
pid_b="$!"
wait "$pid_a"
wait "$pid_b"
con_end="$(date +%s)"
append_phase concurrent "$con_start" "$con_end"

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
