#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

PROJECT_DIR="${PROJECT_DIR:-/workspace/physical-ai/physical_ai_agent}"
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/physical-ai/pip_cache}"
SMOLVLA_MODEL_ID="${SMOLVLA_MODEL_ID:-lerobot/smolvla_libero}"
LANE_A_TASKS="${LANE_A_TASKS:-libero_spatial,libero_object}"
LANE_B_TASKS="${LANE_B_TASKS:-libero_goal,libero_10}"
LIBERO_N_EPISODES="${LIBERO_N_EPISODES:-10}"
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-0}"
if [ -z "${LIBERO_CAMERA_NAME_MAPPING+x}" ]; then
  LIBERO_CAMERA_NAME_MAPPING='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
fi
LIBERO_EXTRA_ARGS="${LIBERO_EXTRA_ARGS:---policy.num_steps=10 --policy.n_action_steps=10 --policy.device=cuda --seed=1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/_workspace/runpod_results/smolvla_two_lane_full_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$OUTPUT_ROOT/lane_a" "$OUTPUT_ROOT/lane_b"

cat > "$OUTPUT_ROOT/README.md" <<EOF
# SmolVLA LIBERO Two-Lane Full Evaluation

- model_id: \`$SMOLVLA_MODEL_ID\`
- lane_a_tasks: \`$LANE_A_TASKS\`
- lane_b_tasks: \`$LANE_B_TASKS\`
- episodes_per_task: \`$LIBERO_N_EPISODES\`
- lane eval settings: batch_size=1, use_async_envs=false, max_parallel_tasks=1
- policy_args: \`$LIBERO_EXTRA_ARGS\`
- camera_name_mapping: \`$LIBERO_CAMERA_NAME_MAPPING\`
- policy_empty_cameras: \`$POLICY_EMPTY_CAMERAS\`
- output_root: \`$OUTPUT_ROOT\`
EOF

run_lane() {
  lane="$1"
  tasks="$2"
  out="$OUTPUT_ROOT/$lane"
  mkdir -p "$out"
  PROJECT_DIR="$PROJECT_DIR" \
  WORK_ROOT="$WORK_ROOT" \
  PY312_VENV="$PY312_VENV" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  SKIP_BOOTSTRAP=1 \
  SMOLVLA_MODEL_ID="$SMOLVLA_MODEL_ID" \
  LIBERO_TASKS="$tasks" \
  LIBERO_N_EPISODES="$LIBERO_N_EPISODES" \
  LIBERO_BATCH_SIZE=1 \
  LIBERO_USE_ASYNC_ENVS=false \
  LIBERO_MAX_PARALLEL_TASKS=1 \
  POLICY_EMPTY_CAMERAS="$POLICY_EMPTY_CAMERAS" \
  LIBERO_CAMERA_NAME_MAPPING="$LIBERO_CAMERA_NAME_MAPPING" \
  LIBERO_EXTRA_ARGS="$LIBERO_EXTRA_ARGS" \
  OUTPUT_ROOT="$out" \
    "$SCRIPT_DIR/eval_smolvla_libero_linux.sh" > "$out.driver.log" 2>&1
}

start_epoch="$(date +%s)"
run_lane lane_a "$LANE_A_TASKS" &
pid_a="$!"
run_lane lane_b "$LANE_B_TASKS" &
pid_b="$!"

set +e
wait "$pid_a"
exit_a="$?"
wait "$pid_b"
exit_b="$?"
set -e
end_epoch="$(date +%s)"

SUMMARY="$OUTPUT_ROOT/two_lane_full_summary.tsv"
cat > "$SUMMARY" <<'EOF'
lane	tasks	exit_code	success	n_episodes	eval_s	eval_ep_s	video_count	output_root
EOF

summarize_lane() {
  lane="$1"
  tasks="$2"
  exit_code="$3"
  out="$OUTPUT_ROOT/$lane"
  success="nan"
  n_episodes="nan"
  eval_s="nan"
  eval_ep_s="nan"
  if [ -f "$out/eval_logs/eval_info.json" ]; then
    "$PY312_VENV/bin/python" - "$out/eval_logs/eval_info.json" > "$out/.summary_values" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    data = json.load(handle).get("overall", {})
print(data.get("pc_success", "nan"))
print(data.get("n_episodes", "nan"))
print(data.get("eval_s", "nan"))
print(data.get("eval_ep_s", "nan"))
PY
    success="$(sed -n '1p' "$out/.summary_values")"
    n_episodes="$(sed -n '2p' "$out/.summary_values")"
    eval_s="$(sed -n '3p' "$out/.summary_values")"
    eval_ep_s="$(sed -n '4p' "$out/.summary_values")"
  fi
  video_count="$(find "$out" -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$lane" "$tasks" "$exit_code" "$success" "$n_episodes" "$eval_s" \
    "$eval_ep_s" "$video_count" "$out" >> "$SUMMARY"
}

summarize_lane lane_a "$LANE_A_TASKS" "$exit_a"
summarize_lane lane_b "$LANE_B_TASKS" "$exit_b"

{
  echo
  echo "## Completion"
  echo
  echo "- wall_clock_sec: \`$((end_epoch - start_epoch))\`"
  echo "- lane_a_exit: \`$exit_a\`"
  echo "- lane_b_exit: \`$exit_b\`"
  echo "- summary: \`$SUMMARY\`"
} >> "$OUTPUT_ROOT/README.md"

cat "$SUMMARY"

if [ "$exit_a" -ne 0 ] || [ "$exit_b" -ne 0 ]; then
  exit 1
fi
