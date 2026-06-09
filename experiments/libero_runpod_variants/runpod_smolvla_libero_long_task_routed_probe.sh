#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"

PROJECT_DIR="${PROJECT_DIR:-/workspace/physical-ai/physical_ai_agent}"
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/physical-ai/pip_cache}"
SMOLVLA_MODEL_ID="${SMOLVLA_MODEL_ID:-lerobot/smolvla_libero}"
LIBERO_N_EPISODES="${LIBERO_N_EPISODES:-10}"
STEPS10_TASK_IDS="${STEPS10_TASK_IDS:-[4,8]}"
STEPS15_TASK_IDS="${STEPS15_TASK_IDS:-[0,1,2,3,5,6,7,9]}"
STEPS10_EXTRA_ARGS="${STEPS10_EXTRA_ARGS:---policy.num_steps=10 --policy.n_action_steps=10 --policy.device=cuda --seed=1000}"
STEPS15_EXTRA_ARGS="${STEPS15_EXTRA_ARGS:---policy.num_steps=10 --policy.n_action_steps=15 --policy.device=cuda --seed=1000}"
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-0}"
if [ -z "${LIBERO_CAMERA_NAME_MAPPING+x}" ]; then
  LIBERO_CAMERA_NAME_MAPPING='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/_workspace/runpod_results/smolvla_long_task_routed_probe_$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$OUTPUT_ROOT/steps10_tasks" "$OUTPUT_ROOT/steps15_tasks"

cat > "$OUTPUT_ROOT/README.md" <<EOF
# SmolVLA LIBERO Long Task-Routed Probe

- model_id: \`$SMOLVLA_MODEL_ID\`
- suite: \`libero_10\`
- steps10_task_ids: \`$STEPS10_TASK_IDS\`
- steps15_task_ids: \`$STEPS15_TASK_IDS\`
- episodes_per_task: \`$LIBERO_N_EPISODES\`
- lane eval settings: batch_size=1, use_async_envs=false, max_parallel_tasks=1
- steps10_policy_args: \`$STEPS10_EXTRA_ARGS\`
- steps15_policy_args: \`$STEPS15_EXTRA_ARGS\`
- camera_name_mapping: \`$LIBERO_CAMERA_NAME_MAPPING\`
- policy_empty_cameras: \`$POLICY_EMPTY_CAMERAS\`
- output_root: \`$OUTPUT_ROOT\`
EOF

run_lane() {
  lane="$1"
  task_ids="$2"
  extra_args="$3"
  out="$OUTPUT_ROOT/$lane"
  mkdir -p "$out"
  PROJECT_DIR="$PROJECT_DIR" \
  WORK_ROOT="$WORK_ROOT" \
  PY312_VENV="$PY312_VENV" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  SKIP_BOOTSTRAP=1 \
  SMOLVLA_MODEL_ID="$SMOLVLA_MODEL_ID" \
  LIBERO_TASKS=libero_10 \
  LIBERO_TASK_IDS="$task_ids" \
  LIBERO_N_EPISODES="$LIBERO_N_EPISODES" \
  LIBERO_BATCH_SIZE=1 \
  LIBERO_USE_ASYNC_ENVS=false \
  LIBERO_MAX_PARALLEL_TASKS=1 \
  POLICY_EMPTY_CAMERAS="$POLICY_EMPTY_CAMERAS" \
  LIBERO_CAMERA_NAME_MAPPING="$LIBERO_CAMERA_NAME_MAPPING" \
  LIBERO_EXTRA_ARGS="$extra_args" \
  OUTPUT_ROOT="$out" \
    "$REPO_ROOT/scripts/eval_smolvla_libero_linux.sh" > "$out.driver.log" 2>&1
}

start_epoch="$(date +%s)"
run_lane steps10_tasks "$STEPS10_TASK_IDS" "$STEPS10_EXTRA_ARGS" &
pid_steps10="$!"
run_lane steps15_tasks "$STEPS15_TASK_IDS" "$STEPS15_EXTRA_ARGS" &
pid_steps15="$!"

set +e
wait "$pid_steps10"
exit_steps10="$?"
wait "$pid_steps15"
exit_steps15="$?"
set -e
end_epoch="$(date +%s)"

SUMMARY="$OUTPUT_ROOT/long_task_routed_summary.tsv"
cat > "$SUMMARY" <<'EOF'
lane	task_ids	exit_code	success	n_episodes	eval_s	eval_ep_s	video_count	output_root
EOF

summarize_lane() {
  lane="$1"
  task_ids="$2"
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
    "$lane" "$task_ids" "$exit_code" "$success" "$n_episodes" "$eval_s" \
    "$eval_ep_s" "$video_count" "$out" >> "$SUMMARY"
}

summarize_lane steps10_tasks "$STEPS10_TASK_IDS" "$exit_steps10"
summarize_lane steps15_tasks "$STEPS15_TASK_IDS" "$exit_steps15"

{
  echo
  echo "## Completion"
  echo
  echo "- wall_clock_sec: \`$((end_epoch - start_epoch))\`"
  echo "- steps10_exit: \`$exit_steps10\`"
  echo "- steps15_exit: \`$exit_steps15\`"
  echo "- summary: \`$SUMMARY\`"
} >> "$OUTPUT_ROOT/README.md"

cat "$SUMMARY"

if [ "$exit_steps10" -ne 0 ] || [ "$exit_steps15" -ne 0 ]; then
  exit 1
fi
