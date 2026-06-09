#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"

PROJECT_DIR="${PROJECT_DIR:-/workspace/physical-ai/physical_ai_agent}"
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/physical-ai/pip_cache}"
SMOLVLA_MODEL_ID="${SMOLVLA_MODEL_ID:-lerobot/smolvla_libero}"
LIBERO_TASKS="${LIBERO_TASKS:-libero_10}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-[0,6,8]}"
LIBERO_N_EPISODES="${LIBERO_N_EPISODES:-2}"
BASELINE_EXTRA_ARGS="${BASELINE_EXTRA_ARGS:---policy.num_steps=10 --policy.n_action_steps=15 --policy.device=cuda --seed=1000}"
RETRY_EXTRA_ARGS="${RETRY_EXTRA_ARGS:---policy.num_steps=10 --policy.n_action_steps=15 --policy.device=cuda --seed=1000}"
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-0}"
if [ -z "${LIBERO_CAMERA_NAME_MAPPING+x}" ]; then
  LIBERO_CAMERA_NAME_MAPPING='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/_workspace/runpod_results/smolvla_agentic_retry_probe_$(date -u +%Y%m%dT%H%M%SZ)}"
if [ "$LIBERO_TASK_IDS" = "all" ]; then
  EVAL_TASK_IDS=""
  DISPLAY_TASK_IDS="all"
else
  EVAL_TASK_IDS="$LIBERO_TASK_IDS"
  DISPLAY_TASK_IDS="$LIBERO_TASK_IDS"
fi

mkdir -p "$OUTPUT_ROOT/baseline" "$OUTPUT_ROOT/retry" "$OUTPUT_ROOT/agentic"

cat > "$OUTPUT_ROOT/README.md" <<EOF
# SmolVLA LIBERO Agentic Retry Probe

- model_id: \`$SMOLVLA_MODEL_ID\`
- task_group: \`$LIBERO_TASKS\`
- initial_task_ids: \`$DISPLAY_TASK_IDS\`
- episodes_per_task: \`$LIBERO_N_EPISODES\`
- baseline_policy_args: \`$BASELINE_EXTRA_ARGS\`
- retry_policy_args: \`$RETRY_EXTRA_ARGS\`
- verifier: \`libero_benchmark_success_flag\`
- retry_policy: \`retry_failed_task_episode_index_once\`
- camera_name_mapping: \`$LIBERO_CAMERA_NAME_MAPPING\`
- policy_empty_cameras: \`$POLICY_EMPTY_CAMERAS\`
- output_root: \`$OUTPUT_ROOT\`
EOF

run_eval() {
  out="$1"
  task_ids="$2"
  extra_args="$3"
  PROJECT_DIR="$PROJECT_DIR" \
  WORK_ROOT="$WORK_ROOT" \
  PY312_VENV="$PY312_VENV" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  SKIP_BOOTSTRAP=1 \
  SMOLVLA_MODEL_ID="$SMOLVLA_MODEL_ID" \
  LIBERO_TASKS="$LIBERO_TASKS" \
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

run_eval "$OUTPUT_ROOT/baseline" "$EVAL_TASK_IDS" "$BASELINE_EXTRA_ARGS"

BASELINE_INFO="$OUTPUT_ROOT/baseline/eval_logs/eval_info.json"
PLAN_JSON="$OUTPUT_ROOT/agentic/retry_plan.json"
RETRY_TASK_IDS="$(
  PYTHONPATH="$PROJECT_DIR/src" "$PY312_VENV/bin/python" -m physical_ai_agent.agent_core.libero_agentic_retry \
    plan "$BASELINE_INFO" \
    --task-group "$LIBERO_TASKS" \
    --output-json "$PLAN_JSON" \
    --print-task-ids
)"

if [ "$RETRY_TASK_IDS" = "[]" ]; then
  cp "$BASELINE_INFO" "$OUTPUT_ROOT/retry/eval_info_no_retry_needed.json"
  PYTHONPATH="$PROJECT_DIR/src" "$PY312_VENV/bin/python" - <<PY
from pathlib import Path

root = Path("$OUTPUT_ROOT")
(root / "agentic" / "agentic_retry_metrics.json").write_text(
    '{"note": "no retry needed; baseline had no failed episodes"}\n',
    encoding="utf-8",
)
(root / "agentic" / "agentic_retry_report.md").write_text(
    "# LIBERO Agentic Retry Probe\n\nNo retry needed; baseline had no failed episodes.\n",
    encoding="utf-8",
)
PY
else
  run_eval "$OUTPUT_ROOT/retry" "$RETRY_TASK_IDS" "$RETRY_EXTRA_ARGS"
  PYTHONPATH="$PROJECT_DIR/src" "$PY312_VENV/bin/python" -m physical_ai_agent.agent_core.libero_agentic_retry \
    aggregate "$BASELINE_INFO" "$OUTPUT_ROOT/retry/eval_logs/eval_info.json" \
    --task-group "$LIBERO_TASKS" \
    --output-json "$OUTPUT_ROOT/agentic/agentic_retry_metrics.json" \
    --output-jsonl "$OUTPUT_ROOT/agentic/agentic_retry_trace.jsonl" \
    --output-md "$OUTPUT_ROOT/agentic/agentic_retry_report.md"
fi

{
  echo
  echo "## Completion"
  echo
  echo "- retry_task_ids: \`$RETRY_TASK_IDS\`"
  echo "- baseline_eval_info: \`$BASELINE_INFO\`"
  echo "- retry_plan: \`$PLAN_JSON\`"
  echo "- agentic_metrics: \`$OUTPUT_ROOT/agentic/agentic_retry_metrics.json\`"
  echo "- agentic_report: \`$OUTPUT_ROOT/agentic/agentic_retry_report.md\`"
} >> "$OUTPUT_ROOT/README.md"

cat "$OUTPUT_ROOT/README.md"
cat "$OUTPUT_ROOT/agentic/agentic_retry_report.md"
