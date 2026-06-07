#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

PROJECT_DIR="${PROJECT_DIR:-/workspace/physical-ai/physical_ai_agent}"
WORK_ROOT="${WORK_ROOT:-/workspace/physical-ai}"
PY312_VENV="${PY312_VENV:-/root/physical-ai/envs/lerobot_py312}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/physical-ai/pip_cache}"
SMOLVLA_MODEL_ID="${SMOLVLA_MODEL_ID:-lerobot/smolvla_libero}"
LIBERO_TASKS="${LIBERO_TASKS:-libero_10}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-all}"
LIBERO_N_EPISODES="${LIBERO_N_EPISODES:-10}"
BASE_SEEDS="${BASE_SEEDS:-1000 1001 1002}"
BASELINE_ACTION_STEPS="${BASELINE_ACTION_STEPS:-15}"
BLIND_RETRY_ACTION_STEPS="${BLIND_RETRY_ACTION_STEPS:-15}"
ALTERNATE_RETRY_ACTION_STEPS="${ALTERNATE_RETRY_ACTION_STEPS:-10}"
RETRY_SEED_OFFSET="${RETRY_SEED_OFFSET:-100}"
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-0}"
if [ -z "${LIBERO_CAMERA_NAME_MAPPING+x}" ]; then
  LIBERO_CAMERA_NAME_MAPPING='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/_workspace/runpod_results/smolvla_agentic_retry_series_$(date -u +%Y%m%dT%H%M%SZ)}"

if [ "$LIBERO_TASK_IDS" = "all" ]; then
  EVAL_TASK_IDS=""
  DISPLAY_TASK_IDS="all"
else
  EVAL_TASK_IDS="$LIBERO_TASK_IDS"
  DISPLAY_TASK_IDS="$LIBERO_TASK_IDS"
fi

mkdir -p "$OUTPUT_ROOT"
MANIFEST="$OUTPUT_ROOT/series_manifest.jsonl"
: > "$MANIFEST"

cat > "$OUTPUT_ROOT/README.md" <<EOF
# SmolVLA LIBERO Agentic Retry Series

- model_id: \`$SMOLVLA_MODEL_ID\`
- task_group: \`$LIBERO_TASKS\`
- task_ids: \`$DISPLAY_TASK_IDS\`
- episodes_per_task: \`$LIBERO_N_EPISODES\`
- base_seeds: \`$BASE_SEEDS\`
- baseline_action_steps: \`$BASELINE_ACTION_STEPS\`
- blind_retry_action_steps: \`$BLIND_RETRY_ACTION_STEPS\`
- alternate_retry_action_steps: \`$ALTERNATE_RETRY_ACTION_STEPS\`
- retry_seed_offset: \`$RETRY_SEED_OFFSET\`
- verifier: \`libero_benchmark_success_flag\`
- metric: \`success_once_rate\`
- output_root: \`$OUTPUT_ROOT\`

## Semantics

- \`blind_new_seed\`: retry failed task/episode indexes with the same action
  horizon and a different seed.
- \`alternate_steps10\`: retry failed task/episode indexes with
  \`n_action_steps=$ALTERNATE_RETRY_ACTION_STEPS\` and a different seed.
- This is an episode-level retry wrapper, not an in-episode subgoal replanner.
EOF

run_eval() {
  out="$1"
  task_ids="$2"
  extra_args="$3"
  mkdir -p "$out"
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
    "$SCRIPT_DIR/eval_smolvla_libero_linux.sh" > "$out.driver.log" 2>&1
}

aggregate_condition() {
  condition="$1"
  base_seed="$2"
  retry_seed="$3"
  baseline_info="$4"
  retry_info="$5"
  run_dir="$6"

  PYTHONPATH="$PROJECT_DIR/src" "$PY312_VENV/bin/python" -m physical_ai_agent.agent_core.libero_agentic_retry \
    aggregate "$baseline_info" "$retry_info" \
    --task-group "$LIBERO_TASKS" \
    --output-json "$run_dir/agentic/agentic_retry_metrics.json" \
    --output-jsonl "$run_dir/agentic/agentic_retry_trace.jsonl" \
    --output-md "$run_dir/agentic/agentic_retry_report.md"

  "$PY312_VENV/bin/python" - <<PY >> "$MANIFEST"
import json
print(json.dumps({
    "condition": "$condition",
    "base_seed": int("$base_seed"),
    "retry_seed": int("$retry_seed"),
    "run_dir": "$condition" + "_seed" + str(int("$base_seed")),
    "baseline_eval_info": "$baseline_info",
    "retry_eval_info": "$retry_info",
}, sort_keys=True))
PY
}

for base_seed in $BASE_SEEDS; do
  baseline_dir="$OUTPUT_ROOT/baseline_seed$base_seed"
  baseline_args="--policy.num_steps=10 --policy.n_action_steps=$BASELINE_ACTION_STEPS --policy.device=cuda --seed=$base_seed"
  echo "running baseline seed=$base_seed"
  run_eval "$baseline_dir" "$EVAL_TASK_IDS" "$baseline_args"

  baseline_info="$baseline_dir/eval_logs/eval_info.json"
  plan_json="$baseline_dir/agentic/retry_plan.json"
  retry_task_ids="$(
    PYTHONPATH="$PROJECT_DIR/src" "$PY312_VENV/bin/python" -m physical_ai_agent.agent_core.libero_agentic_retry \
      plan "$baseline_info" \
      --task-group "$LIBERO_TASKS" \
      --output-json "$plan_json" \
      --print-task-ids
  )"

  if [ "$retry_task_ids" = "[]" ]; then
    echo "baseline seed=$base_seed has no failed episodes; skipping retry conditions"
    continue
  fi

  retry_seed=$((base_seed + RETRY_SEED_OFFSET))

  blind_dir="$OUTPUT_ROOT/blind_new_seed_seed$base_seed"
  blind_args="--policy.num_steps=10 --policy.n_action_steps=$BLIND_RETRY_ACTION_STEPS --policy.device=cuda --seed=$retry_seed"
  echo "running blind_new_seed base_seed=$base_seed retry_seed=$retry_seed task_ids=$retry_task_ids"
  run_eval "$blind_dir/retry" "$retry_task_ids" "$blind_args"
  mkdir -p "$blind_dir/baseline" "$blind_dir/agentic"
  cp "$baseline_info" "$blind_dir/baseline/eval_info.json"
  cp "$plan_json" "$blind_dir/agentic/retry_plan.json"
  aggregate_condition "blind_new_seed" "$base_seed" "$retry_seed" "$blind_dir/baseline/eval_info.json" "$blind_dir/retry/eval_logs/eval_info.json" "$blind_dir"

  alternate_dir="$OUTPUT_ROOT/alternate_steps10_seed$base_seed"
  alternate_args="--policy.num_steps=10 --policy.n_action_steps=$ALTERNATE_RETRY_ACTION_STEPS --policy.device=cuda --seed=$retry_seed"
  echo "running alternate_steps10 base_seed=$base_seed retry_seed=$retry_seed task_ids=$retry_task_ids"
  run_eval "$alternate_dir/retry" "$retry_task_ids" "$alternate_args"
  mkdir -p "$alternate_dir/baseline" "$alternate_dir/agentic"
  cp "$baseline_info" "$alternate_dir/baseline/eval_info.json"
  cp "$plan_json" "$alternate_dir/agentic/retry_plan.json"
  aggregate_condition "alternate_steps10" "$base_seed" "$retry_seed" "$alternate_dir/baseline/eval_info.json" "$alternate_dir/retry/eval_logs/eval_info.json" "$alternate_dir"
done

PYTHONPATH="$PROJECT_DIR/src" "$PY312_VENV/bin/python" "$SCRIPT_DIR/build_agentic_retry_series_report.py" "$OUTPUT_ROOT"

{
  echo
  echo "## Completion"
  echo
  echo "- series_manifest: \`$MANIFEST\`"
  echo "- series_report: \`$OUTPUT_ROOT/agentic_retry_series_report.md\`"
  echo "- series_summary: \`$OUTPUT_ROOT/agentic_retry_series_summary.json\`"
} >> "$OUTPUT_ROOT/README.md"

cat "$OUTPUT_ROOT/agentic_retry_series_report.md"
