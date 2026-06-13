#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/workspace/physical-ai/physical_ai_agent}
PY=${PY:-/workspace/physical-ai/envs/lerobot_py312/bin/python}
VLM_PY=${VLM_PY:-/workspace/physical-ai/envs/risk1b_vlm_py312/bin/python}
OUT=${OUT:-_workspace/runpod_results/ita_risk_probes/risk1b_alt_goal_full_libero10_repair_fallback_seed1201_shallow_osmesa}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
MODEL_SLUG=${MODEL_SLUG:-qwen2_5_vl_7b_instruct}
REPAIR_ATTEMPTS=${REPAIR_ATTEMPTS:-2}
FALLBACK_ON_VALIDATION_ERROR=${FALLBACK_ON_VALIDATION_ERROR:-none}
SELECTOR_STRATEGY=${SELECTOR_STRATEGY:-progress_proxy_or_baseline}
TASKS=${TASKS:-0,1,2,3,4,5,6,7,8,9}

export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-/workspace/physical-ai/libero_config}
export HF_HOME=${HF_HOME:-/workspace/physical-ai/hf_home}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export PYTHONPATH=${PYTHONPATH:-src}

cd "$ROOT" || exit 2
mkdir -p "$OUT"
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source_commit=$(cat .source_commit 2>/dev/null || git rev-parse HEAD 2>/dev/null || true)"
  echo "renderer_backend=osmesa"
  echo "suite=libero_10"
  echo "episodes_per_task=10"
  echo "tasks=$TASKS"
  echo "selector_strategy=$SELECTOR_STRATEGY"
  echo "generation_policy=Qwen initial + ${REPAIR_ATTEMPTS} repair attempts; fallback_policy=${FALLBACK_ON_VALIDATION_ERROR}"
  echo "paper_table_guardrail=any deterministic fallback row invalidates the Qwen-only Risk1-B table run"
  echo "boundary=SECURE/shallow OSMesa data-production lane; non-EGL; full episode rollout success metric"
} > "$OUT/run_metadata.env"

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
for task in "${TASK_ARRAY[@]}"; do
  task=${task//[[:space:]]/}
  if [ -z "$task" ]; then
    continue
  fi
  TASK_DIR="$OUT/libero_10_task${task}_seed1201"
  CTX_DIR="$TASK_DIR/risk1b_context"
  JSON_PATH="$TASK_DIR/risk1b_subgoals_${MODEL_SLUG}_libero_10_task${task}_seed1201.json"
  TRACE_PATH="$TASK_DIR/benchmark_trace.jsonl"
  mkdir -p "$TASK_DIR" "$CTX_DIR"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] task${task} start" | tee -a "$OUT/progress.log"
  if [ ! -s "$CTX_DIR/context_task${task}_seed1201.json" ]; then
    "$PY" -B scripts/capture_risk1b_context.py \
      --backend libero-shallow \
      --suite libero_10 \
      --task-id "$task" \
      --seed 1201 \
      --policy-path lerobot/smolvla_libero \
      --policy-num-steps 10 \
      --policy-n-action-steps 15 \
      --renderer-backend osmesa \
      --output-dir "$CTX_DIR" \
      --json > "$TASK_DIR/context_capture_stdout.json" 2> "$TASK_DIR/context_capture_stderr.log"
    ctx_status=$?
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] task${task} context_exit=${ctx_status}" | tee -a "$OUT/progress.log"
    if [ "$ctx_status" -ne 0 ]; then
      echo "task${task} context failed" > "$TASK_DIR/row_status.txt"
      continue
    fi
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] task${task} context_reused" | tee -a "$OUT/progress.log"
  fi

  if [ ! -s "$JSON_PATH" ]; then
    "$VLM_PY" -B scripts/generate_risk1b_vlm_subgoals.py \
      --backend transformers \
      --model-id "$MODEL" \
      --suite libero_10 \
      --task-id "$task" \
      --seed 1201 \
      --num-subgoals 5 \
      --task-description "Complete the LIBERO long-horizon task from the current observation." \
      --context-image "$CTX_DIR/contact_sheet_task${task}_seed1201.png" \
      --context-json "$CTX_DIR/context_task${task}_seed1201.json" \
      --output-dir "$TASK_DIR" \
      --repair-attempts "$REPAIR_ATTEMPTS" \
      --fallback-on-validation-error "$FALLBACK_ON_VALIDATION_ERROR" \
      --json > "$TASK_DIR/vlm_generation_stdout.json" 2> "$TASK_DIR/vlm_generation_stderr.log"
    gen_status=$?
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] task${task} generation_exit=${gen_status}" | tee -a "$OUT/progress.log"
    if [ "$gen_status" -ne 0 ] || [ ! -s "$JSON_PATH" ]; then
      echo "task${task} generation failed" > "$TASK_DIR/row_status.txt"
      continue
    fi
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] task${task} qwen_json_reused" | tee -a "$OUT/progress.log"
  fi

  "$PY" - "$JSON_PATH" > "$TASK_DIR/qwen_json_gate.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print(json.dumps({"ok": False, "reason": "missing_qwen_json", "path": str(path)}))
    raise SystemExit(2)
payload = json.loads(path.read_text(encoding="utf-8"))
schema_validation = payload.get("schema_validation") if isinstance(payload.get("schema_validation"), dict) else {}
result = {
    "ok": (
        payload.get("provenance") == "external_vlm_json"
        and schema_validation.get("valid") is True
        and payload.get("fallback") in (None, False)
    ),
    "path": str(path),
    "provenance": payload.get("provenance"),
    "schema_valid": schema_validation.get("valid"),
    "schema_errors": schema_validation.get("errors") or [],
    "generation_attempt_count": len(payload.get("generation_attempts", [])),
    "fallback": payload.get("fallback"),
}
if payload.get("fallback") not in (None, False):
    result["reason"] = "fallback_rows_are_incomplete_evidence"
elif schema_validation.get("valid") is not True:
    result["reason"] = "schema_invalid"
elif payload.get("provenance") != "external_vlm_json":
    result["reason"] = "non_external_vlm_provenance"
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["ok"] else 2)
PY
  gate_status=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] task${task} qwen_json_gate_exit=${gate_status}" | tee -a "$OUT/progress.log"
  if [ "$gate_status" -ne 0 ]; then
    echo "task${task} qwen_json_gate failed" > "$TASK_DIR/row_status.txt"
    continue
  fi

  "$PY" -B scripts/run_libero_in_episode_smolvla_instrumented.py \
    --trace-path "$TRACE_PATH" \
    --trigger-mode semantic_no_progress \
    --intervention-mode none \
    --semantic-min-step 220 \
    --semantic-window 20 \
    --semantic-progress-threshold 0.002 \
    --ita-enable \
    --ita-num-candidates 5 \
    --ita-candidate-seeds 1201,1202,1203,1204,1205 \
    --ita-commit-steps 15 \
    --ita-candidate-prompts-json "$JSON_PATH" \
    --ita-selector-strategy "$SELECTOR_STRATEGY" \
    --output_dir="$TASK_DIR/eval_logs" \
    --policy.path=lerobot/smolvla_libero \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids="[$task]" \
    --env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --eval.use_async_envs=false \
    --env.max_parallel_tasks=1 \
    --policy.empty_cameras=0 \
    --policy.num_steps=10 \
    --policy.n_action_steps=15 \
    --seed=1201 > "$TASK_DIR/full_rollout_stdout.log" 2> "$TASK_DIR/full_rollout_stderr.log"
  eval_status=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] task${task} full_rollout_exit=${eval_status}" | tee -a "$OUT/progress.log"
  echo "task${task} full_rollout_exit=${eval_status}" > "$TASK_DIR/row_status.txt"
done

"$PY" - <<'PY' "$OUT" "$MODEL_SLUG" "$SELECTOR_STRATEGY" "$TASKS" | tee "$OUT/summary.json"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model_slug = sys.argv[2]
selector_strategy = sys.argv[3]
tasks = [int(part.strip()) for part in sys.argv[4].split(",") if part.strip()]
rows = []
total_success = 0
total_episodes = 0
provenance_counts: dict[str, int] = {}
for task in tasks:
    task_dir = root / f"libero_10_task{task}_seed1201"
    eval_path = task_dir / "eval_logs" / "eval_info.json"
    json_path = task_dir / f"risk1b_subgoals_{model_slug}_libero_10_task{task}_seed1201.json"
    row = {
        "task_id": task,
        "eval_info_path": str(eval_path),
        "candidate_prompts_json": str(json_path),
        "status": "missing_eval_info",
    }
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text())
            provenance = payload.get("provenance")
            row["candidate_prompt_provenance"] = provenance
            row["generation_attempts"] = payload.get("generation_attempts", [])
            row["fallback"] = payload.get("fallback")
            if provenance:
                provenance_counts[str(provenance)] = provenance_counts.get(str(provenance), 0) + 1
        except Exception as exc:  # noqa: BLE001
            row["candidate_prompt_parse_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    gate_path = task_dir / "qwen_json_gate.json"
    if gate_path.exists():
        try:
            row["qwen_json_gate"] = json.loads(gate_path.read_text())
        except Exception as exc:  # noqa: BLE001
            row["qwen_json_gate_parse_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if eval_path.exists():
        payload = json.loads(eval_path.read_text())
        successes = []
        for item in payload.get("per_task", []):
            metrics = item.get("metrics", {})
            if item.get("task_group") == "libero_10" and item.get("task_id") == task:
                successes = [bool(x) for x in metrics.get("successes", [])]
                break
        if not successes:
            n = (
                payload.get("overall", {}).get("n_episodes")
                or payload.get("per_group", {}).get("libero_10", {}).get("n_episodes")
                or 0
            )
            pc = (
                payload.get("overall", {}).get("pc_success")
                or payload.get("per_group", {}).get("libero_10", {}).get("pc_success")
                or 0
            )
            successes = [True] * round(float(n) * float(pc) / 100.0) + [False] * (
                int(n) - round(float(n) * float(pc) / 100.0)
            )
        success_count = sum(successes)
        n = len(successes)
        total_success += success_count
        total_episodes += n
        row.update(
            {
                "status": "completed",
                "success_count": success_count,
                "n_episodes": n,
                "pc_success": 100.0 * success_count / n if n else None,
            }
        )
    status_path = task_dir / "row_status.txt"
    if status_path.exists():
        row["row_status_text"] = status_path.read_text().strip()
    rows.append(row)
overall_pc = 100.0 * total_success / total_episodes if total_episodes else None
fallback_rows = [
    row["task_id"]
    for row in rows
    if row.get("fallback") not in (None, False)
    or row.get("qwen_json_gate", {}).get("fallback") not in (None, False)
]
invalid_qwen_rows = [
    row["task_id"]
    for row in rows
    if row.get("qwen_json_gate", {}).get("ok") is False
]
payload = {
    "status": "completed" if total_episodes == len(tasks) * 10 and not fallback_rows and not invalid_qwen_rows else "partial_or_blocked",
    "suite": "libero_10",
    "tasks": tasks,
    "renderer_backend": "osmesa",
    "lane": "SECURE/shallow OSMesa data-production lane; non-EGL",
    "method": (
        "Risk1-B alternative-goal candidate prompt full rollout; "
        f"selector_strategy={selector_strategy}; Qwen initial + repair + deterministic fallback"
    ),
    "generation_policy": "Qwen initial + repair attempts + deterministic_locked_fields fallback",
    "total_success": total_success,
    "total_episodes": total_episodes,
    "pc_success": overall_pc,
    "fallback_rows": fallback_rows,
    "invalid_qwen_rows": invalid_qwen_rows,
    "candidate_prompt_provenance_counts": provenance_counts,
    "rows": rows,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] aggregate_done" | tee -a "$OUT/progress.log"
