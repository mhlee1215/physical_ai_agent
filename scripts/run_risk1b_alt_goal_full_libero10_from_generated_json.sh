#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/workspace/physical-ai/physical_ai_agent}
PY=${PY:-/workspace/physical-ai/envs/lerobot_py312/bin/python}
GEN_ROOT=${GEN_ROOT:?GEN_ROOT is required and must point to a zero-fallback Qwen generation root}
OUT=${OUT:-_workspace/runpod_results/ita_risk_probes/risk1b_alt_goal_full_libero10_from_generated_json_seed1201_shallow_osmesa}
MODEL_SLUG=${MODEL_SLUG:-qwen2_5_vl_7b_instruct}
SELECTOR_STRATEGY=${SELECTOR_STRATEGY:-progress_proxy_or_baseline}
FORCED_CANDIDATE_ID=${FORCED_CANDIDATE_ID:-}
TASKS=${TASKS:-0,1,2,3,4,5,6,7,8,9}

export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-/workspace/physical-ai/libero_config}
export HF_HOME=${HF_HOME:-/workspace/physical-ai/hf_home}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export PYTHONPATH=${PYTHONPATH:-src}

cd "$ROOT" || exit 2
mkdir -p "$OUT"
FORCED_CANDIDATE_ARGS=()
if [ -n "$FORCED_CANDIDATE_ID" ]; then
  FORCED_CANDIDATE_ARGS=(--ita-selected-candidate-id "$FORCED_CANDIDATE_ID")
fi
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source_commit=$(cat .source_commit 2>/dev/null || git rev-parse HEAD 2>/dev/null || true)"
  echo "renderer_backend=osmesa"
  echo "suite=libero_10"
  echo "episodes_per_task=10"
  echo "tasks=$TASKS"
  echo "selector_strategy=$SELECTOR_STRATEGY"
  echo "forced_candidate_id=$FORCED_CANDIDATE_ID"
  echo "generation_root=$GEN_ROOT"
  echo "generation_policy=prevalidated Qwen JSON only; fallback rows forbidden"
  echo "paper_table_guardrail=any fallback row blocks this Qwen-only Risk1-B table run"
  echo "boundary=SECURE/shallow OSMesa data-production lane; non-EGL; full episode rollout success metric"
} > "$OUT/run_metadata.env"

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
for task in "${TASK_ARRAY[@]}"; do
  task=${task//[[:space:]]/}
  if [ -z "$task" ]; then
    continue
  fi
  TASK_DIR="$OUT/libero_10_task${task}_seed1201"
  SRC_TASK_DIR="$GEN_ROOT/libero_10_task${task}_seed1201"
  JSON_PATH="$SRC_TASK_DIR/risk1b_subgoals_${MODEL_SLUG}_libero_10_task${task}_seed1201.json"
  CONTEXT_JSON_PATH="$SRC_TASK_DIR/risk1b_context/context_task${task}_seed1201.json"
  TRACE_PATH="$TASK_DIR/benchmark_trace.jsonl"
  mkdir -p "$TASK_DIR"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] task${task} start" | tee -a "$OUT/progress.log"

  "$PY" - "$JSON_PATH" > "$TASK_DIR/qwen_json_gate.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print(json.dumps({"ok": False, "reason": "missing_qwen_json", "path": str(path)}))
    raise SystemExit(2)
payload = json.loads(path.read_text(encoding="utf-8"))
result = {
    "ok": (
        payload.get("provenance") == "external_vlm_json"
        and payload.get("schema_validation", {}).get("valid") is True
        and not payload.get("fallback")
    ),
    "path": str(path),
    "provenance": payload.get("provenance"),
    "schema_valid": payload.get("schema_validation", {}).get("valid"),
    "schema_errors": payload.get("schema_validation", {}).get("errors"),
    "generation_attempt_count": len(payload.get("generation_attempts", [])),
    "fallback": payload.get("fallback"),
}
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["ok"] else 2)
PY
  gate_status=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] task${task} qwen_json_gate_exit=${gate_status}" | tee -a "$OUT/progress.log"
  if [ "$gate_status" -ne 0 ]; then
    echo "task${task} qwen_json_gate failed" > "$TASK_DIR/row_status.txt"
    continue
  fi

  "$PY" - "$CONTEXT_JSON_PATH" "$JSON_PATH" > "$TASK_DIR/semantic_proxy_keys.json" <<'PY'
import json
import pathlib
import re
import sys

context_path = pathlib.Path(sys.argv[1])
json_path = pathlib.Path(sys.argv[2])
if not context_path.exists():
    print(json.dumps({"ok": False, "reason": "missing_context_json", "path": str(context_path)}))
    raise SystemExit(2)

context = json.loads(context_path.read_text(encoding="utf-8"))
payload = json.loads(json_path.read_text(encoding="utf-8"))
observation_summary = context.get("observation_summary") or {}
pos_keys = [
    key
    for key in observation_summary
    if key.endswith("_pos")
    and not key.startswith("robot0")
    and "_to_robot0_eef_" not in key
]
task_text = " ".join(
    str(part)
    for part in (
        context.get("task_description"),
        payload.get("task_description"),
        " ".join(context.get("task_descriptions") or []),
    )
    if part
).lower()
candidate_records = payload.get("candidate_prompts") or payload.get("strategy_variants") or payload.get("subgoals") or []
candidate_text = " ".join(
    " ".join(str(record.get(key, "")) for key in ("target_object", "target_region_or_point", "subgoal_text", "stop_condition"))
    for record in candidate_records
    if isinstance(record, dict)
).lower()
combined = f"{task_text} {candidate_text}"

aliases = [
    ("alphabet soup", "alphabet_soup"),
    ("cream cheese box", "cream_cheese"),
    ("cream cheese", "cream_cheese"),
    ("butter", "butter"),
    ("moka pots", "moka_pot"),
    ("moka pot", "moka_pot"),
    ("black bowl", "akita_black_bowl"),
    ("yellow and white mug", "white_yellow_mug"),
    ("yellow white mug", "white_yellow_mug"),
    ("white mug", "porcelain_mug"),
    ("book", "black_book"),
    ("chocolate pudding", "chocolate_pudding"),
]

def exact_alias_score(alias: str) -> int:
    return 1 if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", combined) else 0

ranked: list[tuple[int, int, int, str, str]] = []
for alias_index, (alias, key_fragment) in enumerate(aliases):
    if not exact_alias_score(alias):
        continue
    for key in pos_keys:
        if key_fragment in key:
            instance_priority = 1 if key.endswith("_1_pos") else 0
            ranked.append((100 - alias_index, instance_priority, -len(key), alias, key))

if not ranked:
    # Last-resort lexical overlap, still constrained to non-robot object position keys.
    words = {word for word in re.findall(r"[a-z0-9]+", combined) if len(word) >= 4}
    for key in pos_keys:
        key_words = set(key.replace("_1_pos", "").replace("_pos", "").split("_"))
        overlap = len(words & key_words)
        if overlap:
            instance_priority = 1 if key.endswith("_1_pos") else 0
            ranked.append((overlap, instance_priority, -len(key), "lexical_overlap", key))

if not ranked:
    print(json.dumps({
        "ok": False,
        "reason": "could_not_resolve_target_object_key",
        "context_json": str(context_path),
        "candidate_prompts_json": str(json_path),
        "available_pos_keys": pos_keys,
    }, indent=2, sort_keys=True))
    raise SystemExit(2)

ranked.sort(reverse=True)
_score, _instance_priority, _len_key, matched_alias, target_key = ranked[0]
print(json.dumps({
    "ok": True,
    "target_object_key": target_key,
    "receptacle_object_key": "",
    "matched_alias": matched_alias,
    "available_pos_keys": pos_keys,
    "task_description": context.get("task_description"),
}, indent=2, sort_keys=True))
PY
  proxy_status=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] task${task} semantic_proxy_key_exit=${proxy_status}" | tee -a "$OUT/progress.log"
  if [ "$proxy_status" -ne 0 ]; then
    echo "task${task} semantic_proxy_key failed" > "$TASK_DIR/row_status.txt"
    continue
  fi
  TARGET_OBJECT_KEY=$("$PY" - "$TASK_DIR/semantic_proxy_keys.json" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["target_object_key"])
PY
)
  RECEPTACLE_OBJECT_KEY=$("$PY" - "$TASK_DIR/semantic_proxy_keys.json" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")).get("receptacle_object_key") or "__missing_receptacle_pos")
PY
)

  "$PY" -B scripts/run_libero_in_episode_smolvla_instrumented.py \
    --trace-path "$TRACE_PATH" \
    --trigger-mode semantic_no_progress \
    --intervention-mode none \
    --semantic-min-step 220 \
    --semantic-window 20 \
    --semantic-progress-threshold 0.002 \
    --target-object-key "$TARGET_OBJECT_KEY" \
    --receptacle-object-key "$RECEPTACLE_OBJECT_KEY" \
    --ita-enable \
    --ita-num-candidates 5 \
    --ita-candidate-seeds 1201,1202,1203,1204,1205 \
    --ita-commit-steps 15 \
    --ita-candidate-prompts-json "$JSON_PATH" \
    --ita-selector-strategy "$SELECTOR_STRATEGY" \
    "${FORCED_CANDIDATE_ARGS[@]}" \
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

"$PY" - <<'PY' "$OUT" "$GEN_ROOT" "$MODEL_SLUG" "$SELECTOR_STRATEGY" "$FORCED_CANDIDATE_ID" "$TASKS" | tee "$OUT/summary.json"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
gen_root = pathlib.Path(sys.argv[2])
model_slug = sys.argv[3]
selector_strategy = sys.argv[4]
forced_candidate_id = sys.argv[5]
tasks = [int(part.strip()) for part in sys.argv[6].split(",") if part.strip()]
rows = []
total_success = 0
total_episodes = 0
provenance_counts: dict[str, int] = {}
fallback_rows = 0
for task in tasks:
    task_dir = root / f"libero_10_task{task}_seed1201"
    eval_path = task_dir / "eval_logs" / "eval_info.json"
    json_path = gen_root / f"libero_10_task{task}_seed1201" / f"risk1b_subgoals_{model_slug}_libero_10_task{task}_seed1201.json"
    row = {
        "task_id": task,
        "eval_info_path": str(eval_path),
        "candidate_prompts_json": str(json_path),
        "status": "missing_eval_info",
    }
    gate_path = task_dir / "qwen_json_gate.json"
    if gate_path.exists():
        row["qwen_json_gate"] = json.loads(gate_path.read_text(encoding="utf-8"))
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            provenance = payload.get("provenance")
            row["candidate_prompt_provenance"] = provenance
            row["generation_attempt_count"] = len(payload.get("generation_attempts", []))
            row["fallback"] = payload.get("fallback")
            if payload.get("fallback"):
                fallback_rows += 1
            if provenance:
                provenance_counts[str(provenance)] = provenance_counts.get(str(provenance), 0) + 1
        except Exception as exc:  # noqa: BLE001
            row["candidate_prompt_parse_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
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
            success_count = round(float(n) * float(pc) / 100.0)
            successes = [True] * success_count + [False] * (int(n) - success_count)
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
        row["row_status_text"] = status_path.read_text(encoding="utf-8").strip()
    rows.append(row)
overall_pc = 100.0 * total_success / total_episodes if total_episodes else None
payload = {
    "status": "completed" if total_episodes == len(tasks) * 10 and fallback_rows == 0 else "partial_or_blocked",
    "suite": "libero_10",
    "tasks": tasks,
    "renderer_backend": "osmesa",
    "lane": "SECURE/shallow OSMesa data-production lane; non-EGL",
    "method": (
        "Risk1-B alternative-goal candidate prompt full rollout; "
        f"selector_strategy={selector_strategy}; prevalidated external-Qwen JSON; fallback forbidden"
    ),
    "forced_candidate_id": forced_candidate_id or None,
    "generation_root": str(gen_root),
    "total_success": total_success,
    "total_episodes": total_episodes,
    "pc_success": overall_pc,
    "fallback_rows": fallback_rows,
    "candidate_prompt_provenance_counts": provenance_counts,
    "rows": rows,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] aggregate_done" | tee -a "$OUT/progress.log"
