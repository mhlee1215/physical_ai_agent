#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/workspace/physical-ai/physical_ai_agent}
PY=${PY:-/workspace/physical-ai/envs/lerobot_py312/bin/python}
GEN_ROOT=${GEN_ROOT:?GEN_ROOT is required and must point to a zero-fallback Qwen generation root}
OUT=${OUT:-_workspace/runpod_results/ita_risk_probes/risk1b_alt_goal_libero10_candidate_ablation_seed1201_shallow_osmesa}
TASKS=${TASKS:-0,6,8}
CANDIDATES=${CANDIDATES:-candidate_00_policy_only,candidate_01,candidate_02,candidate_03,candidate_04,candidate_05}
MODEL_SLUG=${MODEL_SLUG:-qwen2_5_vl_7b_instruct}

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
  echo "candidates=$CANDIDATES"
  echo "generation_root=$GEN_ROOT"
  echo "purpose=forced candidate ablation on weak LIBERO-10 tasks; compare baseline candidate_00 against Qwen strategy variants"
  echo "boundary=SECURE/shallow OSMesa data-production lane; non-EGL; full episode rollout success metric"
} > "$OUT/run_metadata.env"

IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
IFS=',' read -r -a CANDIDATE_ARRAY <<< "$CANDIDATES"

for candidate_id in "${CANDIDATE_ARRAY[@]}"; do
  candidate_id=${candidate_id//[[:space:]]/}
  if [ -z "$candidate_id" ]; then
    continue
  fi
  candidate_out="$OUT/$candidate_id"
  mkdir -p "$candidate_out"
  if [ "$candidate_id" = "candidate_00_policy_only" ]; then
    selector_strategy="baseline_fallback"
    forced_candidate_id=""
  else
    selector_strategy="debug_min_action_norm"
    forced_candidate_id="$candidate_id"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] candidate=${candidate_id} start" | tee -a "$OUT/progress.log"
  TASKS="$TASKS" \
  SELECTOR_STRATEGY="$selector_strategy" \
  FORCED_CANDIDATE_ID="$forced_candidate_id" \
  ROOT="$ROOT" \
  PY="$PY" \
  GEN_ROOT="$GEN_ROOT" \
  OUT="$candidate_out" \
  MODEL_SLUG="$MODEL_SLUG" \
    bash scripts/run_risk1b_alt_goal_full_libero10_from_generated_json.sh \
      > "$candidate_out/ablation_stdout.log" \
      2> "$candidate_out/ablation_stderr.log"
  rc=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] candidate=${candidate_id} exit=${rc}" | tee -a "$OUT/progress.log"
  echo "$rc" > "$candidate_out/ablation_exit_code.txt"
done

"$PY" - <<'PY' "$OUT" "$TASKS" "$CANDIDATES"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
tasks = [int(part.strip()) for part in sys.argv[2].split(",") if part.strip()]
candidates = [part.strip() for part in sys.argv[3].split(",") if part.strip()]
rows = []
by_candidate = {}
by_task = {}
for candidate_id in candidates:
    summary_path = root / candidate_id / "summary.json"
    candidate_total_success = 0
    candidate_total_episodes = 0
    if not summary_path.exists():
        rows.append({"candidate_id": candidate_id, "status": "missing_summary", "summary_path": str(summary_path)})
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary.get("rows", []):
        task_id = int(row.get("task_id"))
        success_count = int(row.get("success_count") or 0)
        n_episodes = int(row.get("n_episodes") or 0)
        candidate_total_success += success_count
        candidate_total_episodes += n_episodes
        record = {
            "candidate_id": candidate_id,
            "task_id": task_id,
            "status": row.get("status"),
            "success_count": success_count,
            "n_episodes": n_episodes,
            "pc_success": row.get("pc_success"),
            "eval_info_path": row.get("eval_info_path"),
            "candidate_prompts_json": row.get("candidate_prompts_json"),
        }
        rows.append(record)
        bucket = by_task.setdefault(str(task_id), {"task_id": task_id, "rows": []})
        bucket["rows"].append(record)
    by_candidate[candidate_id] = {
        "success_count": candidate_total_success,
        "n_episodes": candidate_total_episodes,
        "pc_success": 100.0 * candidate_total_success / candidate_total_episodes if candidate_total_episodes else None,
        "summary_path": str(summary_path),
    }

baseline = by_candidate.get("candidate_00_policy_only", {})
baseline_pc = baseline.get("pc_success")
for candidate_id, bucket in by_candidate.items():
    pc = bucket.get("pc_success")
    bucket["delta_vs_candidate_00_pp"] = (
        None if pc is None or baseline_pc is None else pc - baseline_pc
    )

payload = {
    "status": "completed" if all((root / candidate_id / "summary.json").exists() for candidate_id in candidates) else "partial",
    "tasks": tasks,
    "candidates": candidates,
    "by_candidate": by_candidate,
    "by_task": by_task,
    "rows": rows,
}
(root / "candidate_ablation_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Risk1-B LIBERO-10 Candidate Ablation",
    "",
    "| Candidate | Success | Episodes | Success rate | Delta vs candidate_00 |",
    "|---|---:|---:|---:|---:|",
]
for candidate_id in candidates:
    bucket = by_candidate.get(candidate_id, {})
    pc = bucket.get("pc_success")
    delta = bucket.get("delta_vs_candidate_00_pp")
    lines.append(
        "| {candidate} | {success} | {episodes} | {pc} | {delta} |".format(
            candidate=candidate_id,
            success=bucket.get("success_count", "NA"),
            episodes=bucket.get("n_episodes", "NA"),
            pc="NA" if pc is None else f"{pc:.1f}%",
            delta="NA" if delta is None else f"{delta:+.1f}pp",
        )
    )
lines.extend(["", "## Per Task", ""])
for task_id in tasks:
    lines.extend(
        [
            f"### task{task_id}",
            "",
            "| Candidate | Success | Episodes | Success rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted(by_task.get(str(task_id), {}).get("rows", []), key=lambda item: item["candidate_id"]):
        pc = row.get("pc_success")
        lines.append(
            "| {candidate} | {success} | {episodes} | {pc} |".format(
                candidate=row["candidate_id"],
                success=row["success_count"],
                episodes=row["n_episodes"],
                pc="NA" if pc is None else f"{float(pc):.1f}%",
            )
        )
    lines.append("")
(root / "candidate_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] aggregate_done" | tee -a "$OUT/progress.log"
