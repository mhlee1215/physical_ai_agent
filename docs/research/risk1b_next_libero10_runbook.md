# Risk1-B LIBERO-10 Next Runbook

Purpose: rerun Risk1-B alternative-goal full LIBERO-10 without the unsafe
`candidate_01` auto-selection behavior.

Boundary:

- This is full rollout data production, not smoke-only evidence.
- If `renderer_backend=osmesa`, label the result as non-EGL accepted lane.
- Deterministic fallback candidates are incomplete evidence. If any fallback row
  appears, do not use the run as Qwen-only paper evidence.

## 1. Weak-Task Candidate Ablation

Use this first to verify whether `candidate_01` is actually worse than
`candidate_00_policy_only` on the weak rows from the previous 69.0% run.

```bash
cd /workspace/physical-ai/physical_ai_agent
GEN_ROOT=<zero-fallback-qwen-generation-root> \
OUT=_workspace/runpod_results/ita_risk_probes/risk1b_alt_goal_candidate_ablation_task0_6_8 \
TASKS=0,6,8 \
bash scripts/run_risk1b_alt_goal_libero10_candidate_ablation.sh
```

Key artifacts:

- `candidate_ablation_summary.json`
- `candidate_ablation_summary.md`
- Per-candidate `summary.json` files under `candidate_*/`

Interpretation:

- `candidate_00_policy_only` is the baseline prompt/policy-only chunk.
- Non-baseline candidates are only useful if they beat `candidate_00` on full
  rollout success for the same tasks.

## 2. Full LIBERO-10 Rerun

After ablation confirms the selector rule, run the full 10-task comparison:

```bash
cd /workspace/physical-ai/physical_ai_agent
GEN_ROOT=<zero-fallback-qwen-generation-root> \
OUT=_workspace/runpod_results/ita_risk_probes/risk1b_alt_goal_full_libero10_progress_proxy_seed1201_shallow_osmesa \
SELECTOR_STRATEGY=progress_proxy_or_baseline \
TASKS=0,1,2,3,4,5,6,7,8,9 \
bash scripts/run_risk1b_alt_goal_full_libero10_from_generated_json.sh
```

Required table fields:

- Baseline SmolVLA LIBERO-10: `75.0%`
- Risk1-B full LIBERO-10 success: read from `summary.json pc_success`
- Episodes: must be `100` for a full comparison
- Qwen fallback rows: must be `0`
- Lane label: `SECURE/shallow OSMesa data-production lane; non-EGL` if OSMesa

Build the result package:

```bash
PYTHONPATH=src /workspace/physical-ai/envs/lerobot_py312/bin/python -B \
  scripts/build_risk1b_libero10_result_package.py \
  --summary _workspace/runpod_results/ita_risk_probes/risk1b_alt_goal_full_libero10_progress_proxy_seed1201_shallow_osmesa/summary.json \
  --ablation-summary _workspace/runpod_results/ita_risk_probes/risk1b_alt_goal_candidate_ablation_task0_6_8/candidate_ablation_summary.json \
  --json
```

Key artifacts:

- `risk1b_libero10_result_package.json`
- `risk1b_libero10_result_package.md`

## 3. Stop Policy

After the experiment artifacts are fetched and the result package is produced,
request RunPod stop/no-RUNNING confirmation, because the current user objective
explicitly asks to stop RunPod after experiments.
