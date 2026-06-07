# Agentic Retry Cost and Intervention Plan - 2026-06-07

## Current Claim Boundary

The current LIBERO agentic retry experiments use the same frozen
`lerobot/smolvla_libero` policy as the policy-only baseline. They do not prove
that the model weights or one-shot rollout behavior improved.

The defensible claim is narrower:

> A frozen SmolVLA policy can achieve higher realized benchmark success when a
> verifier detects failed episodes and spends an explicit reset/retry budget.

This must be reported as a retry-budget protocol, not as a stronger one-shot
policy.

## Required Cost-Normalized Metrics

Every retry result should report success together with cost:

- `baseline_success_rate`: one-shot policy-only benchmark success.
- `success_once_rate`: success if baseline or retry succeeds.
- `total_attempts`: baseline episode attempts plus retry episode attempts that
  were actually evaluated.
- `extra_environment_resets`: retry episode attempts.
- `success_once_per_attempt`: realized successes divided by total attempts.
- `total_eval_seconds`: baseline eval seconds plus retry eval seconds.
- `success_once_per_eval_minute`: realized successes divided by eval minutes.
- `action_step_count_available`: whether the run logs per-episode action steps.

Current LeRobot `eval_info.json` records `overall.eval_s`, task successes, and
video paths, but does not record per-episode action-step counts. Therefore,
action-step-normalized metrics require an instrumented rollout path before they
can be used as paper evidence.

Readiness audit:
`docs/research/libero_in_episode_intervention_readiness_report_2026_06_07.md`

Audit result:

- LeRobot `rollout()` returns stacked action, success, and done tensors, so a
  custom rollout can compute action-step-normalized metrics.
- The online verifier hook can be inserted immediately before or after
  `env.step(action_numpy)`.
- `LiberoEnv.step()` exposes `info["is_success"]`, but auto-resets after
  terminal states; in-episode interventions must trigger before `terminated`.
- Default `eval_info.json` does not include per-episode action-step counts.

## Why Blind Retry Is Not Enough

Blind retry is a strong control. If an agentic wrapper only says "try again
with another seed," the result is best framed as retry-budget scaling.

A paper-grade agentic claim needs one of these stricter wins:

- Better success than blind retry under the same retry budget.
- Better success per attempt or per eval minute than blind retry.
- In-episode correction before environment reset.

## Next In-Episode Intervention Experiment

The next method should avoid counting only reset-level retry. The intended
experiment is:

1. Run the frozen SmolVLA policy inside an instrumented LIBERO rollout loop.
2. Log per-step observations, actions, gripper state, object/proprio signals if
   available, reward/success flags, and wall-clock time.
3. Add an online verifier that detects one or more failure precursors:
   - robot/object stagnation over a fixed step window,
   - repeated gripper open/close with no object progress,
   - end-effector moving away from the target after a grasp phase,
   - timeout risk near the final step budget.
4. On verifier trigger, intervene inside the same episode instead of resetting:
   - pause or shorten the current action chunk,
   - switch `n_action_steps` for the next chunk,
   - insert a bounded recovery subgoal such as re-approach or re-grasp,
   - resume SmolVLA after the intervention.
5. Report benchmark success, verifier trigger rate, intervention success,
   false-positive trigger rate, total action steps, total attempts, and eval
   seconds.

Executable preflight:

```bash
python3 scripts/build_libero_in_episode_intervention_readiness_report.py \
  --lerobot-eval-source _workspace/lerobot_source_snapshot_20260607/lerobot_eval.py \
  --libero-env-source _workspace/lerobot_source_snapshot_20260607/libero.py \
  --eval-info _workspace/runpod_results/agentic_retry_portfolio_20260607/smolvla_agentic_retry_portfolio_remaining_suites_seed1001_1002_20260607T012517Z/libero_goal/baseline_seed1001/eval_logs/eval_info.json \
  --output-md docs/research/libero_in_episode_intervention_readiness_report_2026_06_07.md \
  --output-json docs/research/libero_in_episode_intervention_readiness_summary_2026_06_07.json
```

## First Executable Instrumentation Gate

The first repo-local gate is a no-dependency smoke that proves the trace
contract before touching LeRobot internals:

```bash
PYTHONPATH=src:. python3 scripts/run_libero_in_episode_instrumented_smoke.py \
  --output-dir _workspace/libero_in_episode_smoke_20260607
```

Expected evidence:

- `_workspace/libero_in_episode_smoke_20260607/in_episode_metrics.json`
- `_workspace/libero_in_episode_smoke_20260607/in_episode_report.md`
- `_workspace/libero_in_episode_smoke_20260607/in_episode_trace.jsonl`

The smoke uses a toy LIBERO-like step loop that intentionally stagnates, then
fires an online verifier before terminal reset. The intervention scales one
action in-episode, after which the episode can reach environment success. This
does not claim LIBERO task improvement yet; it proves the logging and control
contract needed for the next RunPod implementation.

## First Real LIBERO/SmolVLA Hook Smoke

RunPod smoke report:
`docs/research/libero_in_episode_smolvla_smoke_2026_06_07.md`

Result:

- suite/task: `libero_goal`, task id `[0]`
- episodes: `1`
- policy: `lerobot/smolvla_libero`
- benchmark success: `true`
- action steps: `131`
- verifier triggers: `1`
- interventions: `1`
- environment resets: `1`
- eval seconds: `7.3373`
- success/action-step: `0.007634`

The real smoke monkeypatches the LeRobot `rollout()` function and records
per-step action metadata plus an in-episode intervention marker. The
intervention scale was `1.0`, so this is a no-op hook validation, not an
improvement claim. The next experiment must compare no-op hook against a
non-trivial intervention under fixed task/seed/action budget.

First same-seed ablation:
`docs/research/libero_in_episode_smolvla_ablation_2026_06_07.md`

- no-op hook, scale `1.0`: success `true`, action steps `131`,
  success/action-step `0.007634`
- non-trivial intervention, scale `0.5`: success `true`, action steps `132`,
  success/action-step `0.007576`

Interpretation: scale `0.5` did not improve cost on this single smoke. Keep it
as a negative/neutral ablation and search for verifier/intervention choices
that improve success or cost against the no-op hook.

## Experiment Table Shape

| Condition | Reset budget | In-episode intervention | Success | Attempts | Resets | Eval min | Success/attempt | Success/eval min | Action steps |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| policy-only baseline | 0 | none | required | required | required | required | required | required | required |
| blind retry | 1 | none | required | required | required | required | required | required | required |
| horizon-switch retry | 1 | none | required | required | required | required | required | required | required |
| verifier intervention | 0 | online | required | required | required | required | required | required | required |
| verifier intervention + retry | 1 | online | optional | optional | optional | optional | optional | optional | optional |

The `verifier intervention` row is the first row that can support the stronger
"agentic physical AI" claim. The current retry rows are controls and budget
baselines.
