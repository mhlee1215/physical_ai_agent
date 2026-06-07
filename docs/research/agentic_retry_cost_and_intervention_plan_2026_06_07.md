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
