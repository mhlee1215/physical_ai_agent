# LIBERO SmolVLA In-Episode Instrumented Smoke - 2026-06-07

## Status

- status: passed
- environment: RunPod Linux, CUDA, MuJoCo EGL
- policy: `lerobot/smolvla_libero`
- suite: `libero_goal`
- task ids: `[0]`
- episodes: `1`
- seed: `1200`
- intervention hook: timeout-risk step threshold
- intervention action: scale current action by `1.0`

This is a hook/instrumentation smoke, not a paper-scale benchmark result.

## Evidence

- remote root:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/libero_in_episode_smolvla_smoke_20260607T034632`
- local no-video archive:
  `_workspace/runpod_results/in_episode_20260607/libero_in_episode_smolvla_smoke_20260607T034632_no_videos.tar.gz`
- local extracted trace:
  `_workspace/runpod_results/in_episode_20260607/libero_in_episode_smolvla_smoke_20260607T034632/in_episode_trace.jsonl`
- local eval info:
  `_workspace/runpod_results/in_episode_20260607/libero_in_episode_smolvla_smoke_20260607T034632/eval_logs/eval_info.json`

## Metrics

| Metric | Value |
| --- | ---: |
| benchmark success | true |
| action_step_count | 131 |
| verifier_trigger_count | 1 |
| intervention_count | 1 |
| environment_resets | 1 |
| eval_seconds | 7.3373 |
| success_per_action_step | 0.007634 |

## Trace Contract

The first trace row is a rollout summary:

```json
{"action_step_count": 131, "event": "rollout_summary", "intervention_count": 1, "max_steps": 300, "seeds": [1200], "success": true, "verifier_trigger_count": 1}
```

The hook fired before terminal reset at step `3`:

```json
{"step": 3, "intervention_type": "scale_action_1", "verifier_reason": "timeout_risk_step_threshold", "verifier_triggered": true}
```

## Interpretation

This proves that the existing LeRobot evaluation path can be monkeypatched to
record per-step actions, final benchmark success, verifier triggers, and
in-episode intervention markers during a real LIBERO/SmolVLA rollout.

It does not yet prove that the intervention improves success. The intervention
scale was `1.0`, so it intentionally preserved the policy action while testing
the hook and logging path. The next experiment should compare policy-only,
no-op hook, and non-trivial intervention settings under the same task, seed,
and action budget.
