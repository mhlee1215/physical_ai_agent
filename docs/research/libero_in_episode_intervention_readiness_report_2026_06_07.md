# LIBERO In-Episode Intervention Readiness Report

- lerobot_eval_source: `_workspace/lerobot_source_snapshot_20260607/lerobot_eval.py`
- libero_env_source: `_workspace/lerobot_source_snapshot_20260607/libero.py`
- eval_infos: `_workspace/runpod_results/agentic_retry_portfolio_20260607/smolvla_agentic_retry_portfolio_remaining_suites_seed1001_1002_20260607T012517Z/libero_goal/baseline_seed1001/eval_logs/eval_info.json`

## Checks

| Check | Status | Evidence |
| --- | --- | --- |
| lerobot_rollout_function | pass | _workspace/lerobot_source_snapshot_20260607/lerobot_eval.py: contains def rollout( |
| rollout_records_actions | pass | rollout returns stacked ACTION tensor |
| rollout_records_success_done | pass | rollout returns success and done sequences |
| online_hook_location | pass | hook can be inserted immediately before/after env.step(action_numpy) |
| default_eval_info_action_steps | fail | default eval_info per_episode lacks action_steps |
| libero_step_exposes_success | pass | LiberoEnv.step exposes check_success via info['is_success'] |
| libero_step_auto_resets_on_terminal | warn | LiberoEnv.step auto-resets after terminated; custom intervention loop must account for this |
| eval_info:eval_info.json:eval_seconds | pass | overall.eval_s=438.583 |
| eval_info:eval_info.json:successes | pass | success entries=100 |
| eval_info:eval_info.json:action_step_counts | fail | no per-episode action-step count fields found |

## Summary

| Pass | Warn | Fail | Missing | Unknown |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 1 | 2 | 0 | 0 |

## Next Experiment Contract

- Use a custom rollout path or patch LeRobot rollout to emit per-episode action_step_count.
- Insert the online verifier immediately before or after env.step(action_numpy).
- Log verifier_triggered, trigger_step, intervention_type, action_steps, eval_seconds, and final benchmark success.
- Treat LiberoEnv auto-reset after terminal success/failure as a boundary; in-episode interventions must occur before terminated=True.
- Compare policy-only, blind retry, horizon-switch retry, and in-episode intervention under the same task/seed/action budget.
