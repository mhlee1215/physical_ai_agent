# SmolVLA LIBERO Baseline Report

## Status

Current best internal 4-suite baseline:

- policy: `lerobot/smolvla_libero`
- suites: `libero_spatial,libero_object,libero_goal,libero_10`
- episodes: `10` per task, `400` total
- MuJoCo: `3.3.2`
- batch size: `1`
- action args: `--policy.num_steps=10 --policy.n_action_steps=10`
- device: CUDA on RunPod RTX 4090
- result: `85.25%` average success
- output root:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_full_steps10_two_lane_20260606T171310Z`

This now uses the same high-level evaluation scale as the external LIBERO
table: 4 suites, 10 tasks per suite, 10 episodes per task. It is format
comparable and is now within `3.55` average points of the ActionX Table 1
SmolVLA reference.

## External Reference

ActionX Table 1 reports SmolVLA LIBERO success rates under 10 tasks per suite
and 10 evaluation trials per task:

| Source | Goal | Object | Spatial | Long | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| ActionX Table 1, SmolVLA | 91.0 | 94.0 | 93.0 | 77.0 | 88.8 |

Reference:
<https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1806605/full>

LeRobot issue `#2354` is also tracked as a public-checkpoint reproduction
sanity check. That issue reports `Spatial 0.73`, `Object 0.91`, `Goal 0.83`,
and `Long 0.43` for a public reproduction, and `Spatial 0.90`, `Object 0.96`,
`Goal 0.92`, and `Long 0.71` for the paper line.

Reference:
<https://github.com/huggingface/lerobot/issues/2354>

## Internal Paper-Scale Result

| Run | Policy | Goal | Object | Spatial | Long | Avg | Episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Internal steps10 two-lane full | `lerobot/smolvla_libero` | 85.0 | 92.0 | 91.0 | 73.0 | 85.25 | 400 |
| ActionX Table 1, SmolVLA | SmolVLA | 91.0 | 94.0 | 93.0 | 77.0 | 88.8 | 400 |
| Delta vs ActionX |  | -6.0 | -2.0 | -2.0 | -4.0 | -3.55 |  |
| HF issue #2354 public repro | `HuggingFaceVLA/smolvla_libero` | 83.0 | 91.0 | 73.0 | 43.0 | 72.75 | 400 |
| Delta vs HF repro |  | +2.0 | +1.0 | +18.0 | +30.0 | +12.5 |  |
| HF issue #2354 paper | SmolVLA | 92.0 | 96.0 | 90.0 | 71.0 | 87.25 | 400 |
| Delta vs HF paper |  | -7.0 | -4.0 | +1.0 | +2.0 | -2.0 |  |

Local artifact:
`_workspace/runpod_results/baseline_debug_20260606/smolvla_lerobot_full_steps10_two_lane_20260606T171310Z/merged_eval_info.json`

Network volume artifact:
`/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_full_steps10_two_lane_20260606T171310Z`

## Internal Debug Runs

| Run | Policy | MuJoCo | Batch | Spatial | Notes |
| --- | --- | --- | ---: | ---: | --- |
| Current best steps10 full | `lerobot/smolvla_libero` | `3.3.2` | 1 | 91.0 | Full 4-suite, two process lanes, `n_action_steps=10` |
| Previous paper-scale full | `lerobot/smolvla_libero` | `3.3.2` | 1 | 89.0 | Full 4-suite, `n_action_steps=1`, Avg `76.25` |
| Best current | `lerobot/smolvla_libero` | `3.3.2` | 1 | 88.0 | Best validated internal run |
| Fast full Spatial | `lerobot/smolvla_libero` | `3.3.2` | 10 | 87.0 | One point lower, faster |
| Pre-fix full Spatial | `lerobot/smolvla_libero` | `3.9.0` | 10 | 78.0 | Bad rendering/version setting |
| HF checkpoint | `HuggingFaceVLA/smolvla_libero` | `3.3.2` | 10 | 72.0 | Different feature names, weaker here |
| HF pre-fix | `HuggingFaceVLA/smolvla_libero` | `3.9.0` | 10 | 69.0 | Before MuJoCo downgrade |

## Findings

- MuJoCo version was the largest confirmed fix. Downgrading from `3.9.0` to
  `3.3.2` improved `lerobot/smolvla_libero` Spatial from `78.0%` to `87.0%`,
  and recovered task 5 from `0/10` to `8/10` or `9/10`.
- `batch_size=1` did not close the remaining gap. It improved full Spatial from
  `87.0%` to `88.0%`.
- `POLICY_EMPTY_CAMERAS=1` did not improve the hard-task subset aggregate.
- `n_action_steps=10` was the largest confirmed protocol fix after MuJoCo
  versioning. It improved the full 4-suite average from `76.25%` to `85.25%`
  and improved Long from `59.0%` to `73.0%`.
- The remaining ActionX deltas are Goal `-6`, Object `-2`, Spatial `-2`, Long
  `-4`, and Avg `-3.55`.
- `n_action_steps=15` is registered for the next targeted check because the
  user noted this setting from the pi0.7 paper. Do not mix it into the
  completed steps10 result.
- The ActionX reference likely does not name the exact LeRobot hub checkpoint
  and logs a different action/control protocol. Our LeRobot run logs
  `control_mode=relative`, while ActionX describes normalized absolute Cartesian
  pose and gripper torque actions.

## Per-Task Successes

| Suite | Task 0 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial | 10/10 | 8/10 | 10/10 | 10/10 | 8/10 | 9/10 | 9/10 | 9/10 | 9/10 | 9/10 |
| Object | 9/10 | 9/10 | 9/10 | 9/10 | 9/10 | 9/10 | 10/10 | 9/10 | 9/10 | 10/10 |
| Goal | 10/10 | 9/10 | 10/10 | 5/10 | 10/10 | 9/10 | 7/10 | 10/10 | 7/10 | 8/10 |
| Long | 4/10 | 9/10 | 8/10 | 10/10 | 8/10 | 10/10 | 3/10 | 9/10 | 6/10 | 6/10 |

Use this result as the current internal policy-only baseline for wrapper
experiments unless `n_action_steps=15` or a closer ActionX-compatible
checkpoint/control-mode path improves the remaining gap. For paper claims,
report the gap explicitly instead of implying exact reproduction.
