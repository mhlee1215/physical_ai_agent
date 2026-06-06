# SmolVLA LIBERO Baseline Report

## Status

Completed internal 4-suite baseline:

- policy: `lerobot/smolvla_libero`
- suites: `libero_spatial,libero_object,libero_goal,libero_10`
- episodes: `10` per task, `400` total
- MuJoCo: `3.3.2`
- batch size: `1`
- action args: `--policy.num_steps=10 --policy.n_action_steps=1`
- device: CUDA on RunPod RTX 4090
- result: `76.25%` average success
- output root:
  `/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_libero_all_mj332_b1_10eps_cuda_20260606T044020Z`

This now uses the same high-level evaluation scale as the external LIBERO
table: 4 suites, 10 tasks per suite, 10 episodes per task. It is format
comparable, but it does not reproduce the external SmolVLA numbers.

## External Reference

ActionX Table 1 reports SmolVLA LIBERO success rates under 10 tasks per suite
and 10 evaluation trials per task:

| Source | Goal | Object | Spatial | Long | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| ActionX Table 1, SmolVLA | 91.0 | 94.0 | 93.0 | 77.0 | 88.8 |

Reference:
<https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1806605/full>

## Internal Paper-Scale Result

| Run | Policy | Goal | Object | Spatial | Long | Avg | Episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Internal 4-suite baseline | `lerobot/smolvla_libero` | 79.0 | 78.0 | 89.0 | 59.0 | 76.25 | 400 |
| ActionX Table 1, SmolVLA | SmolVLA | 91.0 | 94.0 | 93.0 | 77.0 | 88.8 | 400 |

Local artifact:
`_workspace/runpod_results/baseline_debug_20260606/smolvla_lerobot_libero_all_mj332_b1_10eps_cuda_20260606T044020Z/eval_logs/eval_info.json`

Network volume artifact:
`/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results/smolvla_lerobot_libero_all_mj332_b1_10eps_cuda_20260606T044020Z`

## Internal Debug Runs

| Run | Policy | MuJoCo | Batch | Spatial | Notes |
| --- | --- | --- | ---: | ---: | --- |
| Paper-scale 4-suite run | `lerobot/smolvla_libero` | `3.3.2` | 1 | 89.0 | Same run as paper-scale table above |
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
- Task 4 remains the main failure source. Action-step sweep on task 4 did not
  solve it: `n_action_steps=5` got `5/10`, `10` got `7/10`, and `50` got `5/10`.
- The 4-suite run improved Spatial to `89.0%`, but Object, Goal, and Long are
  below the ActionX reference. The largest gap is Long/`libero_10`:
  `59.0%` internal vs `77.0%` external.
- The ActionX reference likely does not name the exact LeRobot hub checkpoint
  and logs a different action/control protocol. Our LeRobot run logs
  `control_mode=relative`, while ActionX describes normalized absolute Cartesian
  pose and gripper torque actions.

## Per-Task Successes

| Suite | Task 0 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial | 10/10 | 8/10 | 10/10 | 10/10 | 7/10 | 8/10 | 10/10 | 8/10 | 9/10 | 9/10 |
| Object | 5/10 | 7/10 | 10/10 | 10/10 | 8/10 | 6/10 | 6/10 | 8/10 | 9/10 | 9/10 |
| Goal | 8/10 | 10/10 | 9/10 | 8/10 | 9/10 | 9/10 | 5/10 | 9/10 | 8/10 | 4/10 |
| Long | 2/10 | 5/10 | 10/10 | 8/10 | 6/10 | 10/10 | 2/10 | 5/10 | 2/10 | 9/10 |

Use this result as the current internal policy-only baseline for wrapper
experiments unless a closer ActionX-compatible checkpoint/control-mode path is
found. For paper claims, report the gap explicitly instead of implying exact
reproduction.
