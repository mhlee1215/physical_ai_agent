# Meta-World Very Hard Failure Audit - 2026-06-08

## Scope

This audit uses already-downloaded local Meta-World MT50 evaluation artifacts. No new RunPod evaluation was started.

Very Hard task mapping from the LeRobot eval logs:

| Task id | Task name |
| --- | --- |
| 0 | shelf-place-v3 |
| 1 | disassemble-v3 |
| 2 | stick-pull-v3 |
| 3 | stick-push-v3 |
| 4 | pick-place-wall-v3 |

Local result roots:

- `default`: `_workspace/runpod_results/metaworld_public_full_mt50_10ep_20260608T0650Z/metaworld_public_full_mt50_10ep_20260608T0650Z`
- `nas15`: `_workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z`
- `nas10`: `_workspace/runpod_results/metaworld_public_full_mt50_10ep_nas10_cu124_20260608T1906Z/metaworld_public_full_mt50_10ep_nas10_cu124_20260608T1906Z`
- `nas20`: `_workspace/runpod_results/metaworld_public_full_mt50_10ep_nas20_cu124_20260608T1955Z/metaworld_public_full_mt50_10ep_nas20_cu124_20260608T1955Z`

Failed-only review pack:

```text
_workspace/runpod_results/metaworld_very_hard_failed_pack_nas15/
```

The pack contains failed-video symlinks, a CSV/JSON manifest, and QuickLook thumbnails.

## Split-Level Result

| Run | Weighted overall % | Paper-style avg % | Very Hard % | Very Hard successes | Note |
| --- | --- | --- | --- | --- | --- |
| Default | 55.00 | 42.07 | 24.00 | 12/50 | public resolver, checkpoint/default action horizon |
| n_action_steps=15 | 65.20 | 53.60 | 38.00 | 19/50 | best current paper-style split average |
| n_action_steps=10 CUDA | 66.20 | 52.77 | 34.00 | 17/50 | best current weighted overall, torch 2.5.1+cu124 caveat |
| n_action_steps=20 CUDA | 65.40 | 53.10 | 36.00 | 18/50 | torch 2.5.1+cu124 caveat |

The paper reference reports Very Hard as `60.00%`, but does not give per-task Very Hard numbers in the table used here. Therefore per-task deltas below are diagnostic only, not paper-number comparisons.

## Very Hard Per-Task Matrix

| Task id | Task | Default | n_action_steps=15 | n_action_steps=10 CUDA | n_action_steps=20 CUDA |
| --- | --- | --- | --- | --- | --- |
| 0 | shelf-place-v3 | 2/10 (20.00%) | 2/10 (20.00%) | 5/10 (50.00%) | 5/10 (50.00%) |
| 1 | disassemble-v3 | 7/10 (70.00%) | 4/10 (40.00%) | 4/10 (40.00%) | 3/10 (30.00%) |
| 2 | stick-pull-v3 | 0/10 (0.00%) | 1/10 (10.00%) | 0/10 (0.00%) | 0/10 (0.00%) |
| 3 | stick-push-v3 | 1/10 (10.00%) | 10/10 (100.00%) | 7/10 (70.00%) | 7/10 (70.00%) |
| 4 | pick-place-wall-v3 | 2/10 (20.00%) | 2/10 (20.00%) | 1/10 (10.00%) | 3/10 (30.00%) |

## Baseline Failure Detail (`n_action_steps=15`)

`n_action_steps=15` is the current best paper-style split-average baseline, so this section audits its failed Very Hard episodes.

| Task id | Task | Successes | Success eps | Failed eps | Avg sum reward | Success avg reward | Failed avg reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | shelf-place-v3 | 2/10 | 3, 8 | 0, 1, 2, 4, 5, 6, 7, 9 | 35.7 | 168.9 | 2.4 |
| 1 | disassemble-v3 | 4/10 | 4, 6, 7, 9 | 0, 1, 2, 3, 5, 8 | 281.6 | 167.0 | 358.1 |
| 2 | stick-pull-v3 | 1/10 | 4 | 0, 1, 2, 3, 5, 6, 7, 8, 9 | 65.6 | 12.6 | 71.5 |
| 3 | stick-push-v3 | 10/10 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 | - | 6.6 | 6.6 | n/a |
| 4 | pick-place-wall-v3 | 2/10 | 1, 7 | 0, 2, 3, 4, 5, 6, 8, 9 | 46.9 | 234.0 | 0.1 |

## Failure Type Breakdown (`n_action_steps=15`)

Failure kind is a heuristic based on `sum_reward` and `max_reward`; it is useful for triage, not a substitute for state/action trace instrumentation.

| Task | zero progress | minimal progress | partial shaped reward | near success/contact failure |
| --- | --- | --- | --- | --- |
| shelf-place-v3 | 6 | 0 | 2 | 0 |
| disassemble-v3 | 0 | 0 | 6 | 0 |
| stick-pull-v3 | 0 | 5 | 1 | 3 |
| stick-push-v3 | 0 | 0 | 0 | 0 |
| pick-place-wall-v3 | 7 | 1 | 0 | 0 |

## Why This Misses The Paper Number

The paper's Very Hard reference is `60.00%`, which corresponds to `30/50` successes at this 10-episode-per-task scale. The current paper-style baseline has `19/50`, so the gap is `11` successes.

The missing successes are concentrated in three tasks, not spread evenly. `stick-push-v3` already reaches `10/10`, so it is not the immediate blocker. `stick-pull-v3` contributes `9` failures, `shelf-place-v3` contributes `8`, and `pick-place-wall-v3` contributes `8`.

The failure signatures point to at least two different causes. `shelf-place-v3` and `pick-place-wall-v3` mostly show zero-progress failures, which suggests approach/grasp/contact is failing before the policy gets into the final placement phase. `stick-pull-v3` has several high shaped-reward failures without success, which suggests contact or partial manipulation can happen but the success condition is not reached reliably.

This makes a single global action-horizon fix unlikely to close the gap by itself. The horizon sweep helped the aggregate, but the per-task matrix shows different tasks prefer different settings: `stick-push-v3` peaks at `15`, while `shelf-place-v3` improves under CUDA-pinned `10/20`. The remaining parity gap is therefore more likely task/protocol/reset/contact specific than simply `n_action_steps` being wrong.

## Failure Video Index (`n_action_steps=15`)

### Task 0: `shelf-place-v3`

| Episode | Local video path | Exists |
| --- | --- | --- |
| 0 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_0/eval_episode_0.mp4 | yes |
| 1 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_0/eval_episode_1.mp4 | yes |
| 2 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_0/eval_episode_2.mp4 | yes |
| 4 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_0/eval_episode_4.mp4 | yes |
| 5 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_0/eval_episode_5.mp4 | yes |
| 6 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_0/eval_episode_6.mp4 | yes |
| 7 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_0/eval_episode_7.mp4 | yes |
| 9 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_0/eval_episode_9.mp4 | yes |

### Task 1: `disassemble-v3`

| Episode | Local video path | Exists |
| --- | --- | --- |
| 0 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_1/eval_episode_0.mp4 | yes |
| 1 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_1/eval_episode_1.mp4 | yes |
| 2 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_1/eval_episode_2.mp4 | yes |
| 3 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_1/eval_episode_3.mp4 | yes |
| 5 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_1/eval_episode_5.mp4 | yes |
| 8 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_1/eval_episode_8.mp4 | yes |

### Task 2: `stick-pull-v3`

| Episode | Local video path | Exists |
| --- | --- | --- |
| 0 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_2/eval_episode_0.mp4 | yes |
| 1 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_2/eval_episode_1.mp4 | yes |
| 2 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_2/eval_episode_2.mp4 | yes |
| 3 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_2/eval_episode_3.mp4 | yes |
| 5 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_2/eval_episode_5.mp4 | yes |
| 6 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_2/eval_episode_6.mp4 | yes |
| 7 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_2/eval_episode_7.mp4 | yes |
| 8 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_2/eval_episode_8.mp4 | yes |
| 9 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_2/eval_episode_9.mp4 | yes |

### Task 3: `stick-push-v3`

No failed episodes.

### Task 4: `pick-place-wall-v3`

| Episode | Local video path | Exists |
| --- | --- | --- |
| 0 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_4/eval_episode_0.mp4 | yes |
| 2 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_4/eval_episode_2.mp4 | yes |
| 3 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_4/eval_episode_3.mp4 | yes |
| 4 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_4/eval_episode_4.mp4 | yes |
| 5 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_4/eval_episode_5.mp4 | yes |
| 6 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_4/eval_episode_6.mp4 | yes |
| 8 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_4/eval_episode_8.mp4 | yes |
| 9 | _workspace/runpod_results/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/metaworld_public_full_mt50_10ep_nas15_20260608T1635Z/eval/videos/very_hard_4/eval_episode_9.mp4 | yes |

## Visual Thumbnail Index

These QuickLook thumbnails are lightweight visual aids generated from local `n_action_steps=15` videos. They are not a metric source; use the JSON tables above for success/failure labels.

| Task | Label | Episode | Thumbnail path | Exists |
| --- | --- | --- | --- | --- |
| shelf-place-v3 | failure | episode 0 | _workspace/runpod_results/metaworld_very_hard_audit_thumbnails/shelf_fail_ep0.png | yes |
| shelf-place-v3 | success | episode 3 | _workspace/runpod_results/metaworld_very_hard_audit_thumbnails/shelf_success_ep3.png | yes |
| stick-pull-v3 | failure | episode 0 | _workspace/runpod_results/metaworld_very_hard_audit_thumbnails/stick_pull_fail_ep0.png | yes |
| stick-pull-v3 | success | episode 4 | _workspace/runpod_results/metaworld_very_hard_audit_thumbnails/stick_pull_success_ep4.png | yes |
| pick-place-wall-v3 | failure | episode 0 | _workspace/runpod_results/metaworld_very_hard_audit_thumbnails/pick_wall_fail_ep0.png | yes |
| pick-place-wall-v3 | success | episode 1 | _workspace/runpod_results/metaworld_very_hard_audit_thumbnails/pick_wall_success_ep1.png | yes |

## Findings

1. The remaining paper gap is not evenly distributed. Under the current paper-style baseline (`n_action_steps=15`), `stick-push-v3` is already solved at `10/10`, while `stick-pull-v3`, `shelf-place-v3`, and `pick-place-wall-v3` account for most failed episodes.
2. `stick-pull-v3` is the most severe persistent failure. It scores only `1/10` in the best paper-style baseline and `0/10` in three of the four full MT50 runs inspected here.
3. `stick-push-v3` is horizon-sensitive rather than persistently weak: it moves from `1/10` at the default horizon to `10/10` at `n_action_steps=15`, then drops to `7/10` for CUDA-pinned `10` and `20`.
4. `shelf-place-v3` improves strongly under the CUDA-pinned runs (`5/10`), but the current paper-style baseline remains `2/10`. This suggests the task is not impossible, but the best paper-comparison setting is not the best setting for this task.
5. `pick-place-wall-v3` remains low across horizons (`1/10` to `3/10`). In `n_action_steps=15`, failed episodes have near-zero average reward, which points to early approach/contact failure rather than only final placement precision.

## Recommended Next Debugging Steps

1. Focus visual inspection on `stick-pull-v3`, `pick-place-wall-v3`, and `shelf-place-v3` first. These explain the clearest deficit in the current `n_action_steps=15` baseline.
2. For `stick-pull-v3`, compare the single successful episode `4` against failed episodes `0`, `1`, and `2`. For `pick-place-wall-v3`, compare successful episodes `1` and `7` against failed episodes with near-zero reward.
3. Instrument action/state traces for those three tasks only, rather than rerunning full MT50. The likely next question is whether failures are caused by grasp/contact, object pose reset distribution, or long-horizon target approach.
4. Keep `n_action_steps=15` as the paper-style baseline for agentic-wrapper comparison. Use CUDA `n_action_steps=10` only when weighted overall or runtime matters more than clean paper-style parity.

