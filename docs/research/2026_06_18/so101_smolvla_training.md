# SO101 SmolVLA Training Note - 2026-06-18

## Scope

This note summarizes the current SO101 SmolVLA fine-tuning lane, dataset
contract, RunPod training status, and the new smoothness diagnostics added
during the 2026-06-18 work session.

## Dataset and Input Contract

- Policy inputs are aligned to the SO101 hardware-facing camera contract:
  - `observation.images.camera1`: egocentric camera.
  - `observation.images.camera2`: wrist camera.
  - `observation.images.camera3`: wrist camera duplicate, retained because the
    SmolVLA base config expects three image inputs.
- Image tensors are generated at `3x256x256`, matching the SmolVLA feature
  shape instead of the older low-resolution `96x96` dataset path.
- Dataset config files are the source of truth for train/validation roots,
  prompt, expected counts, cache names, augmentation defaults, and closed-loop
  evaluation mode.
- Dataset contract files explicitly forbid changing dataset roots, camera
  mapping, task semantics, split semantics, or the egocentric camera pose
  without explicit user approval.

## Current Primitive Focus

The active RunPod training lane uses only the `pick_from_top_cube` primitive:

- Train dataset: `pick_from_top_cube_train100_ego_wrist_256_seed112000`
- Validation dataset: `pick_from_top_cube_valid25_ego_wrist_256_seed113000`
- Prompt: `From above the visible cube, grasp it and lift it up.`
- Initial-state condition: fully closed gripper, at least 5 cm above the visible
  cube.
- Closed-loop metric: object is grasped and lifted at least 5 cm.

This primitive split is intentionally narrower than the full pick/pick-place
datasets. It is meant to make the agentic decomposition testable as
`move("cube")` followed by `grip("cube")`.

## Training Harness Updates

- The canonical launcher remains `scripts/start_so101_training.py`.
- PyTorch Lightning remains the training framework, with TensorBoard as the
  training dashboard.
- Dataset viewer and closed-loop rollout viewing stay in the lightweight SO101
  dashboard; scalar training curves live in TensorBoard.
- Validation is scheduled explicitly by step or epoch, not on every training
  step.
- Resume behavior now resolves the latest LeRobot checkpoint when `resume=true`
  and no explicit checkpoint path is supplied.
- DataLoader throughput changes:
  - train and validation dataloaders use `persistent_workers=True` when
    `num_workers > 0`;
  - train and validation can read predecoded image caches;
  - dataset configs expose cache names so RunPod can predecode onto pod-local
    disk instead of repeatedly decoding video from the network volume.
- TensorBoard now records:
  - train/validation loss;
  - important loss mirror with separate train/validation runs for clearer
    coloring;
  - input camera grids;
  - prompt and motor state;
  - batch size, data time, update time, samples per second;
  - GPU utilization and memory when the monitor is running;
  - action chunk smoothness metrics on validation batches.

## Augmentation Policy

SO101 training configs now default to moderate train-time augmentation:

```json
{
  "state_jitter_std": 0.003,
  "state_dropout_prob": 0.02,
  "image_patch_mask_ratio": 0.15,
  "gpu_image_augmentation": true
}
```

Validation and closed-loop tests are unaugmented. Teacher-action dropout was
removed from the config path because it corrupts the behavior-cloning target.
If action chunks are jittery, smoothness should be handled through explicit
predicted-action temporal smoothness loss or inference-time temporal ensembling,
not by dropping teacher labels.

## RunPod Status Snapshot

Active run:

`_workspace/so101_training/runs/pick_from_top_cube_l4_20260617`

Snapshot taken during the 2026-06-18 session:

- Training process was still alive.
- Checkpoints existed through `010800`.
- Best closed-loop checkpoint observed so far was `007650`.
- Closed-loop evaluation at `007650`:
  - `success_rate = 0.625`
  - `grasp_rate = 0.625`
  - 5/8 episodes successful.
- No later closed-loop result had been recorded at the snapshot time because
  the active schedule was best-gated.

## Action Chunk Jitter Diagnostic

A standalone checker was added:

`scripts/evaluate_so101_action_chunk_jitter.py`

The Lightning validation loop also now logs the same family of metrics into
TensorBoard under `val/action_jitter/*`.

Metrics are computed in preprocessed normalized action space:

- `delta`: adjacent action difference, `a[t+1] - a[t]`.
- `jerk`: second temporal difference.
- `path_length`: total action-space path length across the chunk.
- `path_to_endpoint_ratio`: total path length divided by start-to-end distance.

Validation sample comparison, 32 validation samples:

| checkpoint | predicted `jerk_abs_mean` | teacher `jerk_abs_mean` | jerk ratio | path/endpoint ratio vs teacher |
|---|---:|---:|---:|---:|
| `007650` | 0.054700 | 0.019030 | 2.874x | 1.884x |
| `009450` | 0.063282 | 0.019030 | 3.325x | 2.114x |

Interpretation: the later checkpoint had worse chunk-level smoothness despite
continued supervised optimization. This supports tracking smoothness separately
from validation loss and testing explicit smoothness regularization later.

## Current Interpretation

- The pipeline is now much closer to the intended SmolVLA contract than the
  earlier 96x96 training lane.
- Validation loss alone is not enough to select a checkpoint; closed-loop
  success and action smoothness can diverge.
- The current policy can solve the narrow `pick_from_top_cube` primitive in
  some validation cases, but behavior remains inefficient and jittery.
- The next controlled experiment should compare:
  - baseline BC loss only;
  - BC plus small temporal smoothness loss;
  - optional inference-time temporal ensembling.
- The primary metrics for that comparison should be validation loss,
  closed-loop success/grasp rate, and `val/action_jitter/ratio/jerk_abs_mean`.

## Artifact Policy

Datasets, TensorBoard logs, checkpoints, rollout videos, and RunPod experiment
directories remain artifacts and should not be committed to PRs. Completed
remote experiment outputs should be downloaded locally, verified, and then
deleted from the remote artifact directory.
