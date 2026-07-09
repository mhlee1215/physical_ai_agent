# SO101 Move-And-Align Valid-Mask Auxiliary Experiment Cleanup

Date: 2026-06-26

## Scope

This note records the latest local SO101 `move_and_align_cube_edge` experiment
before deleting its local artifacts.

Deleted artifacts are limited to generated run directories under `_workspace`.
Source files, dataset roots, configs, and raw LeRobot exports are not part of
the cleanup.

## Active Run Terminated

- Run id: `move_and_align_cube_edge_train_aligned_validmask_aux_20260625_164927`
- Run root:
  `_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/move_and_align_cube_edge_train_aligned_validmask_aux_20260625_164927`
- Training PID terminated: `16394`
- TensorBoard PID terminated: `16395`
- Approximate run artifact size before cleanup: `1.3G`

The run resumed from:

```text
_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/move_and_align_cube_edge_train_aligned_debug_20260625_075706/model/checkpoints/best_val_loss
```

Resume step:

```text
19170
```

## Training Configuration

- Train dataset:
  `_workspace/so101_lerobot/move_and_align_cube_edge_train100_ego_wrist_256_seed122000`
- Validation dataset:
  `_workspace/so101_lerobot/move_and_align_cube_edge_valid25_ego_wrist_256_seed123000`
- Camera contract:
  - `observation.images.camera1`: `egocentric_cam`
  - `observation.images.camera2`: `wrist_cam`
  - `observation.images.camera3`: `wrist_cam duplicate`
- Prompt:
  `Move above one visible green cube edge and align the gripper jaws around it.`
- Device: `mps`
- Batch size: `32`
- `n_action_steps`: `15`
- Policy chunk size: `50`
- Training target steps: `30000`
- Validation interval: every `10` epochs
- Checkpoint frequency: `2130` steps
- Checkpoint retention policy: `best_val_and_closed_loop`

Training-time augmentation:

- `state_jitter_std=0.003`
- `state_dropout_prob=0.02`
- `image_patch_mask_ratio=0.15`
- `image_affine_degrees=5.0`
- `image_affine_translate=0.05`
- image color jitter enabled
- image sharpness jitter enabled
- GPU image augmentation enabled

Valid-mask auxiliary training:

- `so101_valid_mask_loss_weight=0.05`
- `so101_valid_mask_hidden_dim=128`
- Label source: `action_is_pad_as_termination_proxy`
- The valid-mask head was saved next to checkpoint artifacts as
  `valid_mask_head.pt`.

## Validation And Loop Results

At checkpoint step `21300`, supervised validation and closed-loop testing ran.

Recorded validation-related metrics from the run:

- `val/loss`: around `0.0060` to `0.0080`, depending on the logged source
- `val/valid_mask_loss`: `0.01468`
- `val/valid_mask_accuracy`: `1.0`

Closed-loop test configuration:

- Test id: `move_and_align_cube_edge_train_aligned_debug`
- Episodes: `10`
- Start source: first frames from the supervised train split
- Success metric: `tcp_to_obj_dist <= 0.06`
- Valid-mask threshold: `0.5`
- Valid-mask consecutive count: `2`
- Max steps per primitive: `80`

Observed closed-loop result with valid-mask stop:

```text
final distance success: 1/10
ever distance success: 1/10
mean min distance: 0.09869
mean final distance: 0.09915
all episodes stopped at 30 steps with reason=valid_mask_stop
```

Episode summary:

```text
ep00 min=0.0743 final=0.0743 fail
ep01 min=0.0740 final=0.0740 fail
ep02 min=0.1344 final=0.1344 fail
ep03 min=0.1488 final=0.1488 fail
ep04 min=0.0287 final=0.0321 pass
ep05 min=0.1220 final=0.1232 fail
ep06 min=0.1059 final=0.1059 fail
ep07 min=0.0847 final=0.0847 fail
ep08 min=0.1272 final=0.1272 fail
ep09 min=0.0870 final=0.0870 fail
```

## Full-Horizon Diagnostic

A diagnostic rerun disabled practical valid-mask stopping by setting:

```text
valid_mask_threshold=-1.0
```

This kept the same checkpoint, same starts, same policy, and same prompt, but
allowed the primitive to run to the full `80` step budget.

Full-horizon diagnostic result:

```text
oracle-stop success upper bound: 6/10
final distance success after 80 steps: 1/10
env final_success: 0/10
mean min distance: 0.05987
mean final distance: 0.11054
```

Episode summary:

```text
ep00 min=0.0282 final=0.0736 oracle_pass
ep01 min=0.0268 final=0.0787 oracle_pass
ep02 min=0.1292 final=0.1864 fail
ep03 min=0.0944 final=0.1316 fail
ep04 min=0.0250 final=0.0471 oracle_pass final_pass
ep05 min=0.0547 final=0.0906 oracle_pass
ep06 min=0.0632 final=0.1239 fail
ep07 min=0.0334 final=0.0908 oracle_pass
ep08 min=0.0927 final=0.1806 fail
ep09 min=0.0511 final=0.1021 oracle_pass
```

## Interpretation

The policy can often reach a useful alignment neighborhood, but it does not
hold the target pose. With an oracle stop, the same rollout reaches the distance
threshold in `6/10` episodes. If forced to continue to `80` steps, only `1/10`
episodes remains inside the threshold at the final frame.

Main failure mode:

```text
reach-or-near-reach -> overshoot/drift -> final failure
```

This means the current bottleneck is not only visual recognition or gross
motion. Termination, hold behavior, and near-target stability need to be part of
the data and loss contract.

## Next Dataset Direction

The next dataset should be a separate v2 dataset, not a mutation of the existing
v1 export:

```text
move_and_align_cube_edge_train_v2
  - generated teacher trajectories
  - terminal hold included
  - near-target correction included
```

Recommended construction:

- keep the original v1 train/valid datasets unchanged;
- generate additional teacher trajectories with more start/object diversity;
- append terminal hold frames at the aligned target pose;
- generate near-target perturbation states and teacher correction trajectories;
- include crops or episodes that start near the final alignment state;
- keep training-time augmentation separate from dataset-generation augmentation.

The key distinction:

- training-time augmentation modifies samples during training;
- dataset-generation augmentation creates new labeled teacher trajectories.

## Cleanup Policy Applied

Artifacts deleted after this note:

- current run directory:
  `_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/move_and_align_cube_edge_train_aligned_validmask_aux_20260625_164927`
- failed same-line retry directory if present:
  `_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/move_and_align_cube_edge_train_aligned_validmask_aux_20260625_164819`

Artifacts intentionally preserved:

- source code
- configs
- raw/generated LeRobot datasets under `_workspace/so101_lerobot`
- previous baseline checkpoint run
  `move_and_align_cube_edge_train_aligned_debug_20260625_075706`
