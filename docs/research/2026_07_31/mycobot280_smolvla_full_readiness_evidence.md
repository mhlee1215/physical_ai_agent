# myCobot 280 Full SmolVLA Readiness Evidence - 2026-07-31

This note closes the full-data readiness gate that followed the 2026-07-30
canary. It records the completed 50-train/10-validation teacher dataset,
split-aware LeRobot conversion, two real CUDA optimizer steps, checkpoint
reload, and a same-batch held-out comparison against the unfine-tuned base
policy.

## 2026-07-31 Exact-7D Correction

The original smoke in this note was later found to have loaded the base
checkpoint's six-dimensional state/action processors. The dataset was 7D, but
the RMSE helper silently compared only the first six action dimensions. The
original loss and RMSE numbers are therefore superseded and are not evidence
for myCobot gripper adaptation.

The pipeline now binds SmolVLA to exactly 7D myCobot state/action features,
uses only `observation.images.camera1`, derives normalization from the training
dataset, shuffles training frames deterministically, rejects mismatched action
dimensions, and requires a saved myCobot feature-contract marker when reloading
a local checkpoint. The corrected evidence is recorded below.

## Storage Failure Diagnosis And Fix

The earlier full generation failures were not teacher-task failures. Windows
`C:` reached 0 bytes free while WSL still reported ample logical ext4 space.
The WSL `ext4.vhdx` could not grow, first surfacing as a MuJoCo/EGL `SIGBUS`
and later as filesystem `EIO` while writing `manifest.json`.

Two changes make the pipeline materially more stable and storage-efficient:

- the teacher exporter owns one MuJoCo environment/renderer for the complete
  dataset run and closes it once, instead of creating an EGL context per episode;
- the intermediate LeRobot adapter hard-links source images on the same
  filesystem, with a copy fallback only when hard links are unavailable.

A 10-episode all-frame gate passed before the full retry. The full retry then
completed with exit code 0. Check Windows host storage before long WSL runs;
`df` inside WSL is not sufficient because it does not report VHD backing-disk
headroom.

## Full Teacher Dataset

Source artifact:

```text
_workspace/mycobot_teacher_datasets/
  mycobot_280_ground_pickup_pose_diverse_v1_50train_10val_20260730/
```

| Gate | Result |
| --- | ---: |
| Requested / accepted / passed episodes | 60 / 60 / 60 |
| Train / validation episodes | 50 / 10 |
| Rendered frames | 31,800 |
| Unique object poses / trajectory hashes | 60 / 60 |
| Minimum final cube lift | 0.0541356 m |
| Minimum post-lift hold cube lift | 0.0452369 m |
| Minimum sustained two-pad lift contact | 80 steps |
| Minimum sustained post-lift two-pad contact | 300 steps |
| Maximum pad-cube penetration | 0.00282596 m |
| Rejected attempts / failed episodes | 0 / 0 |

The pinned config validator checked all 60 episode files, all 31,800 rows,
all image paths, 7D exported state, and 7D actions. It returned `status=passed`
with no errors or warnings. The report is:

```text
_workspace/mycobot280_runs/pose_diverse_v1_20260731/
  full_dataset_validation.json
```

## LeRobot Conversion

Train and validation were selected explicitly and converted independently:

| Split | Episodes | Frames | Native size |
| --- | ---: | ---: | ---: |
| Train | 50 | 26,500 | 358 MB |
| Validation | 10 | 5,300 | 72 MB |

All 31,800 intermediate image records report
`image_materialization="hardlink"`. Native artifacts are:

```text
_workspace/mycobot280_lerobot/
  ground_pickup_pose_diverse_v1_train_native/
  ground_pickup_pose_diverse_v1_validation_native/
```

Both native reports declare 7D `observation.state`, 7D `action`, and
256 x 256 RGB `observation.images.camera1` features.

## Corrected Exact-7D Fine-Tune Smoke

The existing approved LeRobot Python environment and cached model weights were
used. No package installation or model download was required. The run pinned:

```bash
HF_HOME=_workspace/local_envs/hf_home
```

| Gate | Result |
| --- | ---: |
| Device | CUDA |
| Training episodes / frames visible | 50 / 26,500 |
| Optimizer steps | 2 |
| State / action dimensions | 7 / 7 |
| Camera inputs | `observation.images.camera1` only |
| Training sampling | deterministic shuffle, seed `20260731` |
| Step 1 loss | 0.7260140 |
| Step 2 loss | 0.6760505 |
| Step 1 / step 2 clipped gradient norm | 21.0531 / 21.6424 |
| Training duration | 33.1435 s |
| TensorBoard error | none |

One constant arm-action dimension had serialized `std=0` and a mean about
`9.5e-7` away from its observed value. The adapter centers zero-variance
dimensions on their observed value before mean/std normalization and records
that adjustment in the checkpoint marker.

The run wrote policy weights, optimizer state, training state, logs,
TensorBoard events, and both saved processor configs:

```text
_workspace/mycobot280_training/ground_pickup_pose_diverse_v1_7d_stable_stats_smoke_20260731/
  tiny_finetune.json
  train.log
  tensorboard/events.out.tfevents.*
  checkpoints/latest/optimizer_state.pt
  checkpoints/latest/training_state.json
  checkpoints/latest/pretrained_model/model.safetensors
  checkpoints/latest/pretrained_model/policy_preprocessor.json
  checkpoints/latest/pretrained_model/policy_postprocessor.json
  checkpoints/latest/pretrained_model/mycobot280_feature_contract.json
```

## Held-Out Reload Comparison

The saved local checkpoint and processor configs reloaded successfully against
the separate 10-episode validation dataset. One held-out batch was evaluated
with the same seed for the base and two-step policies:

| Metric | Unfine-tuned base | Two-step checkpoint | Direction |
| --- | ---: | ---: | --- |
| Supervised loss | 7.0737000 | 7.2547622 | slightly worse |
| Mean postprocessed action RMSE | 0.1066802 | 0.1075550 | slightly worse |
| Step-0 postprocessed action RMSE | 0.1183773 | 0.1187070 | slightly worse |
| Compared action dimensions | 7 | 7 | exact match |

Reports:

```text
_workspace/mycobot280_training/ground_pickup_pose_diverse_v1_7d_stable_stats_smoke_20260731/
  validation_base_1batch.json
  validation_finetuned_2step_1batch.json
```

The slightly worse two-step result is expected to be inconclusive. It proves
reload and exact-7D comparison plumbing, not a learning gain.

## Claim Boundary And Next Gate

This evidence proves the complete readiness path: validated teacher data,
split-aware adaptation, native LeRobot loading, real optimizer updates,
exact-7D dataset-derived processors, complete checkpoint writing, and strict
local checkpoint reload. Two steps and one validation batch do not prove a
learned policy, a favorable supervised change, generalization, closed-loop task
success, randomized-object robustness, or an agentic-method gain.

The next publication-value experiment is a bounded multi-step training run with
a multi-batch held-out loss curve, followed by matched-seed closed-loop
simulation evaluation of the unfine-tuned and fine-tuned policies. Add the
`object_suite_v0` randomized condition after the controlled baseline, and add
verifier/retry only after the policy-only comparison exists.
