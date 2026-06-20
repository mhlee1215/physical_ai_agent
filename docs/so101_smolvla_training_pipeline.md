# SO101 SmolVLA Training Pipeline

This page records the checked training contract for SO101 SmolVLA fine-tuning.
It exists so PRs can test the pipeline before launching another expensive
RunPod run.

## Model Contract

- Base checkpoint: `lerobot/smolvla_base`
- Visual inputs: `observation.images.camera1`, `camera2`, `camera3`
- Image tensor shape: `[3, 256, 256]`
- SmolVLA preprocessing resize target: `[512, 512]`
- State input: `observation.state`, shape `[6]`
- Action output: `action`, shape `[6]`
- Training chunk: `chunk_size=50`, `n_action_steps=50`
- Rollout default: `policy_n_action_steps=15`, `policy_num_steps=10`

The enforceable contract lives in
`src/physical_ai_agent/so101_smolvla_pipeline.py`.

## Augmentation And Cache

Training should use `scripts/lerobot_train_so101_sampling_aug.py`, not a bare
`lerobot-train` command. The wrapper keeps the LeRobot train path intact while
adding SO101 sample-time controls:

- predecoded image cache via `--so101-image-cache-dir`;
- GPU/MPS-side image augmentation via `--so101-gpu-image-augmentation`;
- image color, sharpness, affine jitter;
- camera-level image dropout with `--so101-image-camera-dropout-prob`;
- patch image dropout with `--so101-image-patch-dropout-prob`;
- patch-ratio image masking with `--so101-image-patch-mask-ratio`;
- motor-state jitter with `--so101-state-jitter-std`;
- motor-state dropout with `--so101-state-dropout-prob`.

Image augmentation should run after the batch is moved to CUDA/MPS whenever
possible. CPU-side decoding should be avoided during repeated epochs by using
the predecoded image cache. Training configs should use moderate augmentation
by default. Validation and closed-loop test datasets should remain unaugmented.
Do not use teacher-action dropout for SO101 BC runs; it corrupts the label.

Default SO101 training configs use this moderate preset unless an experiment
explicitly overrides it:

```json
{
  "state_jitter_std": 0.003,
  "state_dropout_prob": 0.02,
  "state_dropout_keep_gripper": true,
  "image_camera_dropout_prob": 0.0,
  "image_patch_dropout_prob": 0.0,
  "image_patch_mask_ratio": 0.15,
  "gpu_image_augmentation": true
}
```

`image_patch_mask_ratio` masks a fraction of an 8x8 image patch grid for every
training image sample. It is distinct from legacy `image_patch_dropout_prob`,
which only masks one random patch for selected samples and should stay `0.0`
unless a specific ablation requires it.

## Action Smoothness

Action smoothness is not data augmentation. Do not use action-label dropout to
make predicted chunks smoother. If generated action chunks are jittery, prefer:

- training-side temporal smoothness loss on predicted chunks, such as
  `lambda_smooth * mean((pred_action[t+1] - pred_action[t]) ** 2)`, starting
  with a small weight like `0.01`;
- inference-side temporal ensembling or chunk-boundary smoothing for rollout
  execution.

Report smoothness loss separately from supervised BC loss in TensorBoard when it
is enabled.

## Dataset Expansion

The next train split target is a doubled dataset:

```text
configs/so101/smolvla_pickplace_contact_train100_manifest.json
```

The manifest requires:

- at least `100` train episodes from the previous `50` episode base;
- `[3, 256, 256]` image inputs;
- 6D state/action;
- no sticky grasp;
- recovery/off-nominal frames, because the teacher is privileged and the student
  is a visual policy.

Use `scripts/so101_dataset_manifest.py validate <manifest>` in CI and after
dataset generation. Use `from-export-report` to turn an exporter report into a
validated manifest.

## Teacher/Student Gap

The pick-place teacher can use privileged simulator state to generate stable
actions. The student only sees SmolVLA runtime inputs. New datasets therefore
must include recovery/off-nominal states, enabled by:

```bash
scripts/export_so101_pickplace_teacher_rollouts_lerobot.py \
  --recovery-steps <N> \
  --recovery-joint-std <STD>
```

Default `--recovery-steps=0` preserves legacy exports. New train data should
turn it on.

## Evaluation Schedule

Use `scripts/monitor_so101_training_dashboard.py` for validation and closed-loop
scheduling. Important flags:

- `--closed-loop-policy best_only`
- `--closed-loop-policy periodic`
- `--closed-loop-policy best_or_periodic`
- `--stop-training-on-overfit`
- `--overfit-patience-checkpoints`
- `--overfit-min-delta`

For `primitive training with qwen validation v1`, supervised validation loss and
Qwen-chain closed-loop validation run on the same checkpoint cadence. In the
standard local setup this means `validation_interval_steps == save_freq ==
steps_per_epoch` and `closed-loop-every-epochs=1`.

For older expensive CUDA-only closed-loop lanes, best-only or manual final
evaluation may still be used when the user has not requested per-validation
closed-loop evidence.

## Runtime Platforms

The canonical launcher must support both local macOS and Linux/RunPod for
training, supervised validation, and closed-loop evaluation:

- macOS local: `--runtime-platform macos` defaults to `policy.device=mps`,
  `--lightning-accelerator=mps`, and closed-loop `--mujoco-gl=glfw`.
- Linux/RunPod: `--runtime-platform linux` defaults to `policy.device=cuda`,
  `--lightning-accelerator=cuda`, and closed-loop `--mujoco-gl=egl`.
- `--runtime-platform auto` detects the host and chooses the matching profile.
- User-provided `--policy.device`, `--lightning-accelerator`,
  `--lightning-devices`, or `--closed-loop-mujoco-gl` remain explicit
  overrides, but monitored training still fails early if validation or
  closed-loop monitoring is disabled.

The local Mac standard is recorded in
`docs/so101_local_training_standard.md`. Every `scripts/start_so101_training.py`
dry-run/start/status payload includes `local_training_standard` so future
training launches see the standard before acting.

Before launching a long run, inspect the dry-run `runtime_contract`:

```bash
PYTHONPATH=src python3 scripts/start_so101_training.py start --dry-run \
  --runtime-platform macos --dataset-config <config.json> -- --policy.type=smolvla

PYTHONPATH=src python3 scripts/start_so101_training.py start --dry-run \
  --runtime-platform linux --dataset-config <config.json> -- --policy.type=smolvla
```

## Required Tests

Before opening or updating a PR for this pipeline:

```bash
PYTHONPATH=src python3 -B -m unittest \
  tests.test_so101_smolvla_pipeline \
  tests.test_lerobot_sampling_augmentation
```

Before launching RunPod training, run the full suite:

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests
```
