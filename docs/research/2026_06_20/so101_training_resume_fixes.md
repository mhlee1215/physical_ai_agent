# SO101 Training Resume Fixes

Date: 2026-06-20

## Context

We resumed the SO101 Qwen-edge SmolVLA training lane from the latest local
checkpoint:

```text
_workspace/so101_training/runs/primitive_training_with_qwen_validation_v1/qwen_edge_primitives/model/checkpoints/003136/pretrained_model
```

The user confirmed the default training contract:

- supervised validation loss runs every epoch;
- closed-loop rollout runs on the same checkpoint cadence;
- closed-loop rollout media/artifact recording stays enabled;
- train-time augmentation applies only in the training pipeline;
- validation and closed-loop inputs stay unaugmented.

## Fixes

### Explicit Affine Augmentation

The train-time augmentation contract now exposes affine augmentation as an
explicit config/CLI/env setting instead of relying on an implicit internal
jitter:

```json
{
  "image_affine_degrees": 5.0,
  "image_affine_translate": 0.05
}
```

The LeRobot wrapper forwards these through:

```text
--so101-image-affine-degrees
--so101-image-affine-translate
```

The implementation uses Torch `affine_grid` and `grid_sample` on the current
image tensor device, so CUDA and MPS use the same path.

### Valid-Mask Head Shape Fix

Valid-mask training hit a real shape bug when `observation.state` arrived with
extra leading dimensions from LeRobot preprocessing. The valid-mask head now
flattens state tensors after the batch dimension before selecting the configured
state width.

### Valid-Mask Training Cache Path

Valid-mask head training only needs state, action chunks, and `action_is_pad`,
but LeRobot still loads image columns through the policy delta timestamps. This
made full valid-mask training decode images slowly.

The valid-mask training script now accepts:

```text
--image-cache-dir
--validation-image-cache-dir
```

so it can reuse the SO101 predecoded image cache instead of repeatedly decoding
images.

### Rollout Media Flag Propagation

The launcher already had `--record-loop-artifacts`, but the closed-loop monitor
did not have a separate media-rendering switch. The launcher and monitor now
both expose `--render-loop-media` and forward it to the Qwen closed-loop runner
with the rollout artifact settings.

### Resume Runtime Fixes

The first resume attempt exposed two launch bugs:

- `scripts/lerobot_train_so101_lightning.py` did not parse
  `--so101-image-affine-degrees` or `--so101-image-affine-translate`, even
  though the dataset config and launcher emitted those options.
- Local macOS training inherited `num_workers=4` from the shared dataset config,
  which broke predecoded-cache loading under spawn multiprocessing. The launcher
  now forces `num_workers=0` when `--runtime-platform macos` is selected, matching
  `docs/so101_local_training_standard.md`.

## Evidence

Predecoded caches were rebuilt locally for the current Qwen-edge train/val
datasets:

```text
train frames: 10148
validation frames: 2535
image shape: [3, 256, 256]
```

A sidecar valid-mask head was trained with the cache-backed path:

```text
output: _workspace/so101_valid_mask_head/qwen_edge_primitives/valid_mask_head.pt
epochs: 5
batch_size: 64
max_train_batches: 40
max_val_batches: 10
device: mps
best_val_loss: 0.0472311873
validation_accuracy: 0.98171875
```

Generated caches and checkpoint artifacts remain under `_workspace/` and are
excluded from the PR.

## Validation

Commands run:

```bash
PYTHONPATH=src python3 -B -m py_compile \
  scripts/start_so101_training.py \
  scripts/lerobot_train_so101_sampling_aug.py \
  scripts/train_so101_valid_mask_head.py \
  src/physical_ai_agent/lerobot_sampling_augmentation.py \
  src/physical_ai_agent/policies/so101_valid_mask.py \
  src/physical_ai_agent/so101_smolvla_pipeline.py \
  tests/test_lerobot_sampling_augmentation.py \
  tests/test_so101_smolvla_pipeline.py

PYTHONPATH=src python3 -B -m unittest \
  tests.test_so101_smolvla_pipeline.SO101SmolVLAPipelineTest.test_default_augmentation_contract_matches_training_configs \
  tests.test_lerobot_sampling_augmentation \
  tests.test_so101_smolvla_pipeline.SO101SmolVLAPipelineTest.test_single_training_launcher_dry_run_builds_one_training_command \
  tests.test_so101_smolvla_pipeline.SO101SmolVLAPipelineTest.test_so101_training_configs_default_to_moderate_augmentation_without_action_dropout \
  tests.test_so101_smolvla_pipeline.SO101SmolVLAPipelineTest.test_so101_harness_documents_augmentation_and_smoothness_contract

PYTHONPATH=src .venv/bin/python -B -m unittest discover \
  -s tests \
  -p 'test_lerobot_sampling_augmentation.py'

PYTHONPATH=src .venv/bin/python -B -m unittest discover \
  -s tests \
  -p 'test_so101_valid_mask.py'

PYTHONPATH=src python3 -B -m unittest \
  tests.test_so101_smolvla_pipeline.SO101SmolVLAPipelineTest.test_training_launcher_forces_zero_workers_for_local_macos \
  tests.test_so101_smolvla_pipeline.SO101SmolVLAPipelineTest.test_training_monitor_qwen_chain_runner_reads_qwen_report \
  tests.test_so101_smolvla_pipeline.SO101SmolVLAPipelineTest.test_training_launcher_resolves_hf_merge_sources_for_train_and_validation \
  tests.test_so101_smolvla_pipeline.SO101SmolVLAPipelineTest.test_so101_training_configs_default_to_moderate_augmentation_without_action_dropout
```

The default `python3` test path skipped Torch-dependent augmentation tests when
Torch was unavailable. The repo `.venv` path ran the Torch-backed augmentation
and valid-mask tests successfully.
