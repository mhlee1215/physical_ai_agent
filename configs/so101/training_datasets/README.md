# SO101 Training Dataset Configs

Use these JSON files with `scripts/start_so101_training.py --dataset-config`.
They define the train/validation LeRobot dataset pair and training defaults in
one place, so long training commands do not need to repeat dataset roots by
hand.

Example:

```bash
PYTHONPATH=src .venv/bin/python scripts/start_so101_training.py start \
  --dataset-config configs/so101/training_datasets/pick_place.json \
  --validation-interval-steps 300 \
  -- \
  --config_path=_workspace/so101_smolvla_pure/official_1ep_smoke_local/checkpoints/000020/pretrained_model/train_config.json \
  --batch_size=32 \
  --steps=50000
```

Config fields:

- `train_dataset.repo_id`, `train_dataset.root`: forwarded to LeRobot as
  `--dataset.repo_id` and `--dataset.root`.
- `validation_dataset.repo_id`, `validation_dataset.root`: forwarded as
  validation dataset args.
- `camera_contract`: human-readable expected model input mapping.
- `tensorboard`: optional default input logging cadence.
- `augmentation`: optional train-time sampling augmentation defaults. Supported
  fields are `state_jitter_std`, `state_jitter_arm_only`,
  `state_dropout_prob`, `state_dropout_keep_gripper`,
  `action_dropout_prob`, `image_camera_dropout_prob`,
  `image_patch_dropout_prob`, and `gpu_image_augmentation`.
  Dropout defaults should usually stay explicit at `0.0` until an experiment
  intentionally enables them. Validation data is not augmented.

Dropout locations:

- `state_dropout_prob`: zeros random dimensions in `observation.state`.
  `state_dropout_keep_gripper=true` preserves the gripper channel.
- `action_dropout_prob`: zeros random dimensions in the teacher `action`
  tensor during training.
- `image_camera_dropout_prob`: with `gpu_image_augmentation=true`, zeros a
  whole `observation.images.*` camera frame for selected batch items.
- `image_patch_dropout_prob`: with `gpu_image_augmentation=true`, zeros one
  random image patch for selected batch items.

CLI args still win. If an arg is already present after `--`, the launcher does
not overwrite it from the dataset config.

Dataset checksums:

- `checksums.json` records compact metadata and SHA-256 checksums for the local
  SO101 LeRobot datasets. Raw dataset files live under `_workspace/` and are
  intentionally excluded from git.
- Regenerate after rebuilding datasets:

```bash
PYTHONPATH=src .venv/bin/python scripts/write_so101_dataset_checksums.py
```
