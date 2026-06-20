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
  `--dataset.repo_id` and `--dataset.root` for legacy single-train-split
  configs.
- `train_datasets`: preferred multi-train-split schema. Each entry defines a
  LeRobot train split with `name`, `repo_id`, `root`, optional HF bundle fields,
  and optional `expected_episodes` / `expected_frames`. When present, this list
  takes priority over `train_dataset`; the launcher resolves each HF subfolder
  independently and the training script uses a virtual `ConcatDataset` instead
  of physically merging shards.
- `validation_dataset.repo_id`, `validation_dataset.root`: forwarded as
  validation dataset args.
- `train_dataset.hf_repo_id`, `train_dataset.hf_path_in_repo`, and matching
  validation fields: optional Hugging Face dataset bundle source. When present,
  `scripts/start_so101_training.py start` downloads only the configured
  subfolder before training and forwards the resolved local subfolder path as
  `dataset.root`.
- `train_dataset.hf_merge_sources` and `validation_dataset.hf_merge_sources`:
  optional lists of HF subfolders to download and compose into the configured
  roots before training. Use this virtual-merge declaration for combined
  multi-task runs while keeping each generated/uploaded dataset as a separate
  HF bundle subfolder.
- `camera_contract`: human-readable expected model input mapping.
- `tensorboard`: optional default input logging cadence.
- `augmentation`: optional train-time sampling augmentation defaults. Supported
  fields are `state_jitter_std`, `state_jitter_arm_only`,
  `state_dropout_prob`, `state_dropout_keep_gripper`,
  `image_camera_dropout_prob`, `image_patch_dropout_prob`,
  `image_patch_mask_ratio`, and `gpu_image_augmentation`. SO101 training configs
  should include moderate train-time augmentation by default. Validation and
  closed-loop test data must stay unaugmented.

Dropout locations:

- `state_dropout_prob`: zeros random dimensions in `observation.state`.
  `state_dropout_keep_gripper=true` preserves the gripper channel.
- `image_camera_dropout_prob`: with `gpu_image_augmentation=true`, zeros a
  whole `observation.images.*` camera frame for selected batch items.
- `image_patch_dropout_prob`: with `gpu_image_augmentation=true`, zeros one
  random image patch for selected batch items.
- `image_patch_mask_ratio`: with `gpu_image_augmentation=true`, masks this
  fraction of an 8x8 patch grid for every image sample.
- Do not add action-label dropout to SO101 BC dataset configs. If action chunks
  need smoothing, use an explicit predicted-action temporal smoothness loss or
  inference-time temporal ensembling/chunk smoothing.

Default moderate train-time preset:

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

CLI args still win. If an arg is already present after `--`, the launcher does
not overwrite it from the dataset config.

Hugging Face dataset workflow:

1. Generate or re-export datasets locally.
2. Upload the checked LeRobot exports to the HF dataset bundle:
   `mhlee1215/so101-nexus-sim-dataset`.
3. Point each training config split at the bundle subfolder, for example
   `hf_path_in_repo="datasets/pick_cube/train"`.
4. Start training through `scripts/start_so101_training.py`; the launcher calls
   `snapshot_download(..., allow_patterns=["<hf_path_in_repo>/**"])` and uses
   the downloaded subfolder under `_workspace/hf_datasets/` as the effective
   LeRobot root.

For a multi-task run, prefer `train_datasets` as in
`all_hf_train_pick_place_closed_loop.json`; the launcher downloads each source
subfolder and passes the list to the training script. The training script
validates schema/count compatibility and samples through a dataset-balanced
random sampler over the virtual concat dataset, so each source split has the
same expected sampling probability even when frame counts differ.

For Qwen primitive validation configs such as `qwen_edge_primitives.json`, use
`hf_merge_sources` for both train and validation splits. The launcher downloads
each source subfolder, composes the shards, and passes the configured
train/validation roots to LeRobot.

For local export debugging, `--use-local-dataset-roots` ignores the HF fields
and forwards the config's `root` values directly. For HF cache debugging,
`--skip-hf-dataset-download` resolves the expected HF cache paths without
network access. Normal training should download from HF first.

Dataset checksums:

- `checksums.json` records compact metadata and SHA-256 checksums for the local
  SO101 LeRobot datasets. Raw dataset files live under `_workspace/` and are
  intentionally excluded from git.
- `dataset_contract.json` is the source of truth for SO101 dataset semantics.
  It defines the required camera mapping and the required train/validation
  splits for:
  `pick_cube`, `pick_and_place_cube`, `pick_cube_grip_focus`, and
  `pick_and_place_cube_grip_focus`.
- `skill_dataset_contract.json` is the additive source of truth for agentic
  primitive datasets. These skill datasets, such as `move_over_cube` and
  `pick_from_top_cube`, must not replace the full-task datasets unless the user
  explicitly approves that change.
- `export_recipes.json` plus `scripts/export_so101_training_datasets.py`
  records the pre-export trajectory generation material for every current
  split. Commit recipes, contracts, tests, and checksum manifests; keep raw
  LeRobot datasets under `_workspace/` out of PRs.
- Do not change dataset roots, camera mapping, task semantics, split names, or
  start-mode semantics without explicit user approval. If a change is needed,
  ask first and record the approval in the PR summary.
- Required camera mapping is fixed as:
  `camera1 = egocentric_cam`, `camera2 = wrist_cam`, and
  `camera3 = wrist_cam duplicate` when the third camera feature is present.
- Required `camera1` pose is the hardware-aligned egocentric view:
  `lookat=[0.245, 0.11, 0.035]`, `distance=0.63`, `azimuth=270`,
  `elevation=-82`, `rotation_degrees=90`.
- Re-export all current full-task and skill-primitive datasets after changing
  the approved camera pose:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python scripts/export_so101_training_datasets.py --overwrite
```

- Regenerate after rebuilding datasets:

```bash
PYTHONPATH=src .venv/bin/python scripts/write_so101_dataset_checksums.py
```
