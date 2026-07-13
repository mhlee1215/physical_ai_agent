# SO101 Grip The Cube V2 Dataset

## Dataset Contract

- train root: `_workspace/so101_lerobot/grip_the_cube_v2`
- validation root: `_workspace/so101_lerobot/grip_the_cube_v2_validation`
- train: 300 episodes / 41,369 frames
- validation: 50 episodes / 6,888 frames
- cameras: camera1 egocentric, camera2 wrist, camera3 wrist duplicate
- image resolution: 256x256
- prompt: `grip the green cube and lift`
- action/state: six-dimensional absolute SO101 qpos targets
- teacher phases: move, roll-align, descend, settle, close, lift, terminal hold
- accepted alignment: jaw line within 3 degrees of the selected contact-face
  normal through the cube center

The train and validation roots contain only successful grasp-and-lift teacher
trajectories. Camera1 grid bins are restricted to the reachable set
`5, 6, 9, 10`.

## Deterministic Split Generation

The exporter uses a 161x161 camera1 world-XY lookup. Train generation consumed
the first lookup ranges for each bin. Validation starts after the complete
train attempt range through `--grid-lookup-start-index`:

| bin | train candidates consumed | validation episodes | validation attempts |
| --- | ---: | ---: | ---: |
| 5 | 253 | 13 | 29 |
| 6 | 200 | 13 | 30 |
| 9 | 275 | 12 | 41 |
| 10 | 295 | 12 | 42 |

Validation generation accepted 50 of 142 attempted candidates (35.2%). It was
written to four independent shard roots and merged with
`scripts/merge_so101_lerobot_shards.py`. The merger rejects duplicate seeds
across shards.

## Split Audit

- train/validation seed overlap: 0
- train/validation forced spawn-XY overlap: 0
- train unique action/state trajectory hashes: 300
- validation unique action/state trajectory hashes: 50
- train/validation action/state trajectory-hash overlap: 0
- train camera1 bins: `{5: 75, 6: 75, 9: 75, 10: 75}`
- validation camera1 bins: `{5: 13, 6: 13, 9: 12, 10: 12}`
- invisible first-frame objects: 0

Both roots have camera-bin sidecars under
`meta/camera_grid_bins/observation_images_camera1_4x4_frame0.parquet`.

## Closed-Loop Starts

`scripts/build_so101_closed_loop_start_report.py` selects ten validation
episodes by round-robin camera bin. The resulting distribution is
`{5: 3, 6: 3, 9: 2, 10: 2}`. Each test restores the selected validation
episode's first `sim_snapshot`; no independently generated reset is used.

The generated start report is:

```text
_workspace/so101_lerobot/grip_the_cube_v2_validation/meta/closed_loop/
grip_the_cube_v2_validation_start10.json
```

Generated datasets, caches, sidecars, and reports remain under `_workspace/`
and are excluded from Git. Source code, configs, tests, and this audit note are
the PR payload.

## Context-Independent Reproduction

The complete generation contract now lives in
`configs/so101/dataset_generation/grip_the_cube_v2.json`. It records the exact
teacher phase lengths, 256x256 camera contract, geometry thresholds, terminal
hold, reachable bins, per-bin counts, seed bases, validation lookup offsets,
and output paths. No chat context is required.

Preview the full command graph:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python \
  scripts/generate_so101_dataset_recipe.py --dry-run
```

Regenerate train and validation artifacts:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python \
  scripts/generate_so101_dataset_recipe.py \
  --split all --workers 3 --overwrite
```

The launcher creates one root per camera bin, merges the shards, builds each
camera1 grid-bin sidecar, derives ten loop-test starts directly from validation
episode-zero states, and fails if train/validation seeds, spawn coordinates, or
action/state trajectory hashes overlap.
