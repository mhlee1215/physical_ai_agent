# SO101 Fixed-Jaw Edge Chain Dataset

This document defines the additive SO101 primitive chain used for agentic
cube-grasp experiments. The raw LeRobot exports live under `_workspace/` and
must stay out of pull requests. The durable sources of truth are:

- `configs/so101/training_datasets/skill_dataset_contract.json`
- `configs/so101/training_datasets/export_recipes.json`
- this document

Do not change the camera mapping, primitive boundaries, start poses, or split
roots without explicit user approval.

## Camera Contract

- `observation.images.camera1`: egocentric camera
- `observation.images.camera2`: wrist camera
- `observation.images.camera3`: wrist camera duplicate, when present
- image shape: `[3, 256, 256]`

The approved `camera1` pose is hardware-aligned:

```json
{
  "type": "free",
  "lookat": [0.245, 0.11, 0.035],
  "distance": 0.63,
  "azimuth": 270,
  "elevation": -82,
  "rotation_degrees": 90
}
```

## Prompt Contract

Each episode uses a color/shape-specific language instruction derived from
the target object metadata at reset time. Do not collapse these prompts back to
generic `visible cube` text without explicit user approval; future multi-object
experiments depend on the language target carrying color, and later shape,
information.

Current templates:

- `move_over_cube_edge`: `Move the static finger pad above one visible {color} {shape} edge.`
- `align_fixed_jaw_cube_edge`: `Align the static finger pad with one visible {color} {shape} edge.`
- `grip_from_edge_cube`: `Keep the static finger pad at the {color} {shape} edge, close the gripper, and lift.`

The export report records `task_generation`, `task_template`,
`object_color`, `object_shape`, and `target_object` for every episode summary.
The LeRobot `meta/tasks.parquet` file may therefore contain multiple task rows
per split, with frame rows selecting the right prompt via `task_index`.

## Chain Contract

The chain is intentionally split into three short primitives so an agentic
controller can call them as `move -> align -> grip`.

1. `move_over_cube_edge`
   - Starts from the home reset pose with the gripper fully closed.
   - Moves the static finger pad above one visible cube edge.
   - Ends with the gripper still closed.
   - Ends at least 5 cm above the cube target neighborhood in the policy view.
   - The final pose must keep the cube visible and roughly centered in
     `camera2` so `align_fixed_jaw_cube_edge` can start from that state.

2. `align_fixed_jaw_cube_edge`
   - Starts from the `move_over_cube_edge` final pose.
   - Opens and aligns the static finger pad with one cube edge.
   - Ends at the exact state used as the `grip_from_edge_cube` start pose.
   - The wrist camera should still see the cube during the correction.

3. `grip_from_edge_cube`
   - Starts from the `align_fixed_jaw_cube_edge` final pose.
   - Keeps the static finger pad at the cube edge.
   - Closes the gripper, grasps the cube, and lifts it.
   - A successful episode has a grasped cube and positive lift height.

The handoff constraints are the important part:

```text
move_over_cube_edge.final_qpos == align_fixed_jaw_cube_edge.start_qpos
align_fixed_jaw_cube_edge.final_qpos == grip_from_edge_cube.start_qpos
```

The datasets are generated independently but each primitive is generated from
the same pose-construction contract, so the start/end distributions are
compatible for closed-loop chaining.

## Current Splits

Each primitive has a train/validation split:

| Dataset | Train episodes | Validation episodes |
| --- | ---: | ---: |
| `move_over_cube_edge` | 100 | 25 |
| `align_fixed_jaw_cube_edge` | 100 | 25 |
| `grip_from_edge_cube` | 100 | 25 |

The checked LeRobot exports are mirrored to the HF dataset repository
`mhlee1215/so101-nexus-sim-dataset` under:

- `datasets/move_over_cube_edge/train`
- `datasets/move_over_cube_edge/validation`
- `datasets/align_fixed_jaw_cube_edge/train`
- `datasets/align_fixed_jaw_cube_edge/validation`
- `datasets/grip_from_edge_cube/train`
- `datasets/grip_from_edge_cube/validation`

Regenerate only this chain with:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python scripts/export_so101_training_datasets.py \
  --overwrite \
  --only move_over_cube_edge_train \
  --only move_over_cube_edge_val \
  --only align_fixed_jaw_cube_edge_train \
  --only align_fixed_jaw_cube_edge_val \
  --only grip_from_edge_cube_train \
  --only grip_from_edge_cube_val
```

After regeneration, refresh checksums:

```bash
PYTHONPATH=src .venv/bin/python scripts/write_so101_dataset_checksums.py
```
