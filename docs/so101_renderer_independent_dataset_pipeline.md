# SO101 Renderer-Independent Dataset Pipeline

Future SO101 datasets use one versioned JSON recipe under
`configs/so101/dataset_generation/` as the source of truth for teacher export,
simulation state capture, rendering, and LeRobot derivative construction.
Existing datasets are not rewritten.

## Contract

Recipes are parsed by the strict Pydantic models in
`src/physical_ai_agent/so101_dataset_generation_schema.py`. Unknown fields are
rejected. The generated JSON Schema is stored at
`configs/so101/schemas/dataset_generation_recipe.schema.json`.

A renderer-independent recipe has two split kinds:

- `generated`: executes the MuJoCo teacher and writes the ordinary LeRobot data.
- `render_derivative`: renders an existing generated split and replaces only the
  image columns in a new LeRobot dataset root.

Run the complete recipe with:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python \
  scripts/generate_so101_dataset_recipe.py \
  --recipe configs/so101/dataset_generation/<recipe>.json
```

Dataset generation is complete only after the final training-ready registry
gate succeeds.

## Render Replay

When `common.capture_render_replay` and `render_replay.enabled` are true, the
teacher captures the exact pre-action state used for each source image. It does
not reconstruct the scene later by replaying actions.

The generated split contains:

```text
render_replay/
  manifest.json
  episode_snapshots.parquet
  frame_world_state.parquet
  frame_camera_specs.parquet
  asset_checksums.json
  assets/meshes/
```

The sidecar records full MuJoCo integration state, qpos/qvel/ctrl/act, frame-0
mocap and RNG state, active object slots and collision masks, per-frame geom
world transforms/visibility/material color, and per-frame camera transforms and
intrinsics. The manifest records model dimensions, joint index mappings,
physics options, runtime versions, environment factory inputs, mesh hashes, and
source dataset hashes.

## Immutable And Mutable Data

The following values are immutable across render derivatives:

- episode and frame timeline
- simulation and geom world state
- camera extrinsics and intrinsics
- robot/object meshes and their checksums
- state, action, prompt, timestamps, and episode indices

The `splits.<name>.render` object is the mutable render profile. It controls:

- material and scene profiles
- Cycles samples, seed, and denoising
- lighting profile, key/fill power, world strength, and HDRI rotation
- exposure, color management, look, and gamma
- PNG or JPEG output

Rendered asset checksums are written into the Blender preview report. The
photoreal builder consumes that report and changes only declared image columns.

## Validation

Required checks are automated:

- recipe and renderer options pass strict Pydantic validation
- exact sidecar frame count equals the source parquet frame count
- state and camera rows are both complete
- source data and scene asset checksums are recorded
- derivative cameras remain 256x256
- final recipe dataset passes registry training-readiness validation

For release candidates, render the same frame twice with the same render profile
and compare image hashes. Blender, macOS, GPU, and driver versions should also be
fixed when byte-identical output is required.
