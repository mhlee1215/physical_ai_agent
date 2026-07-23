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

New recipes use `schema_version: 2` and must choose one source mode:

```json
{
  "schema_version": 2,
  "source": {"mode": "from_scratch"}
}
```

`from_scratch` means that MuJoCo state, spawn lookup, teacher trajectory, and
images are all constructed by the recipe without reading another dataset.

When only object placement is reused, extract and declare a seed-free spawn
catalog instead of making the old dataset look like the trajectory source:

```json
{
  "schema_version": 2,
  "source": {
    "mode": "from_spawn_catalog",
    "catalogs": ["configs/so101/spawn_catalogs/task_v1.json"]
  }
}
```

The catalog stores only `bin -> [[world_x, world_y], ...]`. The new export owns
its simulator state, teacher trajectory, images, actions, and episode seeds.

```json
{
  "schema_version": 2,
  "source": {
    "mode": "from_existing_dataset",
    "operation": "regenerate_teacher",
    "datasets": ["_workspace/so101_lerobot/source_v1"]
  }
}
```

`regenerate_teacher` may reuse the declared source export reports to build a
spawn lookup, but it writes new simulator trajectories into new append-only
roots. Use `operation: render_derivative` when state/action/timeline stay
immutable and only declared image streams are rerendered. Pydantic requires the
lookup-report roots or render-source roots to exactly match `source.datasets`.
Historical schema-v1 recipes remain readable so an existing dataset can be
reproduced without silently changing its semantics.

New `grip_the_cube_v1` recipes also require a constructive alignment gate:

```json
{
  "common": {
    "inspection_gates": [
      {
        "kind": "geometry_contact_alignment",
        "contract": "jaw_line_vs_contact_face_normal_through_cube_center",
        "max_pre_close_error_deg": 3.0
      },
      {
        "kind": "camera2_visual_alignment",
        "camera_key": "observation.images.camera2",
        "edge_mode": "top_contact",
        "strategy": "constructive_refine_then_probe",
        "mode": "preclose_and_early_trace",
        "pre_close_max_deg": 8.0,
        "close_25_max_deg": 8.0,
        "close_50_max_deg": 8.0
      }
    ]
  }
}
```

The geometry threshold and camera2 trace limits are forwarded to the teacher
exporter and participate in episode acceptance. They are not merely a
post-export filter.

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

- renderer mode and output location
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
- the same canary frame is rendered twice and stays within the declared pixel tolerance
- derivative cameras remain 256x256
- final recipe dataset passes registry training-readiness validation

When `determinism_probe` is enabled, the generator rerenders episode 0/frame 0
before building the derivative and writes `render_determinism_report.json`. The
build fails when `determinism_max_channel_diff` or
`determinism_max_changed_pixels` is exceeded. Blender, macOS, GPU, and driver
versions must also be fixed when byte-identical output is required.
