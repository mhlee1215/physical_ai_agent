# Deterministic Render Contract

## Determinism Levels

Keep these claims separate:

1. **Trajectory-equivalent:** robot and object state follow the source episode.
2. **Camera-equivalent:** each policy camera uses the source intrinsics,
   extrinsics, resolution, clipping, and postprocess.
3. **Render-repeatable:** the same frozen render profile produces stable pixels
   on the pinned Blender/device stack.
4. **Byte-identical:** output bytes match. This additionally requires pinned
   Blender, OS, GPU backend/driver, asset files, color management, and encoder.

Training derivatives require levels 1-3. Do not claim level 4 from a fixed seed
alone.

## Immutable Dataset Identity

Record:

- source root and repo id;
- dataset format and schema version;
- episode/frame counts, fps, timestamps, and camera keys;
- source manifest, audit, generation recipe, Git commit, and checksums;
- task prompt and target object semantics;
- action/state names, shapes, dtypes, and normalization contract.

## Environment Manifest

Persist all environment-factory arguments that affect model structure or RNG:

```json
{
  "factory": "make_high_contrast_picklift_env",
  "env_id": "MuJoCoPickLift-v1",
  "target_object_color": "green",
  "object_half_sizes": [0.0125, 0.015, 0.0175],
  "object_pool_order": [],
  "spawn_center": [0.15, 0.0],
  "spawn_min_radius": 0.1,
  "spawn_max_radius": 0.3,
  "spawn_angle_half_range_deg": 90.0,
  "n_distractors": 0
}
```

Also record MuJoCo and environment-package versions, MJCF hash, timestep,
substeps, solver, integrator, gravity, friction/contact parameters, control
mode, action repeat, and robot/object asset hashes.

## Episode Start State

Capture after reset/settling and immediately before frame 0:

- complete `qpos`, `qvel`, `ctrl`, and `act`;
- mocap position/quaternion and relevant userdata when present;
- environment and NumPy RNG state when later stochastic events are possible;
- target slot, active slots, object identity/color/size, and target geom id;
- per-geom collision masks and visibility;
- lift baseline such as `initial_object_z`;
- joint names plus qpos/qvel address ranges.

Validate snapshot lengths against `model.nq`, `model.nv`, and `model.nu` before
assignment. A snapshot with a lifted target or large hidden-body velocity is not
a valid start snapshot even if its dimensions match.

## Frame State Sidecar

For simulator-version-independent rerendering, store one row per source frame:

```text
episode_index
frame_index
timestamp
simulation.qpos
simulation.qvel
simulation.ctrl
camera1.world_from_camera
camera1.intrinsics
camera2.world_from_camera
camera2.intrinsics
```

The strongest render-only contract stores visual geom identity, visibility,
world position, and world quaternion per frame. Then Blender does not need to
rerun MuJoCo. This sidecar is small relative to RGB images and prevents contact
or object-pose drift after simulator upgrades.

## Camera Contract

Store, per camera and frame when moving:

- camera key and semantic role;
- 4x4 world-from-camera matrix or position/forward/up;
- `fx`, `fy`, `cx`, `cy` or exact vertical FOV;
- width, height, aspect, clipping, distortion;
- image rotation/crop/postprocess;
- focus distance, aperture, and DOF state.

Wrist-camera extrinsics are frame-dependent. Do not reconstruct them from a
new robot model when source matrices are available.

## Timeline Semantics

Freeze the frame/action ordering:

1. restore episode start;
2. render frame 0 before action 0;
3. apply source action 0 exactly once;
4. render frame 1;
5. continue until the source episode ends.

Record control timestep, action delay/repeat, terminal handling, and whether
state is pre-action or post-action.

## Asset and Renderer Manifest

Hash every mesh, texture, HDRI, material profile, table asset, and visual prop
definition. Record Blender/Cycles version, Metal device report, samples, seed,
animated-seed flag, bounces, clamp, denoiser and version, pixel filter, lights,
exposure, view transform, resolution, and output encoding.

## Preflight Gates

Before full rendering require:

- no missing or duplicate episode/frame/camera key;
- full-state dimensions and named-joint mapping match;
- `max_state_replay_error == 0` unless an explicit tolerance is justified;
- target slot/object/color and collision state match;
- final grasp, lift, TCP distance, and object pose match source evidence;
- camera matrices and source camera roles match;
- two repeated canary runs are pixel-stable under the frozen profile.

If the full state is missing, reconstruct once from source evidence and write a
reviewed sidecar. Keep the reconstruction label and provenance; do not present
an approximate replay as exact.
