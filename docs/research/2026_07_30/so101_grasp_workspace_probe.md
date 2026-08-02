# SO101 Grasp Workspace Probe

## Goal

Measure the base-relative area where the current SO101 teacher can:

1. approach a 30 mm green cube from the hardware home pose,
2. align and close the gripper,
3. grasp the cube,
4. lift it,
5. preserve the terminal hold and dataset camera contract.

This is an empirical workspace for the current robot, teacher, cameras, and
quality gates. It is not a theoretical kinematic maximum.

## Locked Contract

- Base origin: world projection of the `shoulder_pan` joint axis
- Base axes: `+X` forward, `+Y` left
- Base world origin: `[0.0154783813, 0.0000052560, 0.0696]` m
- Start qpos: `[0, -pi/2, pi/2, 0.66, pi/2, -0.1745329252]`
- Cube: green, half-size `0.015` m
- Camera rig:
  `official_32x32_uvc_photoreal_v10_fov_calibrated_direct_square.json`
- Policy images: camera1 egocentric and camera2 wrist, `256 x 256`
- Minimum gripper-to-floor clearance: `0.01` m
- Operational lift threshold: `0.060` m
- Strict dataset lift threshold: `0.065` m
- Terminal hold: 12 steps
- Prompt: `grip the green cube and lift`

The probe uses the full teacher trajectory but does not write LeRobot image
frames. Every worker creates its own MuJoCo environment and renderers.

## Existing Catalog Diagnosis

The previous dense catalog appeared large but did not cover the physical
workspace:

- candidates: 15,505
- non-empty camera bins: only 9 and 10
- world X span: 3 mm
- world Y span: 23 mm
- base-relative radius: approximately 25.55 to 26.85 cm
- base-relative angle: approximately 24.24 to 29.02 degrees
- object yaw: fixed at `-53.567` degrees

The old process balanced image-plane bins after using one fixed cube yaw. It
did not balance base-relative physical space. A fixed yaw also made teacher
feasibility strongly direction-dependent.

## Probe Method

The replacement probe samples base-polar cells and assigns the contacted cube
face per cell:

```text
object_yaw_deg = base_angle_deg - 80 deg
```

Each cell is evaluated in stages:

1. teacher preflight and floor-clearance gate,
2. physical grasp,
3. operational lift of at least 6.0 cm,
4. strict lift of at least 6.5 cm,
5. teacher geometry and terminal-hold contract,
6. camera1 visibility and full dataset contract.

The broad survey covered radius 18 to 40 cm and azimuth -170 to 180 degrees.
Boundary probes then refined the useful region, followed by a canonical
2.5-degree survey.

## Results

### Broad Physical Survey

At the broad 2 cm radial and 10-degree angular resolution, the teacher reached
the 26 cm ring from about -110 to +110 degrees. Camera1 visibility reduced the
dataset-ready range. This broad result is only a search envelope, not the
catalog used for sampling.

### Canonical Camera-Usable Survey

The final survey contains 224 tested cells:

- angles: -40 to +97.5 degrees, every 2.5 degrees
- radii: 25.0, 25.5, 26.0, and 26.5 cm
- preflight passed: 224 / 224
- grasp succeeded: 224 / 224
- operational lift succeeded: 224 / 224
- strict physical lift succeeded: 168 / 224
- strict dataset contract succeeded: 163 / 224

The 25.5 cm ring consistently lifted to approximately 6.464 to 6.468 cm. It
therefore passes the 6.0 cm operational criterion but misses the strict 6.5 cm
criterion by roughly 0.33 mm. Increasing the controller Z error from 1.5 to
2.0 cm did not close that gap, so it remains outside the strict catalog.

The conservative strict dataset-ready region is:

- one edge cell at angle -40 degrees and radius 25.0 cm,
- every 2.5-degree angle from -37.5 through +95 degrees,
- radii 25.0, 26.0, and 26.5 cm at those angles.

This gives:

- strict cells: 163
- estimated strict area: `0.0091847952 m^2`, approximately `91.85 cm^2`
- base-relative X bounds: approximately -2.31 to +26.50 cm
- base-relative Y bounds: approximately -16.13 to +26.50 cm

The five strict physical cells rejected by the dataset contract were all
camera1 visibility failures at the angular boundaries.

## Dataset Sampling Contract

The next dataset generator should not treat camera 4x4 bins as the primary
spatial sampler. It should:

1. sample a verified base-polar cell by `uniform_area_weight`,
2. use that cell's `world_xy_m`,
3. preserve that cell's `object_yaw_deg`,
4. retry teacher failures inside the same target cell,
5. audit per-cell episode counts or area-normalized density after export,
6. use camera bins only as a secondary visual-distribution audit.

The current `from_spawn_catalog` generator accepts XY positions but one global
fixed yaw. The strict workspace catalog is therefore evidence-ready but not
yet a drop-in training source. The next implementation must add
position-conditioned yaw support, or partition the workspace into
fixed-yaw sectors without losing the base-space sampling weights.

## Reproduction

Canonical config:

```text
configs/so101/workspace_probes/grip_the_cube_v3_hardware_workspace_final_usable_v1.json
```

Command:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python \
  scripts/probe_so101_grasp_workspace.py \
  --config configs/so101/workspace_probes/grip_the_cube_v3_hardware_workspace_final_usable_v1.json \
  --workers 3
```

Artifacts:

```text
_workspace/so101_workspace_probes/
  grip_the_cube_v3_hardware_workspace_final_usable_v1/
    manifest.json
    points.jsonl
    summary.json
    successful_workspace_catalog.json
    workspace_map.png
```

`successful_workspace_catalog.json` stores each accepted base/world position,
required object yaw, camera1 centroid/bin, cell area, and normalized
uniform-area sampling weight.

## Reliability Fix

The multi-process canary exposed a generated camera-asset race: one worker
could truncate an STL while another worker read it. Generated camera XML, STL,
and stamp files now use process-local temporary files followed by atomic
`os.replace`. A repeated three-worker canary initialized and completed without
the prior empty-STL failure.
