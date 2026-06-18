# SO101 Camera Contract

## Official Reference

The official LeRobot dataset `lerobot/svla_so100_pickplace` exposes policy image
features as:

- `observation.images.top`
- `observation.images.wrist`

Cached SmolVLA SO101 preprocessing artifacts on this workstation record this
rename map:

```json
{
  "observation.images.top": "observation.images.camera1",
  "observation.images.wrist": "observation.images.camera2"
}
```

## Local SO101 Pick-Place Export Contract

For SO101 pick-place datasets generated in this repo, use:

- `observation.images.camera1` = `egocentric_cam`
- `observation.images.camera2` = `wrist_cam`
- `observation.images.camera3` = `wrist_cam duplicate` only when a three-camera
  SmolVLA base config requires a third image feature.

Current approved `camera1` egocentric pose for dataset generation:

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

This pose is part of the local real-hardware-aligned dataset contract. Do not
change it, or replace `camera1` with `top_down`, without explicit user approval.
The executable source of truth is
`physical_ai_agent.sim.so101_camera_input.EGOCENTRIC_CAMERA1_POSE`; export
recipes and checksum manifests must agree with that constant.

`top_down` is debug/teacher evidence only. Do not feed `top_down` to SmolVLA
for the local SO101/real-hardware-aligned training lane.
