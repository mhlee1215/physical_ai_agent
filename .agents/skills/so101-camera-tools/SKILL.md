---
name: so101-camera-tools
description: Use when tuning, validating, or wiring SO101 visual input cameras, especially camera1 egocentric view and camera2 wrist view for SmolVLA/LeRobot datasets.
---

# SO101 Camera Tools

Use this skill when the user wants to inspect or adjust SO101 camera views before generating visual-policy datasets.

## Camera Contract

- `camera1`: egocentric hardware-like view.
- `camera2`: wrist view.
- Do not silently change dataset camera poses or remap camera names after sign-off. If a saved camera preset should become the dataset/export default, ask the user before wiring it into exporters or camera contracts.

## Interactive Camera1 Tuner

Run:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python scripts/tune_so101_camera1_view.py --host 127.0.0.1 --port 8770
```

Open `http://127.0.0.1:8770/`.

The tuner renders a candidate camera1 view next to camera2/wrist, top/debug, and scene reference views. Use it to adjust MuJoCo free-camera `lookat`, `distance`, `azimuth`, `elevation`, and postprocess `rotation_degrees` until camera1 matches the real SO101 hardware setup.

The `Save` button writes `_workspace/so101_camera_tuner/camera1_preset.json`. Treat that file as a reviewed candidate preset, not an automatic dataset contract change.

## Validation

Before publishing changes to the tuner, run:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python -B -m py_compile scripts/tune_so101_camera1_view.py
```
