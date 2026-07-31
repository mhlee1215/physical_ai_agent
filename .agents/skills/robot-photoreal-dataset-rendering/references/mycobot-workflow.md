# MyCobot Photoreal Workflow

## Model Contract

MyCobot visual evidence defaults to the official adaptive gripper. For the
current 320 lane use:

- model profile: `320-m5-2022-adaptive-gripper`;
- pose preset: `adaptive-table`;
- source assets: official `elephantrobotics/mycobot_ros2` tree;
- gripper semantics: adaptive fingers and mimic motion, not a synthetic
  parallel gripper.

Use `280-jn` or `320-m5-2022-gripper` only for explicitly labeled legacy/debug
checks. For MyCobot 280 Pi dataset work use the dedicated 280 Pi adaptive
export/conversion contract; do not substitute the 320 model.

## Blender Probe

```bash
PYTHONPATH=src:.:scripts .venv/bin/python \
  scripts/render_mycobot_blender_probe.py \
  --model-profile 320-m5-2022-adaptive-gripper \
  --pose-preset adaptive-table \
  --official-gripper-root <mycobot_ros2-root> \
  --output-dir _workspace/mycobot_photoreal/<version> \
  --width 640 --height 480 --samples 256 \
  --robot-material matte_pla \
  --render-asset-root _workspace/photoreal_assets
```

This command is a render probe. Do not call it a full dataset conversion unless
every source episode/frame/camera is replayed and a strict dataset builder has
replaced the training image bytes.

## Required Visual Gates

Before publishing MyCobot adaptive-gripper evidence, verify the upstream mesh
transform, kinematic tree, mimic motion, collision proxy, and table pose with
the repo's `verify_mycobot_*adaptive*` scripts. The final image must show the
adaptive gripper connected to the correct arm and in a physically plausible
pose.

## Dataset Preservation

Preserve the source action/state/timestamp and camera contract exactly as for
SO101. A MyCobot render sidecar is not a training dataset. Keep raw renders in
`_workspace`, register completed derivatives in the Robot Experiment Manager,
and label 280 Pi versus 320 M5 provenance in the manifest and viewer.
