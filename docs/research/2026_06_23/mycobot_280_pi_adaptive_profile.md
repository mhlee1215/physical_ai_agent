# myCobot 280 Pi Adaptive Gripper Profile

## Hardware Target

User hardware is the Elephant Robotics myCobot 280 Raspberry Pi 2023 arm with flat base, adaptive gripper, and Camera Flange 2.0.

This is not the same profile as the PR #20 `320-m5-2022-adaptive-gripper` path. The 320 path starts from a combined myCobot 320 M5 2022 adaptive-gripper URDF. The 280 Pi hardware should instead compose the official ROS1 280 Pi arm model with the official ROS1 adaptive gripper accessory model.

## Official Source Paths

Expected `MYCOBOT_ROS_ROOT` layout:

- `mycobot_description/urdf/mycobot_280_pi/mycobot_280_pi.urdf`
- `mycobot_description/urdf/mycobot_280_pi/*.dae`
- `mycobot_description/urdf/adaptive_gripper/mycobot_adaptive_gripper.urdf`
- `mycobot_description/urdf/adaptive_gripper/*.dae`

The new model profile is:

```text
280-pi-adaptive-gripper
```

## Current Branch Support

This branch adds dry scene-building support for the 280 Pi adaptive profile:

1. Parse the official 280 Pi arm URDF.
2. Parse the official adaptive gripper URDF.
3. Attach the adaptive gripper to the terminal 280 Pi flange link when available.
4. Convert referenced Collada meshes into local OBJ files for MuJoCo XML generation.
5. Add the same Nexus cube/table scene and transparent finger-pad contact proxies used by the current adaptive gripper POC.
6. Expose the profile in the CLI and dry contract.

This is a profile/asset-routing gate. It does not yet prove physical fidelity or dataset quality.



## Capture Contract Gate

Before running the dataset exporter, verify the capture itself:

```bash
PYTHONPATH=src:. python3 scripts/verify_mycobot_280_pi_capture_contract.py \
  --input-trace path/to/ros_gazebo_trace.jsonl \
  --camera-manifest path/to/camera_manifest.json \
  --output-dir _workspace/mycobot_280pi_capture_contract_verify
```

This gate checks that every frame has:

- monotonic timestamp evidence;
- the 280 Pi joint order plus a matching action or trajectory point;
- object pose evidence for the success oracle;
- contact evidence from gripper/object contacts;
- existing `top` and `wrist` camera image files.

A passing capture-contract report is still not a calibration claim. It only proves the recorded artifacts are complete enough for the stricter LeRobot-style dataset export.

## Dataset Pipeline Gate

This branch also adds a stricter dataset exporter for the 280 Pi adaptive profile:

```bash
PYTHONPATH=src:. python3 scripts/export_mycobot_280_pi_adaptive_lerobot_dataset.py \
  --root _workspace/mycobot_280pi_adaptive_dataset \
  --input-trace path/to/ros_gazebo_trace.jsonl \
  --camera-manifest path/to/camera_manifest.json \
  --repo-id physical-ai-agent/mycobot-280pi-adaptive \
  --overwrite
```

The exporter requires two external inputs:

- a JSONL trace with `joint_state`, action or trajectory point, object pose, and contact evidence;
- a camera manifest whose `top` and `wrist` image paths already exist on disk.

It writes a LeRobot-style folder with `data/frames.jsonl`, `data/episodes.jsonl`, `meta/info.json`, `meta/tasks.jsonl`, `meta/stats.json`, and `meta/smolvla_tiny_smoke_plan.json`.

The success label is computed from object lift plus gripper/object contact evidence, not from a teacher attachment proxy. The exporter refuses missing camera-frame files, so placeholder-image generation is no longer part of this stricter gate.

The SmolVLA output is currently a smoke-plan artifact, not an executed train/eval run, because this local workspace does not have a LeRobot/SmolVLA training environment installed.

## Remaining Dataset-Quality Work

To turn this from a myCobot POC into a dataset-quality pipeline, the next gates are:

1. Validate against the real official `mycobot_ros` checkout and inspect generated scene geometry.
2. Capture real ROS/Gazebo traces for joint states, gripper command/state, object pose, contact evidence, and camera timestamps.
3. Point the dataset exporter at real Camera Flange 2.0 or calibrated external-camera frames.
4. Calibrate the object pose/contact oracle against the actual object and gripper collision/contact topics.
5. Run the generated LeRobot-style dataset through an installed LeRobotDataset loader.
6. Execute the tiny SmolVLA train/eval smoke from `meta/smolvla_tiny_smoke_plan.json` in a real LeRobot/SmolVLA environment.

## Known Blockers In This Local Environment

The current Codex workspace does not have the official vendor asset checkout or MuJoCo installed. Local verification can therefore cover parser/contract/scene-generation behavior using tiny test fixtures, but cannot yet claim real 280 Pi simulation readiness.
