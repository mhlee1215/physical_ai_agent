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

## Remaining Dataset-Quality Work

To turn this from a myCobot POC into a dataset-quality pipeline, the next gates are:

1. Validate against the real official `mycobot_ros` checkout and inspect generated scene geometry.
2. Capture real ROS/Gazebo traces for joint states, gripper command/state, object pose, and camera timestamps.
3. Replace placeholder rendered frames with real camera frames from the Camera Flange 2.0 or calibrated external camera.
4. Add an object pose/contact success oracle rather than only teacher attachment or heuristic lift labels.
5. Convert episodes into a real LeRobotDataset with synchronized actions, observations, images, timestamps, and metadata.
6. Run a tiny SmolVLA train/eval smoke over the generated dataset.

## Known Blockers In This Local Environment

The current Codex workspace does not have the official vendor asset checkout or MuJoCo installed. Local verification can therefore cover parser/contract/scene-generation behavior using tiny test fixtures, but cannot yet claim real 280 Pi simulation readiness.
