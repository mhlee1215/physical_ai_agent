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
6. Add a 280-specific collision-proxy verifier for the official adaptive gripper finger pads.`n7. Expose the profile in the CLI and dry contract.

This is a profile/asset-routing and collision-proxy gate. It does not by itself prove physical fidelity or dataset quality.




## Gate 7/8 Parity Target

The merged 320 adaptive path has two physics gates that this 280 Pi branch is now wired to reuse:

For the side-by-side evidence boundary, see
[`mycobot_280_vs_320_adaptive_parity_report.md`](./mycobot_280_vs_320_adaptive_parity_report.md).




### Gate 5 Mimic/Jaw-Gap Validation

The 280 Pi profile now has a dedicated mimic-motion verifier for the official ROS1 adaptive gripper URDF:

```bash
PYTHONPATH=src:. python3 scripts/verify_mycobot_280_pi_adaptive_mimic_motion.py   --official-gripper-root _vendor/mycobot_ros   --output-dir _workspace/parity_280_gate5_mimic_motion
```

Current local real-asset result: the verifier runs and writes JSON, Markdown, and SVG evidence, but the strict monotonic jaw-gap check reports failed for the full official controller range. The measured jaw gap goes from about 0.0266 m at the lower limit to a near-closed minimum around 0.0021 m, then opens slightly to about 0.00418 m at the upper limit. That means 280 jaw motion is not yet documented with the same clean Gate 5 parity claim as the 320 ladder.

This does not invalidate Gate 7 contact by itself; it narrows the next investigation to the effective 280 gripper command range and lift-phase contact retention.

### Cube Mass / Contact Note

The Nexus task cube in the generated MuJoCo scene is a 0.03 m cube with mass 0.005 kg and high contact friction. MuJoCo can absolutely make an object too heavy or too light relative to actuator strength, solver settings, pad friction, and lift acceleration. In the current 280 failure, though, the cube already reaches two-pad contact during close and then loses it at the lift transition, so mass is a secondary tuning axis after jaw-gap range, pad-frame evidence, and lift trajectory/contact retention.

Verify the 280 collision proxy before running physics gates:

```bash
PYTHONPATH=src:. python3 scripts/verify_mycobot_280_pi_adaptive_collision_proxy.py \
  --official-gripper-root _vendor/mycobot_ros \
  --output-dir _workspace/mycobot_280_pi_collision_proxy_verify
```

This Gate 6-style check compares the generated `left_finger_pad` and `right_finger_pad` geoms against the 280 adaptive gripper source links, including parent link, local pad position, size, friction, `condim`, contact type, and contact affinity. It verifies our own conversion-layer contact proxy contract, not manufacturer-supplied friction calibration.

Preflight local readiness before running either physics gate:

```bash
PYTHONPATH=src:. python3 scripts/check_mycobot_280_pi_gate8_readiness.py \
  --asset-root _vendor/mycobot_mujoco \
  --official-gripper-root _vendor/mycobot_ros \
  --output _workspace/mycobot_280_pi_gate8_readiness/report.json
```

If this preflight is blocked, that is an execution-environment or asset-checkout blocker, not evidence that the 280 profile failed physics. In this branch, the 280 collision-proxy verifier itself is covered by standard-library tests and does not require importing MuJoCo.

```bash
PYTHONPATH=src:. python3 scripts/mycobot_280_pi_adaptive_static_contact_smoke.py \
  --asset-root _vendor/mycobot_mujoco \
  --official-gripper-root _vendor/mycobot_ros \
  --output-dir _workspace/mycobot_280_pi_gate7_static_contact

PYTHONPATH=src:. python3 scripts/mycobot_280_pi_adaptive_grasp_lift_smoke.py \
  --asset-root _vendor/mycobot_mujoco \
  --official-gripper-root _vendor/mycobot_ros \
  --output-dir _workspace/mycobot_280_pi_gate8_grasp_lift
```

Gate 7 checks fixed-arm table contact: place the cube under the validated adaptive finger pads, close slowly, and require sustained two-pad contact. Gate 8 checks short grasp-lift: hold/pregrasp, close, lift, and require sustained close contact, sustained lift contact, two final contact pads, and final cube lift above threshold.

The 280 wrappers default to `model_profile=280-pi-adaptive-gripper` and the ROS1 `mycobot_ros` asset root, while the shared 320 scripts keep their proven 320 default. This means the next real execution can compare 320 and 280 with the same contact/lift metrics rather than a different scoring rule.

A 280 Gate 8 teacher dataset wrapper is also wired:

```bash
PYTHONPATH=src:. python3 scripts/export_mycobot_280_pi_adaptive_teacher_dataset.py \
  --asset-root _vendor/mycobot_mujoco \
  --official-gripper-root _vendor/mycobot_ros \
  --output-dir _workspace/mycobot_teacher_datasets/mycobot_280_pi_adaptive_gate8_10eps \
  --episodes 10 \
  --render-every 4
```

This is not yet a raw-contact-only 280 physics claim. It is the parity entrypoint and manifest wiring. Actual raw Gate 7/8 pass evidence still requires a MuJoCo-capable Python plus the official `mycobot_mujoco` and `mycobot_ros` assets, followed by metric and render inspection.

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

When a LeRobot runtime is available, convert that stricter JSONL export into a native `LeRobotDataset`:

```bash
PYTHONPATH=src:. python3 scripts/convert_mycobot_280_pi_adaptive_jsonl_to_lerobot.py \
  --source-root _workspace/mycobot_280pi_adaptive_dataset \
  --output-root _workspace/mycobot_280pi_adaptive_lerobot_native \
  --repo-id physical-ai-agent/mycobot-280pi-adaptive \
  --require-lerobot \
  --overwrite
```

Without `--require-lerobot`, the converter writes a blocked report if the `lerobot` package is not installed. With `--require-lerobot`, it fails fast so a cloud or robot workstation job cannot silently skip the native dataset conversion.

After native conversion, run the tiny SmolVLA supervised-loss smoke:

```bash
PYTHONPATH=src:. python3 scripts/run_mycobot_280_pi_smolvla_tiny_smoke.py \
  --dataset-root _workspace/mycobot_280pi_adaptive_lerobot_native \
  --dataset-repo-id physical-ai-agent/mycobot-280pi-adaptive \
  --policy-path lerobot/smolvla_base \
  --output-path _workspace/mycobot_280pi_adaptive_lerobot_native/smolvla_tiny_smoke.json \
  --max-batches 1 \
  --require-runtime
```

Without `--require-runtime`, the smoke runner writes a blocked report if the native dataset layout or SmolVLA runtime is unavailable. With `--require-runtime`, it fails fast so a real LeRobot workstation cannot skip the one-batch policy smoke.

The SmolVLA step is now an executable tiny supervised-loss smoke gate. In this local workspace it is expected to report `blocked` until a LeRobot/SmolVLA runtime and native 280 LeRobotDataset are available.

## End-to-End Status Gate

Run the full 280 readiness/status summary at any point:

```bash
PYTHONPATH=src:. python3 scripts/check_mycobot_280_pi_dataset_pipeline_status.py \
  --asset-root _vendor/mycobot_mujoco \
  --official-gripper-root _vendor/mycobot_ros \
  --jsonl-dataset-root _workspace/mycobot_280pi_adaptive_dataset \
  --native-dataset-root _workspace/mycobot_280pi_adaptive_lerobot_native \
  --smolvla-smoke-report _workspace/mycobot_280pi_adaptive_lerobot_native/smolvla_tiny_smoke.json \
  --output _workspace/mycobot_280pi_pipeline_status/report.json
```

Pass `--input-trace` and `--camera-manifest` once a real ROS/Gazebo capture exists. The report identifies the first blocked stage and prints the next exact command for each stage.

## Remaining Dataset-Quality Work
To turn this from a myCobot POC into a dataset-quality pipeline, the next gates are:

1. Validate against the real official `mycobot_ros` checkout and inspect generated scene geometry.
2. Capture real ROS/Gazebo traces for joint states, gripper command/state, object pose, contact evidence, and camera timestamps.
3. Point the dataset exporter at real Camera Flange 2.0 or calibrated external-camera frames.
4. Calibrate the object pose/contact oracle against the actual object and gripper collision/contact topics.
5. Convert the generated LeRobot-style JSONL export through an installed native `LeRobotDataset` loader.
6. Execute the tiny SmolVLA supervised-loss smoke in a real LeRobot/SmolVLA environment.

## Known Blockers In This Local Environment

The current Codex workspace does not have the official vendor asset checkout or MuJoCo installed. Local verification can therefore cover parser/contract/scene-generation behavior using tiny test fixtures, but cannot yet claim real 280 Pi simulation readiness.
