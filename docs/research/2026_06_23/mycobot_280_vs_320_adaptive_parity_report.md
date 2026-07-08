# myCobot 280 Pi vs 320 Adaptive Gate 8 Parity Report

## Purpose

This note separates three claims that are easy to blur:

1. what the merged 320 M5 2022 adaptive path has actually demonstrated;
2. what this 280 Pi adaptive branch has already wired to reuse;
3. what still needs a MuJoCo-capable environment and official vendor assets.

The target hardware for this branch is the user's myCobot 280 Raspberry Pi
2023 with flat base, adaptive gripper, and Camera Flange 2.0.

## Current Evidence Matrix

| Area | 320 M5 2022 adaptive path | 280 Pi adaptive branch |
| --- | --- | --- |
| Official source profile | ROS2 Humble `mycobot_320_m5_2022_adaptive_gripper.urdf` plus `pro_adaptive_gripper` meshes. | ROS1 `mycobot_280_pi.urdf` plus ROS1 `adaptive_gripper/mycobot_adaptive_gripper.urdf`. |
| Source/mesh/pose gates | Gates 1-6 passed for source routing, kinematic tree, mesh transform, visual pose, mimic motion, and collision proxy. | Profile builder plus 280-specific collision-proxy verifier exist for the 280 Pi + adaptive gripper composition. Gate 6-style proxy verification checks parent link, local pad position, size, friction, `condim`, contact type, and contact affinity. |
| Gate 7 static contact | Passed. Evidence: `gripper_cube_contact_pads=2`, `gripper_cube_contacts=6`, `best_sustained_contact_steps=45` with requirement 15. | Passed locally with bundled MuJoCo runtime. Evidence: `gripper_cube_contact_pads=2`, `gripper_cube_contacts=8`, `best_sustained_contact_steps=106` with requirement 15. |
| Gate 8 grasp-lift | Passed. Evidence: `close_best_sustained_contact_steps=27`, `lift_best_sustained_contact_steps=60`, `lift_two_pad_contact_steps=60`, `final_gripper_cube_contact_pads=2`, `final_gripper_cube_contacts=6`, `final_cube_lift=0.0367 m`. | Teacher-attachment Gate 8 passed previously; raw-contact-only Gate 8 now has an explicit `--disable-teacher-attachment` mode and currently fails because lift contact is not retained. Best raw run after pose search: close contact passes, but lift contact remains below the 30-step threshold. |
| Teacher dataset POC | Exported 10 episodes, 1600 frames, 400 rendered frames, and `failed_episodes=[]` from the Gate 8 scripted trajectory. | Wrapper is wired: `scripts/export_mycobot_280_pi_adaptive_teacher_dataset.py`. It should be run only after 280 Gate 7/8 pass in a MuJoCo-capable environment. |
| Dataset-quality exporter | Current 320 artifact is a local `mycobot_jsonl_v1` teacher POC, not a full LeRobotDataset training result. | Adds stricter capture verification and LeRobot-style export entrypoints that require real trace fields and real `top`/`wrist` image files. |
| Learned policy evidence | No 320 SmolVLA training/eval claim from this Gate 8 POC. | No 280 SmolVLA training/eval claim yet. The exporter writes a tiny-smoke plan, not an executed training run. |

## What The 10-Episode Teacher Dataset Means

For the 320 path, the 10-episode export is a scripted teacher-data proof of
concept. It repeats the Gate 8 short grasp-lift trajectory and records
timestamped observations/actions plus contact/lift metadata. The important
numbers are:

- `episodes=10`
- `frames=1600`
- `rendered_frames=400`
- `failed_episodes=[]`

That is useful because it proves the Gate 8 trajectory can be converted into a
consistent offline dataset artifact. It is not the same as proving a learned
policy succeeds. It also does not yet prove real-robot transfer.

For the 280 branch, parity means first showing the same Gate 7/8 contact/lift
evidence on the 280 Pi adaptive model, then exporting the same style of
10-episode teacher artifact with the 280 profile label and joint names.

## Current Local Blocker

The current Codex WSL workspace is blocked before physics execution:

- Python cannot import `mujoco`.
- `_vendor/mycobot_mujoco` is missing.
- `_vendor/mycobot_ros` is missing.

The branch therefore cannot honestly claim 280 Gate 7/8 success from this
machine. What it can claim is profile wiring, CLI parity, tests around the
280-specific defaults, capture-contract verification, and a readiness command
that reports these missing dependencies explicitly.

## Next Commands

Run the 280 collision proxy gate first:

```bash
PYTHONPATH=src:. python3 scripts/verify_mycobot_280_pi_adaptive_collision_proxy.py \
  --official-gripper-root _vendor/mycobot_ros \
  --output-dir _workspace/mycobot_280_pi_collision_proxy_verify
```

Then run the readiness gate:

```bash
PYTHONPATH=src:. python3 scripts/check_mycobot_280_pi_gate8_readiness.py \
  --asset-root _vendor/mycobot_mujoco \
  --official-gripper-root _vendor/mycobot_ros \
  --output _workspace/mycobot_280_pi_gate8_readiness/report.json
```

If readiness passes, run static contact:

```bash
PYTHONPATH=src:. python3 scripts/mycobot_280_pi_adaptive_static_contact_smoke.py \
  --asset-root _vendor/mycobot_mujoco \
  --official-gripper-root _vendor/mycobot_ros \
  --output-dir _workspace/mycobot_280_pi_gate7_static_contact
```

Then run grasp-lift:

```bash
PYTHONPATH=src:. python3 scripts/mycobot_280_pi_adaptive_grasp_lift_smoke.py \
  --asset-root _vendor/mycobot_mujoco \
  --official-gripper-root _vendor/mycobot_ros \
  --output-dir _workspace/mycobot_280_pi_gate8_grasp_lift
```

Only after those pass, export the 280 teacher POC:

```bash
PYTHONPATH=src:. python3 scripts/export_mycobot_280_pi_adaptive_teacher_dataset.py \
  --asset-root _vendor/mycobot_mujoco \
  --official-gripper-root _vendor/mycobot_ros \
  --output-dir _workspace/mycobot_teacher_datasets/mycobot_280_pi_adaptive_gate8_10eps \
  --episodes 10 \
  --render-every 4
```

## Recommended Merge Boundary

This branch is reasonable as a small "280 parity wiring and readiness" PR. It
should not be described as a passed 280 simulation PR until the real Gate 7 and
Gate 8 commands run with the official assets and produce the corresponding
contact/lift metrics and inspected render artifacts.

The next highest-value contribution is to own that evidence run: make the 280
preflight pass, run Gate 7, run Gate 8, inspect the rendered views, then export
the 10-episode teacher dataset if the metrics match the 320 success criteria.
