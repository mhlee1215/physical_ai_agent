# myCobot 280 SmolVLA Training Pipeline

This page records the readiness contract for myCobot 280 Pi adaptive-gripper
SmolVLA fine-tuning. It mirrors the SO101 training pipeline shape while keeping
the myCobot embodiment, action space, and contact-quality evidence explicit.

## Claim Boundary

This pipeline is readiness work only until a real training/evaluation run is
completed. A tiny smoke run may prove dataset loading, feature compatibility,
loss computation, and artifact writing. It must not be reported as policy
success, generalization, or an agentic-method result.

## Model Contract

- Base checkpoint: `lerobot/smolvla_base`
- Robot: myCobot 280 Pi with adaptive gripper
- Scenario: `ground_pickup_cube`
- Default task prompt: `pick up the cube from the work mat with the myCobot 280 Pi adaptive gripper`
- Object standard: `object_suite_v0`, cube-from-mat pickup profile
- State input: `observation.state`, shape `[7]`
- Action output: `action`, shape `[7]`
- Joint/action order:
  - `joint2_to_joint1`
  - `joint3_to_joint2`
  - `joint4_to_joint3`
  - `joint5_to_joint4`
  - `joint6_to_joint5`
  - `joint7_to_joint6`
  - `gripper_controller`
- Initial camera feature for smoke conversion: `observation.images.camera1`
  mapped from the rendered teacher frame.

The myCobot 280 contract must not silently inherit SO101's 6D action/state
assumptions. Any future real-camera or multi-camera path must declare its own
camera mapping in config before training.

## Dataset Contract

The first readiness dataset is the deterministic ground-pickup teacher POC:

```text
configs/mycobot280/training_datasets/ground_pickup_tiny_smoke.json
```

The source dataset must report:

- `format="mycobot_jsonl_v1"`
- `generation_mode="deterministic_fixed_task"` for the tiny smoke path
- `teacher_attachment_enabled=false`
- `object_teleport_during_pickup_lift=false`
- `randomization_enabled=false` for the tiny smoke path
- 7D state/action rows
- rendered image paths for smoke conversion
- per-episode pass/fail summaries
- aggregate contact/lift/hold/penetration metrics

The future randomized dataset should keep the same schema and change only the
generation mode, split declarations, and object/pose randomization metadata.

## Readiness Milestones

1. Validate the config and deterministic source dataset.
2. Convert or plan conversion from myCobot teacher JSONL into a
   LeRobot/SmolVLA-loadable dataset.
3. Produce a config-first fine-tuning dry-run report.
4. Optionally run a tiny smoke fine-tune only when the required LeRobot/SmolVLA
   runtime is already available or explicitly approved for installation.
5. Prepare closed-loop simulation evaluation as a command/report contract before
   making any learning claim.

## Training Dry-Run

Use the dry-run planner first:

```bash
PYTHONPATH=src:. python3 scripts/plan_mycobot280_smolvla_training.py \
  --config configs/mycobot280/training_datasets/ground_pickup_tiny_smoke.json \
  --output _workspace/mycobot280_training/ground_pickup_tiny_smoke/dry_run.json
```

The dry-run must emit the resolved dataset root, source format, conversion
output, base checkpoint, feature schema, state/action dimensions, output
directory, checkpoint/log/TensorBoard paths, and closed-loop evaluation stub.

## Dependency Policy

Do not silently install or upgrade Torch, LeRobot, SmolVLA, MuJoCo, LIBERO, or
system packages. First inspect available environments. If anything is missing,
write a blocker with the exact install or bootstrap command and ask for
approval before installing or downloading.

## Smoke Runtime Status

As of 2026-07-15, the repo-local WSL runtime `_workspace/local_envs/lerobot_py312`
passed the myCobot 280 tiny smoke path with:

- `torch==2.11.0+cu129`
- editable `lerobot==0.6.1` from the local vendor checkout
- `datasets==4.8.5`, `pandas==2.3.3`, `av==15.1.0`
- `transformers==5.13.1`
- `mujoco==3.10.0`

Native LeRobot conversion wrote 10 episodes and 5300 frames to
`_workspace/mycobot280_lerobot/ground_pickup_tiny_smoke_native`. The tiny
supervised-loss smoke loaded `lerobot/smolvla_base`, evaluated one CPU batch,
and wrote `_workspace/mycobot280_training/ground_pickup_tiny_smoke/tiny_smoke.json`.
This is still a plumbing result only, not a learned-policy or closed-loop
success claim.

## Verification

No-dependency verification:

```bash
PYTHONPATH=src:. python3 -B -m unittest tests.test_mycobot280_smolvla_readiness
PYTHONPATH=src:. python3 scripts/validate_mycobot280_training_dataset.py \
  --config configs/mycobot280/training_datasets/ground_pickup_tiny_smoke.json
PYTHONPATH=src:. python3 scripts/plan_mycobot280_smolvla_training.py \
  --config configs/mycobot280/training_datasets/ground_pickup_tiny_smoke.json
```

The validator may return `blocked` when the source dataset root is not present.
That is acceptable for readiness plumbing and should include the exact dataset
generation command to run before a real smoke fine-tune.
