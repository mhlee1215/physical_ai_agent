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

The readiness lane has two config-first datasets:

```text
configs/mycobot280/training_datasets/ground_pickup_tiny_smoke.json
configs/mycobot280/training_datasets/ground_pickup_pose_diverse_v1.json
```

The pose-diverse baseline freezes 50 train and 10 validation episodes before
the later object-suite randomization condition.

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

For split manifests, the adapter and validator read
`splits.<name>.episode_summaries`; train and validation conversion commands must
select their split explicitly. The future randomized dataset should keep the
same schema and change only the
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

As of 2026-07-30, the repo-local WSL runtime `_workspace/local_envs/lerobot_py312`
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

The pose-diverse canary then proved the next contract on CUDA:

- generated one train and one validation episode with all 1,060 frames rendered
- converted each split into a native LeRobot dataset with 7D state/action
- performed one real optimizer step from cached `lerobot/smolvla_base` weights
- saved policy weights, optimizer state, logs, TensorBoard, preprocessor, and postprocessor
- reloaded the saved checkpoint and evaluated one held-out validation batch

See `docs/research/2026_07_30/mycobot280_smolvla_readiness_evidence.md` for
exact metrics and artifact paths.

## Full Readiness Status - 2026-07-31

The frozen full baseline passed end to end:

- generated and validated 50 train plus 10 validation episodes and 31,800 frames
- reused one MuJoCo environment/renderer for the full dataset run
- hard-linked all 31,800 intermediate adapter images
- converted native LeRobot splits with 26,500 train and 5,300 validation frames
- performed two CUDA optimizer steps, with loss `0.9219151 -> 0.5983112`
- saved and reloaded the complete local checkpoint on held-out validation
- reduced same-batch held-out loss from `0.8851919` to `0.5125332` versus the
  unfine-tuned base policy

This remains a smoke comparison, not a learned-policy claim. Exact evidence is
in `docs/research/2026_07_31/mycobot280_smolvla_full_readiness_evidence.md`.

For offline runs, pin `HF_HOME=_workspace/local_envs/hf_home`; the successful
run used the already-downloaded cache and installed no packages. Before long
WSL generation, check free space on the Windows drive backing `ext4.vhdx`,
not only WSL `df`. Start with at least 10 GB of host headroom and pass a
10-episode stability gate. Keep adapter output on the source filesystem so
hard-link materialization avoids duplicate image blocks.

The lifecycle regression gate is:

```bash
MYCOBOT_PYTHON=_workspace/worktrees/mycobot-280pi-adaptive/_workspace/venvs/mycobot-mujoco-py312/bin/python
PYTHONPATH=src:. "$MYCOBOT_PYTHON" -B -m unittest \
  tests.test_mycobot_280_ground_pickup_teacher_dataset_lifecycle
```

## Verification

No-dependency verification:

```bash
PYTHONPATH=src:. python3 -B -m unittest \
  tests.test_mycobot280_smolvla_readiness \
  tests.test_mycobot280_smolvla_tiny_finetune
PYTHONPATH=src:. python3 scripts/validate_mycobot280_training_dataset.py \
  --config configs/mycobot280/training_datasets/ground_pickup_tiny_smoke.json
PYTHONPATH=src:. python3 scripts/plan_mycobot280_smolvla_training.py \
  --config configs/mycobot280/training_datasets/ground_pickup_tiny_smoke.json
```

The validator may return `blocked` when the source dataset root is not present.
That is acceptable for readiness plumbing and should include the exact dataset
generation command to run before a real smoke fine-tune.
