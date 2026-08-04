# myCobot 280 Training Dataset Configs

These configs define myCobot 280 Pi adaptive-gripper datasets for SmolVLA
fine-tuning readiness. They intentionally mirror the SO101 config-first
workflow while preserving the myCobot-specific 7D action/state contract.

Use configs here for dataset validation and dry-run planning before any real
training launch. Do not use ad hoc CLI-only dataset roots for repeated runs.

Required fields:

- `schema_version`
- `name`
- `robot`
- `scenario`
- `task_prompt`
- `object_suite`
- `feature_contract`
- `source_dataset`
- `lerobot_conversion`
- `training_smoke`
- `closed_loop_stub`

Dependency policy:

- Do not silently install LeRobot, Torch, MuJoCo, or SmolVLA dependencies.
- Validate configs and source datasets with standard-library scripts first.
- If a dependency is missing, write a blocked report with the exact next command.

Tiny smoke config:

```bash
PYTHONPATH=src:. python3 scripts/validate_mycobot280_training_dataset.py \
  --config configs/mycobot280/training_datasets/ground_pickup_tiny_smoke.json

PYTHONPATH=src:. python3 scripts/plan_mycobot280_smolvla_training.py \
  --config configs/mycobot280/training_datasets/ground_pickup_tiny_smoke.json
```

Pose-diverse controlled baseline:

```bash
PYTHONPATH=src:. python3 scripts/validate_mycobot280_training_dataset.py \
  --config configs/mycobot280/training_datasets/ground_pickup_pose_diverse_v1.json

PYTHONPATH=src:. python3 scripts/plan_mycobot280_smolvla_training.py \
  --config configs/mycobot280/training_datasets/ground_pickup_pose_diverse_v1.json \
  --output _workspace/mycobot280_training/ground_pickup_pose_diverse_v1/dry_run.json
```

`ground_pickup_pose_diverse_v1.json` freezes a 50-train/10-validation
deterministic pose-diverse baseline. The planner emits separate train and
validation conversion commands. The validator is expected to report `blocked`
until the source dataset exists, then requires all 60 episodes, 31,800 rendered
frames, non-overlapping split membership, and the configured contact/lift/hold/
penetration gates.

Matched-camera deterministic control:

```bash
PYTHONPATH=src:. python3 scripts/validate_mycobot280_training_dataset.py \
  --config configs/mycobot280/training_datasets/ground_pickup_pose_diverse_closecam_v1.json \
  --dataset-root _workspace/mycobot_teacher_datasets/mycobot_280_ground_pickup_pose_diverse_closecam_v1_50train_10val_png_20260731 \
  --require-present

PYTHONPATH=src:. python3 scripts/audit_mycobot_280_camera_contract.py \
  --dataset-root _workspace/mycobot_teacher_datasets/mycobot_280_ground_pickup_pose_diverse_closecam_v1_50train_10val_png_20260731 \
  --output _workspace/mycobot280_runs/pose_diverse_closecam_v1_20260731/camera_contract_audit.json \
  --require-pass

PYTHONPATH=src:. python3 scripts/summarize_mycobot280_camera_ablation.py \
  --eval-root _workspace/mycobot280_eval/camera_ablation_20260731 \
  --output-json _workspace/mycobot280_runs/pose_diverse_closecam_v1_20260731/camera_ablation_summary.json \
  --output-figure docs/research/2026_07_31/mycobot280_camera_ablation.png \
  --wide-frame _workspace/mycobot_teacher_datasets/mycobot_280_ground_pickup_pose_diverse_v1_50train_10val_20260730/splits/train/frames/episode_0000/frame_0000.bmp \
  --close-frame _workspace/mycobot_teacher_datasets/mycobot_280_ground_pickup_pose_diverse_closecam_v1_50train_10val_png_20260731/splits/train/frames/episode_0000/frame_0000.png
```

`ground_pickup_pose_diverse_closecam_v1.json` changes only the observation
camera and lossless image encoding relative to the deterministic wide-camera
control. It freezes the close-camera geometry, PNG provenance, exact 7D
state/action contract, 50/10 split, 100-step recipe, and 11-seed closed-loop
schedule. Do not compare it with a randomized policy unless both evaluations
use this same close-camera contract.
