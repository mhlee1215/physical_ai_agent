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
