# myCobot 280 randomized ground-pickup source audit

Date: 2026-07-31

## Scope

This audit validates the already-generated randomized teacher dataset at:

`_workspace/mycobot_teacher_datasets/mycobot_280_ground_pickup_randomized_v1_50train_10val_allframes_verified_20260731`

It does not regenerate simulation episodes. The checker reads every JSONL source row and samples the first and last rendered frame in every required rollout phase.

## Result

- Status: passed.
- Accepted episodes: 60 of 73 attempts (50 train, 10 validation; 82.19% acceptance).
- Source rows checked: 31,800.
- Visibility samples checked: 600 (120 per required phase).
- Image resolution: 256 x 256 BMP.
- Minimum sampled red-cube area: 2.571% of the frame; required minimum: 2.0%.
- Sampled cube border touches: 0.
- Unique attempt indices and seeds: 73 of 73.
- Train/validation seed, pose, factor, and trajectory-hash overlaps: 0.
- Minimum final lift: 57.13 mm.
- Minimum post-lift hold height: 47.67 mm.
- Minimum post-lift two-pad hold: 300 steps.
- Maximum pad/cube penetration: 2.841 mm; allowed maximum: 3.0 mm.

The audit also independently checks row-level teacher-attachment state, cube/mat, pad/mat and visual gripper/mat guards, terminal lift/contact, post-lift retention, candidate provenance, rejected-attempt provenance, camera contract, and summary-to-source consistency.

## Rejection-conditioned factor coverage

The checker also bins every accepted and rejected attempt into four equal-width yaw, cube-mass, and cube-friction ranges. These are advisory warnings: they do not invalidate otherwise correct source episodes, but they prevent the accepted corpus from being mistaken for uniform coverage of the declared sampling ranges.

| Factor | Quartile accepted / attempted | Validation examples per quartile |
| --- | --- | --- |
| Yaw delta | 13/17, 12/16, 17/20, 18/20 | 3, 2, 1, 4 |
| Cube mass | 13/13, 27/27, 18/18, 2/15 | 4, 5, 1, 0 |
| Cube friction | 12/14, 11/15, 25/27, 12/17 | 0, 0, 5, 5 |

The material caveat is cube mass. All 13 rejected attempts fall in the highest mass quartile, 0.034-0.036 kg, where only 2 of 15 attempts passed (13.3%) and no accepted example entered validation. The dataset is therefore a teacher-success-conditioned demonstration distribution, not uniform evidence over the declared physics range.

For a later robustness dataset, oversample or recalibrate the high-mass range and require stratified validation coverage. The current corpus should remain unchanged as the provenance-preserving source used by the completed pilot.

## Verification

```bash
PYTHONPATH=src:. MUJOCO_GL=egl \
/home/villy/code/physical_ai_agent/_workspace/worktrees/mycobot-280pi-adaptive/_workspace/venvs/mycobot-mujoco-py312/bin/python \
scripts/check_mycobot_280_ground_pickup_randomized_dataset.py \
  --existing-dataset-root /home/villy/code/physical_ai_agent/_workspace/mycobot_teacher_datasets/mycobot_280_ground_pickup_randomized_v1_50train_10val_allframes_verified_20260731
```

Focused unit gate:

```bash
PYTHONPATH=src:. MUJOCO_GL=egl \
/home/villy/code/physical_ai_agent/_workspace/worktrees/mycobot-280pi-adaptive/_workspace/venvs/mycobot-mujoco-py312/bin/python \
  -m unittest -v tests.test_mycobot_280_ground_pickup_randomized_dataset
```

The checker exits nonzero for camera-contract drift, tiny or missing red-cube visibility, image-border contact, source-row guard regression, teacher attachment, split leakage, malformed provenance, or summary/source inconsistency. Factor-coverage skew is emitted as an advisory warning.
