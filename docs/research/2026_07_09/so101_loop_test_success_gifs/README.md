# SO101 Closed-Loop Success GIFs

This folder preserves successful closed-loop rollout GIFs from the stopped local
SO101 training run:

- Run: `_workspace/so101_training/runs/grip_from_above_edge_cube_no_aug_rmse_v1`
- Test id: `grip_from_above_edge_cube`
- Best observed checkpoint in this run: `001080`
- Success rate at that checkpoint: `2/6`
- Prompt: `From just above one visible green cube edge, move down, close the gripper, and lift.`

Included evidence:

- `step_001080_episode_002_tensorboard_style_success.gif`
- `step_001080_episode_003_tensorboard_style_success.gif`
- `manifest.json`

These GIFs are generated through the same side-by-side camera1/camera2 renderer
used for TensorBoard tag
`closed_loop/grip_from_above_edge_cube/rollout_camera1_camera2_episode_*`.

The large run directory remains outside the PR. Only the two successful
TensorBoard-style GIFs and their manifest are checked in as review evidence.
