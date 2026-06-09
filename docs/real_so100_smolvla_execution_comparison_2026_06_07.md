# Real SO-100 SmolVLA Execution Comparison

Date: 2026-06-07

## Rebase Status

- Branch: `codex/real-so100-green-doll-dryrun`
- Command: `git rebase main`
- Result: current branch was already up to date with local `main`.
- Temporary tracked stash was restored after the rebase check.

## Baseline Source

Reference:

`docs/research/smolvla_baseline_handoff_2026_06_07.md`

The validated baseline uses:

- Model: `lerobot/smolvla_libero`
- Runner: LeRobot `lerobot-eval`
- Wrapper: `scripts/run_libero_in_episode_smolvla_instrumented.py`
- Policy call sequence:
  1. `observation = env_preprocessor(observation)`
  2. `observation = preprocessor(observation)`
  3. `action = policy.select_action(observation)`
  4. `action = postprocessor(action)`
  5. `action = env_postprocessor({ACTION: action})[ACTION]`
  6. `env.step(action)`
- Required camera mapping:
  `--env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'`
- Required empty camera setting:
  `--policy.empty_cameras=0`

## Current Real SO-100 Path

Current real observation dry-run uses:

- Script: `scripts/real_so100_smolvla_dry.py`
- Default model: `lerobot/smolvla_base`
- Policy call sequence:
  1. Build a manual batch with SO-100 raw state and camera frames.
  2. Call `policy.predict_action_chunk(batch)` directly.
  3. Save `raw_action_chunk`.
  4. A separate executor attempts to convert chunk values into SO-100 motor targets.

## Match / Mismatch

### Matches

- Both paths load LeRobot `SmolVLAPolicy`.
- Both paths feed visual observations and language/task context into SmolVLA.
- The real path now correctly requests a multi-step action chunk instead of using only one isolated action.
- Camera role intent is aligned: two policy cameras plus non-policy debug/observer evidence.

### Mismatches

- The validated baseline uses `lerobot/smolvla_libero`; the real SO-100 path defaults to `lerobot/smolvla_base`.
- The baseline goes through LeRobot policy `preprocessor` and `postprocessor`; the real path bypasses those processors.
- The baseline executes postprocessed env actions; the real path stores direct `predict_action_chunk()` tensors.
- The baseline has explicit camera feature mapping from env images to policy camera names; the real path uses a local manual mapping that must be checked against the loaded policy config.
- The baseline action semantics are defined by the env/postprocessor contract; the real path still needs a robot-specific action contract for SO-100 follower joint order, gripper semantics, and calibration ranges.

## Important Conclusion

The current real SO-100 SmolVLA execution method is not yet equivalent to the validated baseline execution method.

The main issue is not action magnitude. The main issue is that the correct baseline path relies on LeRobot preprocessing and postprocessing around the policy. Calling `predict_action_chunk()` directly gives a policy chunk before the same execution contract is established for the real SO-100 follower.

## Required Fix Direction

The real SO-100 path should be changed to mirror the validated baseline as closely as possible:

1. Load the policy config and processor pipeline using LeRobot factory utilities where available.
2. Build observations with policy feature names that match the checkpoint config.
3. Run the policy through the same preprocessor/postprocessor contract used by `lerobot-eval`.
4. For chunk execution, use a chunk-capable variant of the postprocessed action path, not arbitrary raw tick scaling.
5. Only after postprocessed action units are known, map them to SO-100 follower joints and gripper semantics.
6. Clip final motor targets to the SO-100 calibration range.
7. Execute 10-step chunks only after the metadata and processor contract is proven.

## Current Gate

Physical execution should remain blocked until the real path proves:

- model id is the intended robot/checkpoint model, not accidentally `smolvla_base`
- policy camera feature names match the checkpoint
- action output has passed the correct LeRobot postprocessor or an equivalent verified unnormalization path
- SO-100 follower joint order is confirmed
- gripper open/close semantics are confirmed
- calibration clipping is applied after unit conversion

