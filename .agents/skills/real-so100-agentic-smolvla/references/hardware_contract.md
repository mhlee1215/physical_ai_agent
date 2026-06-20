# Real SO-100 Hardware Contract

This is the durable contract for the current real SO-100 follower arm
agentic SmolVLA development environment.

## Physical Setup

- Robot: SO-100 follower arm.
- Motor topology: chained serial servo bus; commands affect follower joints in
  the calibrated joint order.
- Current serial port:
  `/dev/cu.usbmodem5AE60824791`
- Calibration manifest:
  `_workspace/real_so100/calibration_manifest.json`
- Calibration file:
  `_workspace/real_so100/calibration/so100_local.json`
- Current lab power note: the user set the supply to `5V` during this session.
  Do not request or assume voltage changes during a run. If voltage matters,
  stop and verify the live setup with the user before moving hardware.
- Calibration operator rule: the user performs physical calibration. Codex may
  launch the calibration script, observe prompts, verify the resulting files,
  and record `operator`, but should not describe itself as having physically
  calibrated the arm.
- Terminal policy: do not open a new Terminal window for routine real-SO-100
  work. Use the current Codex command path for non-interactive camera checks,
  analysis, report generation, tests, and helper scripts whenever permissions
  allow. The calibration script is the canonical exception because the user
  performs an interactive physical procedure. External Terminal use is also
  allowed only for documented macOS camera/TCC, GUI, or runtime-permission
  boundaries, and must write deterministic logs/artifacts under
  `_workspace/real_so100/`.
- Canonical home pose was defined by the user from the live physical robot
  state on 2026-06-07. Do not infer home pose from camera appearance or
  calibration midpoints. Use
  `_workspace/real_so100/home_pose/canonical_home_pose_2026_06_07.json`.
  Current raw home targets are:
  `shoulder_pan=2042`, `shoulder_lift=2049`, `elbow_flex=2046`,
  `wrist_flex=2028`, `wrist_roll=2052`, `gripper=1789`.
  The helper command is `scripts/real_so100_move_to_home_pose.py`.

## Camera Contract

Current camera routing is fixed unless the user explicitly changes it:

| Index | Device / role | Policy use |
| ---: | --- | --- |
| `0` | Innomaker-U20CAM-1080p, wrist/end-effector style view | SmolVLA policy input |
| `1` | Innomaker-U20CAM-1080p, egocentric/object context view | SmolVLA policy input |
| `3` | iPhone webcam observer view | Codex observer/debug only |

Rules:

- Use only camera indexes `0` and `1` as SmolVLA/VLA inputs.
- Use camera index `3` to verify whether the experiment actually worked.
- Camera `3` evidence is for Codex observer feedback, qualitative debugging,
  and task-level verifier inputs. It must not be fed to SmolVLA.
- Temporary observer-off mode: when the user says camera `3` is off, continue
  only with camera `0` and camera `1` for no-actuation agentic-layer
  development. Camera `1` is the wide context view and may be used as a
  Pseudo-LLM feedback source, but not as a replacement for observer-backed
  physical success evidence.
- Legacy camera index `2` artifacts exist, but index `2` is not part of the
  current experiment loop unless the user explicitly re-enables it.
- If camera `0` clips the target, the agentic layer should create a robot
  subgoal/prompt to improve visibility using camera `1` context when helpful.
  It should not produce an in-loop instruction to the human operator.
- If no policy camera sees the object, the agent should search with robot
  behavior through the policy/agent layer, not ask the human to reposition
  things as an autonomous-loop action.

## Agentic-Layer Semantics

The work is to develop an agentic layer around a lightweight VLA, not to
manually teleoperate the arm.

- Initial human task examples:
  - `Pick up the green Android figure and move it to the right.`
  - `Pick up the red box and move it left.`
- The human prompt becomes an input to the in-loop agent/SmolVLA stack.
- Codex may act as a development-time Pseudo-LLM for diagnosis, feedback, and
  next-version generation.
- The final runtime decision-maker should be an on-device lightweight LLM/VLM
  or an equivalent replaceable local model.
- Do not hard-code semantic decisions with rules when they should belong to
  the LLM/VLM layer.
- Do not translate object motion goals into fixed robot-arm directions. Use
  object-frame verifier language, e.g. `object center moved right in observer
  image frame by at least threshold`.

## SmolVLA Execution Contract

Validated LIBERO baseline runs use LeRobot's full execution contract:

1. `env_preprocessor(observation)`
2. `preprocessor(observation)`
3. `policy.select_action(observation)`
4. `postprocessor(action)`
5. `env_postprocessor({ACTION: action})`
6. environment step

For real SO-100, do not treat direct `policy.predict_action_chunk(batch)`
values as motor commands until an equivalent processor/postprocessor or
verified unnormalization contract exists.

Current known mismatch to fix:

- baseline model path is `lerobot/smolvla_libero`;
- current real path defaults to `lerobot/smolvla_base`;
- current real path has manually built policy batches;
- direct chunk tensors bypass the LeRobot postprocessor;
- action normalization / feature metadata must be checked before conversion.

Required before any real SmolVLA chunk execution:

1. Inspect checkpoint `output_features`, `normalization_mapping`,
   `chunk_size`, and `n_action_steps`.
2. Confirm whether the postprocessed output is absolute joint position, joint
   delta, or another robot action representation.
3. Confirm SO-100 follower joint order:
   `["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]`
4. Confirm gripper semantics:
   `higher_raw_opens` or `higher_raw_closes`.
5. Use authoritative action stats or saved LeRobot processor state when
   normalization is `MEAN_STD`.
6. Clip final SO-100 raw targets to calibrated ranges.
7. Execute only bounded chunks, normally 10 steps.
8. After any robot-arm movement task, disable torque on all SO-100 motors
   before returning control to the user. If a full-chain torque-off fails,
   directly address any stiff/responding motor IDs and save a recovery report.

Blocked is a valid and preferred outcome when this contract is incomplete.

## Physical Execution Protocol

Every physical movement attempt must follow this evidence order:

1. Capture before observer frame from camera `3`.
2. Start camera `3` motion recording.
3. Execute the approved limited command chunk.
4. Stop camera `3` recording.
5. Capture after observer frame from camera `3`.
6. Save raw joint readbacks before and after.
7. Save per-step target commands.
8. At task end, return to canonical home pose with
   `scripts/real_so100_move_to_home_pose.py`.
9. Disable torque on all motors and verify/report the result.
10. Run task-level verifier if the task includes object relocation or grasp.
11. Write feedback and next-version plan.

Required report fields:

- `send_action_called`
- `policy_actions_executed`
- `action_chunk_steps`
- `executed_action_steps`
- `readback_before_raw`
- `readback_after_raw`
- `observed_delta_raw`
- `motion_video`
- `visual_check.before`
- `visual_check.after`
- `task_home_return`
- `post_task_torque_disabled`
- `blockers`
- `status`

Never claim physical movement from joint readback alone. Camera `3` before/after
or video evidence must be inspected.

When camera `3` is temporarily unavailable:

- keep `observer_camera_indexes=[]`;
- set `observer_camera_status=temporarily_unavailable` in manifests;
- capture only cameras `0` and `1`;
- keep `send_action_called=false`;
- keep `physical_robot_motion=false`;
- use camera `1` as policy-context feedback for the agentic layer;
- do not run contact probes;
- do not claim grasp, relocation, or final task success.

## Current Historical Evidence Boundaries

- Earlier raw-tick scaling executions were useful for proving serial/motor
  communication, but they are invalid as final SmolVLA execution evidence.
- Earlier camera index `2` videos are legacy debugging artifacts and cannot
  stand in for the current iPhone camera `3` observer contract.
- Earlier gripper close probes did not achieve a green-object grasp; do not
  claim grasp success from those runs.
- Dashboard/video artifacts are debugging aids. They are secondary to the
  agentic-layer loop and must not become the main work.

## Canonical Files to Check First

- `docs/research/smolvla_baseline_handoff_2026_06_07.md`
- `docs/real_so100_smolvla_execution_comparison_2026_06_07.md`
- `docs/real_so100_agentic_smolvla_handoff_2026_06_07.md`
- `docs/harness/physical-ai/team-spec.md`
- `_workspace/real_so100/calibration/so100_local.json`
- `_workspace/real_so100/home_pose/canonical_home_pose_2026_06_07.json`

## Preferred Artifact Names

Use deterministic paths under `_workspace/real_so100/`:

- `agentic_smolvla_iter_<NNN>/pre_observe/episode.jsonl`
- `agentic_smolvla_iter_<NNN>/smolvla_chunk/smolvla_action_chunk.json`
- `agentic_smolvla_iter_<NNN>/chunk_execute_report.json`
- `agentic_smolvla_iter_<NNN>/observer_camera_3/motion.mp4`
- `agentic_smolvla_iter_<NNN>/post_observe/episode.jsonl`
- `reports/real_so100_prompt_iteration_<name>.json`
- `reports/real_so100_relocation_verifier_packet_<name>.json`
- `reports/agentic_smolvla_iterations_001_010_report.md`
