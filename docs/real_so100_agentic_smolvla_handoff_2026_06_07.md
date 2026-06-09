# Real SO-100 Agentic SmolVLA Handoff

Date: 2026-06-07

## Goal

Develop a real SO-100 agentic layer around SmolVLA for the task:

`Pick up the green Android figure and move it to the right.`

The target experiment loop is:

1. Capture policy observations from cameras `0` and `1`.
2. Run SmolVLA with the in-loop prompt and produce an action chunk.
3. Execute a limited real robot movement from that chunk.
4. Record camera `3` before/during/after execution for Codex observer feedback only.
5. Use camera `3` evidence plus policy-camera intermediate data to write feedback.
6. Produce the next agentic-layer version.
7. Repeat through iteration 10.
8. Generate a result report with success/failure, videos, readbacks, and lessons.

## Non-Negotiable Design Rules

- Cameras `0` and `1` are the VLA / agentic-layer policy inputs.
- Camera `3` is Codex observer/debug feedback only. It must not be treated as SmolVLA policy input.
- In-loop prompts are for the in-loop agent or SmolVLA, not for the human operator and not for Codex.
- Do not make semantic decisions with hard-coded rules.
- During development, Codex may act as a Pseudo-LLM for feedback and version updates.
- Runtime design must keep the semantic decision-maker replaceable by an on-device lightweight LLM/VLM.
- If camera `0` clips the target, the agent should produce a robot subgoal/prompt to improve visibility, using camera `1` context if helpful.
- If neither policy camera sees the target, the agent should search with the robot. It should not ask the human to move the camera as an in-loop action.
- SmolVLA output must be consumed as an action chunk. Do not use only one `select_action()` output for real iteration.
- Do not interpret SmolVLA output with arbitrary raw tick scaling. Before any real execution, inspect checkpoint action normalization metadata, confirm action semantics, map SO-100 follower joint order and gripper semantics, unnormalize with authoritative stats, then execute a 10-step chunk inside calibration bounds.

## Hardware Setup Assumptions

- Robot: SO-100 follower arm.
- Serial port from calibration manifest:
  `/dev/cu.usbmodem5AE60824791`
- Calibration manifest:
  `_workspace/real_so100/calibration_manifest.json`
- Calibration file:
  `_workspace/real_so100/calibration/so100_local.json`
- Policy cameras:
  - `0`: wrist / policy camera
  - `1`: egocentric / policy camera
- Observer camera:
  - `3`: iPhone / Codex observer camera

## Current Important Code Paths

### SmolVLA Chunk Proposal

File:

`scripts/real_so100_smolvla_dry.py`

Current contract:

- Uses `policy.predict_action_chunk(batch)`.
- Default `--action-steps 10`.
- Writes `raw_action_chunk` with 10 actions.
- Preserves `raw_action` only as first-step legacy compatibility.
- Writes:
  - `raw_action_chunk_steps`
  - `predicted_chunk_size`
  - `planned_action_steps`
  - `executed_action_steps`
  - `action_chunk_semantics`
- Never sends robot actions.

Latest successful chunk artifact:

`_workspace/real_so100/smolvla_proposal_move_right_u20cam_007_chunk10_allow_download/smolvla_action_chunk.json`

Observed values:

- `predicted_chunk_size`: 50
- `raw_action_chunk_steps`: 10
- `send_action_called`: false

### Chunk-Aware Action Loader

File:

`src/physical_ai_agent/safety/so100_action_gate.py`

Function:

`load_action_chunk_payload(path, action_steps=10)`

Behavior:

- Prefer `raw_action_chunk`.
- Slice to requested `action_steps`.
- Fall back to legacy `raw_action` as a one-step chunk only when no chunk exists.

### Deprecated Chunk-Aware Command Adapter

File:

`src/physical_ai_agent/safety/so100_command_adapter.py`

Function:

`build_so100_command_chunk_plan(...)`

Behavior:

- Converts each SmolVLA chunk step into a conservative sequential raw-tick target plan.
- Applies per-step raw delta limit.
- Applies calibration range clipping when calibration is supplied.
- Deprecated for SmolVLA hardware execution because it treats normalized SmolVLA outputs as raw tick deltas.
- Use this only as historical evidence that motor communication worked, not as the correct SmolVLA adapter.

Latest no-actuation command chunk plan:

`_workspace/real_so100/smolvla_proposal_move_right_u20cam_007_chunk10_allow_download/so100_command_chunk_plan.json`

Observed values:

- `action_chunk_steps`: 10
- `step_plans`: 10
- `ready_for_execution`: false unless experimental adapter confirmation and human confirmation are supplied.

### Correct Metadata-Based SmolVLA Adapter

Files:

- `scripts/inspect_smolvla_action_metadata.py`
- `src/physical_ai_agent/safety/so100_smolvla_metadata_adapter.py`

Required sequence before physical execution:

1. Inspect SmolVLA checkpoint config.
2. Confirm output feature dimension and normalization mode.
3. Provide authoritative action stats if normalization is `MEAN_STD`.
4. Confirm action semantics as `absolute_joint_position` or `joint_delta`.
5. Confirm SO-100 follower joint order:
   `["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]`
6. Confirm gripper open/close semantics for the follower.
7. Unnormalize action chunk values.
8. Clip targets inside calibration range.
9. Execute 10 chunk steps with camera `3` evidence.

Known checkpoint metadata for local `lerobot/smolvla_base` snapshot:

- `output_features.action.shape`: `[6]`
- `normalization_mapping.ACTION`: `MEAN_STD`
- `chunk_size`: `50`
- `n_action_steps`: `50`
- `use_delta_joint_actions_aloha`: `false`
- Local snapshot currently contains `config.json` and `model.safetensors`, but no local action stats file.

Current consequence:

- Physical execution is blocked until matching action mean/std stats, action semantics, SO-100 joint order, and gripper semantics are supplied.
- Iterations 1-3 used the deprecated raw tick scaling bridge and should be treated as invalid for the final SmolVLA adapter claim.
- Iteration 4 chunk was generated but must not be executed until the metadata adapter is unblocked.

### Real Chunk Executor

File:

`scripts/real_so100_execute_chunk.py`

Purpose:

Execute a limited 10-step SmolVLA action chunk on the real SO-100 with camera `3` evidence.

Default execution bridge:

- Requires metadata-based unnormalization.
- Blocks deprecated raw tick scaling unless `--allow-deprecated-raw-tick-scaling` is explicitly supplied.
- Requires `--metadata-config`, `--action-stats`, `--action-semantics`, `--gripper-semantics`, and `--confirm-so100-joint-order` for the correct path.

Safety requirements before serial write:

- `--execute`
- `--human-confirmed`
- metadata adapter `ready_for_execution=true`
- `--record-video`
- `--camera-index 3`
- `--visual-output-dir <dir>`

The executor records:

- before observer frame
- motion video
- after observer frame
- before/after raw joint readbacks
- observed raw deltas
- per-step target writes

## Tests Added / Updated

Run these before continuing:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_smolvla_dry.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_so100_action_gate.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_so100_command_adapter.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_so100_smolvla_metadata_adapter.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_execute_chunk.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_build_real_so100_agentic_layer_contract.py'
```

Already passed in this session before the handoff:

- `test_real_so100_smolvla_dry.py`
- `test_so100_action_gate.py`
- `test_so100_command_adapter.py`
- `test_build_real_so100_agentic_layer_contract.py`

`test_so100_smolvla_metadata_adapter.py` was added after correcting the execution direction and should be included before the next hardware run.

## Immediate Next Commands

### 1. Inspect SmolVLA Action Metadata

```bash
PYTHONPATH=src:. .venv/bin/python -B scripts/inspect_smolvla_action_metadata.py \
  --output-dir _workspace/real_so100/smolvla_action_metadata
```

Expected current result:

- `status=blocked`
- blocker says `ACTION=MEAN_STD` but action mean/std stats are unavailable
- blocker says action semantics are not confirmed
- blocker says SO-100 follower joint order is not confirmed
- blocker says SO-100 gripper semantics are not confirmed

### 2. Run Adapter / Executor Tests

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_so100_smolvla_metadata_adapter.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_execute_chunk.py'
```

### 3. Resolve Metadata Blocker

Do not execute iteration 4 or any future chunk until all are known:

- action mean/std stats for the exact SmolVLA checkpoint
- whether unnormalized output is follower joint position or follower joint delta
- SO-100 follower joint order and gripper open/close semantics

### 4. Execute Physical Iteration Only After Metadata Gate Passes

Use this shape only after supplying real stats:

```bash
PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_execute_chunk.py \
  --port /dev/cu.usbmodem5AE60824791 \
  --action _workspace/real_so100/agentic_smolvla_iter_004/smolvla_chunk/smolvla_action_chunk.json \
  --output _workspace/real_so100/agentic_smolvla_iter_004/chunk_execute_report.json \
  --calibration _workspace/real_so100/calibration/so100_local.json \
  --metadata-config /Users/minhaeng/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/config.json \
  --action-stats <authoritative-action-stats.json> \
  --action-semantics <absolute_joint_position-or-joint_delta> \
  --gripper-semantics <higher_raw_opens-or-higher_raw_closes> \
  --confirm-so100-joint-order \
  --execute \
  --human-confirmed \
  --action-steps 10 \
  --step-settle-seconds 0.15 \
  --camera-index 3 \
  --visual-output-dir _workspace/real_so100/agentic_smolvla_iter_004/observer_camera_3 \
  --record-video \
  --video-fps 12
```

### 5. Observe After Each Valid Iteration

Use the latest post-observe episode and frame index, usually the last frame.

```bash
PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_smolvla_dry.py \
  --episode _workspace/real_so100/agentic_smolvla_iter_001/post_observe/episode.jsonl \
  --frame-index 1 \
  --output-dir _workspace/real_so100/agentic_smolvla_iter_005/smolvla_chunk \
  --instruction 'Pick up the green Android figure and move it to the right.' \
  --allow-download \
  --wrist-camera-index 0 \
  --egocentric-camera-index 1 \
  --observer-camera-index 3 \
  --action-steps 10
```

Then repeat execute / observe / feedback through iteration 10.

## Iteration Report Target

After iteration 10, generate a report under:

`_workspace/real_so100/reports/agentic_smolvla_iterations_001_010_report.md`

The report should include:

- iteration number
- SmolVLA chunk path
- execution report path
- camera 3 video path
- before/after camera 3 frames
- raw joint deltas
- whether visual motion was detected
- whether the green object moved right
- whether grasp happened
- whether the final task succeeded
- Pseudo-LLM feedback for the next version
- clear separation between:
  - policy/action execution success
  - verifier/internal progress
  - final task success

## Current State At Handoff

- The main bug found by the user was real: old SmolVLA real dry artifacts used a single 6D `raw_action`, not a 10-step chunk.
- The dry/proposal path has been changed to 10-step chunk output.
- A successful 10-step SmolVLA proposal exists.
- A no-actuation 10-step command chunk plan exists.
- A real 10-step chunk executor exists, but the correct metadata-based execution gate is currently blocked by missing action stats / semantics.
- Iterations 1-3 physically moved the arm via deprecated raw scaling and should be excluded from the correct-adapter success claim.
- Iteration 4 chunk exists but has not been executed through the correct adapter.
- Valid metadata-based iterations 1-10 have not yet been run.
- Final task success has not been achieved or claimed.

## Do Not Forget

The purpose is not to produce dashboards. The purpose is to develop the agentic layer around SmolVLA.

The artifact loop should serve the robot iteration:

`SmolVLA chunk -> limited execution -> observer evidence -> LLM feedback -> next prompt/version`

Keep the work focused there.

## 2026-06-07 Observer-Off Continuation

The iPhone observer camera index `3` is temporarily off. Continue only
no-actuation agentic-layer development with camera indexes `0` and `1`.
Camera `0` and `1` remain the SmolVLA policy inputs; camera `1` is the wide
context view for Pseudo-LLM feedback. Do not claim physical motion, grasp,
relocation, or task success while observer camera `3` is unavailable.

Latest observer-off artifacts:

- SmolVLA 10-step proposal:
  `_workspace/real_so100/smolvla_proposal_move_right_u20cam_observer_off_001/`
- Agentic layer contract:
  `_workspace/real_so100/reports/real_so100_agentic_layer_contract_move_right_u20cam_observer_off_002.json`
- Contract markdown:
  `_workspace/real_so100/reports/real_so100_agentic_layer_contract_move_right_u20cam_observer_off_002.md`
- Agentic iteration manifest:
  `_workspace/real_so100/reports/real_so100_agentic_iteration_move_right_u20cam_observer_off_002.json`
- Prompt iteration:
  `_workspace/real_so100/reports/real_so100_prompt_iteration_move_right_u20cam_observer_off_002.json`
- VLA prompt packet:
  `_workspace/real_so100/reports/real_so100_vla_prompt_packet_move_right_u20cam_observer_off_002.json`

Current state:

- `send_action_called=false`
- `physical_robot_motion=false`
- `observer_camera_status=temporarily_unavailable`
- `policy_camera_indexes=["0","1"]`
- `raw_action_chunk_steps=10`
- `next_stage=smolvla_proposal_only`
- `task_success_claim_allowed=false`

Current execution blockers:

- Camera `3` observer evidence is unavailable.
- Camera `0` jaw/object framing is still blocked because the green object is
  clipped in the wrist/end-effector view.
- SmolVLA action normalization is `MEAN_STD`. The authoritative
  `so100.buffer` action stats have now been extracted from the LeRobot
  `policy_postprocessor`, but the postprocessed action units still must be
  confirmed before they can be written as Feetech raw `Goal_Position` ticks.
- The direct real path still must be made equivalent to the validated
  LeRobot preprocessor/postprocessor execution contract before any chunk is
  converted into SO-100 motor targets.

## 2026-06-07 Postprocessor Stats Update

The missing action-stats blocker has been narrowed. The Hub repo for
`lerobot/smolvla_base` includes:

- `policy_postprocessor.json`
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors`

Extracted stats artifact:

`_workspace/real_so100/smolvla_proposal_move_right_u20cam_observer_off_001/action_metadata/policy_postprocessor_action_stats_so100_buffer.json`

Available stats keys:

- `so100-blue.buffer`
- `so100-red.buffer`
- `so100.buffer`

Current selected key:

- `so100.buffer`

Updated metadata report:

`_workspace/real_so100/smolvla_proposal_move_right_u20cam_observer_off_001/action_metadata_with_stats/smolvla_action_metadata_report.json`

Updated agentic artifacts:

- Contract:
  `_workspace/real_so100/reports/real_so100_agentic_layer_contract_move_right_u20cam_observer_off_003.json`
- Iteration manifest:
  `_workspace/real_so100/reports/real_so100_agentic_iteration_move_right_u20cam_observer_off_003.json`
- Prompt packet:
  `_workspace/real_so100/reports/real_so100_vla_prompt_packet_move_right_u20cam_observer_off_003.json`
- Execute gate dry run:
  `_workspace/real_so100/reports/real_so100_execute_chunk_dry_gate_move_right_u20cam_observer_off_003.json`

Important: the extracted stats values look like LeRobot SO-100 joint-space
actions, not Feetech raw ticks. Do not set `--command-units feetech_raw_ticks`
or execute until the conversion from postprocessed LeRobot action units to
SO-100 follower motor commands is verified against the calibrated follower.

## 2026-06-07 LeRobot SO-100 Unit Update

The command-unit blocker has been narrowed again. LeRobot's `SO100Follower`
uses:

- body joints: `MotorNormMode.DEGREES`
- gripper: `MotorNormMode.RANGE_0_100`
- `send_action()` / `sync_write("Goal_Position", goal_pos)` with
  `normalize=True`

The real adapter now supports `--command-units lerobot_so100_position` and
keeps a raw-tick estimate for safety evidence. This is still no-actuation while
camera `3` is unavailable.

Updated unit metadata:

`_workspace/real_so100/smolvla_proposal_move_right_u20cam_observer_off_001/action_metadata_lerobot_units/smolvla_action_metadata_report.json`

Current metadata status:

- `status=passed`
- `action_stats_available=true`
- `selected_action_stats_key=so100.buffer`
- `command_units=lerobot_so100_position`

Current execute dry gate:

`_workspace/real_so100/reports/real_so100_execute_chunk_dry_gate_move_right_u20cam_observer_off_004.json`

Current v004 artifacts:

- Contract:
  `_workspace/real_so100/reports/real_so100_agentic_layer_contract_move_right_u20cam_observer_off_004.json`
- Iteration manifest:
  `_workspace/real_so100/reports/real_so100_agentic_iteration_move_right_u20cam_observer_off_004.json`
- Prompt packet:
  `_workspace/real_so100/reports/real_so100_vla_prompt_packet_move_right_u20cam_observer_off_004.json`

Current authoritative execution blocker:

- The current 10-step SmolVLA proposal maps several joints outside the
  calibrated follower ranges when interpreted as LeRobot SO-100 positions.
  Step 0 already maps:
  - `shoulder_lift` to raw target about `4771`, outside `[2001, 3695]`
  - `elbow_flex` to raw target about `3327`, outside `[468, 2048]`
  - `wrist_flex` to raw target about `2489`, outside `[1315, 2047]`
  - `gripper` to raw target about `1606`, outside `[1658, 2747]`

Do not execute this chunk. The next agentic-layer improvement should use this
as policy feedback: the current prompt/frame/proposal drives the policy toward
an unreachable posture, so rerun proposal generation only after improving the
policy input context, model/checkpoint choice, or prompt strategy.

Validated after the observer-off update:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_build_real_so100_agentic_layer_contract.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_iteration.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_build_real_so100_vla_prompt_packet.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_smolvla_dry.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_so100_smolvla_metadata_adapter.py'
```

## 2026-06-07 v005 Agentic Proposal Sweep

Camera `3` is temporarily off, so the current loop is intentionally
no-actuation. Policy cameras remain Innomaker U20CAM indexes `0` and `1`; index
`1` is the wide-context camera. The agentic layer now has a proposal-sweep
stage that asks SmolVLA for multiple 10-step chunks, maps each through the
LeRobot-compatible SO-100 action stats and calibrated follower ranges, and
chooses the best candidate as feedback for the next in-loop prompt strategy.
This is prompt/adapter feedback for SmolVLA, not an instruction to the human
operator.

New code:

- `scripts/real_so100_agentic_proposal_sweep.py`
- `tests/test_real_so100_agentic_proposal_sweep.py`

SmolVLA local loading was also fixed for offline Mac execution:

- `src/physical_ai_agent/policies/smolvla_real.py`
  now resolves the cached `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
  snapshot for LeRobot's internal processor/model load and language tokenizer.

Latest sweep artifact:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_007/proposal_sweep_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_007/proposal_sweep_report.md`

Sweep result:

- All four SmolVLA dry runs passed and produced 10-step chunks.
- No candidate was ready for physical execution.
- `send_action_called=false`
- `policy_actions_executed=false`
- `physical_robot_motion=false`
- `task_success_claim_allowed=false`

Candidate ranking by calibrated-range gate:

| Rank | Candidate | Range violations | Total excess raw ticks | Max excess raw ticks |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 01 | 28 | 9385.6253 | 684.7609 |
| 2 | 02 | 30 | 9356.1539 | 630.3149 |
| 3 | 04 | 35 | 18727.8326 | 1073.9078 |
| 4 | 03 | 39 | 30104.7037 | 1654.4687 |

Selected feedback:

- Best prompt family: `Pick up the green Android figure and move it to the right.`
- Current dominant blockers:
  - `shoulder_lift` targets above calibrated max, up to about `4379.76` raw
    against max `3695`;
  - `elbow_flex` targets above calibrated max on multiple candidates;
  - some prompt variants increase wrist or elbow excess rather than improving
    executability.

Interpretation for the next version:

- The issue is no longer just missing metadata. SmolVLA is producing valid
  chunks, but the current prompt/frame/checkpoint combination pushes the SO-100
  follower toward unreachable upper-body postures under the current conservative
  calibration.
- The next agentic-layer version should generate candidate prompts or
  observation strategies that bias toward a lower, less extended pre-grasp
  posture before retrying the 10-step dry gate.
- Do not execute any candidate until a no-actuation sweep candidate is ready,
  camera `3` observer evidence is available again, and the normal confirmation
  gates pass.

Validated after v005:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_smolvla_dry.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_smolvla_real_batch.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_execute_chunk.py'
```

## 2026-06-07 v006 State-Unit Fix and Feedback Sweep

v005 and the first lower-pregrasp retry showed that prompt changes alone did
not make the SmolVLA chunks executable. The important bug found in this pass was
that real SO-100 observations were feeding raw Feetech ticks into the SmolVLA
state vector while the action path was interpreting outputs as LeRobot
SO-100 position units. The dry path now converts raw episode state to LeRobot
SO-100 position units with the same calibration used by the execution gate.

Code updates:

- `src/physical_ai_agent/safety/so100_smolvla_metadata_adapter.py`
  - added `raw_to_lerobot_so100_position(...)`
- `scripts/real_so100_smolvla_dry.py`
  - accepts `--calibration`
  - records `episode_state_units=raw_ticks`
  - records `policy_state_units=lerobot_so100_position`
- `scripts/real_so100_agentic_proposal_sweep.py`
  - passes calibration into dry inference
  - supports `--feedback-report` and `--prompt-profile lower_pregrasp`
  - ranks candidates by weighted range penalty:
    `total_excess + 3 * max_excess + 50 * violation_count`

Important invalidated intermediate:

- `_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_008/`
  was produced before the CLI feedback-profile bug was fixed and should not be
  treated as the authoritative lower-pregrasp sweep.
- `_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_009/`
  used the intended lower-pregrasp prompts but still used the old raw-state
  policy input path. It is useful as a before-state-unit-fix comparison only.

Current authoritative state-unit-fixed sweep:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_010/proposal_sweep_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_010/proposal_sweep_report.md`

v010 result:

- All five feedback-generated SmolVLA dry runs passed and produced 10-step
  chunks.
- No candidate was ready for physical execution.
- `send_action_called=false`
- `policy_actions_executed=false`
- `physical_robot_motion=false`
- `task_success_claim_allowed=false`
- Best candidate: `03`
- Best prompt:
  `Approach the green Android figure from the side at table height. Stop in a nearby pre-grasp pose before closing the gripper.`

Best v010 dry report:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_010/candidate_03/smolvla/smolvla_dry_report.json`

Key evidence from the best dry report:

- `instruction_tokenized=true`
- `language_token_count=29`
- `raw_action_chunk_steps=10`
- `predicted_chunk_size=50`
- `episode_state_units=raw_ticks`
- `policy_state_units=lerobot_so100_position`

Best v010 gate score:

- `range_violation_count=34`
- `total_range_excess_raw_ticks=12628.9636`
- `max_range_excess_raw_ticks=900.0272`
- `range_penalty_score=17029.0452`
- dominant joint excess:
  - `shoulder_lift=5056.7583`
  - `wrist_flex=4078.6799`
  - `elbow_flex=3460.1638`
  - `gripper=33.3616`

Comparison:

| Sweep | Meaning | Best candidate | Best penalty | Notes |
| --- | --- | ---: | ---: | --- |
| v005 / `007` | first weighted baseline, raw state path | 02 by weighted score | 12747.0986 | lower total/max than candidate 01, but still raw-state input |
| v009 | lower-pregrasp prompts, raw state path | 02 | 20098.0093 | prompt-only retry got worse |
| v010 | lower-pregrasp prompts + LeRobot state units | 03 | 17029.0452 | improved over v009, still not executable |

Interpretation for the next version:

- The state-unit fix is required and should remain part of all future real
  SmolVLA dry or execution paths.
- Prompt-only steering is not enough to pass the calibrated range gate.
- Next useful agentic-layer improvement is a projection/distortion analysis:
  clip or project postprocessed targets into calibrated SO-100 ranges in
  proposal-only mode, measure how much each 10-step chunk is distorted, and use
  low-distortion candidates as the next feedback signal. Do not physically
  execute clipped/projection candidates until camera `3` observer evidence is
  available again and the normal confirmation gates pass.

Validated after v006/v010:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_smolvla_dry.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_so100_smolvla_metadata_adapter.py'
```

## 2026-06-07 v011-v013 Projection-Aware Agentic Loop

Camera `3` remains temporarily off, so this phase stayed no-actuation and used
only policy camera indexes `0` and `1`. The goal was to turn calibrated range
failures into agentic feedback for SmolVLA prompt generation, without treating
clipped/projection commands as executable robot actions.

New code:

- `scripts/real_so100_projection_analysis.py`
  - reads a proposal sweep
  - clamps each postprocessed raw target to the calibrated follower range
  - converts the projected raw target back to LeRobot SO-100 position units
  - measures raw and command-unit distortion per joint and candidate
- `scripts/real_so100_agentic_proposal_sweep.py`
  - now accepts projection-analysis feedback
  - supports `--prompt-profile projection_aware`
  - generates projection-aware prompt candidates from dominant distortion joints
- `tests/test_real_so100_projection_analysis.py`
- `tests/test_real_so100_agentic_proposal_sweep.py`

v011 projection analysis of v010:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_011/projection_analysis_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_011/projection_analysis_report.md`

Best v011 projected candidate:

- Candidate: `03`
- Prompt:
  `Approach the green Android figure from the side at table height. Stop in a nearby pre-grasp pose before closing the gripper.`
- Shape-only projected ready: `true`
- Range violations clipped: `34`
- Total raw distortion: `12628.9636`
- Max raw distortion: `900.0272`
- Mean raw distortion: `210.4827`
- Projection penalty score: `17029.0452`

v012 projection-aware sweep:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_012/proposal_sweep_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_012/proposal_sweep_report.md`

Best v012 candidate:

- Candidate: `03`
- Prompt:
  `Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, extending the elbow, or flexing the wrist upward.`
- Ready for execution: `false`
- Range violations: `27`
- Total excess raw ticks: `10559.945`
- Max excess raw ticks: `607.8465`
- Range penalty score: `13733.4845`

v013 projection analysis of v012:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_013/projection_analysis_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_013/projection_analysis_report.md`

Best v013 projected candidate:

- Candidate: `03`
- Shape-only projected ready: `true`
- Range violations clipped: `27`
- Total raw distortion: `10559.945`
- Max raw distortion: `607.8465`
- Mean raw distortion: `175.9991`
- Projection penalty score: `13733.4845`
- Dominant remaining distortion:
  - `wrist_flex`: total raw distortion `4747.2159`, violations `10`
  - `shoulder_lift`: total raw distortion `3844.8755`, violations `10`
  - `elbow_flex`: total raw distortion `1938.5055`, violations `5`
  - `gripper`: total raw distortion `29.3481`, violations `2`

Comparison:

| Stage | Best candidate | Best penalty | Range violations | Max excess/distortion |
| --- | ---: | ---: | ---: | ---: |
| v010 lower-pregrasp + LeRobot state units | 03 | 17029.0452 | 34 | 900.0272 |
| v012 projection-aware prompt sweep | 03 | 13733.4845 | 27 | 607.8465 |

Interpretation:

- The projection-aware feedback loop improved the best no-actuation candidate
  by reducing the weighted range penalty from `17029.0452` to `13733.4845`.
- The improvement is real agentic-layer progress: v011 analyzed why SmolVLA
  was unreachable, v012 generated new prompt candidates from that feedback, and
  v013 measured the new residual distortion.
- The candidate is still not physically executable because calibrated range
  violations remain. Projection is analysis-only and must not be used as a
  motor command while camera `3` is off or while the normal execution gates are
  blocked.

Recommended next agentic-layer step:

- Generate another projection-aware prompt/profile that specifically reduces
  `wrist_flex` and `shoulder_lift` saturation while preserving the better
  elbow behavior of v012 candidate `03`.
- Alternatively add a model-side candidate type that asks SmolVLA for a
  shorter horizon or a hold/current-pose-biased pre-grasp proposal, then score
  it with the same 10-step calibrated range gate.

Validated after v012/v013:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_projection_analysis.py'
```

## 2026-06-07 v014-v016 Residual Retry and Candidate Memory

After v013, the dominant residual distortion was `wrist_flex`,
`shoulder_lift`, and then `elbow_flex`. The next agentic-layer change added a
`residual_distortion` prompt profile to test whether explicitly asking SmolVLA
to keep the wrist straight and shoulder from rising would improve the 10-step
calibrated range gate.

Code updates:

- `scripts/real_so100_agentic_proposal_sweep.py`
  - added `--prompt-profile residual_distortion`
  - added residual prompts for wrist/shoulder saturation
- `scripts/real_so100_agentic_candidate_memory.py`
  - compares multiple sweep/projection reports on the same penalty axis
  - records best-so-far, latest-best, regression, and next prompt-family action
- `tests/test_real_so100_agentic_candidate_memory.py`
- `tests/test_real_so100_agentic_proposal_sweep.py`

v014 residual sweep:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_014/proposal_sweep_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_014/proposal_sweep_report.md`

Best v014 candidate:

- Candidate: `02`
- Prompt:
  `Hold the current table-height approach and make only a tiny side alignment toward the green Android figure; keep the wrist straight, shoulder from rising, elbow midrange, and gripper open.`
- Ready for execution: `false`
- Range violations: `31`
- Total excess raw ticks: `11448.0766`
- Max excess raw ticks: `769.1897`
- Range penalty score: `15305.6457`

v015 projection analysis of v014:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_015/projection_analysis_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_015/projection_analysis_report.md`

v016 candidate memory:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_016/candidate_memory_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_016/candidate_memory_report.md`

Candidate-memory result:

- Best-so-far remains v012/v013 candidate `03`.
- Best-so-far prompt:
  `Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, extending the elbow, or flexing the wrist upward.`
- Best-so-far penalty: `13733.4845`
- Latest v014/v015 best penalty: `15305.6457`
- Regression delta: `1572.1612`
- Next step type: `reuse_best_historical_prompt_family`

Interpretation:

- The residual wrist/shoulder profile did not improve the best candidate. It
  remained better than v010, but regressed relative to v012/v013.
- This is still useful agentic-layer progress because the loop now has memory:
  it can reject a regressed prompt family instead of blindly continuing from the
  latest attempt.
- Continue future observer-off prompt generation from the v012/v013 prompt
  family, not from the v014 residual profile.
- Physical execution remains blocked: `send_action_called=false`,
  `physical_robot_motion=false`, `task_success_claim_allowed=false`, and camera
  `3` observer evidence is unavailable.

Validated after v014-v016:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_candidate_memory.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_projection_analysis.py'
```

## 2026-06-07 v017-v019 Memory-Refine Improvement

v016 candidate memory recommended returning to the best historical prompt
family instead of continuing the regressed residual profile. The proposal sweep
now supports a `memory_refine` prompt profile that reads
`next_agentic_layer_step.selected_prompt` from candidate memory and generates
small prompt-family variants around that best-so-far prompt.

Code updates:

- `scripts/real_so100_agentic_proposal_sweep.py`
  - added `--prompt-profile memory_refine`
  - added `memory_refine_prompts(...)`
  - preserves the selected best historical prompt as candidate `01`
- `tests/test_real_so100_agentic_proposal_sweep.py`
  - verifies memory-refine prompt generation and sweep profile plumbing

v017 memory-refine sweep:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_017/proposal_sweep_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_017/proposal_sweep_report.md`

Best v017 candidate:

- Candidate: `01`
- Prompt:
  `Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, extending the elbow, or flexing the wrist upward.`
- Ready for execution: `false`
- Range violations: `24`
- Total excess raw ticks: `4581.1389`
- Max excess raw ticks: `393.0481`
- Range penalty score: `6960.2832`

v018 projection analysis:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_018/projection_analysis_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_018/projection_analysis_report.md`

Best v018 projected candidate:

- Candidate: `01`
- Shape-only projected ready: `true`
- Range violations clipped: `24`
- Total raw distortion: `4581.1389`
- Max raw distortion: `393.0481`
- Mean raw distortion: `76.3523`
- Projection penalty score: `6960.2832`
- Remaining dominant distortion:
  - `shoulder_lift`: total raw distortion `2892.371`, violations `10`
  - `wrist_flex`: total raw distortion `1499.1736`, violations `5`
  - `gripper`: total raw distortion `73.4454`, violations `8`
  - `elbow_flex`: total raw distortion `116.1489`, violations `1`

v019 candidate memory:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_019/candidate_memory_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_019/candidate_memory_report.md`

Candidate-memory result:

- Best-so-far is now v017/v018 candidate `01`.
- Latest is also best; `regression_from_best.is_regression=false`.
- Next step type: `continue_from_latest_best_prompt_family`.

Comparison:

| Stage | Best penalty | Range violations | Max excess/distortion |
| --- | ---: | ---: | ---: |
| v010 lower-pregrasp + LeRobot state units | 17029.0452 | 34 | 900.0272 |
| v012/v013 projection-aware | 13733.4845 | 27 | 607.8465 |
| v014/v015 residual retry | 15305.6457 | 31 | 769.1897 |
| v017/v018 memory-refine | 6960.2832 | 24 | 393.0481 |

Interpretation:

- Candidate memory plus prompt-family refinement produced the largest
  no-actuation improvement so far.
- The current prompt family is a useful stable anchor for the agentic layer.
- The candidate is still not executable because calibrated range violations
  remain. Projection remains analysis-only; do not execute while camera `3` is
  off or while normal execution gates are blocked.

Recommended next agentic-layer step:

- Continue from v017/v018 candidate `01`.
- Target the remaining `shoulder_lift` and `wrist_flex` excess without adding
  prompt clauses that introduce elbow or gripper regressions.
- If camera `3` returns, do not jump directly to physical execution; first
  rerun observation, rebuild the no-actuation gate, and require the normal
  observer-backed execution protocol.

Validated after v017-v019:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_candidate_memory.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_projection_analysis.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_smolvla_dry.py'
```

## 2026-06-07 v020-v022 Memory-Sample Regression Check

Because v017 improved sharply while reusing the same best prompt family, the
proposal sweep now supports a `memory_sample` prompt profile. It reads
`next_agentic_layer_step.selected_prompt` from candidate memory and deliberately
repeats that exact prompt five times, so the agentic layer can rank multiple
SmolVLA 10-step chunks produced under the same in-loop instruction. Duplicate
prompts are intentional for this profile.

Code updates:

- `scripts/real_so100_agentic_proposal_sweep.py`
  - added `--prompt-profile memory_sample`
  - added `memory_sample_prompts(...)`
  - preserves duplicate prompts for repeated action-chunk sampling
- `tests/test_real_so100_agentic_proposal_sweep.py`
  - verifies repeated selected-prompt sampling and sweep plumbing

v020 memory-sample sweep:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_020/proposal_sweep_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_020/proposal_sweep_report.md`

Best v020 candidate:

- Candidate: `04`
- Prompt:
  `Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, extending the elbow, or flexing the wrist upward.`
- Ready for execution: `false`
- Range violations: `23`
- Total excess raw ticks: `6760.5698`
- Max excess raw ticks: `631.9307`
- Range penalty score: `9806.3619`

v021 projection analysis:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_021/projection_analysis_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_021/projection_analysis_report.md`

Best v021 projected candidate:

- Candidate: `04`
- Shape-only projected ready: `true`
- Range violations clipped: `23`
- Total raw distortion: `6760.5698`
- Max raw distortion: `631.9307`
- Mean raw distortion: `112.6762`
- Projection penalty score: `9806.3619`
- Remaining dominant distortion:
  - `wrist_flex`: total raw distortion `4264.1817`, violations `10`
  - `shoulder_lift`: total raw distortion `1828.8534`, violations `10`
  - `elbow_flex`: total raw distortion `667.5347`, violations `3`
  - `gripper`: total raw distortion `0.0`, violations `0`

v022 candidate memory:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_022/candidate_memory_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_022/candidate_memory_report.md`

Candidate-memory result:

- Best-so-far remains v017/v018 candidate `01`.
- Latest v020/v021 candidate `04` is a regression from best.
- Regression penalty delta: `2846.0787`.
- Next step type: `reuse_best_historical_prompt_family`.

Comparison:

| Stage | Best penalty | Range violations | Max excess/distortion |
| --- | ---: | ---: | ---: |
| v010 lower-pregrasp + LeRobot state units | 17029.0452 | 34 | 900.0272 |
| v012/v013 projection-aware | 13733.4845 | 27 | 607.8465 |
| v014/v015 residual retry | 15305.6457 | 31 | 769.1897 |
| v017/v018 memory-refine | 6960.2832 | 24 | 393.0481 |
| v020/v021 memory-sample | 9806.3619 | 23 | 631.9307 |

Interpretation:

- Repeated sampling under the best prompt reduced violation count by one, but
  increased total and max distortion; it is not the new best.
- The agentic memory should keep v017/v018 candidate `01` as the current
  anchor.
- Continue no-actuation development while camera `3` is off. Do not execute the
  projected/clipped action and do not claim physical motion or task success.

Validated after v020-v022:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
```

## 2026-06-07 v023-v028 Residual-Memory Improvement and Regression Guard

The proposal sweep now supports a `memory_residual` prompt profile. It reads the
selected best prompt from candidate memory, summarizes normalized joint
blockers from both sweep reports and candidate-memory reports, and generates
variants around residual joints. The feedback summarizer now accepts both raw
sweep names (`violation_joint_counts`, `violation_joint_excess_raw_ticks`) and
candidate-memory names (`joint_violation_counts`, `joint_excess_raw_ticks`).

Code updates:

- `scripts/real_so100_agentic_proposal_sweep.py`
  - added `--prompt-profile memory_residual`
  - added `memory_residual_prompts(...)`
  - generalized `summarize_feedback(...)` for candidate-memory reports
  - added duplicate-clause prevention so prompt memory does not keep appending
    the same natural-language clause across iterations
- `tests/test_real_so100_agentic_proposal_sweep.py`
  - verifies residual prompt generation, memory-report field names, and
    duplicate-clause prevention

v023 memory-residual sweep:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_023/proposal_sweep_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_023/proposal_sweep_report.md`

Best v023 candidate:

- Candidate: `02`
- Prompt:
  `Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, extending the elbow, or flexing the wrist upward. Preserve the same side pre-grasp, but keep the wrist neutral and keep the upper arm low.`
- Ready for execution: `false`
- Range violations: `11`
- Total excess raw ticks: `2445.7501`
- Max excess raw ticks: `542.1252`
- Range penalty score: `4622.1257`

v024 projection analysis:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_024/projection_analysis_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_024/projection_analysis_report.md`

Best v024 projected candidate:

- Candidate: `02`
- Shape-only projected ready: `true`
- Range violations clipped: `11`
- Total raw distortion: `2445.7501`
- Max raw distortion: `542.1252`
- Mean raw distortion: `40.7625`
- Projection penalty score: `4622.1257`
- Remaining dominant distortion:
  - `elbow_flex`: total raw distortion `946.6906`, violations `3`
  - `shoulder_lift`: total raw distortion `848.8153`, violations `2`
  - `wrist_flex`: total raw distortion `612.3239`, violations `3`
  - `gripper`: total raw distortion `37.9203`, violations `3`

v025 candidate memory:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_025/candidate_memory_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_025/candidate_memory_report.md`

Candidate-memory result:

- Best-so-far became v023/v024 candidate `02`.
- Latest was also best; `regression_from_best.is_regression=false`.
- Next step type: `continue_from_latest_best_prompt_family`.

v026-v028 regression guard:

- v026 reran `memory_residual` from v025 after adding elbow-residual clauses.
- The best v026/v027 candidate regressed to penalty `14071.936`, with `27`
  range violations, total distortion `10265.5663`, max distortion `818.7899`,
  and mean distortion `171.0928`.
- v028 candidate memory marks v026/v027 as regression with penalty delta
  `9449.8103`.
- Authoritative current best remains v023/v024 candidate `02`.

Comparison:

| Stage | Best penalty | Range violations | Max excess/distortion |
| --- | ---: | ---: | ---: |
| v010 lower-pregrasp + LeRobot state units | 17029.0452 | 34 | 900.0272 |
| v012/v013 projection-aware | 13733.4845 | 27 | 607.8465 |
| v014/v015 residual retry | 15305.6457 | 31 | 769.1897 |
| v017/v018 memory-refine | 6960.2832 | 24 | 393.0481 |
| v020/v021 memory-sample | 9806.3619 | 23 | 631.9307 |
| v023/v024 memory-residual | 4622.1257 | 11 | 542.1252 |
| v026/v027 elbow-expanded residual | 14071.936 | 27 | 818.7899 |

Interpretation:

- `memory_residual` produced the strongest no-actuation improvement so far.
- Expanding residual clauses too aggressively caused prompt bloat and
  regression; duplicate-clause prevention is now in code for future iterations.
- The current best is still not physically executable because calibrated range
  violations remain.
- While camera `3` is off, keep `send_action_called=false`,
  `physical_robot_motion=false`, and `task_success_claim_allowed=false`.

Recommended next agentic-layer step:

- Continue from v023/v024 candidate `02`, but avoid stacking more natural
  language constraints onto the same prompt.
- Prefer a smaller next mutation surface: either sample the v023/v024 best
  prompt family with de-duplicated clauses or introduce a structured local
  prompt template with separate anchor, residual-joint hints, and forbidden
  repeated clauses.
- Do not execute on hardware until the normal execution gate has zero calibrated
  range violations and camera `3` observer evidence is available again.

Validated after v023-v028:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
```

## 2026-06-07 v029-v031 Structured-Prompt Regression Check

The proposal sweep now supports a `memory_structured` prompt profile. It tries
to avoid natural-language prompt bloat by compacting the selected best prompt
into a short `Goal: ... Constraints: ...` format with residual joint hints.

Code updates:

- `scripts/real_so100_agentic_proposal_sweep.py`
  - added `--prompt-profile memory_structured`
  - added `memory_structured_prompts(...)`
  - added compact prompt anchors and structured joint hint generation
- `tests/test_real_so100_agentic_proposal_sweep.py`
  - verifies compact anchor generation and sweep plumbing

v029 structured sweep:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_029/proposal_sweep_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_029/proposal_sweep_report.md`

Best v029 candidate:

- Candidate: `02`
- Prompt:
  `Goal: Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, extending the elbow, or flexing the wrist upward Constraints: low upper arm; neutral wrist; elbow midrange; gripper open. Move as one short 10-step pre-grasp chunk.`
- Ready for execution: `false`
- Range violations: `26`
- Total excess raw ticks: `15525.8209`
- Max excess raw ticks: `1143.6223`
- Range penalty score: `20256.6878`

v030 projection analysis:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_030/projection_analysis_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_030/projection_analysis_report.md`

v031 candidate memory:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_031/candidate_memory_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_031/candidate_memory_report.md`

Candidate-memory result:

- Structured profile is a regression.
- Latest v029/v030 penalty: `20256.6878`.
- Regression penalty delta from v023/v024 best: `15634.5621`.
- Authoritative best remains v023/v024 candidate `02` with penalty
  `4622.1257`, `11` range violations, total excess `2445.7501`, and max excess
  `542.1252`.

Interpretation:

- SmolVLA responded poorly to the explicit `Goal: ... Constraints: ...` format
  in this real SO-100 observer-off context.
- Do not continue structured-prompt mutation as the next primary path.
- Keep the v023/v024 natural-language prompt as the current anchor.
- A better next observer-off direction is to analyze the v023/v024 action chunk
  trajectory itself, or to run de-duplicated repeated sampling around exactly
  the v023/v024 best prompt without adding new constraint clauses.

Validated after v029-v031:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
```

## 2026-06-07 v032-v034 Exact Best-Prompt Sampling Regression Check

After structured prompting regressed, v032 reran `memory_sample` using the
current authoritative v023/v024 best prompt exactly:

`Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, extending the elbow, or flexing the wrist upward. Preserve the same side pre-grasp, but keep the wrist neutral and keep the upper arm low.`

This tests whether repeated SmolVLA sampling under the current best
natural-language prompt can produce a better 10-step action chunk without adding
new prompt clauses.

v032 exact-repeat sweep:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_032/proposal_sweep_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_032/proposal_sweep_report.md`

Best v032 candidate:

- Candidate: `04`
- Ready for execution: `false`
- Range violations: `22`
- Total excess raw ticks: `14893.8786`
- Max excess raw ticks: `959.2787`
- Range penalty score: `18871.7147`

v033 projection analysis:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_033/projection_analysis_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_033/projection_analysis_report.md`

Best v033 projected candidate:

- Candidate: `04`
- Shape-only projected ready: `true`
- Range violations clipped: `22`
- Total raw distortion: `14893.8786`
- Max raw distortion: `959.2787`
- Mean raw distortion: `248.2313`
- Projection penalty score: `18871.7147`
- Remaining dominant distortion:
  - `elbow_flex`: total raw distortion `8289.5801`, violations `10`
  - `shoulder_lift`: total raw distortion `6603.2443`, violations `10`
  - `gripper`: total raw distortion `1.0542`, violations `2`
  - `wrist_flex`: total raw distortion `0.0`, violations `0`

v034 candidate memory:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_034/candidate_memory_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_034/candidate_memory_report.md`

Candidate-memory result:

- Exact-repeat sampling is a regression.
- Latest v032/v033 penalty: `18871.7147`.
- Regression penalty delta from v023/v024 best: `14249.589`.
- Authoritative best remains v023/v024 candidate `02` with penalty
  `4622.1257`, `11` range violations, total excess `2445.7501`, and max excess
  `542.1252`.

Interpretation:

- Repeating the exact best prompt did not reproduce the v023/v024 action
  quality.
- The repeat sample removed wrist-flex violations, but elbow and shoulder
  violations became much worse.
- The agentic layer should stop chasing natural-language prompt variants for
  this observer-off state and inspect the v023/v024 action chunk trajectory
  directly.
- Still no hardware execution: `send_action_called=false`,
  `physical_robot_motion=false`, and `task_success_claim_allowed=false`.

Recommended next agentic-layer step:

- Build an action-trajectory diagnostic around the v023/v024 best chunk:
  compare per-step projected targets against calibrated ranges, identify the
  first violating steps, and propose either a shorter prefix chunk or a
  verifier-gated hold action.
- Keep this analysis-only while camera `3` is off and while any calibrated range
  violation remains.

Validated after v032-v034:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
```

## 2026-06-07 v035 Trajectory Diagnostic

After prompt mutation and repeated sampling both regressed, the agentic layer
moved from prompt search to trajectory diagnosis around the authoritative
v023/v024 best chunk.

Code updates:

- `scripts/real_so100_trajectory_diagnostic.py`
  - reads a projection-analysis report
  - selects the best candidate by default
  - summarizes step-level range violations
  - computes safe prefix length and longest safe contiguous run
  - emits a next agentic-layer action without enabling actuation
- `tests/test_real_so100_trajectory_diagnostic.py`
  - verifies unsafe-prefix / safe-late-run detection
  - verifies observer-off no-actuation contract preservation

v035 trajectory diagnostic:

`_workspace/real_so100/agentic_smolvla_trajectory_diagnostic_move_right_observer_off_035/trajectory_diagnostic_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_trajectory_diagnostic_move_right_observer_off_035/trajectory_diagnostic_report.md`

Diagnostic result for v023/v024 best candidate `02`:

- Action chunk steps: `10`
- Safe prefix length: `0`
- Violation step count: `3`
- Safe step count: `7`
- Longest safe run: starts at step `3`, length `7`
- First violation step: step `0`
- First violation joints:
  - `shoulder_lift`: raw distortion `539.0894`
  - `elbow_flex`: raw distortion `542.1252`
  - `wrist_flex`: raw distortion `324.2242`
  - `gripper`: raw distortion `19.8232`
- Dominant violation joints:
  - `elbow_flex`: `3` violations, total raw distortion `946.6906`
  - `wrist_flex`: `3` violations, total raw distortion `612.3239`
  - `gripper`: `3` violations, total raw distortion `37.9203`
  - `shoulder_lift`: `2` violations, total raw distortion `848.8153`

Next agentic-layer step:

- Type: `do_not_execute_prefix_replan_to_safe_late_pose`
- Reason: the chunk starts with calibrated range violations, even though a
  later contiguous run is range-safe.

Interpretation:

- v023/v024 is not executable as a full 10-step chunk.
- There is no safe prefix from the current state.
- The useful signal is that steps `3-9` form a range-safe target region; the
  agentic layer should replan the initial transition into that late safe region
  rather than keep changing the language prompt.
- This remains no-actuation while camera `3` is off and while the initial
  transition violates calibrated range.

Validated after v035:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_trajectory_diagnostic.py'
```

## 2026-06-07 v036 Late Safe Pose Bridge

The v035 trajectory diagnostic found no safe prefix but did find a contiguous
range-safe late run from steps `3-9`. v036 converts that signal into an
explicit agentic-layer bridge target.

Code updates:

- `scripts/real_so100_late_safe_pose_bridge.py`
  - reads a projection-analysis report plus a trajectory-diagnostic report
  - selects the candidate and safe-run start step
  - extracts the late safe pose joint targets from the projected chunk
  - preserves the observer-off no-actuation contract
  - emits the next agentic-layer action to generate a transition plan
- `tests/test_real_so100_late_safe_pose_bridge.py`
  - verifies safe late-pose extraction
  - verifies blocked behavior when no safe late run exists
  - verifies no hardware execution flags remain false

v036 bridge artifact:

`_workspace/real_so100/agentic_smolvla_late_safe_pose_bridge_move_right_observer_off_036/late_safe_pose_bridge_report.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_late_safe_pose_bridge_move_right_observer_off_036/late_safe_pose_bridge_report.md`

Bridge result for v023/v024 best candidate `02`:

- Status: `passed`
- Source candidate: `02`
- Source prompt:
  `Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, extending the elbow, or flexing the wrist upward. Preserve the same side pre-grasp, but keep the wrist neutral and keep the upper arm low.`
- Safe run start step: `3`
- Safe run length: `7`
- Bridge target step: `3`
- All bridge targets in range: `true`
- Bridge target raw values:
  - `shoulder_pan`: `1787.0526`
  - `shoulder_lift`: `3453.6021`
  - `elbow_flex`: `1777.6110`
  - `wrist_flex`: `2016.2251`
  - `wrist_roll`: `3343.5387`
  - `gripper`: `1867.7250`

Next agentic-layer step:

- Type: `generate_transition_to_late_safe_pose_without_executing`
- Reason: the source chunk has an unsafe prefix, but step `3` is a calibrated
  in-range late safe pose. Generate a new transition plan to this pose, rerun
  the dry gate, and keep physical execution disabled until observer camera `3`
  returns.

Interpretation:

- v036 does not make the task executable yet.
- It turns the best chunk's useful safe region into a concrete target for the
  next in-loop planner/SmolVLA prompt generation step.
- The next version should create a short transition-to-bridge candidate, run
  projection and trajectory diagnostics again, and only proceed toward
  hardware execution after camera `3` observer evidence is available again.
- Still no hardware execution: `send_action_called=false`,
  `physical_robot_motion=false`, and `task_success_claim_allowed=false`.

Validated after v036:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_late_safe_pose_bridge.py'
```

## 2026-06-07 v037-v038 Bridge Transition Plan

v036 produced an in-range late safe pose, but the transition into that pose
still needed a bounded plan from the current episode state. v037-v038 add an
analysis-only transition planner.

Code updates:

- `scripts/real_so100_bridge_transition_plan.py`
  - reads the v036 bridge target plus the current observation episode state
  - builds a calibrated raw-position transition toward the bridge target
  - converts each raw target back to LeRobot SO-100 command units
  - blocks when per-step raw delta exceeds the configured limit
  - optionally keeps 10-step chunk semantics while auto-splitting into multiple
    chunks
- `tests/test_real_so100_bridge_transition_plan.py`
  - verifies a bounded no-actuation transition
  - verifies large single-chunk deltas are blocked
  - verifies `--auto-chunks --chunk-size 10` preserves 10-step chunk groups

v037 single-chunk transition artifact:

`_workspace/real_so100/agentic_smolvla_bridge_transition_move_right_observer_off_037/bridge_transition_plan.json`

v037 result:

- Status: `blocked`
- Requested step count: `10`
- Blockers:
  - `shoulder_lift` required `130.7602` raw ticks per step, above limit `80`
  - `wrist_roll` required `110.0539` raw ticks per step, above limit `80`

v038 auto-chunk transition artifact:

`_workspace/real_so100/agentic_smolvla_bridge_transition_move_right_observer_off_038/bridge_transition_plan.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_bridge_transition_move_right_observer_off_038/bridge_transition_plan.md`

v038 result:

- Status: `passed`
- Auto chunks: `true`
- Chunk size: `10`
- Transition chunk count: `2`
- Transition step count: `20`
- Max per-step raw delta limit: `80`
- Largest per-step deltas:
  - `shoulder_lift`: `65.3801`
  - `wrist_roll`: `55.0269`
  - `shoulder_pan`: `-32.5474`
- All transition targets are inside calibrated ranges.
- Still no hardware execution: `send_action_called=false`,
  `physical_robot_motion=false`, and `task_success_claim_allowed=false`.

Next agentic-layer step:

- Type: `run_projection_and_trajectory_diagnostics_on_transition_candidate`
- Reason: a bounded transition-to-bridge candidate exists, but it is still
  analysis-only. Validate it through the same projection/trajectory gate and
  keep physical execution disabled while camera `3` is off.

Validated after v037-v038:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_bridge_transition_plan.py'
```

## 2026-06-07 v039 Transition Candidate Gate

v038 produced a bounded two-chunk transition plan. v039 adds a formal
observer-off transition candidate gate so the agentic layer can distinguish
internal candidate validity from real hardware execution readiness.

Code updates:

- `scripts/real_so100_transition_candidate_gate.py`
  - validates that the transition source passed
  - validates observer-off no-actuation contract fields
  - validates contiguous 10-step chunk structure
  - validates every transition target remains inside calibrated ranges
  - validates per-step raw deltas remain under the configured limit
  - emits an explicit observer-camera blocker for real execution
- `tests/test_real_so100_transition_candidate_gate.py`
  - verifies a valid two-chunk candidate passes
  - verifies non-10-step chunks are blocked
  - verifies observer-off contract violations are blocked

v039 gate artifact:

`_workspace/real_so100/agentic_smolvla_transition_candidate_gate_move_right_observer_off_039/transition_candidate_gate.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_transition_candidate_gate_move_right_observer_off_039/transition_candidate_gate.md`

v039 result:

- Status: `passed`
- Source transition: v038
- Transition chunk count: `2`
- Transition step count: `20`
- Expected chunk size: `10`
- Chunk `0`: `10` steps, all targets in range, max abs raw delta `65.3802`
- Chunk `1`: `10` steps, all targets in range, max abs raw delta `65.3801`
- Execution ready with observer: `false`
- Execution blocker: camera `3` observer evidence is temporarily unavailable
- Still no hardware execution: `send_action_called=false`,
  `physical_robot_motion=false`, and `task_success_claim_allowed=false`.

Next agentic-layer step:

- Type: `wait_for_observer_camera_3_before_physical_execution_gate`
- Reason: the transition candidate is internally valid as two bounded 10-step
  chunks, but real execution still requires camera `3` observer evidence, live
  readback regeneration, and user confirmation.

Validated after v039:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_transition_candidate_gate.py'
```

## 2026-06-07 v040 Observer Return Preflight

v039 proved the transition candidate is internally valid, but it still cannot
be physically executed while camera `3` is off. v040 formalizes the exact
observer-return preflight requirements before the next physical execution gate
can open.

Code updates:

- `scripts/real_so100_observer_return_preflight.py`
  - reads a transition candidate gate
  - requires observer camera `3` availability
  - requires live SO-100 readback-based transition regeneration
  - requires workspace-clear and user confirmations
  - preserves no-actuation fields while preflight is incomplete
  - lists the execution artifacts that must be produced by the next physical
    attempt
- `tests/test_real_so100_observer_return_preflight.py`
  - verifies the current camera-3-off state remains blocked
  - verifies readiness only when observer, live-readback, workspace, and user
    confirmations are present
  - verifies camera `3` cannot be used as a SmolVLA policy input

v040 preflight artifact:

`_workspace/real_so100/agentic_smolvla_observer_return_preflight_move_right_040/observer_return_preflight.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_observer_return_preflight_move_right_040/observer_return_preflight.md`

v040 result:

- Status: `blocked`
- Source transition gate: v039
- Source transition gate status: `passed`
- Required observer camera index: `3`
- Observer camera status: `off`
- Execution ready with observer: `false`
- Still no hardware execution: `send_action_called=false`,
  `physical_robot_motion=false`, and `task_success_claim_allowed=false`.

Current blockers:

- Observer camera `3` must be available for before/during/after evidence.
- Transition must be regenerated from live SO-100 readback before execution.
- Workspace must be confirmed clear before any physical movement.
- User confirmation is required before any physical SO-100 execution.

Required execution artifacts when camera `3` returns:

- `camera_3_before_frame`
- `camera_3_motion_video`
- `camera_3_after_frame`
- `live_readback_before_raw`
- `live_readback_regenerated_transition_plan`
- `per_step_target_commands`
- `readback_after_raw`
- `observed_delta_raw`
- `task_level_grasp_outcome_if_contact_attempted`
- `task_level_object_relocation_if_transport_attempted`
- `agentic_feedback_report`

Next agentic-layer step:

- Type: `wait_for_observer_camera_3_and_live_readback_regeneration`
- Reason: the transition candidate is valid, but physical execution preflight
  is not complete.

Validated after v040:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_observer_return_preflight.py'
```

## 2026-06-07 v041-v042 Readback Transition Regeneration

v040 identified live readback regeneration as a required pre-execution step.
v041-v042 add that regeneration path in replay-safe form so the same code can
be rerun from a live read-only SO-100 probe when camera `3` returns.

Code updates:

- `scripts/real_so100_readback_transition_regenerator.py`
  - reads a passed bridge target
  - reads current raw joint state from one of:
    - read-only probe payload `positions_raw`
    - execution report `readback_before_raw`
    - transition report `source_current_raw`
    - JSON payload `state`
    - observation payload `observation.state`
    - JSONL episode frame
  - regenerates a bounded transition into 10-step chunks
  - preserves no-actuation fields
  - marks whether the transition came from `live` readback or replay/static
    state
- `tests/test_real_so100_readback_transition_regenerator.py`
  - verifies read-only probe payload handling
  - verifies replay readback still requires live rerun before execution
  - verifies missing joint blockers
  - verifies JSONL episode frame handling

v041 replay-regenerated transition:

`_workspace/real_so100/agentic_smolvla_readback_transition_regen_move_right_replay_041/readback_transition_regen.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_readback_transition_regen_move_right_replay_041/readback_transition_regen.md`

v041 result:

- Status: `passed`
- Readback source: `replay`
- Live readback regenerated: `false`
- Source readback: `_workspace/real_so100/agentic_iteration_move_right_u20cam_observer_off_001/episode.jsonl`
- Source frame index: `0`
- Transition chunk count: `2`
- Transition step count: `20`
- Max per-step raw delta limit: `80`
- Largest per-step deltas:
  - `shoulder_lift`: `65.3801`
  - `wrist_roll`: `55.0269`
  - `shoulder_pan`: `-32.5474`
- Still no hardware execution: `send_action_called=false`,
  `physical_robot_motion=false`, and `task_success_claim_allowed=false`.

v041 next agentic-layer step:

- Type: `rerun_with_live_readback_before_execution`
- Reason: the transition was regenerated from replay/static readback only;
  physical execution still requires a live readback source.

v042 candidate gate for v041:

`_workspace/real_so100/agentic_smolvla_transition_candidate_gate_move_right_replay_042/transition_candidate_gate.json`

v042 result:

- Status: `passed`
- Confirms the regenerated transition has two contiguous 10-step chunks.
- Confirms every target remains inside calibrated range.
- Confirms max chunk delta remains about `65.38` raw ticks.
- Execution ready with observer remains `false` because camera `3` is still
  unavailable and the readback source was replay, not live.

Validated after v041-v042:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_readback_transition_regenerator.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_transition_candidate_gate.py'
```

## 2026-06-07 v043 Observer Return Refresh Orchestration

v043 adds a single refresh wrapper for the next camera-3-return loop. The
wrapper chains readback acquisition/regeneration, transition candidate gating,
and observer-return preflight while preserving no-actuation semantics.

Code updates:

- `scripts/real_so100_observer_return_refresh.py`
  - `mode=replay`: consumes an existing readback/episode artifact
  - `mode=live_readonly`: runs `scripts/real_so100_read_only_probe.py` to
    capture live `positions_raw`
  - runs `real_so100_readback_transition_regenerator.py`
  - runs `real_so100_transition_candidate_gate.py`
  - runs `real_so100_observer_return_preflight.py`
  - writes a top-level summary artifact with the next agentic-layer action
- `tests/test_real_so100_observer_return_refresh.py`
  - verifies replay refresh stays blocked but exercises the chain
  - verifies mocked live-readonly refresh can reach execution preflight
  - verifies replay mode requires a replay readback input

v043 replay refresh artifact:

`_workspace/real_so100/agentic_smolvla_observer_return_refresh_move_right_replay_043/observer_return_refresh.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_observer_return_refresh_move_right_replay_043/observer_return_refresh.md`

Child artifacts:

- `_workspace/real_so100/agentic_smolvla_observer_return_refresh_move_right_replay_043/readback_transition_regen.json`
- `_workspace/real_so100/agentic_smolvla_observer_return_refresh_move_right_replay_043/transition_candidate_gate.json`
- `_workspace/real_so100/agentic_smolvla_observer_return_refresh_move_right_replay_043/observer_return_preflight.json`

v043 result:

- Status: `blocked`
- Mode: `replay`
- Readback source: `replay`
- Regenerated transition status: `passed`
- Transition gate status: `passed`
- Observer preflight status: `blocked`
- Transition chunk count: `2`
- Transition step count: `20`
- Still no hardware execution: `send_action_called=false`,
  `physical_robot_motion=false`, and `task_success_claim_allowed=false`.

Current blocker:

- `Observer return preflight is not ready.`

Next agentic-layer step:

- Type: `rerun_refresh_in_live_readonly_mode_when_camera_3_returns`
- Reason: replay refresh validates the orchestration path, but physical
  execution still needs live readback and observer camera `3`.

When camera `3` returns, run the same wrapper in live-readonly mode first:

```bash
PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_observer_return_refresh.py \
  --bridge-report _workspace/real_so100/agentic_smolvla_late_safe_pose_bridge_move_right_observer_off_036/late_safe_pose_bridge_report.json \
  --output-dir _workspace/real_so100/agentic_smolvla_observer_return_refresh_move_right_live_<NNN> \
  --port /dev/cu.usbmodem5AE60824791 \
  --mode live_readonly \
  --observer-camera-index 3 \
  --observer-camera-status available \
  --workspace-clear-confirmed \
  --user-confirmed \
  --max-abs-raw-delta-per-step 80 \
  --chunk-size 10
```

That command is still read-only for robot state acquisition; it does not send
motor commands. If it reaches `ready_for_execution_gate`, the next stage may
build an observer-backed execution report with camera `3` before/during/after
evidence.

Validated after v043:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_observer_return_refresh.py'
```

## 2026-06-07 v044 Transition Execution Packet

v044 turns the refreshed transition artifacts into an execution-packet shape
that a future observer-backed executor can consume. It does not execute motor
commands and remains blocked unless the observer-return refresh, live readback,
transition gate, and observer preflight are all ready.

Code updates:

- `scripts/real_so100_transition_execution_packet.py`
  - reads an observer-return refresh summary
  - loads the regenerated transition, transition gate, and observer preflight
  - converts transition steps into two execution chunks with per-step
    `target_command`, `target_raw_estimate`, and `write_normalize=true`
  - carries required camera `3` observer artifacts and readback artifact slots
  - blocks unless the refresh came from live readback and observer preflight is
    ready
- `tests/test_real_so100_transition_execution_packet.py`
  - verifies blocked replay/preflight behavior
  - verifies a mocked ready refresh produces a ready execution packet
  - verifies non-10-step chunks are blocked

v044 execution packet:

`_workspace/real_so100/agentic_smolvla_transition_execution_packet_move_right_replay_044/transition_execution_packet.json`

Human-readable companion:

`_workspace/real_so100/agentic_smolvla_transition_execution_packet_move_right_replay_044/transition_execution_packet.md`

v044 result:

- Status: `blocked`
- Source refresh: v043 replay refresh
- Transition chunk count: `2`
- Transition step count: `20`
- Execution ready: `false`
- Live readback regenerated: `false`
- Required observer artifacts:
  - `camera_3_before_frame`
  - `camera_3_motion_video`
  - `camera_3_after_frame`
- Still no hardware execution: `send_action_called=false`,
  `physical_robot_motion=false`, and `task_success_claim_allowed=false`.

Current blockers:

- Observer return refresh is not ready for execution gate.
- Transition was not regenerated from live readback.
- Observer preflight did not pass.

What is now prepared:

- Chunk `0`: `10` per-step target commands and raw estimates.
- Chunk `1`: `10` per-step target commands and raw estimates.
- Readback slots:
  - `readback_before_raw=null`
  - `readback_after_raw=null`
  - `observed_delta_raw=null`

Next agentic-layer step:

- Type: `resolve_refresh_preflight_before_execution_packet`
- Reason: execution packet is blocked until refresh, live readback,
  transition gate, and observer preflight are all ready.

Validated after v044:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_transition_execution_packet.py'
```

## 2026-06-07 v045 Transition Packet Executor

v045 adds the executor boundary for transition execution packets. The executor
can consume the v044 packet shape, but it refuses to connect to the robot unless
the packet is ready and all observer-backed execution flags are present.

Code updates:

- `scripts/real_so100_execute_transition_packet.py`
  - reads a transition execution packet
  - validates packet readiness, two 10-step chunks, complete joint targets, and
    observer camera index `3`
  - in dry-run mode, never connects to the robot and never writes
  - in execution mode, requires:
    - packet status `ready_for_observer_backed_execution`
    - `--execute`
    - `--human-confirmed`
    - `--workspace-clear-confirmed`
    - `--record-video`
    - `--visual-output-dir`
    - observer camera index `3`
  - when all gates pass, it will record before/video/after observer evidence,
    read live raw state, write per-step target commands, read after state, and
    preserve observed deltas
- `tests/test_real_so100_execute_transition_packet.py`
  - verifies blocked packets stop before bus connection
  - verifies ready packets dry-run without connecting
  - verifies mocked ready execution writes all `20` steps with camera evidence

v045 dry-run executor artifact:

`_workspace/real_so100/agentic_smolvla_execute_transition_packet_move_right_replay_045/execute_transition_packet_report.json`

v045 result:

- Status: `dry_run`
- Packet status: `blocked`
- Packet execution ready: `false`
- Execute requested: `false`
- Transition chunk count: `2`
- Transition step count: `20`
- Blocker:
  - `Transition execution packet is not ready.`
- Still no hardware execution: `send_action_called=false`,
  `policy_actions_executed=false`, and `physical_robot_motion=false`.

Next real execution path:

1. Bring camera `3` back.
2. Run v043 refresh in `live_readonly` mode.
3. Build a new v044-style execution packet from that ready refresh.
4. Run v045 executor first without `--execute` and inspect the dry-run report.
5. Only after the dry-run report is clean, rerun with `--execute`,
   `--human-confirmed`, `--workspace-clear-confirmed`, `--record-video`, and a
   `--visual-output-dir`.

Validated after v045:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_execute_transition_packet.py'
```

## 2026-06-07 v046 Transition Execution Feedback

v046 closes the agentic-layer loop after the transition packet executor. It
turns a blocked, dry-run, failed, or executed transition report into a
normalized feedback artifact for the next SmolVLA agentic iteration.

Code updates:

- `scripts/real_so100_transition_execution_feedback.py`
  - consumes a v045-style execution report
  - loads the linked transition execution packet when available
  - optionally consumes grasp and relocation verifier outputs after a real
    physical attempt
  - preserves the camera contract:
    - policy cameras: `0`, `1`
    - observer camera: `3`, not a policy input
  - separates execution-gate blockers from SmolVLA prompt-quality feedback
  - keeps `does_not_prompt_operator=true`
  - blocks prompt mutation when no policy action executed
  - requires grasp and object-relocation verifiers before any task-success
    candidate for the move-right transport task
- `tests/test_real_so100_transition_execution_feedback.py`
  - verifies v045-like dry-run feedback preserves the current candidate and
    does not mutate the prompt
  - verifies executed chunks without verifiers request task-level verifiers
  - verifies grasp plus relocation verifier success can become a task-success
    candidate

v046 feedback artifact:

`_workspace/real_so100/agentic_smolvla_transition_execution_feedback_move_right_replay_046/transition_execution_feedback.json`

v046 result:

- Status: `passed` as feedback generation
- Execution status: `dry_run`
- No hardware execution:
  - `send_action_called=false`
  - `policy_actions_executed=false`
  - `physical_robot_motion=false`
- Camera contract:
  - `policy_camera_indexes=[0, 1]`
  - `observer_camera_indexes=[]`
  - `observer_camera_status=off`
  - `camera_3_policy_input=false`
- Failure modes:
  - `execution_packet_not_ready`
  - `observer_or_live_readback_preflight_incomplete`
  - `no_policy_action_executed`
  - `task_success_not_verified`
- Prompt mutation: `false`
- Task success claim: `false`

Next agentic-layer step:

- Type: `rerun_observer_return_refresh_live_readonly_when_camera_3_available`
- Reason: the transition candidate remains useful, but the current blocker is
  observer/live-readback preflight rather than SmolVLA prompt quality.

While camera `3` remains off, continue no-actuation agentic-layer development
with policy cameras `0` and `1` only. Do not execute the transition packet and
do not claim grasp, relocation, or final task success.

Validated after v046:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_transition_execution_feedback.py'
```

## 2026-06-07 v047 Prompt-Mutation Gate

v047 updates the observer-off proposal sweep so execution feedback can block
unnecessary SmolVLA prompt mutation. This prevents the agentic layer from
treating an observer/live-readback preflight blocker as evidence that the
current prompt or transition candidate is bad.

Code updates:

- `scripts/real_so100_agentic_proposal_sweep.py`
  - detects feedback artifacts with
    `operation=real_so100_transition_execution_feedback`
  - when `prompt_mutation_allowed=false` and no explicit prompts are supplied,
    it does not call SmolVLA dry inference
  - writes a no-actuation proposal-sweep report with zero new candidates
  - preserves policy camera indexes `0` and `1`
  - keeps observer camera index `3` unavailable and observer-only
  - emits `preserve_transition_candidate_until_observer_live_readback_gate`
- `tests/test_real_so100_agentic_proposal_sweep.py`
  - verifies v046-style feedback prevents calls to SmolVLA dry inference and
    execution dry gates

v047 artifact:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_047/proposal_sweep_report.json`

v047 result:

- Status: `passed` as a guardrail artifact
- New candidates: `0`
- `feedback_gate.prompt_mutation_allowed=false`
- Source feedback step:
  `rerun_observer_return_refresh_live_readonly_when_camera_3_available`
- Next agentic-layer step:
  `preserve_transition_candidate_until_observer_live_readback_gate`
- Still no hardware execution:
  - `send_action_called=false`
  - `policy_actions_executed=false`
  - `physical_robot_motion=false`
- Task success claim remains blocked.

This means the current best transition candidate should be preserved. The next
meaningful real-world progress is not another natural-language prompt mutation;
it is live readback regeneration plus observer-backed execution preflight after
camera `3` returns. While camera `3` stays off, camera `0` and `1` can still be
used for no-actuation evidence and state inspection, but not for physical
success claims.

Validated after v047:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
```

## 2026-06-07 v048 Feedback Router

v048 adds a reusable feedback router for the real SO-100 agentic layer. The
router consumes feedback artifacts from different loop stages and decides
whether the next work is prompt mutation, execution preflight, task verifier
execution, success accounting, or holding the current candidate.

Code updates:

- `scripts/real_so100_agentic_feedback_router.py`
  - accepts one or more `--feedback-report` inputs
  - normalizes camera roles, execution flags, failure modes, and
    `next_agentic_layer_step`
  - routes feedback into:
    - `resolve_execution_preflight`
    - `run_task_verifiers`
    - `mutate_smolvla_prompt_or_plan`
    - `success_accounting`
    - `hold_current_candidate`
  - preserves the guardrails that cameras `0` and `1` are policy inputs and
    camera `3` is observer/debug only
  - explicitly blocks prompt mutation when feedback is about
    observer/live-readback preflight
- `tests/test_real_so100_agentic_feedback_router.py`
  - verifies execution-preflight feedback routes to the observer/live-readback
    gate
  - verifies executed chunks without verifiers route to task verifiers
  - verifies candidate-memory feedback can route to no-actuation prompt
    mutation

v048 artifact:

`_workspace/real_so100/agentic_smolvla_feedback_router_move_right_observer_off_048/feedback_router_report.json`

Inputs:

- v046 transition execution feedback
- v047 prompt-mutation guardrail report

v048 result:

- Selected route: `resolve_execution_preflight`
- Prompt mutation allowed: `false`
- Next step:
  `rerun_observer_return_refresh_live_readonly_when_camera_3_available`
- Policy cameras: `[0, 1]`
- Observer cameras: `[]`
- Observer status: `off`
- Still no hardware execution:
  - `send_action_called=false`
  - `policy_actions_executed=false`
  - `physical_robot_motion=false`
- Task success claim remains blocked.

This is now the authoritative current agentic-layer decision: preserve the
current transition candidate and reopen only the observer/live-readback
execution gate when camera `3` becomes available.

Validated after v048:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_feedback_router.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
```

## 2026-06-07 v049 Execution Preflight Runbook

v049 converts the v048 routing decision into a concrete camera-3-return runbook.
It prepares the exact live-readonly commands needed to reopen the execution
preflight path without moving the robot.

Code updates:

- `scripts/real_so100_execution_preflight_runbook.py`
  - consumes the v048 feedback router report
  - consumes the v036 late-safe bridge report
  - writes a JSON/Markdown runbook for the preserved transition candidate
  - generates three commands:
    - live-readonly observer-return refresh
    - transition execution packet build
    - executor dry-run
  - deliberately does not generate a physical `--execute` command
- `tests/test_real_so100_execution_preflight_runbook.py`
  - verifies the live-readonly runbook is generated only when the router
    selected `resolve_execution_preflight`
  - verifies the generated commands do not include `--execute`

v049 artifact:

`_workspace/real_so100/agentic_smolvla_execution_preflight_runbook_move_right_observer_off_049/execution_preflight_runbook.json`

v049 result:

- Status: `passed`
- Selected route: `resolve_execution_preflight`
- Commands prepared:
  - `observer_return_refresh_live_readonly`
  - `build_transition_execution_packet`
  - `executor_dry_run`
- All commands preserve:
  - policy cameras `[0, 1]`
  - observer camera `3`
  - no motor writes
  - no physical robot motion
- The runbook includes the current v036 bridge target:
  - `shoulder_pan=1787.0526`
  - `shoulder_lift=3453.6021`
  - `elbow_flex=1777.6110`
  - `wrist_flex=2016.2251`
  - `wrist_roll=3343.5387`
  - `gripper=1867.7250`

Next agentic-layer step:

- Type: `wait_for_camera_3_then_run_live_readonly_refresh`
- Reason: the current transition candidate is preserved; when observer camera
  `3` returns, run live-readonly refresh before any physical execution.

Validated after v049:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_execution_preflight_runbook.py'
```

## 2026-06-07 v050 Execution Preflight Runbook Audit

v050 adds a safety audit for the v049 camera-3-return runbook. The audit checks
that the prepared commands are truly no-actuation, live-readonly-first, and
observer-camera-3-gated before anyone uses the runbook.

Code updates:

- `scripts/audit_real_so100_execution_preflight_runbook.py`
  - loads a v049-style execution preflight runbook
  - verifies no-actuation fields are false
  - verifies policy cameras are `[0, 1]`
  - verifies required observer camera is `3`
  - verifies command order:
    - `observer_return_refresh_live_readonly`
    - `build_transition_execution_packet`
    - `executor_dry_run`
  - verifies generated commands do not include `--execute`
  - verifies the first command uses `--mode live_readonly`
  - verifies the refresh and packet commands use observer camera `3`
  - verifies the executor command is a dry-run shape with `--record-video`
  - verifies the v036 bridge target is in calibrated range
- `tests/test_audit_real_so100_execution_preflight_runbook.py`
  - passes a no-execute live-readonly runbook
  - fails a runbook that accidentally includes `--execute`

v050 artifact:

`_workspace/real_so100/agentic_smolvla_execution_preflight_runbook_audit_move_right_observer_off_050/execution_preflight_runbook_audit.json`

v050 result:

- Status: `passed`
- Check count: `15`
- Failed checks: `0`
- Confirmed:
  - no `--execute`
  - live-readonly refresh first
  - camera `3` is the required observer camera
  - policy cameras are `[0, 1]`
  - v036 bridge target is in range
  - still no motor write and no physical motion

Next agentic-layer step:

- Type: `safe_to_run_live_readonly_refresh_when_camera_3_returns`
- Reason: the runbook is safe as a no-actuation live-readonly preflight path,
  but it should only be run after observer camera `3` is available.

Validated after v050:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_audit_real_so100_execution_preflight_runbook.py'
```

## 2026-06-07 v051 Feedback Router After Runbook Audit

v051 teaches the feedback router to consume the v050 execution preflight
runbook audit. A passed audit no longer routes back to prompt mutation or the
older unresolved execution-preflight blocker; it records the current state as
waiting for observer camera `3`.

Code updates:

- `scripts/real_so100_agentic_feedback_router.py`
  - recognizes `operation=real_so100_execution_preflight_runbook_audit`
  - routes a passed audit with `failed_check_count=0` to
    `await_observer_camera_3`
  - routes a failed audit to `fix_execution_preflight_runbook`
  - preserves `prompt_mutation_allowed=false` for both cases
- `tests/test_real_so100_agentic_feedback_router.py`
  - verifies a passed v050-style audit supersedes older execution-preflight
    blockers
  - verifies a failed audit repairs the runbook instead of mutating prompts

v051 artifact:

`_workspace/real_so100/agentic_smolvla_feedback_router_move_right_observer_off_051/feedback_router_report.json`

v051 result:

- Status: `passed`
- Selected route: `await_observer_camera_3`
- Next agentic-layer step: `wait_for_camera_3_then_run_live_readonly_refresh`
- Prompt mutation allowed: `false`
- Policy cameras: `[0, 1]`
- Observer cameras: `[]`
- Observer camera status: `off`
- Send action called: `false`
- Policy actions executed: `false`
- Physical robot motion: `false`
- Task success claim allowed: `false`

Current interpretation:

- The best transition candidate remains preserved.
- The no-actuation runbook has passed audit.
- Camera `3` is still required before any physical execution or task-success
  evidence.
- While camera `3` is off, continue only no-actuation agentic-layer work with
  policy cameras `0` and `1`.

Validated after v051:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_feedback_router.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_audit_real_so100_execution_preflight_runbook.py'
```

## 2026-06-07 v052-v055 Policy-Camera Feedback While Observer Is Off

v052-v055 continues agentic-layer development with camera `3` temporarily off.
The loop uses only policy cameras `0` and `1`, keeps camera `3` excluded from
SmolVLA, and still sends no robot actions.

Code updates:

- `scripts/real_so100_observe.py`
  - adds explicit `--allow-camera-only-without-robot`
  - default behavior still blocks on serial connection failure
  - when the flag is set, camera-only evidence can be saved with
    `observation.state=null` if live readback is unavailable
- `scripts/build_real_so100_policy_camera_feedback.py`
  - packages Codex/Pseudo-LLM visual feedback into a replaceable LLM/VLM
    feedback artifact
  - targets `in_loop_agent_or_smolvla`, not the operator
  - keeps `does_not_prompt_operator=true`
  - keeps `task_success_claim_allowed=false`
- `scripts/real_so100_agentic_proposal_sweep.py`
  - adds `--prompt-profile policy_camera_feedback`
  - reads `pseudo_llm_feedback.next_smolvla_prompt` as the no-actuation
    SmolVLA candidate prompt only when `does_not_prompt_operator=true`

v052 observation artifact:

`_workspace/real_so100/agentic_smolvla_policy_cams_0_1_observer_off_052/manifest.json`

v052 observation result:

- Status: `ok=true`
- Frames recorded: `3`
- Mode: `live_readback_and_camera`
- Policy cameras: `[0, 1]`
- Observer cameras: `[]`
- Live readback available: `true`
- Latest frame state:
  - `shoulder_pan=2438`
  - `shoulder_lift=2146`
  - `elbow_flex=2043`
  - `wrist_flex=1940`
  - `wrist_roll=2243`
  - `gripper=1788`
- `send_action_called=false`
- `physical_robot_motion=false`

Visual inspection:

- Camera `0` shows wrist/gripper context and a partial green target near the
  upper-left image edge.
- Camera `1` clearly shows the green Android figure and the robot arm in a
  wide scene view, so it is useful for policy-context feedback while observer
  camera `3` is unavailable.
- This is not observer-backed success evidence.

v052 Pseudo-LLM feedback artifact:

`_workspace/real_so100/agentic_smolvla_policy_camera_feedback_observer_off_052/policy_camera_feedback.json`

v053 no-actuation SmolVLA proposal:

`_workspace/real_so100/agentic_smolvla_proposal_sweep_move_right_observer_off_053/proposal_sweep_report.json`

v053 result:

- Prompt source: v052 policy-camera Pseudo-LLM feedback
- Candidates: `1`
- Action steps: `10`
- Dry inference: `passed`
- Execute gate: `dry_run`
- Ready for execution: `false`
- Range violations: `31`
- Penalty score: `20925.3096`
- Dominant range excess:
  - `elbow_flex=8026.0614`
  - `shoulder_lift=6360.8705`
  - `wrist_flex=2170.7335`
- `send_action_called=false`
- `physical_robot_motion=false`
- `task_success_claim_allowed=false`

v054 projection artifact:

`_workspace/real_so100/agentic_smolvla_projection_analysis_move_right_observer_off_054/projection_analysis_report.json`

v055 candidate memory artifact:

`_workspace/real_so100/agentic_smolvla_candidate_memory_move_right_observer_off_055/candidate_memory_report.json`

v055 memory result:

- Best historical candidate remains v023/v024 candidate `02`
- Best penalty: `4622.1257`
- Latest v054 penalty: `20925.3096`
- Regression delta: `16303.1839`
- Next step: `reuse_best_historical_prompt_family`
- Selected prompt:
  `Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, extending the elbow, or flexing the wrist upward. Preserve the same side pre-grasp, but keep the wrist neutral and keep the upper arm low.`

Current interpretation:

- Camera `1` is useful wide-context policy feedback, but the v052 feedback
  prompt caused SmolVLA to overreach badly.
- Do not use v053 as an execution candidate.
- Preserve the v023/v024 best prompt family and the later v038/v039 transition
  candidate path.
- Physical execution remains blocked until camera `3` observer evidence
  returns and live-readonly refresh passes.

Validated after v052-v055:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_observe.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_build_real_so100_policy_camera_feedback.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_proposal_sweep.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_candidate_memory.py'
```

## 2026-06-07 v056 Feedback Router With Regression Memory

v056 combines the v051 observer-wait routing state with the v055 candidate
memory regression. This prevents the agentic loop from interpreting
`reuse_best_historical_prompt_family` as another immediate prompt-mutation
sweep when a safe transition candidate already exists downstream of that best
historical prompt.

Code updates:

- `scripts/real_so100_agentic_feedback_router.py`
  - consumes prior `real_so100_agentic_feedback_router` artifacts and preserves
    `await_observer_camera_3`
  - routes candidate-memory reports with
    `regression_from_best.is_regression=true` to
    `preserve_best_historical_candidate`
  - keeps `prompt_mutation_allowed=false` for regressed policy-camera prompts
- `tests/test_real_so100_agentic_feedback_router.py`
  - verifies regressed candidate memory preserves the best candidate rather
    than mutating prompts
  - verifies prior await-observer router artifacts remain await-observer inputs

v056 artifact:

`_workspace/real_so100/agentic_smolvla_feedback_router_move_right_observer_off_056/feedback_router_report.json`

v056 result:

- Status: `passed`
- Source reports:
  - v051 feedback router
  - v055 candidate memory
- v055 recommended route: `preserve_best_historical_candidate`
- Selected route: `await_observer_camera_3`
- Next agentic-layer step: `wait_for_camera_3_then_run_live_readonly_refresh`
- Prompt mutation allowed: `false`
- `send_action_called=false`
- `physical_robot_motion=false`
- `task_success_claim_allowed=false`

Current authoritative state:

- v053 policy-camera feedback prompt is rejected as a regression.
- v023/v024 candidate `02` remains the best historical prompt/action family.
- v038/v039 transition path remains the preserved execution candidate path.
- The next real step is still observer-camera-3 return followed by
  live-readonly refresh, not another prompt sweep and not physical execution.

Validated after v056:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_feedback_router.py'
```

## 2026-06-07 v057 Agentic Loop State

v057 creates a single machine-readable state artifact for the current real
SO-100 SmolVLA agentic loop. It combines the latest router, candidate memory,
and policy-camera observation so the next loop does not need to reinterpret
several reports by hand.

Code updates:

- `scripts/real_so100_agentic_loop_state.py`
  - consumes a router report, candidate memory, and optional observation
    manifest
  - emits `allowed_next_actions`, `blocked_actions`, success requirements, and
    the best historical candidate
  - keeps physical execution and task-success claims blocked while camera `3`
    is unavailable
- `tests/test_real_so100_agentic_loop_state.py`
  - verifies observer-camera wait state
  - verifies regressed policy-camera prompts are blocked
  - verifies physical execution and task-success claims are blocked

v057 artifact:

`_workspace/real_so100/agentic_smolvla_loop_state_move_right_observer_off_057/agentic_loop_state.json`

v057 result:

- Status: `passed`
- Selected route: `await_observer_camera_3`
- Allowed next action:
  `wait_for_camera_3_then_run_live_readonly_refresh`
- Blocked actions:
  - `physical_execution`
  - `task_success_claim`
  - `rerun_regressed_policy_camera_prompt`
  - `prompt_mutation_before_observer_refresh`
- Best historical candidate:
  - source: v024 projection report
  - candidate: `02`
  - penalty: `4622.1257`
  - range violations: `11`
- Latest regression:
  - v054 regressed by `16303.1839`
- `send_action_called=false`
- `physical_robot_motion=false`
- `task_success_claim_allowed=false`

Success requirements preserved by v057:

- observer camera `3` before/during/after evidence for any physical movement
- live readback-regenerated transition gate passes
- human and workspace-clear confirmation before motor writes
- grasp outcome verifier after contact
- object relocation verifier shows the object moved right in image space

Validated after v057:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_loop_state.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_feedback_router.py'
```

## 2026-06-07 v058 State-Derived Command Plan

v058 converts the v057 single loop state into the next allowed command plan.
This is a state-driven guard: it does not execute anything, and it refuses to
plan from states that authorize prompt mutation, record prior physical motion,
or fail to explicitly block physical execution.

Code updates:

- `scripts/real_so100_agentic_state_command_plan.py`
  - consumes v057-style loop state plus the v036 bridge target
  - emits the next allowed no-actuation command plan
  - carries forward blocked actions from v057
  - requires observer camera `3`
  - emits no `--execute` command
- `tests/test_real_so100_agentic_state_command_plan.py`
  - verifies observer-wait states produce live-readonly command plans
  - verifies prompt-mutation states are blocked
  - verifies states that already record physical motion are blocked

v058 artifact:

`_workspace/real_so100/agentic_smolvla_state_command_plan_move_right_observer_off_058/state_command_plan.json`

v058 result:

- Status: `passed`
- Command count: `3`
- Has `--execute`: `false`
- Next agentic-layer step: `run_first_command_only_when_camera_3_available`
- Required observer camera: `3`
- Requires observer camera available: `true`
- `send_action_called=false`
- `physical_robot_motion=false`
- `task_success_claim_allowed=false`

Prepared commands:

1. `observer_return_refresh_live_readonly`
2. `build_transition_execution_packet`
3. `executor_dry_run`

Important execution rule:

- Only command 1 is the next allowed command, and only after camera `3` is
  available.
- Commands 2 and 3 depend on command 1 output and remain non-execute/dry-run
  planning steps.
- Physical motor writes remain blocked until a later ready packet, observer
  evidence, and explicit execution confirmation exist.

Validated after v058:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_state_command_plan.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_loop_state.py'
```

## 2026-06-07 v059 State Command Plan Audit

v059 audits the v058 state-derived command plan before use. This is the current
guard immediately before any future camera-3-return command is run.

Code updates:

- `scripts/audit_real_so100_agentic_state_command_plan.py`
  - verifies the v058 operation and status
  - verifies no-actuation flags
  - verifies policy cameras `[0, 1]`
  - verifies observer camera `3` is required and must be available
  - verifies command order
  - verifies no command contains `--execute`
  - verifies the first command is live-readonly observer refresh
  - verifies packet/dry-run commands use camera `3`
  - verifies blocked actions from v057 are carried forward
- `tests/test_audit_real_so100_agentic_state_command_plan.py`
  - passes a no-execute live-readonly plan
  - fails a plan with `--execute`
  - fails a plan missing required blocked-action carry-forward

v059 artifact:

`_workspace/real_so100/agentic_smolvla_state_command_plan_audit_move_right_observer_off_059/state_command_plan_audit.json`

v059 result:

- Status: `passed`
- Check count: `15`
- Failed checks: `0`
- Next agentic-layer step:
  `safe_to_run_first_command_when_camera_3_available`
- `send_action_called=false`
- `physical_robot_motion=false`
- `task_success_claim_allowed=false`

Current execution boundary:

- The only safe next command is command 1 from v058:
  `observer_return_refresh_live_readonly`.
- It may be run only after observer camera `3` is available.
- It is no-actuation/read-only.
- No physical movement, grasp claim, relocation claim, or task-success claim is
  allowed from v059 alone.

Validated after v059:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_audit_real_so100_agentic_state_command_plan.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_state_command_plan.py'
```

## 2026-06-07 v060 First Command Launch Packet

v060 extracts only the first safe command from the v058 plan after the v059
audit passed. It does not run the command and does not authorize physical
execution.

Code updates:

- `scripts/real_so100_first_command_launch_packet.py`
  - consumes the v058 state command plan and v059 audit
  - requires audit status `passed` and `failed_check_count=0`
  - requires the first command to be `observer_return_refresh_live_readonly`
  - blocks if the first command contains `--execute`
  - preserves follow-up commands only as blocked/dependent commands
- `tests/test_real_so100_first_command_launch_packet.py`
  - verifies only the first live-readonly command is extracted
  - verifies failed audit blocks launch-packet creation
  - verifies wrong first command blocks
  - verifies `--execute` in the first command blocks

v060 artifact:

`_workspace/real_so100/agentic_smolvla_first_command_launch_packet_move_right_observer_off_060/first_command_launch_packet.json`

v060 result:

- Status: `passed`
- Launch command name: `observer_return_refresh_live_readonly`
- Launch command allowed when: `observer_camera_3_available`
- Follow-up commands blocked as `depend_on_refresh_output`
- `does_not_run_command=true`
- `not_a_physical_execution_authorization=true`
- `send_action_called=false`
- `physical_robot_motion=false`
- `task_success_claim_allowed=false`

Current next step:

- Wait until observer camera `3` is available again.
- Then run only the v060 `launch_command` and capture its output.
- Do not run the packet-build or dry-run commands until the refresh output
  exists and is inspected.

Validated after v060:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_first_command_launch_packet.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_audit_real_so100_agentic_state_command_plan.py'
```

## 2026-06-07 v061 Agentic Development Lane

v061 separates the real-robot loop into two explicit lanes while camera `3` is
temporarily off:

- Execution lane: waits for observer camera `3`, then may run only the v060
  `observer_return_refresh_live_readonly` command.
- Policy-camera development lane: remains active with cameras `0` and `1`, but
  only for no-actuation agentic-layer work.

Code updates:

- `scripts/real_so100_agentic_development_lane.py`
  - consumes v057 loop state, v060 launch packet, v052 policy-camera feedback,
    and v055 candidate memory
  - preserves camera `0` and `1` as the only policy cameras
  - preserves camera `3` as observer-only and currently unavailable
  - records the execution lane separately from policy-camera development
  - blocks physical execution and task-success claims from policy cameras alone
  - blocks prompt mutation from the latest regressed policy-camera prompt
  - recommends building a policy-camera task-state packet next
- `tests/test_real_so100_agentic_development_lane.py`
  - verifies the execution lane waits for camera `3`
  - verifies policy-camera no-actuation development remains active
  - verifies regressed prompt mutation is blocked
  - verifies prompt sweeps are allowed only when feedback has no regression
  - verifies a launch packet that authorizes execution is rejected

v061 artifact:

`_workspace/real_so100/agentic_smolvla_development_lane_move_right_observer_off_061/development_lane.json`

v061 result:

- Status: `passed`
- Execution lane: `waiting_for_observer_camera_3`
- Policy-camera development lane: `active_no_actuation`
- Next agentic-layer step: `build_policy_camera_task_state_packet`
- Prompt mutation allowed: `false`
- Blocked no-actuation action:
  `mutate_prompt_from_latest_policy_camera_feedback`
- `send_action_called=false`
- `physical_robot_motion=false`
- `task_success_claim_allowed=false`

Current development rule:

- Continue improving the general agentic layer by converting cameras `0` and
  `1` into task-state packets for the in-loop agent/SmolVLA.
- Do not generate another latest-feedback prompt sweep until candidate memory no
  longer marks the latest feedback as regressed.
- Do not physically execute anything until camera `3` is available again and
  the v060 launch command has produced live-readonly refresh output.

Validated after v061:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_development_lane.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_first_command_launch_packet.py'
```

## 2026-06-07 v062 Policy-Camera Task-State Packet

v062 converts camera `0` and camera `1` policy observations into a structured
task-state packet for the replaceable in-loop LLM/VLM. This packet is not an
action decision and does not authorize physical execution.

Code updates:

- `scripts/real_so100_policy_camera_task_state_packet.py`
  - consumes the v061 development lane and v052 policy-camera observation
  - runs the pre-grasp probe over camera `0` and camera `1`
  - runs jaw readiness on the wrist camera
  - represents the task as object-frame pick-and-transport:
    green Android figure, move object right
  - emits allowed and forbidden LLM/VLM reasoning outputs
  - keeps camera `3` out of policy input
  - records `send_action_called=false` and `physical_robot_motion=false`
- `tests/test_real_so100_policy_camera_task_state_packet.py`
  - verifies structured LLM/VLM packet creation from policy cameras
  - verifies edge-clipped object observations request reframe/approach reasoning
  - verifies a blocked source lane blocks packet status

v062 artifact:

`_workspace/real_so100/agentic_smolvla_policy_camera_task_state_move_right_observer_off_062/task_state_packet.json`

v062 result:

- Status: `passed`
- Frame: `2`
- Policy cameras: `[0, 1]`
- Camera `3` as policy input: `false`
- Object visible: `true`
- Usable pre-grasp camera: `1`
- Camera `1` object center: `[1111.54, 592.96]`
- Camera `0` wrist/jaw context: `blocked`
- Camera `0` blocker: `green object touches image boundary`
- Next agentic-layer step:
  `ask_llm_vlm_for_no_actuation_jaw_alignment_prompt_packet`
- `send_action_called=false`
- `physical_robot_motion=false`
- `task_success_claim_allowed=false`

Visual inspection after v062:

- `camera_0_000002.jpg`: wrist/jaw region visible, but the green target is
  clipped at the upper-left image edge.
- `camera_1_000002.jpg`: wide egocentric context clearly shows the green
  Android figure and surrounding workspace.

Current development rule:

- The next packet should ask the in-loop LLM/VLM for a no-actuation jaw
  alignment / approach prompt packet.
- That packet should preserve object-frame wording and should not instruct the
  human operator.
- Do not run physical execution until observer camera `3` returns and the
  execution lane is reopened.

Validated after v062:

```bash
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_policy_camera_task_state_packet.py'
PYTHONPATH=src:. .venv/bin/python -B -m unittest discover -s tests -p 'test_real_so100_agentic_development_lane.py'
```
