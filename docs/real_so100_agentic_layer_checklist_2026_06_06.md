# Real SO-100 Agentic Layer Startup Checklist

Date: 2026-06-06

## Goal

- [ ] Use the already-installed SO-100 and an iPhone webcam to build a safe real-robot loop for agentic-layer optimization.
- [ ] Keep the lightweight VLA policy frozen.
- [ ] Optimize only the surrounding agentic layer: task decomposition, prompts, verifier predicates, retry/replan policy, camera selection, action filtering, and failure taxonomy.
- [ ] Use Codex as an outer-loop reviewer/optimizer that reads artifacts and patches the agentic layer.
- [ ] Avoid imitation training in the first real-robot phase.

## Current Hardware Assumption

- [ ] SO-100 is physically assembled.
- [ ] iPhone will be used as the first webcam through macOS Continuity Camera.
- [ ] Extra fixed cameras may be added later, but the first loop must work with one iPhone camera.
- [ ] No real robot action should be sent until camera capture, robot discovery, logging, safety limits, and dry-run policy output all pass.

## Phase 0: Workspace and Safety Preparation

- [ ] Clear a dedicated table area for the robot.
- [ ] Remove fragile, sharp, reflective, or unstable objects from the workspace.
- [ ] Place the robot on a stable, non-slippery surface.
- [ ] Mark the robot base position with tape so it can be restored after movement.
- [ ] Mark an allowed workspace rectangle on the table.
- [ ] Keep the first object lightweight, matte, and soft-edged.
- [ ] Use a red box or red block that is easy to segment visually.
- [ ] Prepare a left target region using tape or a colored marker.
- [ ] Make sure the robot can move without hitting the laptop, phone, cables, table edge, or your hand.
- [ ] Keep one hand near power or emergency stop during the first powered tests.
- [ ] Decide a manual stop action before running any code.
  - [ ] Unplug power.
  - [ ] Disable motor torque.
  - [ ] Kill the control process.
- [ ] Document the manual stop method in the real-robot run notes.

## Phase 1: Repository State Check

- [ ] Start from the repo root:

```bash
cd /Users/minhaeng/workspace/physical_ai_agent
```

- [ ] Check the working tree before changing code:

```bash
git status --short --branch
```

- [ ] Confirm the existing local camera note is available:

```bash
sed -n '1,220p' docs/iphone_continuity_camera_capture.md
```

- [ ] Confirm the existing SO101 browser-only SmolVLA path is still documented:

```bash
sh scripts/view_so101_live.sh --browser-only --policy smolvla --allow-download --smolvla-action-steps 15 --show-inputs --fps 2
```

- [ ] Treat the browser-only path as the last sim-side sanity check before touching real robot actuation.

## Phase 2: iPhone Camera Bring-Up

- [ ] Put the iPhone on a fixed mount.
- [ ] Use landscape orientation unless a task-specific reason requires portrait.
- [ ] Frame the full robot workspace.
- [ ] Include the robot gripper, red object, left target region, and table boundary in view.
- [ ] Avoid backlighting and strong reflections.
- [ ] Lock the phone position once framing is good.
- [ ] Open a camera-using app once if needed so macOS exposes Continuity Camera.
- [ ] Grant camera access to Terminal:
  - [ ] Open `System Settings -> Privacy & Security -> Camera`.
  - [ ] Enable camera access for `Terminal`.
  - [ ] If permission is stuck, run `tccutil reset Camera` and try again.
- [ ] Scan camera indexes from Terminal, not only from Codex sandbox:

```bash
.venv/bin/python _workspace/camera_permission_probe/scan_opencv_cameras.py \
  --start 0 \
  --end 8 \
  --output-dir _workspace/camera_permission_probe/opencv_scan \
  --report _workspace/camera_permission_probe/opencv_scan_report.json
```

- [ ] Identify the iPhone camera index.
- [ ] Capture one still image:

```bash
.venv/bin/python _workspace/camera_permission_probe/probe_camera_opencv.py \
  --index 1 \
  --output _workspace/camera_permission_probe/iphone_camera.jpg \
  --report _workspace/camera_permission_probe/iphone_camera_report.json \
  --timeout-seconds 15
```

- [ ] Replace `--index 1` if the scan reports a different iPhone index.
- [ ] Open the captured image and verify:
  - [ ] Robot base is visible.
  - [ ] Robot arm is visible.
  - [ ] Gripper is visible.
  - [ ] Red box is visible.
  - [ ] Left target region is visible.
  - [ ] Workspace boundary is visible.
  - [ ] No critical object is cropped out.
- [ ] Record a 5-second test video:

```bash
.venv/bin/python _workspace/camera_permission_probe/record_opencv_video.py \
  --index 1 \
  --output _workspace/camera_permission_probe/iphone_video.mp4 \
  --report _workspace/camera_permission_probe/iphone_video_report.json \
  --duration-seconds 5 \
  --fps 15
```

- [ ] Verify the video has stable exposure and no severe dropped frames.
- [ ] Save the selected camera index in the real-robot config once a config file exists.

## Phase 3: SO-100 Hardware Discovery Without Motion

- [ ] Connect SO-100 to the Mac.
- [ ] Confirm the device appears at the OS level.
- [ ] List serial devices:

```bash
ls /dev/tty.* /dev/cu.*
```

- [ ] Identify the SO-100 device path.
- [ ] Record the device path in run notes.
- [ ] Confirm the robot control package needed for SO-100 is installed.
- [ ] If using LeRobot hardware APIs, confirm the exact SO-100 robot type and config name.
- [ ] Run only a read-only discovery command first.
- [ ] Confirm the code can read robot metadata without enabling torque.
- [ ] Confirm the code can read joint positions without commanding movement.
- [ ] Save a discovery report with:
  - [ ] timestamp
  - [ ] device path
  - [ ] robot model/config
  - [ ] joint names
  - [ ] joint position vector
  - [ ] gripper position
  - [ ] any firmware or calibration metadata
- [ ] Do not proceed if joint reads are unstable, missing, or delayed.

## Phase 4: Real-Robot Config Files

- [ ] Complete SO-100 device calibration before treating joint positions as policy-ready observations.
- [ ] Do not run policy dry-runs or observation datasets as "calibrated" until the calibration file path is saved in the robot config.
- [ ] Add a real robot config under `configs/robot/`.
- [ ] Include:
  - [ ] robot kind: `so100`
  - [ ] serial/device path
  - [ ] action dimension
  - [ ] joint names
  - [ ] joint limits
  - [ ] velocity limits
  - [ ] acceleration limits if available
  - [ ] gripper limits
  - [ ] home pose
  - [ ] safe observation pose
  - [ ] emergency stop instructions
  - [ ] calibration file path
  - [ ] calibration timestamp
  - [ ] calibration operator
- [ ] Add a camera config under `configs/camera/`.
- [ ] Include:
  - [ ] camera kind: `iphone_continuity`
  - [ ] OpenCV backend: `AVFOUNDATION`
  - [ ] camera index
  - [ ] width
  - [ ] height
  - [ ] FPS
  - [ ] camera role: `external_workspace`
- [ ] Add an experiment config under `configs/eval/`.
- [ ] Include:
  - [ ] task: `move_red_box_left`
  - [ ] prompt: `붉은색 상자를 집어서 왼쪽으로 옮겨줘`
  - [ ] object description
  - [ ] start region
  - [ ] target region
  - [ ] max episode seconds
  - [ ] max robot steps
  - [ ] max retries
  - [ ] actuation mode: `disabled` for first dry runs

## Phase 4A: SO-100 Device Calibration Gate

- [ ] Confirm the SO-100 serial port:

```bash
.venv/bin/python -m serial.tools.list_ports -v
```

- [ ] Use the SO-100 controller port, not monitor or Bluetooth ports.
- [ ] Current observed SO-100 candidate:

```text
/dev/cu.usbmodem5AE60824791
```

- [ ] Confirm read-only joint positions still work:

```bash
.venv/bin/python scripts/real_so100_read_only_probe.py \
  --port /dev/cu.usbmodem5AE60824791 \
  --output _workspace/real_so100/read_only_discovery_report_so100_pre_calibration.json
```

- [ ] Run LeRobot SO follower calibration only when the arm can be safely moved by hand.
- [ ] Keep robot policy/action execution disabled during calibration.
- [ ] Follow the LeRobot calibration prompts exactly:
  - [ ] move the arm to the middle of its range when prompted
  - [ ] move all joints except `wrist_roll` through their ranges when prompted
  - [ ] press ENTER only after each requested manual positioning step
- [ ] Preserve the generated calibration file.
- [ ] Record the calibration file path.
- [ ] Record the calibration id used by LeRobot.
- [ ] Record whether this is first-time calibration or reuse of an existing calibration file.
- [ ] After calibration, run read-only joint position probe again:

```bash
.venv/bin/python scripts/real_so100_read_only_probe.py \
  --port /dev/cu.usbmodem5AE60824791 \
  --output _workspace/real_so100/read_only_discovery_report_so100_post_calibration.json
```

- [ ] Confirm post-calibration joint reads are stable across three samples.
- [ ] Save calibration metadata to:

```text
_workspace/real_so100/calibration_manifest.json
```

- [ ] Do not proceed to policy dry-run until the calibration manifest exists.

Suggested calibration manifest fields:

```json
{
  "robot_kind": "so100_follower",
  "port": "/dev/cu.usbmodem5AE60824791",
  "calibration_file": "<path from LeRobot>",
  "calibration_id": "<robot id>",
  "calibrated_at": "YYYY-MM-DDTHH:MM:SS",
  "operator": "human",
  "pre_calibration_report": "_workspace/real_so100/read_only_discovery_report_so100_pre_calibration.json",
  "post_calibration_report": "_workspace/real_so100/read_only_discovery_report_so100_post_calibration.json",
  "notes": "No policy actions were executed during calibration."
}
```

## Phase 5: Observation Recorder

- [ ] Implement or add a script for synchronized real-robot episode recording.
- [ ] The first recorder mode must be observation-only.
- [ ] Each frame should save:
  - [ ] monotonic timestamp
  - [ ] wall-clock timestamp
  - [ ] camera frame path
  - [ ] camera index
  - [ ] frame width/height
  - [ ] robot joint positions
  - [ ] gripper position
  - [ ] optional robot status flags
  - [ ] task instruction
  - [ ] episode id
  - [ ] step index
- [ ] Use this output layout:

```text
_workspace/real_so100/
  YYYYMMDD_HHMMSS_move_red_box_left_observation_only/
    config_snapshot.json
    manifest.json
    frames/
      camera_external_000000.jpg
    robot_states.jsonl
    episode.jsonl
    notes.md
```

- [ ] Record a 10-second observation-only episode.
- [ ] Move the red box by hand during the episode to test visual change.
- [ ] Confirm the frame count roughly matches the target FPS.
- [ ] Confirm timestamps are monotonic.
- [ ] Confirm robot state is logged at every step or with documented missing samples.
- [ ] Confirm the episode can be replayed visually.

## Phase 6: Task Scene Calibration

- [ ] Define the start region for the red box.
- [ ] Define the target-left region.
- [ ] Put tape markers on the table.
- [ ] Capture a clean background image with no object.
- [ ] Capture an initial task image with the red box at the start region.
- [ ] Capture a goal image with the red box manually placed in the left target region.
- [ ] Save the three calibration images:
  - [ ] empty workspace
  - [ ] start state
  - [ ] goal state
- [ ] Record approximate pixel bounding boxes for:
  - [ ] red object at start
  - [ ] target-left region
  - [ ] robot gripper at home
- [ ] Write down the physical dimensions of the red box.
- [ ] Write down the approximate distance from robot base to object.
- [ ] Write down the approximate distance from robot base to target-left region.

## Phase 7: Verifier v0 Before Policy Execution

- [ ] Implement a simple visual verifier before running SmolVLA on hardware.
- [ ] First verifier predicates:
  - [ ] `red_object_visible`
  - [ ] `red_object_bbox`
  - [ ] `red_object_center_px`
  - [ ] `red_object_in_start_region`
  - [ ] `red_object_in_left_target_region`
  - [ ] `gripper_visible`
  - [ ] `workspace_visible`
- [ ] Use simple color segmentation first.
- [ ] Save verifier output as JSON per frame.
- [ ] Run verifier on:
  - [ ] empty workspace image
  - [ ] start state image
  - [ ] goal state image
  - [ ] 10-second observation-only video
- [ ] Confirm the verifier does not mark success on the start image.
- [ ] Confirm the verifier marks target success on the goal image.
- [ ] Confirm the verifier reports uncertainty when the red object is occluded or cropped.
- [ ] Do not use this verifier as final truth for publication; use it as a retry/progress signal.

## Phase 8: SmolVLA Real-Image Dry Run With No Actuation

- [ ] Use the real iPhone frame as policy input.
- [ ] Use the Korean task prompt first:

```text
붉은색 상자를 집어서 왼쪽으로 옮겨줘
```

- [ ] Also test an English equivalent to check policy sensitivity:

```text
Pick up the red box and move it to the left.
```

- [ ] Run SmolVLA in dry-run mode only.
- [ ] Save the predicted action chunk.
- [ ] Save policy latency.
- [ ] Save the exact input image path.
- [ ] Save the exact prompt.
- [ ] Save the model id/checkpoint.
- [ ] Save preprocessing metadata.
- [ ] Save output to:

```text
_workspace/real_so100/
  YYYYMMDD_HHMMSS_move_red_box_left_smolvla_dry/
    manifest.json
    input_frames/
    policy_actions.jsonl
    verifier.jsonl
    notes.md
```

- [ ] Manually inspect whether the action dimensions and magnitudes are plausible.
- [ ] Do not send these actions to the robot yet.

## Phase 9: Action Adapter and Safety Filter

- [ ] Implement a real SO-100 action adapter separate from the policy.
- [ ] The adapter must accept a policy action and return either:
  - [ ] a bounded robot command
  - [ ] a rejection with reason
- [ ] Add hard action limits:
  - [ ] maximum joint delta per step
  - [ ] maximum gripper delta per step
  - [ ] maximum command frequency
  - [ ] maximum episode duration
  - [ ] maximum number of executed steps
  - [ ] workspace bounds
  - [ ] forbidden joint ranges if any
- [ ] Add command smoothing.
- [ ] Add a deadman switch flag.
- [ ] Add a dry-run flag that is enabled by default.
- [ ] Add a `--require-explicit-actuation` flag for any command that can move the robot.
- [ ] Log every rejected action with reason.
- [ ] Unit test the safety filter with:
  - [ ] zero action
  - [ ] small valid action
  - [ ] too-large action
  - [ ] invalid dimension
  - [ ] NaN/Inf action
  - [ ] gripper over-limit
  - [ ] workspace over-limit

## Phase 10: First Motion Without SmolVLA

- [ ] Start with robot home/neutral pose.
- [ ] Run a read-only state check.
- [ ] Enable torque only if required and safe.
- [ ] Execute one tiny scripted motion.
- [ ] Use a motion so small that it cannot hit the object.
- [ ] Record camera and robot state during motion.
- [ ] Return to home pose.
- [ ] Confirm the robot stopped.
- [ ] Confirm the logged action matches the visible movement.
- [ ] Confirm the action adapter records command and state.
- [ ] Do not combine SmolVLA with actuation until scripted bounded motion is verified.

## Phase 11: First Bounded SmolVLA-Assisted Motion

- [ ] Put the red box far enough that a bad tiny motion will not collide dangerously.
- [ ] Use SmolVLA to propose an action chunk.
- [ ] Pass the action through the safety adapter.
- [ ] Execute only the first safe bounded step, not the full chunk.
- [ ] Record before/during/after frames.
- [ ] Record robot state before/during/after.
- [ ] Run verifier after the step.
- [ ] Stop after one step.
- [ ] Inspect the episode manually.
- [ ] Classify the result:
  - [ ] harmless no-op
  - [ ] moved toward object
  - [ ] moved away from object
  - [ ] moved in wrong axis
  - [ ] unsafe/rejected
  - [ ] camera/verifier failure
- [ ] Patch only adapter/verifier/prompting logic, not model weights.

## Phase 12: Agentic Layer v0

- [ ] Define the agentic controller inputs:
  - [ ] task instruction
  - [ ] current camera frame
  - [ ] current robot state
  - [ ] verifier predicates
  - [ ] previous action
  - [ ] previous failure label
- [ ] Define the agentic controller outputs:
  - [ ] subgoal instruction
  - [ ] policy prompt
  - [ ] allowed action scale
  - [ ] retry decision
  - [ ] stop decision
  - [ ] failure label
- [ ] Start with explicit subgoals:
  - [ ] observe red box
  - [ ] move gripper near red box
  - [ ] align gripper with red box
  - [ ] close gripper
  - [ ] lift slightly
  - [ ] move left
  - [ ] release
  - [ ] verify red box in left target region
- [ ] Keep each subgoal bounded by max steps.
- [ ] Require verifier progress before moving to the next subgoal.
- [ ] Stop when verifier reports object lost, workspace lost, or unsafe motion.

## Phase 13: Codex Outer-Loop Optimization

- [ ] After each episode batch, produce a compact evidence pack.
- [ ] Evidence pack must include:
  - [ ] config snapshot
  - [ ] prompts/subgoals used
  - [ ] policy action chunks
  - [ ] executed commands
  - [ ] rejected commands
  - [ ] camera video or contact sheet
  - [ ] verifier timeline
  - [ ] failure labels
  - [ ] final human annotation
- [ ] Ask Codex to review only the evidence pack and repo code.
- [ ] Codex may patch:
  - [ ] prompt templates
  - [ ] subgoal order
  - [ ] retry thresholds
  - [ ] verifier predicates
  - [ ] camera selection
  - [ ] action scaling
  - [ ] safety rejection reasons
  - [ ] logging schema
- [ ] Codex must not:
  - [ ] edit SmolVLA weights
  - [ ] use imitation demonstrations as training data in this phase
  - [ ] silently loosen safety limits to get success
  - [ ] count verifier self-judgment as final task success
- [ ] Every Codex patch must include:
  - [ ] hypothesis
  - [ ] changed files
  - [ ] expected failure reduction
  - [ ] verification command
  - [ ] before/after comparison plan

## Phase 14: First Real Task Evaluation Protocol

- [ ] Use the same physical scene for all first comparisons.
- [ ] Use the same red box.
- [ ] Use the same phone position.
- [ ] Use the same table markers.
- [ ] Use the same prompt.
- [ ] Run condition A: policy-only dry or bounded execution.
- [ ] Run condition B: static agentic wrapper.
- [ ] Run condition C: Codex-optimized agentic wrapper.
- [ ] Use the same action budget per condition.
- [ ] Use the same retry budget per condition.
- [ ] Use a human final label for early real-world episodes.
- [ ] Record final outcome categories:
  - [ ] success
  - [ ] partial progress
  - [ ] no progress
  - [ ] wrong object/region
  - [ ] unsafe stopped
  - [ ] perception failure
  - [ ] hardware/control failure
- [ ] Report verifier predicates separately from final success.

## Phase 15: Minimum Scripts To Add Next

- [ ] `scripts/real_so100_scan.sh`
  - [ ] scans camera indexes
  - [ ] scans serial devices
  - [ ] writes hardware discovery report
- [ ] `scripts/real_so100_observe.sh`
  - [ ] records camera + robot state
  - [ ] never actuates
- [ ] `scripts/real_so100_smolvla_dry.sh`
  - [ ] runs real-image SmolVLA action prediction
  - [ ] never actuates
- [ ] `scripts/real_so100_scripted_motion.sh`
  - [ ] executes one tiny scripted safe motion
  - [ ] requires explicit actuation flag
- [ ] `scripts/real_so100_agentic_episode.sh`
  - [ ] runs one bounded agentic episode
  - [ ] requires explicit actuation flag
- [ ] `scripts/build_real_so100_episode_report.py`
  - [ ] turns logs/videos/verifier outputs into a Codex-reviewable evidence pack

## Phase 16: Stop Conditions

- [ ] Stop immediately if robot state cannot be read reliably.
- [ ] Stop immediately if camera frame is missing, frozen, or badly cropped.
- [ ] Stop immediately if the verifier reports workspace not visible.
- [ ] Stop immediately if an action has invalid dimension.
- [ ] Stop immediately if an action contains NaN or Inf.
- [ ] Stop immediately if an action exceeds safety limits.
- [ ] Stop immediately if a cable is near the arm.
- [ ] Stop immediately if the robot moves opposite to the expected safe direction during scripted motion.
- [ ] Stop immediately if the object leaves the camera view.
- [ ] Stop immediately if any unexpected collision occurs.

## Phase 17: First-Day Definition Of Done

- [ ] iPhone camera index is known.
- [ ] iPhone still image capture succeeds.
- [ ] iPhone video capture succeeds.
- [ ] SO-100 device path is known.
- [ ] SO-100 read-only state discovery succeeds.
- [ ] Observation-only real episode is recorded.
- [ ] Calibration images are saved.
- [ ] Verifier v0 runs on calibration images.
- [ ] SmolVLA dry-run predicts actions from a real camera frame.
- [ ] No robot actuation happens from SmolVLA yet.
- [ ] All evidence is saved under `_workspace/real_so100/`.
- [ ] Next code checkpoint is clear.

## Suggested Next Checkpoint Names

- [ ] CP26A: SO-100 hardware discovery and iPhone camera scan.
- [ ] CP26B: SO-100 device calibration manifest.
- [ ] CP26C: synchronized real observation recorder.
- [ ] CP26D: real-image SmolVLA dry-run with no actuation.
- [ ] CP26E: SO-100 action adapter and safety filter.
- [ ] CP26F: scripted bounded real motion.
- [ ] CP26G: verifier-gated agentic real episode.
- [ ] CP26H: Codex outer-loop improvement report.
