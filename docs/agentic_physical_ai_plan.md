# Agentic Physical AI Plan

## Goal

- [ ] Build a Mac-local evaluation stack for agentic physical AI.
- [ ] Use MuJoCo, LIBERO, and LeRobot as the first simulation and policy evaluation path.
- [ ] Compare a standalone policy against an agentic wrapper with planner, verifier, retry, and replan logic.
- [ ] Keep the architecture compatible with local VLM/LLM inference on a MacBook Pro M5 Pro with 64 GB unified memory.

Target MVP:

> Evaluate whether a weak or medium VLA/policy, such as ACT or SmolVLA, can achieve better long-horizon task success when wrapped with an agentic planner-verifier-retry loop in LIBERO simulation.

## Recommended Stack

- [ ] Simulation: MuJoCo
- [ ] Benchmark: LIBERO
- [ ] Robot learning framework: LeRobot
- [ ] First baseline policy: random policy and ACT
- [ ] Second baseline policy: SmolVLA
- [ ] Agent runtime: custom Python runtime
- [ ] Planner: rule-based first, local LLM later
- [ ] Verifier: simulation-state verifier first, VLM verifier later
- [ ] Local inference: Ollama or llama.cpp first, MLX later
- [ ] Debug UI: FastAPI + WebSocket + React or Vite
- [ ] Optional observability: Foxglove if ROS2 is added later
- [ ] Cloud expansion: GPU training for SmolVLA/OpenPI/pi0 later

## Architecture

```text
Task Instruction
        |
        v
Agentic Runtime
  - planner
  - subgoal manager
  - verifier
  - retry/replan policy
  - trace logger
        |
        v
Policy Executor
  - random policy
  - ACT
  - SmolVLA
        |
        v
Simulation Interface
  - LIBERO
  - MuJoCo
        |
        v
Metrics, traces, videos, replay data
```

## Phase 0: Project Definition

- [ ] Write the first MVP sentence.
  - Suggested: "Compare policy-only execution against agentic retry/replan execution on LIBERO long-horizon tasks."
- [ ] Confirm the first target machine.
  - [ ] MacBook Pro M5 Pro
  - [ ] 64 GB unified memory
  - [ ] No CUDA dependency for the first milestone
- [ ] Confirm first simulation path.
  - [ ] Primary: MuJoCo + LIBERO + LeRobot
  - [ ] Secondary: Meta-World
  - [ ] Later: raw robosuite custom environments
  - [ ] Later: ManiSkill
  - [ ] Later: Isaac Lab on cloud NVIDIA GPU
- [ ] Choose first evaluation suites.
  - [ ] LIBERO-Spatial for easier baseline checks
  - [ ] LIBERO-Object for object grounding checks
  - [ ] LIBERO-Goal for goal-conditioned tasks
  - [ ] LIBERO-Long or `libero_10` for agentic long-horizon evaluation
- [ ] Define comparison conditions.
  - [ ] A: policy-only execution
  - [ ] B: policy plus simple retry
  - [ ] C: planner plus policy plus verifier plus retry/replan
- [ ] Define success criteria.
  - [ ] Task success rate
  - [ ] Subgoal success rate
  - [ ] Retry count
  - [ ] Recovery success rate
  - [ ] Average episode length
  - [ ] Policy latency
  - [ ] Planner latency
  - [ ] Verifier latency
  - [ ] False positive verification rate
  - [ ] False negative verification rate
  - [ ] Invalid or unsafe action count

## Phase 1: Repository Skeleton

- [ ] Create the initial directory layout.

```text
physical_ai_agent/
  apps/
    web/
    agent/
  packages/
    sim/
    policies/
    agent_core/
    skills/
    inference/
    evaluation/
    data/
    safety/
    observability/
  configs/
    sim/
    policy/
    agent/
    eval/
  scripts/
  experiments/
  docs/
```

- [ ] Create `configs/sim/libero.yaml`.
- [ ] Create `configs/policy/random.yaml`.
- [ ] Create `configs/policy/act.yaml`.
- [ ] Create `configs/policy/smolvla.yaml`.
- [ ] Create `configs/agent/rule_planner.yaml`.
- [ ] Create `configs/agent/verifier.yaml`.
- [ ] Create `configs/eval/libero_baseline.yaml`.
- [ ] Add a config loader.
  - [ ] YAML loading
  - [ ] CLI overrides
  - [ ] experiment output directory creation
  - [ ] config snapshot saved with each run
- [ ] Define experiment output layout.

```text
experiments/
  YYYY-MM-DD_libero_long_agentic/
    config.yaml
    metrics.json
    episodes.jsonl
    traces.jsonl
    summary.md
    videos/
    logs/
```

- [ ] Add `README.md` with first-run instructions.
- [ ] Add `docs/architecture.md`.
- [ ] Add `docs/evaluation.md`.

## Phase 2: Local Environment Setup

- [ ] Choose Python version.
  - [ ] Recommended: Python 3.10 or 3.11
- [ ] Choose package manager.
  - [ ] Recommended: `uv`
  - [ ] Alternative: `poetry`
- [ ] Add core dependencies.
  - [ ] `lerobot`
  - [ ] `mujoco`
  - [ ] `robosuite`
  - [ ] `libero`
  - [ ] `torch`
  - [ ] `transformers`
  - [ ] `numpy`
  - [ ] `opencv-python`
  - [ ] `pydantic`
  - [ ] `pyyaml`
  - [ ] `rich`
  - [ ] `fastapi`
  - [ ] `websockets`
  - [ ] `uvicorn`
  - [ ] `wandb` or `mlflow`
- [ ] Check Apple Silicon compatibility.
  - [ ] PyTorch import works
  - [ ] PyTorch MPS availability checked
  - [ ] MuJoCo import works
  - [ ] MuJoCo rendering works
  - [ ] Headless rendering works or fallback is documented
  - [ ] CPU-only fallback works
- [ ] Add smoke tests.
  - [ ] MuJoCo import test
  - [ ] robosuite import test
  - [ ] LIBERO environment reset test
  - [ ] LeRobot import test
  - [ ] Camera observation shape check
  - [ ] Random action step check

## Phase 3: LIBERO and LeRobot Baseline

- [ ] Run a LIBERO environment directly.
  - [ ] Create environment
  - [ ] Reset environment
  - [ ] Step random action
  - [ ] Inspect observation keys
  - [ ] Inspect reward, done, and info
  - [ ] Save one rendered image
- [ ] Verify LeRobot LIBERO integration.
  - [ ] Single-suite evaluation runs
  - [ ] Observation format is documented
  - [ ] Action format is documented
  - [ ] Camera keys are documented
  - [ ] Robot state keys are documented
- [ ] Implement baseline policy adapters.
  - [ ] Random policy
  - [ ] ACT adapter
  - [ ] SmolVLA adapter
  - [ ] Scripted oracle adapter if available
- [ ] Implement baseline evaluator.
  - [ ] `run_episode(policy, env, task)`
  - [ ] `run_eval(policy, suite, num_episodes)`
  - [ ] Save metrics
  - [ ] Save episode traces
  - [ ] Save videos
  - [ ] Save per-step observation summaries
  - [ ] Save per-step actions
- [ ] Collect baseline metrics.
  - [ ] Random policy success rate
  - [ ] ACT success rate
  - [ ] SmolVLA success rate
  - [ ] Average episode length
  - [ ] Average policy latency
  - [ ] Failure breakdown

## Phase 4: Agentic Runtime Design

- [ ] Define the agent entrypoint.

```python
agent.run(task_instruction, env_context) -> EpisodeResult
```

- [ ] Define agent state fields.
  - [ ] Current task
  - [ ] Current subgoal
  - [ ] Step index
  - [ ] Observation summary
  - [ ] Policy result
  - [ ] Verifier result
  - [ ] Retry count
  - [ ] Failure reason
  - [ ] Trace log
- [ ] Define subgoal schema.

```json
{
  "id": "subgoal_001",
  "instruction": "move the gripper above the red mug",
  "success_condition": "gripper is above red mug",
  "max_attempts": 2,
  "policy": "smolvla",
  "timeout_steps": 80
}
```

- [ ] Define trace schema.
  - [ ] Planner input
  - [ ] Planner output
  - [ ] Verifier input
  - [ ] Verifier output
  - [ ] Policy actions
  - [ ] Observations
  - [ ] Retries
  - [ ] Replans
  - [ ] Final outcome
- [ ] Define failure taxonomy.
  - [ ] Object not found
  - [ ] Wrong object selected
  - [ ] Grasp failed
  - [ ] Object dropped
  - [ ] Timeout
  - [ ] Collision or unsafe action
  - [ ] Verifier uncertain
  - [ ] Invalid policy output

## Phase 5: Planner

- [ ] Implement planner interface.

```python
planner.plan(task, observation_context) -> list[Subgoal]
```

- [ ] Implement rule-based planner first.
  - [ ] Parse known LIBERO task instructions
  - [ ] Generate deterministic subgoals
  - [ ] Use known task templates
  - [ ] Avoid external LLM dependency for first MVP
- [ ] Add LLM-compatible planner later.
  - [ ] Prompt template
  - [ ] JSON schema output
  - [ ] Output validation
  - [ ] Retry malformed JSON
  - [ ] Fallback to rule-based planner
- [ ] Validate planner output.
  - [ ] Required fields
  - [ ] Maximum subgoal count
  - [ ] Allowed policies
  - [ ] Allowed skills
  - [ ] Timeout bounds
  - [ ] Retry bounds
- [ ] Evaluate planner quality.
  - [ ] Invalid output rate
  - [ ] Unsupported skill rate
  - [ ] Subgoal count distribution
  - [ ] Manual review of sampled plans

## Phase 6: Policy Executor

- [ ] Implement policy adapter interface.

```python
policy.act(observation, instruction, state) -> ActionChunk
```

- [ ] Implement action chunk executor.
  - [ ] Execute action step by step
  - [ ] Allow interruption
  - [ ] Enforce timeout
  - [ ] Run safety checks before stepping
  - [ ] Record per-step actions
- [ ] Implement adapters.
  - [ ] Random policy adapter
  - [ ] ACT adapter
  - [ ] SmolVLA adapter
  - [ ] Scripted adapter if available
- [ ] Measure latency.
  - [ ] Action generation time
  - [ ] Environment step time
  - [ ] Subgoal execution time
  - [ ] Episode wall-clock time

## Phase 7: Verifier

- [ ] Implement verifier interface.

```python
verifier.verify(subgoal, before_obs, after_obs, state) -> VerificationResult
```

- [ ] Implement simulation-state verifier first.
  - [ ] Object pose checks
  - [ ] Gripper pose checks
  - [ ] Distance threshold checks
  - [ ] Lifted-object checks
  - [ ] Object-in-target-zone checks
- [ ] Define verification result schema.

```json
{
  "success": false,
  "confidence": 0.72,
  "reason": "object was not lifted",
  "failure_type": "grasp_failed"
}
```

- [ ] Implement vision-based verifier later.
  - [ ] Rendered image input
  - [ ] Local VLM prompt
  - [ ] Structured success/failed/uncertain output
  - [ ] Confidence score
- [ ] Handle uncertainty.
  - [ ] Low confidence triggers retry
  - [ ] Repeated uncertainty triggers abort
  - [ ] Compare VLM verifier to simulation-state verifier
- [ ] Measure verifier quality.
  - [ ] False positive rate
  - [ ] False negative rate
  - [ ] Uncertain rate
  - [ ] Latency

## Phase 8: Retry, Replan, and Recovery

- [ ] Implement retry policies.
  - [ ] Same subgoal retry
  - [ ] Modified instruction retry
  - [ ] Viewpoint reset then retry
  - [ ] Roll back to previous subgoal
  - [ ] Abort
- [ ] Map failure types to recovery strategies.
  - [ ] Object not found -> inspect scene
  - [ ] Wrong object selected -> re-detect target
  - [ ] Grasp failed -> regrasp
  - [ ] Object dropped -> locate object again
  - [ ] Timeout -> use shorter subgoal
  - [ ] Verifier uncertain -> collect another view or use state verifier
  - [ ] Invalid policy output -> abort or switch policy
- [ ] Enforce retry limits.
  - [ ] Per-subgoal max attempts
  - [ ] Per-episode max retries
  - [ ] Total step limit
  - [ ] Total wall-clock limit
- [ ] Add agentic evaluation variants.
  - [ ] No retry
  - [ ] Simple retry
  - [ ] Typed recovery
  - [ ] Full replan

## Phase 9: Evaluation Harness

- [ ] Create common evaluation command.

```bash
python -m physical_ai_agent.evaluation.run \
  --suite libero_long \
  --policy smolvla \
  --agent full \
  --episodes 50
```

- [ ] Support evaluation modes.
  - [ ] `policy_only`
  - [ ] `policy_retry`
  - [ ] `agentic_rule_planner`
  - [ ] `agentic_llm_planner`
  - [ ] `agentic_vlm_verifier`
- [ ] Save outputs.
  - [ ] `metrics.json`
  - [ ] `episodes.jsonl`
  - [ ] `traces.jsonl`
  - [ ] `summary.md`
  - [ ] videos
  - [ ] logs
- [ ] Build metric aggregation.
  - [ ] Overall success rate
  - [ ] Per-task success rate
  - [ ] Per-subgoal success rate
  - [ ] Retry histogram
  - [ ] Recovery success rate
  - [ ] Latency percentiles
  - [ ] Failure breakdown
  - [ ] Verification error rates
- [ ] Generate comparison report.
  - [ ] Policy-only vs retry vs agentic
  - [ ] ACT vs SmolVLA
  - [ ] Short task suites vs long-horizon suite
  - [ ] Cost of agentic wrapper in latency and retries

## Phase 9A: Research-Relevant Mac-Local Benchmarks

- [ ] Add ManiSkill / ManiSkill-HAB as the first research-relevant Mac-local benchmark.
  - [x] Bootstrap `mani_skill` and `gymnasium` dependencies.
  - [x] Run a ManiSkill reset/step rollout on `PickCube-v1`.
  - [x] Save `episodes.jsonl`, `metrics.json`, and `summary.md`.
  - [x] Add `random` and `zero` baseline policy metrics for `PickCube-v1`.
  - [x] Run small Mac-local HAB partial probes on `ReplicaCADSetTableVal` and `ReplicaCADPrepareGroceriesVal`.
  - [x] Add a dry ManiSkill observation-to-LeRobot feature bridge for SmolVLA.
  - [x] Add a dry SmolVLA action-chunk-to-ManiSkill action bridge.
  - [x] Add a minimal real pretrained SmolVLA inference probe on ManiSkill state observations and action-space clipping.
  - [x] Add a minimal real ManiSkill RGB observation bridge for SmolVLA using `sensor_data.base_camera.rgb`.
  - [ ] Scale CP24 real-image SmolVLA evaluation beyond the one-camera local probe before treating it as task-quality VLA performance.
  - [ ] Reuse the `policy_only` vs `agentic_retry` comparison contract.
- [ ] Add RoboCasa / RoboCasa365 as the long-horizon household manipulation benchmark.
  - [ ] Keep asset download and install separate from the lightweight ManiSkill gate.
  - [ ] Run one dependency probe and reset/step rollout first.
  - [ ] Add task success metrics, traces, videos, and comparison reports.
  - [ ] Use RoboCasa for planner, verifier, retry, and replan stress tests after CP24 passes.

## Phase 10: Web UI and Debugging

- [ ] Choose backend.
  - [ ] Recommended: FastAPI + WebSocket
- [ ] Choose frontend.
  - [ ] Recommended: React/Vite or Next.js
- [ ] Implement live run view.
  - [ ] Live simulation frame
  - [ ] Current task
  - [ ] Current subgoal
  - [ ] Agent state
  - [ ] Latest verifier result
  - [ ] Retry count
  - [ ] Episode status
- [ ] Implement episode replay view.
  - [ ] Timeline
  - [ ] Frames
  - [ ] Actions
  - [ ] Subgoals
  - [ ] Verifier decisions
  - [ ] Failure reason
- [ ] Implement metrics dashboard.
  - [ ] Success rate
  - [ ] Failure types
  - [ ] Latency
  - [ ] Policy comparison
  - [ ] Per-task table
- [ ] Implement trace viewer.
  - [ ] Planner prompt/output
  - [ ] Verifier prompt/output
  - [ ] Action chunk
  - [ ] State diff
  - [ ] Retry/replan decision

## Phase 11: Local VLM and LLM Integration

- [ ] Define local inference provider interface.

```python
vlm.describe(image, prompt) -> str
vlm.classify_success(image, question) -> VerificationResult
llm.plan(task, context) -> list[Subgoal]
```

- [ ] Implement Ollama provider first.
  - [ ] Text LLM call
  - [ ] Vision model call
  - [ ] Timeout
  - [ ] JSON parsing
  - [ ] Error handling
- [ ] Design llama.cpp provider.
- [ ] Design MLX provider later for speed.
- [ ] Add optional cloud fallback.
  - [ ] OpenAI
  - [ ] Anthropic
  - [ ] Gemini
- [ ] Benchmark providers.
  - [ ] Image QA latency
  - [ ] Planner latency
  - [ ] Verifier accuracy
  - [ ] JSON validity
  - [ ] Memory usage

## Phase 12: Safety Layer

- [ ] Implement simulation action validator.
  - [ ] Action dimension check
  - [ ] NaN and inf check
  - [ ] Action range clamp
  - [ ] Max delta limit
- [ ] Implement policy output guard.
  - [ ] Invalid action abort
  - [ ] Repeated stuck action detection
  - [ ] Oscillation detection
- [ ] Implement agent-level guard.
  - [ ] Max retries
  - [ ] Max episode duration
  - [ ] Forbidden skill block
  - [ ] Fail-closed behavior
- [ ] Prepare real-robot safety checklist for later.
  - [ ] Workspace bounds
  - [ ] Emergency stop
  - [ ] Speed limit
  - [ ] Dry-run mode
  - [ ] Human confirmation mode

## Phase 13: First Experiment

- [ ] Run random policy on LIBERO-Spatial.
- [ ] Run ACT baseline on LIBERO-Spatial.
- [ ] Confirm SmolVLA inference or document blocker.
- [ ] Run policy-only evaluation on LIBERO-Long.
- [ ] Add rule-based planner.
- [ ] Add simulation-state verifier.
- [ ] Add one retry per failed subgoal.
- [ ] Compare policy-only against retry.
- [ ] Generate first comparison report.
- [ ] Manually inspect 10 failed episodes.
- [ ] Record what failures agentic retry fixes.
- [ ] Record what failures require better policy training.

## Phase 14: Second Experiment

- [ ] Add VLM verifier.
- [ ] Compare VLM verifier against simulation-state verifier.
- [ ] Measure local VLM latency.
- [ ] Reduce verifier frequency if latency is high.
- [ ] Run verifier only at subgoal boundaries.
- [ ] Add every-N-step visual check option.
- [ ] Compare simple retry against VLM-verifier retry.

## Phase 15: Third Experiment

- [ ] Replace rule-based planner with LLM planner.
- [ ] Validate LLM planner output.
- [ ] Add fallback to rule-based planner.
- [ ] Measure LLM planner latency.
- [ ] Measure long-horizon success improvement.
- [ ] Measure bad-plan rate.
- [ ] Compare LLM planner to rule-based planner.

## Phase 16: Real Robot Preparation

- [ ] Define common robot interface.

```python
robot.get_observation()
robot.step(action)
robot.reset()
robot.stop()
```

- [ ] Implement `SimRobotInterface`.
- [ ] Add `RealRobotInterface` stub.
- [ ] Keep safety layer shared between sim and real.
- [ ] Add dry-run logger for real robot mode.
- [ ] Test planner and verifier on real camera frames without executing actions.
- [ ] Test policy output without sending commands to robot.
- [ ] Add human confirmation mode before real actuation.

## Immediate Task Order

- [ ] 1. Create repository skeleton.
- [ ] 2. Add config system.
- [x] 3. Add Mac-local MuJoCo environment smoke test.
- [x] 4. Add random policy baseline.
- [x] 5. Add baseline evaluator.
- [x] 6. Save metrics, traces, and frames.
- [x] 7. Add policy adapter/action-chunk interface.
- [x] 8. Add SmolVLA adapter readiness probe or document local blocker.
- [x] 9. Add SO101-Nexus Mac-local robot-arm simulation gate.
- [x] 10. Add SO101 rollout trace and visualization artifacts.
- [x] 11. Add LeRobot-compatible SO101 environment factory.
- [x] 12. Add SmolVLA dry input mapping and dry rollout visualization.
- [x] 13. Add SO101 demo dataset generation.
- [x] 14. Add SO101-Nexus 3D MuJoCo render output.
- [x] 15. Add pretrained SmolVLA inference rollout through SO101-Nexus with 3D output.
- [x] 16. Add SO101 camera input capture and preview.
- [x] 17. Add SO101 wrist plus top-down multi-camera input capture and preview.
- [x] 18. Add SO101 wrist plus egocentric policy inputs and top-down debug input.
- [x] 19. Feed real SO101 camera inputs into SmolVLA rollout.
- [x] 20. Add rule-based planner.
- [x] 21. Add simulation-state verifier.
- [x] 22. Add retry loop.
- [x] 23. Produce first comparison report.
- [ ] 24. Add Web UI.
- [ ] 25. Add local VLM verifier.

## Minimal MVP Checklist

- [x] Mac-local MuJoCo environment runs on the Mac.
- [x] Random policy can run one full episode.
- [x] Episode frame is saved.
- [x] Episode trace is saved.
- [x] Baseline success rate is computed.
- [x] Policy adapter/action-chunk contract can be evaluated.
- [x] SmolVLA readiness probe runs and documents blockers.
- [x] SO101-Nexus robot-arm simulation runs on the Mac.
- [x] SO101 simulation rollout visualization is saved.
- [x] SmolVLA dry action chunk can be stepped through SO101-Nexus and visualized.
- [x] SO101 demo data is generated in a LeRobot-like intermediate format.
- [x] SO101-Nexus real 3D MuJoCo render PNG/GIF is saved.
- [x] Pretrained SmolVLA can produce an action that steps SO101-Nexus and saves a 3D rollout.
- [x] SO101 camera input frames and state/action previews are saved.
- [x] SO101 `wrist_cam` and virtual `top_down` RGB inputs are captured together.
- [x] SO101 `wrist_cam` and virtual `egocentric_cam` are recorded as policy inputs.
- [x] SmolVLA receives real SO101 camera frames instead of zero image tensors.
- [x] Rule-based planner creates subgoals.
- [x] Simulation-state verifier judges success or failure.
- [x] Retry once after failed subgoal works.
- [x] `policy_only` and `agentic_retry` results are compared.
- [x] A Markdown summary report is generated.

## Notes

- Keep the first milestone CUDA-free.
- Prefer MuJoCo over Isaac Lab for local Mac development. Keep LIBERO as a Linux/cloud strict gate.
- Use simulation-state verification first because it gives reliable ground truth.
- Add VLM verification only after the basic agentic loop is measurable.
- Treat the VLA/policy as an executor, not as the whole agent.
- Keep planner decisions, verifier decisions, policy actions, and simulation state in one trace format.
- Use failed episodes as future training data.
