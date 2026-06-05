# Physical AI Harness Team Spec

## Goal

Make the agentic physical AI workflow repeatable. Every implementation checkpoint must include an executable verification step before the work is considered complete.

## Inputs

- Project plan: `docs/agentic_physical_ai_plan.md`
- Simulation config: `configs/sim/libero.yaml`
- Checkpoint smoke command: `sh scripts/checkpoint_01.sh`

## Outputs

- Source changes under `src/physical_ai_agent/`
- Tests under `tests/`
- Checkpoint evidence in the final response or `_workspace/` when a long run needs preserved artifacts

## Roles

| Role | Responsibility | Reusable skill | Writes |
| --- | --- | --- | --- |
| Orchestrator | Select the next checkpoint and enforce validation | `.agents/skills/physical-ai-orchestrator/SKILL.md` | final response or `_workspace/*` |
| Implementer | Add code, config, and tests for the checkpoint | n/a | repo files |
| Verifier | Run required commands and report pass/fail/blockers | n/a | command evidence |

## Phase Order

### Phase 1: Select Checkpoint

- input sources: `docs/agentic_physical_ai_plan.md`
- actions: choose the smallest unchecked checkpoint that advances the MVP
- output files: none required
- completion criteria: target checkpoint and verification command are known

### Phase 2: Implement

- input sources: selected checkpoint, existing package scaffold
- actions: add focused code, config, docs, and tests
- output files: repo-local source and test files
- completion criteria: code path is runnable without hidden setup when possible

### Phase 3: Verify

- input sources: implementation output
- actions: run the required checkpoint command and relevant tests
- output files: command output summarized in final response
- completion criteria: verification passes, or blocker is explicit and reproducible

## Required Checkpoint 01 Verification

Run these commands before completing checkpoint 01:

```bash
sh scripts/bootstrap_checkpoint_01.sh
sh scripts/checkpoint_01.sh
sh scripts/checkpoint_01.sh --strict-local-sim --probe-mujoco
sh scripts/checkpoint_01.sh --strict-sim-deps --probe-libero-env
```

The bootstrap command creates `.venv` and installs MuJoCo if needed. The lightweight command must pass in the scaffold environment and writes evidence to `_workspace/checkpoints/checkpoint_01_smoke.json`. The Mac-local simulation command writes evidence to `_workspace/checkpoints/checkpoint_01_local_sim.json`; it must pass before claiming checkpoint 01 works on the target Mac. The full LIBERO/LeRobot command writes evidence to `_workspace/checkpoints/checkpoint_01_libero_strict.json`; current LeRobot documentation says LIBERO requires Linux, so on macOS this gate should be reported as a future Linux/cloud blocker rather than the Mac-local checkpoint.

## Failure Policy

- retry policy: fix local code/config failures and rerun verification once
- partial completion policy: scaffold smoke can pass while Mac-local MuJoCo smoke is blocked by missing external dependencies
- conflict resolution policy: the Mac-local MuJoCo smoke is authoritative for claiming checkpoint 01 works on the target Mac; the LIBERO strict gate is authoritative only for claiming full LIBERO execution
- escalation trigger: request permission before installing or downloading simulation dependencies

## RunPod Orchestration

Use RunPod as the Linux/NVIDIA execution lane for LIBERO, LeRobot/SmolVLA,
ManiSkill GPU rendering, Isaac, and paper-comparable evaluation runs.

Maintain `docs/runpod_worklog.md` as the durable handoff journal for cloud work.
Update it whenever a RunPod evaluation, setup fix, result fetch, or lifecycle
decision changes the state a future conversation needs to recover. Do not
include API keys, private SSH keys, full `.env` contents, or raw secret-bearing
API responses.

Default workflow:

```bash
set -a
. ./.env
set +a
sh scripts/runpod_pod.sh start
sh scripts/runpod_pod.sh status
RUNPOD_SSH="$RUNPOD_SSH" sh scripts/runpod_check.sh
```

Then, on the Pod:

```bash
cd /workspace/physical-ai/physical_ai_agent
git fetch origin
git checkout main
git pull --ff-only origin main
```

Run heavy evaluation under `/workspace` only. Keep environments, model caches,
datasets, logs, videos, and benchmark outputs under `/workspace`, not `/`.

Before stopping the Pod, consolidate outputs under a network-volume result
directory and fetch them back to the local repo:

```bash
cd /workspace/physical-ai/physical_ai_agent
mkdir -p _workspace/runpod_results
cp -R _workspace/checkpoints _workspace/runpod_results/checkpoints
cp -R outputs _workspace/runpod_results/outputs 2>/dev/null || true
```

Write a concise Markdown report under `_workspace/runpod_results/` with:

- git commit evaluated on RunPod
- exact command line
- policy checkpoint/model id
- benchmark suite and episode count
- success-rate table
- blocker list, if any
- artifact paths for logs, videos, and JSON metrics

Fetch results locally before stopping:

```bash
set -a
. ./.env
set +a
sh scripts/runpod_fetch_results.sh
```

Then stop the Pod:

```bash
set -a
. ./.env
set +a
sh scripts/runpod_pod.sh stop
sh scripts/runpod_pod.sh status
```

RunPod API responses can include sensitive env values. `scripts/runpod_pod.sh`
redacts API response fields by default; do not set `RUNPOD_RAW_RESPONSE=1`
unless raw JSON is explicitly needed for debugging.

For paper-comparable numbers, only count runs from a committed Git revision
that has been pulled on RunPod. Ad hoc `rsync` or uncommitted remote edits may
be used for quick debugging, but they are not acceptable evidence for a
reported benchmark table.

## Validation Checks

- `sh scripts/bootstrap_checkpoint_01.sh`
- `sh scripts/checkpoint_01.sh`
- `sh scripts/checkpoint_01.sh --strict-local-sim --probe-mujoco`
- `sh scripts/checkpoint_01.sh --strict-sim-deps --probe-libero-env`
- `sh scripts/checkpoint_02_04.sh`
- `sh scripts/bootstrap_checkpoint_05_06.sh`
- `sh scripts/checkpoint_05_06.sh`
- `sh scripts/checkpoint_05_06.sh --require-real-smolvla --output-dir _workspace/checkpoints/checkpoint_05_06_require_real`
- `sh scripts/bootstrap_checkpoint_07_13.sh`
- `sh scripts/checkpoint_07_13.sh`
- `sh scripts/bootstrap_checkpoint_14_15.sh`
- `sh scripts/checkpoint_14_15.sh --allow-download --require-3d-render --require-real-smolvla`
- `sh scripts/checkpoint_16.sh`
- `sh scripts/checkpoint_17.sh`
- `sh scripts/checkpoint_18.sh`
- `sh scripts/checkpoint_19.sh --allow-download --require-real-smolvla`
- `sh scripts/view_so101_live.sh --browser-only --policy smolvla --allow-download --smolvla-action-steps 15 --show-inputs --fps 2 --max-steps 1`
- `sh scripts/checkpoint_20.sh`
- `sh scripts/checkpoint_21.sh`
- `sh scripts/checkpoint_22.sh`
- `sh scripts/checkpoint_23.sh`
- `sh scripts/bootstrap_checkpoint_24.sh`
- `sh scripts/checkpoint_24.sh`
- `sh scripts/checkpoint_24.sh --require-maniskill`
- `sh scripts/checkpoint_24.sh --require-maniskill --episodes 1 --steps 1 --policy smolvla_real --allow-download --output-dir _workspace/checkpoints/checkpoint_24_pickcube_smolvla_real_1ep_1step`
- `sh scripts/checkpoint_24.sh --require-maniskill --episodes 1 --steps 1 --policy smolvla_real --allow-download --real-images --output-dir _workspace/checkpoints/checkpoint_24_pickcube_smolvla_real_images_1ep_1step`
- `sh scripts/eval_smolvla_libero_mac.sh`
- `LIBERO_TASKS=libero_spatial LIBERO_N_EPISODES=1 sh scripts/eval_smolvla_libero_linux.sh`
- `LIBERO_TASKS=libero_spatial LIBERO_TASK_IDS='[0,1,2]' LIBERO_N_EPISODES=5 sh scripts/eval_smolvla_libero_linux.sh`
- `sh scripts/runpod_smolvla_libero_eval.sh`
- `PYTHONPATH=src /Users/minhaeng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -m unittest discover -s tests`
- `PYTHONPATH=src python3 -B -m pytest` when pytest is available
- `python3 -B -c "import ast, pathlib; ..."` as a no-dependency fallback syntax check

## Required Checkpoint 07-13 Verification

Run these commands before completing checkpoints 07-13:

```bash
sh scripts/bootstrap_checkpoint_07_13.sh
sh scripts/checkpoint_07_13.sh
```

The bootstrap command installs SO101-Nexus MuJoCo into `.venv`. The checkpoint command runs a real SO101-Nexus reset/step, saves a deterministic rollout trace, writes schematic PNG/GIF visualizations, verifies a LeRobot EnvHub-compatible `make_env` surface, executes an SO101 action-chunk policy, validates the SO101-to-SmolVLA dry input mapping, runs a dry SmolVLA action chunk through the simulator, and writes a LeRobot-like JSONL demo dataset. Native MuJoCo RGB rendering can fail on headless macOS, so CP08/CP12 visualization is intentionally trace-derived while the physics steps still execute in SO101-Nexus.

## Required Checkpoint 14-15 Verification

Run these commands before completing checkpoints 14-15:

```bash
sh scripts/bootstrap_checkpoint_14_15.sh
sh scripts/checkpoint_14_15.sh --allow-download --require-3d-render --require-real-smolvla
```

The strict checkpoint command must save a real SO101-Nexus 3D render PNG/GIF, load LeRobot's pretrained `lerobot/smolvla_base` through `SmolVLAPolicy.from_pretrained()`, execute `select_action()`, step SO101-Nexus with the resulting action for at least one rollout, and save a 3D SmolVLA rollout PNG/GIF plus JSONL trace. A non-strict run may pass with documented blockers, but it is not sufficient to claim CP14 or CP15 are complete.

## Required Checkpoint 16 Verification

Run this command before completing checkpoint 16:

```bash
sh scripts/checkpoint_16.sh
```

The checkpoint command must list available SO101 camera inputs, capture real MuJoCo camera RGB frames, save the state/action/camera input manifest, and write a preview PNG/GIF showing what the policy input would contain. This checkpoint must pass before changing SmolVLA rollout code to depend on visual input.

## Required Checkpoint 17 Verification

Run this command before completing checkpoint 17:

```bash
sh scripts/checkpoint_17.sh
```

The checkpoint command must save both `wrist_cam` and `top_down` RGB frames for each captured step, write a multi-input manifest, and save a preview PNG/GIF showing both visual inputs alongside the state/action summary. The manifest should record planned LeRobot image feature keys for the later real SmolVLA image-input rollout.

## Required Checkpoint 18-19 Verification

Run these commands before completing checkpoints 18-19:

```bash
sh scripts/checkpoint_18.sh
sh scripts/checkpoint_19.sh --allow-download --require-real-smolvla
```

Checkpoint 18 must record `wrist_cam` and `egocentric_cam` as policy inputs and `top_down` as a debug input. Checkpoint 19 must load pretrained `lerobot/smolvla_base`, feed real SO101 RGB frames into SmolVLA image features without zero image tensors, step SO101-Nexus with the resulting action, and save input preview, rollout trace, and 3D rollout PNG/GIF artifacts.

## Required Checkpoint 20-23 Verification

Run these commands before completing checkpoints 20-23:

```bash
sh scripts/checkpoint_20.sh
sh scripts/checkpoint_21.sh
sh scripts/checkpoint_22.sh
sh scripts/checkpoint_23.sh
```

Checkpoint 20 must write a rule-based SO101 subgoal plan. Checkpoint 21 must execute a real SO101 simulator step and save a simulation-state verifier decision based on `tcp_to_target_dist` and `success`. Checkpoint 22 must execute planned subgoals with a retry event after a verifier failure and save planner/verifier/retry trace records. Checkpoint 23 must compare `policy_only` and `agentic_retry` outputs and save both JSON metrics and a Markdown comparison report.

## Required Checkpoint 24 Verification

Run these commands before completing checkpoint 24:

```bash
sh scripts/bootstrap_checkpoint_24.sh
sh scripts/checkpoint_24.sh --require-maniskill
sh scripts/checkpoint_24.sh --require-maniskill --episodes 20 --steps 100 --policy zero --output-dir _workspace/checkpoints/checkpoint_24_pickcube_baselines_20ep_100step
sh scripts/checkpoint_24.sh --require-maniskill --episodes 10 --steps 50 --policy zero --policy smolvla_dry --output-dir _workspace/checkpoints/checkpoint_24_pickcube_smolvla_dry_10ep_50step
sh scripts/checkpoint_24.sh --require-maniskill --episodes 1 --steps 1 --policy smolvla_real --allow-download --output-dir _workspace/checkpoints/checkpoint_24_pickcube_smolvla_real_1ep_1step
sh scripts/checkpoint_24.sh --require-maniskill --episodes 1 --steps 1 --policy smolvla_real --allow-download --real-images --output-dir _workspace/checkpoints/checkpoint_24_pickcube_smolvla_real_images_1ep_1step
sh scripts/checkpoint_24.sh --require-maniskill --no-fallback-env --env-id ReplicaCADSetTableVal_SceneManipulation-v1 --episodes 2 --steps 50 --policy zero --output-dir _workspace/checkpoints/checkpoint_24_hab_settable_val_2ep_50step
sh scripts/checkpoint_24.sh --require-maniskill --no-fallback-env --env-id ReplicaCADPrepareGroceriesVal_SceneManipulation-v1 --episodes 2 --steps 50 --policy zero --output-dir _workspace/checkpoints/checkpoint_24_hab_preparegroceries_val_2ep_50step
sh scripts/checkpoint_24.sh --require-maniskill --no-fallback-env --env-id ReplicaCADSetTableVal_SceneManipulation-v1 --episodes 2 --steps 50 --policy zero --policy smolvla_dry --output-dir _workspace/checkpoints/checkpoint_24_hab_settable_val_smolvla_dry_2ep_50step
sh scripts/checkpoint_24.sh --require-maniskill --no-fallback-env --env-id ReplicaCADPrepareGroceriesVal_SceneManipulation-v1 --episodes 2 --steps 50 --policy zero --policy smolvla_dry --output-dir _workspace/checkpoints/checkpoint_24_hab_preparegroceries_val_smolvla_dry_2ep_50step
```

Checkpoint 24 must register ManiSkill / ManiSkill-HAB as the first research-relevant Mac-local evaluation gate, execute a real ManiSkill reset/step rollout when dependencies are installed, save rollout JSONL metrics and summary artifacts, and write a SmolVLA-to-ManiSkill evaluation plan. The non-strict command may pass with a documented dependency blocker, but `--require-maniskill` is required before claiming the benchmark pipeline is executable on the target Mac. The longer baseline command must produce per-policy `random` and `zero` metrics on `PickCube-v1`, including success rate, mean reward sum, and mean episode steps. `smolvla_dry` must write `maniskill_rollout/smolvla_dry_bridge_manifest.json` and is only a bridge-shape baseline, not pretrained model performance. `smolvla_real` must load LeRobot's pretrained `lerobot/smolvla_base`, execute `select_action()`, clip the action into the ManiSkill action space, and step the environment; the smallest probe may use zero image tensors, while the `--real-images` probe must map ManiSkill `sensor_data.base_camera.rgb` into SmolVLA image features and save `maniskill_rollout/smolvla_real_input.png`. The HAB partial probes must use `--no-fallback-env` so the requested `ReplicaCAD...SceneManipulation` task is authoritative. They require `ReplicaCAD`, `ReplicaCADRearrange`, and `ycb` assets. On macOS, install `vulkan-loader`, `vulkan-tools`, and `molten-vk` with Homebrew before strict manipulation-task evaluation. If the default `PickCube-v1` manipulation task is blocked by a headless SAPIEN render-device failure, CP24 may execute the fallback `Empty-v1` rollout and must preserve the `PickCube-v1` blocker in `maniskill_blocker.md`.

## Planned Checkpoint 25 Verification

Checkpoint 25 will register RoboCasa / RoboCasa365 as the heavier long-horizon household manipulation benchmark. Its first gate should probe dependencies and execute one reset/step rollout; its strict gate should save success metrics, trace/video artifacts, and a `policy_only` vs `agentic_retry` comparison report. Keep CP25 separate from CP24 because RoboCasa assets are large and should not be part of the lightweight Mac-local smoke.

## Planned Checkpoint 26 Verification

Checkpoint 26 will register the real SO-100 pilot. The first gate should use a calibrated `so100_follower` arm with two arm-mounted OpenCV cameras as policy inputs and the MacBook built-in webcam as a separate Codex/operator observer channel. The smoke gate must capture all camera streams, save a manifest with camera indexes, feature keys, frame sizes, calibration file paths, and observer-frame artifacts, then run planner/verifier logic without sending robot actions. The strict gate may execute physical actions only after emergency stop, joint limits, action rate limits, safety clipping, and human confirmation are wired and evidenced. Any paper claim from CP26 must distinguish quantitative policy results from qualitative Codex observer-assisted demonstrations.

## AI & Coding-Agent Disclosure Policy for Papers

Purpose:

- Keep publication artifacts reproducible and transparent.
- Prevent policy drift between simulation runs, benchmark claims, and manuscript text.
- Ensure top-tier journal expectations are met for LLM/coding-agent usage.

Mandatory rules for any manuscript draft in this repo:

- Never list AI tools as authors.
- Always disclose AI contributions in a stable manuscript section (`Methods`, `Acknowledgements`, `Author contributions`, or a dedicated `AI Disclosure` section).
- Record at least:
  - tool name + model/version,
  - date used,
  - task type (e.g., drafting, proofreading, code assist, experiment interpretation, figure caption drafting),
  - scope/amount (entire draft vs. partial section, prompt count, or file list),
  - whether outputs were independently verified.
- Keep a human-in-the-loop trace:
  - all AI outputs must be reviewed and edited by the authors;
  - generated code must be executed, and reported metrics/visuals linked as evidence.

Top-tier examples to mirror (Nature portfolio examples already observed):

- Nature Communications 2025: `Acknowledgements` disclosure of ChatGPT-based assistance and explicit statement that core scientific conclusions were written by authors.
- Nature Communications 2025 (another article): script assistance with explicit tool/version/task disclosure in `Acknowledgements`.
- Communications Biology 2026: explicit `Acknowledgements` disclosure of ChatGPT/Perplexity/Elicit for idea generation and literature scan.
- Scientific Reports 2024: explicit AI-use section with model/version and non-authorship statement.

Example templates (adapt per journal):

- `We used [Tool] ([model/version], [date]) for [task], and then manually reviewed, edited, and finalized all outputs before submission.`
- `No text or claims in the manuscript were copied directly from model output without human verification; all figures and code were executed/validated, and evidence is reported in _workspace/checkpoints.`
- `No AI system is listed as an author.`

Venue alignment:

- For journal-specific policies (Nature/Elsevier/IEEE/OUP/MDPI/ACM), place disclosure in the required location and mirror wording in final manuscript evidence notes.

Required checkpoint-local record:

- Add a short disclosure note to the checkpoint evidence bundle whenever AI or coding-agent outputs were used during manuscript preparation.
- Add the same note to final response artifacts when publishing claims or benchmark tables are produced.
