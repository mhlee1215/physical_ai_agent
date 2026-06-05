# physical_ai_agent

Agentic physical AI evaluation stack for Mac-local simulation.

The first milestone is to evaluate whether an agentic wrapper can improve task success for a weak or medium robot policy in LIBERO simulation.

## Initial Direction

- Simulation: MuJoCo + LIBERO
- Robot learning: LeRobot
- Baselines: random policy, ACT, SmolVLA
- Agent loop: planner -> policy executor -> verifier -> retry/replan
- Local inference: Ollama or llama.cpp first, MLX later
- Target machine: MacBook Pro M5 Pro, 64 GB unified memory

## Repository Layout

```text
apps/
  agent/                 # future agent service entrypoint
  web/                   # future debug UI
configs/
  agent/                 # planner/verifier configs
  eval/                  # evaluation configs
  policy/                # policy configs
  sim/                   # simulation configs
docs/
  agentic_physical_ai_plan.md
src/
  physical_ai_agent/
    agent_core/
    data/
    evaluation/
    inference/
    observability/
    policies/
    safety/
    sim/
    skills/
tests/
```

## Planned MVP

- [ ] Run a LIBERO environment locally.
- [ ] Execute a random policy and save episode traces.
- [ ] Add ACT or SmolVLA baseline evaluation.
- [ ] Add a rule-based planner.
- [ ] Add a simulation-state verifier.
- [ ] Add retry after failed subgoals.
- [ ] Compare `policy_only` against `agentic_retry`.

See [docs/agentic_physical_ai_plan.md](docs/agentic_physical_ai_plan.md) for the full checklist.

## Development

This repo is scaffolded but dependencies are intentionally light for now.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m physical_ai_agent
pytest
```

## Checkpoint 01

Use the repo-local script so the checkpoint runs with a Python version compatible with `pyproject.toml`.

```bash
sh scripts/bootstrap_checkpoint_01.sh
sh scripts/checkpoint_01.sh
sh scripts/checkpoint_01.sh --strict-local-sim --probe-mujoco
sh scripts/checkpoint_01.sh --strict-sim-deps --probe-libero-env
```

The bootstrap command creates `.venv` and installs MuJoCo for the first Mac-local simulation gate. The first checkpoint command verifies the lightweight scaffold and writes evidence to `_workspace/checkpoints/checkpoint_01_smoke.json`. The second checkpoint command verifies the Mac-local MuJoCo simulation path and writes evidence to `_workspace/checkpoints/checkpoint_01_local_sim.json`. The third checkpoint command writes evidence to `_workspace/checkpoints/checkpoint_01_libero_strict.json`; it is expected to fail on macOS or until the full LIBERO/LeRobot dependency path is available on Linux/cloud.

## Checkpoints 02-04

Run a random policy episode, save artifacts, and compute baseline evaluator metrics:

```bash
sh scripts/checkpoint_02_04.sh
```

Artifacts are written to `_workspace/checkpoints/checkpoint_02_04/`, including `metrics.json`, `summary.md`, per-episode JSONL traces, final PPM frames, and `checkpoint_report.json`.

Current CP02-04 status:

- CP02 random policy episode: implemented and verified.
- CP03 episode trace/frame/metrics artifacts: implemented and verified.
- CP04 baseline evaluator metrics and summary: implemented and verified.

## Checkpoints 05-06

Create the policy adapter/action-chunk contract and probe SmolVLA readiness:

```bash
sh scripts/bootstrap_checkpoint_05_06.sh
sh scripts/checkpoint_05_06.sh
sh scripts/checkpoint_05_06.sh --require-real-smolvla --output-dir _workspace/checkpoints/checkpoint_05_06_require_real
```

The bootstrap command installs `lerobot[smolvla]` into `.venv`. Artifacts are written to `_workspace/checkpoints/checkpoint_05_06/`, including `checkpoint_report.json` and `smolvla_blocker.md`. The probe passes when the adapter contract works and either the LeRobot SmolVLA import path is ready or the missing dependency/model blocker is explicitly documented. Use `--require-real-smolvla` when the LeRobot SmolVLA import path must be available. This does not download model weights or prove task-quality inference yet.

## Checkpoints 07-13

Run SO101-Nexus MuJoCo locally, visualize the rollout, validate a LeRobot-compatible environment surface, and produce a SmolVLA dry-run rollout plus demo dataset:

```bash
sh scripts/bootstrap_checkpoint_07_13.sh
sh scripts/checkpoint_07_13.sh
```

Artifacts are written to `_workspace/checkpoints/checkpoint_07_13/`, including `rollout/so101_rollout.jsonl`, `rollout/so101_rollout.png`, `rollout/so101_rollout.gif`, `smolvla_dry_rollout/smolvla_dry_rollout.jsonl`, `smolvla_dry_rollout/smolvla_dry_rollout.png`, `smolvla_dry_rollout/smolvla_dry_rollout.gif`, `demo_dataset/episodes.jsonl`, `demo_dataset/metadata.json`, and `checkpoint_report.json`.

Current CP07-13 status:

- CP07 SO101-Nexus reset/step: implemented and verified by the checkpoint command.
- CP08 SO101 rollout trace and visualization: implemented with trace-derived PNG/GIF output.
- CP09 LeRobot EnvHub-compatible `make_env`: implemented as `physical_ai_agent.envhub.so101_env.make_env`.
- CP10 SO101 action-chunk policy: implemented as a center-action chunk policy.
- CP11 SmolVLA dry input mapping: implemented without downloading model weights.
- CP12 SmolVLA dry rollout visualization: implemented by feeding the dry chunk through SO101-Nexus.
- CP13 demo dataset generation: implemented as a LeRobot-like JSONL intermediate artifact.

## Checkpoints 14-15

Render the SO101-Nexus MuJoCo scene as real 3D RGB frames, then load LeRobot's pretrained SmolVLA and use its action output to step SO101-Nexus:

```bash
sh scripts/bootstrap_checkpoint_14_15.sh
sh scripts/checkpoint_14_15.sh --allow-download --require-3d-render --require-real-smolvla
```

The strict command may download `lerobot/smolvla_base` from Hugging Face on first run. Artifacts are written to `_workspace/checkpoints/checkpoint_14_15/`, including `render_3d/so101_3d_render.png`, `render_3d/so101_3d_render.gif`, `smolvla_real/smolvla_real_rollout.jsonl`, `smolvla_real/smolvla_real_rollout_3d.png`, `smolvla_real/smolvla_real_rollout_3d.gif`, and `checkpoint_report.json`.

Current CP14-15 status:

- CP14 SO101 3D MuJoCo render: implemented and verified with 640x480 PNG/GIF output.
- CP15 pretrained SmolVLA inference rollout: implemented and verified by loading `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")`, executing `select_action()`, stepping SO101-Nexus for six steps, and saving a 3D rollout PNG/GIF.
- The current SmolVLA observation shim uses SO101 state plus zero image tensors and synthetic language tokens, so it proves execution wiring rather than task-quality policy behavior.

## Live SO101 Viewer

Open a real-time MuJoCo viewer from a normal macOS Terminal session:

```bash
sh scripts/view_so101_live.sh
```

On macOS, MuJoCo's live viewer requires `mjpython`. If the repo `.venv` was created from a bundled/symlinked Python and `mjpython` cannot start, create a separate viewer venv from Homebrew or python.org Python:

```bash
/opt/homebrew/bin/python3 -m venv .venv-viewer
.venv-viewer/bin/python -m pip install -e ".[so101]"
PYTHONPATH=src .venv-viewer/bin/mjpython -B -m physical_ai_agent.sim.so101_live_viewer
```

Optional finite demo:

```bash
sh scripts/view_so101_live.sh --max-steps 300 --fps 30
```

Open the native MuJoCo viewer with a browser-based real-time camera-input stream:

```bash
sh scripts/view_so101_live.sh --show-inputs --fps 15
```

When `--show-inputs` is enabled, the script prints a local URL such as `http://127.0.0.1:8765`. Open that URL in a browser to watch `wrist_cam` and `egocentric_cam` as policy inputs plus `top_down` as a debug view. The live viewer uses a deterministic smooth action policy and is meant for visual inspection. GUI windows may not open from headless agent sessions; use the CP14/15 GIF artifacts or CP18/19 input previews when running in a headless context.

Run the browser-based live simulator with pretrained SmolVLA action chunks:

```bash
sh scripts/view_so101_live.sh --browser-only --policy smolvla --allow-download --smolvla-action-steps 15 --show-inputs --fps 2
```

This path avoids the macOS `mjpython` native-window trampoline by streaming the MuJoCo 3D scene, `wrist_cam`, `egocentric_cam`, `top_down`, action bars, image-feature mapping, chunk status, and inference latency to `http://127.0.0.1:8765`. It loads `lerobot/smolvla_base` in an isolated worker process, predicts an action chunk, executes 15 actions from that chunk, steps SO101-Nexus with each selected action, and refreshes the chunk after 15 executed actions so the sim does not blindly consume all 50 predicted actions from a stale observation.

## Checkpoint 16

Capture and preview the actual inputs available to policies before adding planner or verifier logic:

```bash
sh scripts/checkpoint_16.sh
```

Artifacts are written to `_workspace/checkpoints/checkpoint_16/`, including `so101_inputs/input_manifest.json`, `so101_inputs/input_preview.png`, `so101_inputs/input_preview.gif`, and per-step camera frames under `so101_inputs/frames/`. Current SO101-Nexus environments expose one camera input, `wrist_cam`; observation vectors are shape 6 for reach/move, 18 for pick-lift, and 24 for pick-and-place.

## Checkpoint 17

Capture a two-view visual input bundle for future LeRobot/SmolVLA multi-image batches:

```bash
sh scripts/checkpoint_17.sh
```

Artifacts are written to `_workspace/checkpoints/checkpoint_17/`, including `so101_multi_inputs/input_manifest.json`, `so101_multi_inputs/input_preview.png`, `so101_multi_inputs/input_preview.gif`, and per-step `wrist_cam` plus `top_down` frames. `wrist_cam` is the SO101-Nexus named camera; `top_down` is a MuJoCo virtual camera rendered without editing the SO101 XML. The planned LeRobot feature keys are `observation.images.wrist_cam` and `observation.images.top_down`.

## Checkpoints 18-19

Capture the policy/debug visual-input split, then run pretrained SmolVLA with real SO101 camera frames instead of zero image tensors:

```bash
sh scripts/checkpoint_18.sh
sh scripts/checkpoint_19.sh --allow-download --require-real-smolvla
```

CP18 writes `wrist_cam` and `egocentric_cam` as policy inputs plus `top_down` as a debug input. CP19 maps those real frames into LeRobot's pretrained SmolVLA image feature keys: `observation.images.camera1 <- wrist_cam`, `observation.images.camera2 <- egocentric_cam`, and `observation.images.camera3 <- egocentric_cam` as a non-zero duplicate fallback. Artifacts are written under `_workspace/checkpoints/checkpoint_18/` and `_workspace/checkpoints/checkpoint_19/`.

## Checkpoints 20-23

Add the first agentic wrapper around SO101 evaluation: rule-based planning, simulation-state verification, retry, and a policy-only vs agentic-retry comparison report.

```bash
sh scripts/checkpoint_20.sh
sh scripts/checkpoint_21.sh
sh scripts/checkpoint_22.sh
sh scripts/checkpoint_23.sh
```

CP20 writes a deterministic subgoal plan for `reach_target`. CP21 verifies a subgoal from SO101 simulator state using `tcp_to_target_dist` and `success`. CP22 executes the planned subgoals with one retry budget per failed subgoal and records verifier decisions in the trace. CP23 writes `_workspace/checkpoints/checkpoint_23/comparison/comparison_report.md` comparing `policy_only` and `agentic_retry` runs.

## Checkpoint 24: ManiSkill / ManiSkill-HAB

Add the first research-relevant Mac-local benchmark gate beyond the SO101 smoke path:

```bash
sh scripts/bootstrap_checkpoint_24.sh
sh scripts/checkpoint_24.sh
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

CP24 targets ManiSkill first and records the ManiSkill-HAB expansion path as the next suite target. The non-strict command writes `_workspace/checkpoints/checkpoint_24/checkpoint_report.json`, documents a dependency blocker when `mani_skill` is not installed, and always writes `smolvla_maniskill_eval_plan.md` plus `checkpoint_25_robocasa_plan.md`. The strict command requires a real ManiSkill reset/step rollout and writes `maniskill_rollout/episodes.jsonl`, `metrics.json`, and `summary.md`.

For a first Mac-local baseline number, run the longer command above. It evaluates `random` and `zero` policies on the same `PickCube-v1` seeds and writes per-policy success rate, mean reward sum, and mean episode steps under `maniskill_rollout/metrics.json`.

For a small Mac-local HAB probe, install the extra scene/object assets once:

```bash
PYTHONPATH=src .venv/bin/python -B -m mani_skill.utils.download_asset ReplicaCAD -y
PYTHONPATH=src .venv/bin/python -B -m mani_skill.utils.download_asset ReplicaCADRearrange -y
PYTHONPATH=src .venv/bin/python -B -m mani_skill.utils.download_asset ycb -y
```

Then run the two `ReplicaCAD...SceneManipulation` commands above. The `--no-fallback-env` flag keeps these probes honest: if the HAB task cannot load, the checkpoint fails instead of silently falling back to `Empty-v1`.

The default research target is `PickCube-v1`. On macOS, SAPIEN needs a working Vulkan loader plus MoltenVK ICD to create the render device used by ManiSkill manipulation assets. Install them once with:

```bash
/opt/homebrew/bin/brew install vulkan-loader vulkan-tools molten-vk
```

`scripts/checkpoint_24.sh` automatically exports the Homebrew Vulkan loader and MoltenVK ICD paths when they exist. In Codex sandboxed sessions, Metal may still be hidden from MoltenVK; run the strict command from a normal macOS Terminal or with an unsandboxed command runner. If `PickCube-v1` is still blocked, CP24 falls back to `Empty-v1` to prove the ManiSkill evaluation pipeline is executable while preserving the `PickCube-v1` blocker in `maniskill_blocker.md`.

The SmolVLA layer is not treated as task-quality ManiSkill evaluation until two bridges exist: a ManiSkill observation-to-LeRobot feature bridge and a SmolVLA action-chunk-to-ManiSkill action bridge. CP24 records that installation and adapter plan explicitly.

`smolvla_dry` validates that bridge shape without loading model weights. It writes `maniskill_rollout/smolvla_dry_bridge_manifest.json` with the mapped feature keys, state dimension, synthetic image feature shape, instruction, and ManiSkill action-space shape. Treat it as a wiring baseline, not as pretrained SmolVLA task performance.

`smolvla_real` loads LeRobot's pretrained `lerobot/smolvla_base`, builds a ManiSkill-shaped batch, calls `select_action()`, clips the resulting action into the ManiSkill action space, and steps the environment. By default it uses state plus zero image tensors for the smallest model-call probe. Add `--real-images` to create the ManiSkill env with `obs_mode=rgb`, map `sensor_data.base_camera.rgb` into SmolVLA image features, and save `maniskill_rollout/smolvla_real_input.png`, per-step frames under `maniskill_rollout/smolvla_real_frames/`, and `maniskill_rollout/smolvla_real_rollout.gif`. This proves real model inference with real ManiSkill camera frames, but it is still a one-camera Mac-local probe rather than a full paper-scale visual policy evaluation.

## Checkpoint 25: RoboCasa

RoboCasa is registered as the heavier long-horizon household manipulation checkpoint. CP25 should remain separate from CP24 because RoboCasa assets are large and the benchmark is better used after the lighter ManiSkill local gate passes. The planned strict gate is a RoboCasa reset/step rollout, task success metrics, trace/video artifacts, and the same `policy_only` vs `agentic_retry` comparison contract used by CP23/CP24.
