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

The live viewer uses a deterministic smooth action policy and is meant for visual inspection. GUI windows may not open from headless agent sessions; use the CP14/15 GIF artifacts when running in a headless context.

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
