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

See [docs/agentic_physical_ai_plan.md](docs/agentic_physical_ai_plan.md) for historical checkpoint context. Completed CP01-CP24 smoke/checkpoint gates have been retired from the active repo surface. Current maintained paths are the unified SmolVLA LIBERO/Meta-World evaluation entrypoint, live SO101 viewer, RoboCasa CP25 probe, real SO-100 CP26 gate, and RunPod-backed benchmark evaluation work.

## Development

This repo is scaffolded but dependencies are intentionally light for now.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m physical_ai_agent
pytest
```

## Retired Checkpoint Gates

CP01-CP24 were completed milestone gates and have been removed from the active command surface to keep the repo focused on maintained evaluation and real-robot paths. Historical plans and paper-facing notes remain under `docs/` where they are still useful as research context, but deleted `scripts/checkpoint_*.sh` commands should not be used as active verification.

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

## Checkpoint 25: RoboCasa

RoboCasa is registered as the heavier long-horizon household manipulation checkpoint. CP25 should remain separate from CP24 because RoboCasa assets are large and the benchmark is better used after the lighter ManiSkill local gate passes. The planned strict gate is a RoboCasa reset/step rollout, task success metrics, trace/video artifacts, and the same `policy_only` vs `agentic_retry` comparison contract used by CP23/CP24.
