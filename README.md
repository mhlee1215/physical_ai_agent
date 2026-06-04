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
