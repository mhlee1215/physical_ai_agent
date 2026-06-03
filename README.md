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

