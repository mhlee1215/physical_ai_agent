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
