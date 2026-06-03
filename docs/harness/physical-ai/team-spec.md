# Physical AI Harness Team Spec

## Goal

Make the agentic physical AI workflow repeatable. Every implementation checkpoint must include an executable verification step before the work is considered complete.

## Inputs

- Project plan: `docs/agentic_physical_ai_plan.md`
- Simulation config: `configs/sim/libero.yaml`
- Checkpoint smoke command: `python3 -B -m physical_ai_agent.checkpoints.checkpoint_01`

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

Run both commands before completing checkpoint 01:

```bash
PYTHONPATH=src python3 -B -m physical_ai_agent.checkpoints.checkpoint_01
PYTHONPATH=src python3 -B -m physical_ai_agent.checkpoints.checkpoint_01 --strict-sim-deps --probe-libero-env
```

The first command must pass in the lightweight scaffold environment. The second command is the real simulation readiness gate; if MuJoCo, LIBERO, robosuite, or LeRobot are not installed yet, report it as the next environment blocker instead of treating the checkpoint as fully simulation-ready.

## Failure Policy

- retry policy: fix local code/config failures and rerun verification once
- partial completion policy: scaffold smoke can pass while strict simulation smoke is blocked by missing external dependencies
- conflict resolution policy: the strict simulation smoke is authoritative for claiming LIBERO is executable
- escalation trigger: request permission before installing or downloading simulation dependencies

## Validation Checks

- `PYTHONPATH=src python3 -B -m physical_ai_agent.checkpoints.checkpoint_01`
- `PYTHONPATH=src python3 -B -m physical_ai_agent.checkpoints.checkpoint_01 --strict-sim-deps --probe-libero-env`
- `PYTHONPATH=src python3 -B -m unittest discover -s tests`
- `PYTHONPATH=src python3 -B -m pytest` when pytest is available
- `python3 -B -c "import ast, pathlib; ..."` as a no-dependency fallback syntax check
