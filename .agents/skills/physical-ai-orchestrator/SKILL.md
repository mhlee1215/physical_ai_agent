---
name: physical-ai-orchestrator
description: Orchestrate checkpoint-driven development for the agentic physical AI simulation stack, including mandatory executable verification steps.
---

# Physical AI Orchestrator

## When to Use

Use this skill when implementing or reviewing milestones for the agentic physical AI project, especially LIBERO, LeRobot, policy evaluation, planner/verifier/retry, or local VLM checkpoints.

## Required Inputs

- Current checkpoint from `docs/agentic_physical_ai_plan.md`
- Team spec: `docs/harness/physical-ai/team-spec.md`
- Existing code/config under `src/physical_ai_agent/` and `configs/`

## Workflow

1. Select the smallest unchecked checkpoint that moves the MVP forward.
2. Implement only the code, config, docs, and tests needed for that checkpoint.
3. Register or update the executable verification command in the team spec when the checkpoint adds a new runnable path.
4. Run the required verification command before marking the checkpoint complete.
5. Report whether the checkpoint is passed, failed, or blocked by missing external dependencies.

## Checkpoint 01 Required Verification

Always run:

```bash
PYTHONPATH=src python3 -B -m physical_ai_agent.checkpoints.checkpoint_01
```

When claiming LIBERO itself is executable, also run:

```bash
PYTHONPATH=src python3 -B -m physical_ai_agent.checkpoints.checkpoint_01 --strict-sim-deps --probe-libero-env
```

If the strict command fails because MuJoCo, LIBERO, robosuite, or LeRobot are missing, treat the checkpoint as scaffold-passed but simulation-blocked.

## Expected Outputs

- Updated repo files
- Verification command results
- Clear next blocker or next checkpoint

## Validation Notes

- Do not claim simulation readiness from import-free tests alone.
- Do not install or download simulation dependencies without user approval.
- Keep validation commands deterministic and repo-local.

