# Repository Agents Guide

## What

- This repo builds a Mac-local agentic physical AI evaluation stack.
- The first stack target is MuJoCo + LIBERO + LeRobot, with ACT/SmolVLA policies wrapped by planner, verifier, retry, and replan logic.
- Use `docs/agentic_physical_ai_plan.md` for the full checkpoint plan and `docs/harness/physical-ai/team-spec.md` for checkpoint orchestration.

## Why

- The project should prove whether agentic wrapping improves weak or medium policy success before real robot integration.
- Simulation readiness must be verified by executable commands, not just by static code.

## How

- Lightweight checkpoint 01 smoke:
  `PYTHONPATH=src python3 -B -m physical_ai_agent.checkpoints.checkpoint_01`
- Strict checkpoint 01 simulation gate:
  `PYTHONPATH=src python3 -B -m physical_ai_agent.checkpoints.checkpoint_01 --strict-sim-deps --probe-libero-env`
- Standard-library tests:
  `PYTHONPATH=src python3 -B -m unittest discover -s tests`
- No-dependency syntax fallback:
  `PYTHONPATH=src python3 -B -c "import ast, pathlib; files=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text()) for p in files]; print(f'parsed {len(files)} files')"`
