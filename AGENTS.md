# Repository Agents Guide

## What

- This repo builds a Mac-local agentic physical AI evaluation stack.
- The first stack target is MuJoCo + LIBERO + LeRobot, with ACT/SmolVLA policies wrapped by planner, verifier, retry, and replan logic.
- Use `docs/agentic_physical_ai_plan.md` for the full checkpoint plan and `docs/harness/physical-ai/team-spec.md` for checkpoint orchestration.

## Why

- The project should prove whether agentic wrapping improves weak or medium policy success before real robot integration.
- Simulation readiness must be verified by executable commands, not just by static code.

## How

- Bootstrap checkpoint 01 local sim dependency:
  `sh scripts/bootstrap_checkpoint_01.sh`
- Lightweight checkpoint 01 smoke:
  `sh scripts/checkpoint_01.sh`
- Mac-local checkpoint 01 simulation gate:
  `sh scripts/checkpoint_01.sh --strict-local-sim --probe-mujoco`
- Strict checkpoint 01 simulation gate:
  `sh scripts/checkpoint_01.sh --strict-sim-deps --probe-libero-env`
- Standard-library tests:
  `PYTHONPATH=src /Users/minhaeng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -m unittest discover -s tests`
- No-dependency syntax fallback:
  `PYTHONPATH=src python3 -B -c "import ast, pathlib; files=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text()) for p in files]; print(f'parsed {len(files)} files')"`
- Checkpoint 02-04 random policy evaluation:
  `sh scripts/checkpoint_02_04.sh`
- Checkpoint 05-06 policy adapter and SmolVLA probe:
  `sh scripts/checkpoint_05_06.sh`
- Bootstrap CP05-06 SmolVLA dependencies:
  `sh scripts/bootstrap_checkpoint_05_06.sh`
- Checkpoint 14-15 3D render and real SmolVLA rollout:
  `sh scripts/checkpoint_14_15.sh --allow-download --require-3d-render --require-real-smolvla`
- Live SO101 MuJoCo viewer:
  `sh scripts/view_so101_live.sh`
- Checkpoint 16 SO101 camera input preview:
  `sh scripts/checkpoint_16.sh`
- Checkpoint 17 SO101 multi-camera input preview:
  `sh scripts/checkpoint_17.sh`
