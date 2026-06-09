# Repository Agents Guide

## What

- This repo builds a Mac-local agentic physical AI evaluation stack.
- The first stack target is MuJoCo + LIBERO + LeRobot, with ACT/SmolVLA policies wrapped by planner, verifier, retry, and replan logic.
- Use `docs/harness/physical-ai/team-spec.md` for current orchestration.

## Why

- The project should prove whether agentic wrapping improves weak or medium policy success before real robot integration.
- Simulation readiness must be verified by executable commands, not just by static code.

## How

- Standard-library tests:
  `PYTHONPATH=src /Users/minhaeng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -m unittest discover -s tests`
- No-dependency syntax fallback:
  `PYTHONPATH=src python3 -B -c "import ast, pathlib; files=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text()) for p in files]; print(f'parsed {len(files)} files')"`
- Live SO101 MuJoCo viewer:
  `sh scripts/view_so101_live.sh`
- Live SO101 MuJoCo viewer with camera inputs:
  `sh scripts/view_so101_live.sh --show-inputs --fps 15`
- Live SO101 browser simulator with SmolVLA inference:
  `sh scripts/view_so101_live.sh --browser-only --policy smolvla --allow-download --smolvla-action-steps 15 --show-inputs --fps 2`
- RoboCasa / RoboCasa365 probe:
  `sh scripts/checkpoint_25.sh`
- RoboCasa strict reset/step probe:
  `sh scripts/checkpoint_25.sh --probe-reset-step --require-robocasa --task CloseFridge`
- RunPod cloud workstation check:
  `RUNPOD_SSH='<pod-user>@ssh.runpod.io' sh scripts/runpod_check.sh`
- RunPod Pod lifecycle:
  `RUNPOD_API_KEY='<api-key>' RUNPOD_POD_ID='<pod-id>' sh scripts/runpod_pod.sh stop`
- RunPod baseline note:
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` verifies CUDA/PyTorch, but Python 3.11 blocks `lerobot>=0.5.1` SmolVLA.
