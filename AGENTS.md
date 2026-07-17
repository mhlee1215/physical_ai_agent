# Repository Agents Guide

## What

- This repo builds a Mac-local agentic physical AI evaluation stack.
- The first stack target is MuJoCo + LIBERO + LeRobot, with ACT/SmolVLA policies wrapped by planner, verifier, retry, and replan logic.
- New collaborators should read `Summary.md` first for the current paper target, thread topology, RunPod volume state, and active blockers.
- New conversations should use normal Codex collaboration by default. Enter postdoc/orchestrator mode only when the user says `포닥 모드로 전환해서 PM과 스레드들을 오케스트레이션해줘.` or `포닥 모드 켜줘.`
- `리서치 모드` is a RunPod debugging override: preserve the live Pod by default and let RunPod Researcher and Tech Lead coordinate 1:1 while PM tracks progress/cost/ownership.
- Any user-requested orchestration policy change must be persisted to repo docs in the same turn. Update `Summary.md` for current state and `docs/harness/physical-ai/team-spec.md` for durable role/routing rules; thread messages alone are not enough.
- Dataset/viewer servers must reuse the existing user-visible port/session by default. Do not avoid a live server by starting another one on a different port unless the user explicitly asks for a separate server.
- For deterministic photoreal robot dataset generation or rerendering, use `.agents/skills/robot-photoreal-dataset-rendering/SKILL.md`. Do not start a full render until the source identity, all-episode replay preflight, camera contract, and representative-frame canary pass.
- Top paper-result priority: produce experiment data for manuscript tables as quickly and efficiently as possible. Prioritize actual table-ready rows, metrics, and artifacts over repeated smoke diagnostics. Use only the minimal gates needed to unblock data-producing runs.
- RunPod experiment-data lifecycle: past remote results are not needed. For every new RunPod experiment, download completed datasets/results/checkpoints locally, verify the local copy, then delete the remote artifact directory.
- SO101 datasets are append-only by default. Unless the user explicitly requests destructive replacement or cleanup in the current turn, create a new versioned dataset root/HF path and never overwrite, rename, or delete an existing dataset.
- Every new SO101 dataset must be declared in `configs/so101/dataset_generation/*.json`. Dataset generation is not complete until the recipe-backed split appears in the Robot Experiment Manager dataset viewer and both `/api/datasets` and one `/api/frame` request succeed.
- Use `PYTHONPATH=src .venv/bin/python scripts/so101_dataset_registry.py validate --require-training-ready` as the dataset completion gate. `training_ready` requires LeRobot metadata/data, 256x256 camera1/2, 6D state/action, prompt/stats, passed audit, train grid sidecar, and any declared validation loop-start.
- Use `docs/agentic_physical_ai_plan.md` for the full checkpoint plan and `docs/harness/physical-ai/team-spec.md` for checkpoint orchestration.

## Why

- The project should prove whether agentic wrapping improves weak or medium policy success before real robot integration.
- Simulation readiness must be verified by executable commands, not just by static code.

## How

- Bootstrap checkpoint 01 local sim dependency:
  `sh scripts/install/local_install.sh --checkpoint 01`
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
  `sh scripts/install/local_install.sh --checkpoint 05-06`
- Checkpoint 14-15 3D render and real SmolVLA rollout:
  `sh scripts/checkpoint_14_15.sh --allow-download --require-3d-render --require-real-smolvla`
- Live SO101 MuJoCo viewer:
  `sh scripts/view_so101_live.sh`
- Live SO101 MuJoCo viewer with camera inputs:
  `sh scripts/view_so101_live.sh --show-inputs --fps 15`
- Live SO101 browser simulator with SmolVLA inference:
  `sh scripts/view_so101_live.sh --browser-only --policy smolvla --allow-download --smolvla-action-steps 15 --show-inputs --fps 2`
- Checkpoint 16 SO101 camera input preview:
  `sh scripts/checkpoint_16.sh`
- Checkpoint 17 SO101 multi-camera input preview:
  `sh scripts/checkpoint_17.sh`
- Checkpoint 18 SO101 egocentric policy input preview:
  `sh scripts/checkpoint_18.sh`
- Checkpoint 19 SmolVLA real camera input rollout:
  `sh scripts/checkpoint_19.sh --allow-download --require-real-smolvla`
- Checkpoint 20 rule-based planner:
  `sh scripts/checkpoint_20.sh`
- Checkpoint 21 simulation-state verifier:
  `sh scripts/checkpoint_21.sh`
- Checkpoint 22 agentic retry loop:
  `sh scripts/checkpoint_22.sh`
- Checkpoint 23 first comparison report:
  `sh scripts/checkpoint_23.sh`
- Checkpoint 24 ManiSkill / ManiSkill-HAB evaluation:
  `sh scripts/checkpoint_24.sh --require-maniskill`
- Checkpoint 24 real SmolVLA image probe:
  `sh scripts/checkpoint_24.sh --require-maniskill --episodes 1 --steps 1 --policy smolvla_real --allow-download --real-images`
- RunPod cloud workstation check:
  `RUNPOD_SSH='<pod-user>@ssh.runpod.io' sh scripts/runpod_check.sh`
- RunPod Pod lifecycle:
  `RUNPOD_API_KEY='<api-key>' RUNPOD_POD_ID='<pod-id>' sh scripts/runpod_pod.sh stop`
- RunPod baseline note:
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` verifies CUDA/PyTorch and CP24 `Empty-v1`, but Python 3.11 blocks `lerobot>=0.5.1` SmolVLA and the tested Pod did not expose NVIDIA Vulkan for `PickCube-v1`.
