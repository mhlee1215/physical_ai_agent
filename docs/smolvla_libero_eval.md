# SmolVLA LIBERO Evaluation

## Goal

Generate paper-comparable SmolVLA baseline numbers on LIBERO before testing
agentic wrappers.

## Comparable Protocol

Use LeRobot's LIBERO benchmark integration:

- policy: `lerobot/smolvla_libero`
- environment: `libero`
- suites: `libero_spatial,libero_object,libero_goal,libero_10`
- episodes: `10` per task
- tasks: all 10 tasks per suite
- total trials: `4 suites * 10 tasks * 10 episodes = 400`
- metric: binary task success rate, reported per suite and averaged
- rendering: `MUJOCO_GL=egl` on headless cloud
- control mode: use the checkpoint's default unless the model card or config
  specifies otherwise

The quick smoke protocol may use one suite and one episode per task, but those
numbers are not paper-comparable.

## RunPod Command

After starting the Pod and pulling the latest repo:

```bash
cd /workspace/physical-ai/physical_ai_agent
git pull --ff-only origin main
sh scripts/runpod_smolvla_libero_eval.sh
```

For a quick smoke:

```bash
LIBERO_TASKS=libero_spatial \
LIBERO_N_EPISODES=1 \
sh scripts/runpod_smolvla_libero_eval.sh
```

## Result Handoff

The script writes to:

```text
_workspace/runpod_results/smolvla_libero_<timestamp>/
```

Before stopping RunPod, fetch results locally:

```bash
set -a
. ./.env
set +a
RUNPOD_REMOTE_RESULT_DIR=/workspace/physical-ai/physical_ai_agent/_workspace/runpod_results \
  sh scripts/runpod_fetch_results.sh
sh scripts/runpod_pod.sh stop
```

## Blocking Conditions

- If RunPod cannot start the stopped Pod, create a new Pod with the same
  network volume attached.
- The previous official PyTorch 2.4 / Python 3.11 template is insufficient for
  current LeRobot SmolVLA because `lerobot>=0.5.1` requires Python `>=3.12`.
- Prefer a Python 3.12-capable image. If unavailable, bootstrap Python 3.12
  inside the Pod and create the LeRobot environment under `/workspace`.
