# LeRobot Evaluation Entrypoints

This repo keeps paper-facing SmolVLA evaluation focused on two core benchmarks:

- LIBERO
- Meta-World

Other benchmark probes and LIBERO throughput/retry variants should live under
`experiments/` unless they become part of the fixed comparison protocol.

## Single Linux Entrypoint

The canonical Linux/RunPod entrypoint is:

```bash
sh scripts/eval_smolvla_lerobot_linux.sh --benchmark libero --agentic-layer baseline
sh scripts/eval_smolvla_lerobot_linux.sh --benchmark metaworld --agentic-layer baseline
```

`scripts/eval_smolvla_libero_linux.sh` and
`scripts/eval_smolvla_metaworld_linux.sh` are compatibility wrappers around the
same entrypoint.

Agentic behavior is selected by `--agentic-layer`. The current registry is:

- `baseline`: policy-only SmolVLA evaluator.
- `episode_retry`: LIBERO episode-level retry planning layer. It records retry
  planning/debug artifacts from failed `eval_info.json` episodes; it is not an
  in-episode controller.

New planner/verifier/retry methods should be implemented as another
`AgenticLayer` class under `physical_ai_agent.evaluation.agentic_layers`, then
exposed through the same flag.

Every run writes intermediate debug artifacts:

- `debug_artifacts/eval_manifest.json`
- `debug_artifacts/command_argv.json`
- `debug_artifacts/agentic_layer.json`
- `debug_artifacts/events.jsonl`
- `debug_artifacts/environment_probe.txt`
- `debug_artifacts/runpod_preflight.txt`

RunPod defaults are intentionally strict:

- the Python 3.12 virtualenv defaults to `/root/physical-ai/envs/lerobot_py312`
  because earlier installs on the network volume were slow and quota-heavy;
- durable caches and results stay under `/workspace`;
- `REQUIRE_CUDA=1` is the default, so a paper-facing run fails instead of
  silently falling back to CPU;
- set `REQUIRE_CUDA=0` only for explicit plumbing checks.

## Shared Command Builder

Both core shell runners use one shared command generator:

```bash
PYTHONPATH=src python -m physical_ai_agent.evaluation.lerobot_eval \
  --benchmark libero \
  --output-dir /tmp/libero/eval_logs \
  --print-command
```

The same module also supports the project CLI:

```bash
PYTHONPATH=src python -m physical_ai_agent lerobot-eval \
  --benchmark metaworld \
  --output-dir /tmp/metaworld/eval_logs \
  --n-action-steps 15 \
  --print-command
```

## LIBERO

Canonical Linux/RunPod entrypoint:

```bash
sh scripts/eval_smolvla_libero_linux.sh
```

Default command settings:

- policy: `lerobot/smolvla_libero`
- environment: `libero`
- task suites: `libero_spatial,libero_object,libero_goal,libero_10`
- camera mapping:
  `{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}`
- `MUJOCO_GL=egl`

Paper-comparable runs should keep all four standard suites, all tasks, and 10
episodes per task unless the comparison table explicitly defines a smaller
subset.

## Meta-World

Canonical Linux/RunPod entrypoint:

```bash
sh scripts/eval_smolvla_metaworld_linux.sh
```

Default command settings:

- policy: `lerobot/smolvla_metaworld`
- environment: `metaworld`
- task groups: `easy,medium,hard,very_hard`
- rename map: `{"observation.image":"observation.images.camera1"}`
- `policy.empty_cameras=0`
- `policy.device=cuda`
- `policy.use_amp=false`
- `seed=0`
- `MUJOCO_GL=egl`

Use `POLICY_N_ACTION_STEPS=15` when testing the pi0.7-style action-step setting,
but report it as a protocol variant unless it matches the referenced table.

## Experiments

Exploratory LIBERO launchers moved to:

```text
experiments/libero_runpod_variants/
```

Those scripts still call the core LIBERO evaluator. Treat their results as
throughput, retry-budget, routing, or scheduling experiments, not policy-only
baseline numbers.
