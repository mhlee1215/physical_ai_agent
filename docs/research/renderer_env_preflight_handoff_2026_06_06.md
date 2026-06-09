# Renderer Environment Preflight Handoff

Date: 2026-06-06

## Purpose

Before running the two-stage true-oracle gate in a remote or Linux/GPU
environment, run a non-mutating preflight:

```bash
OUTPUT_DIR=_workspace/checkpoints/renderer_env_preflight \
sh scripts/run_renderer_env_preflight.sh
```

Then render the report:

```bash
PYTHONPATH=src python3 -B scripts/build_renderer_env_preflight_report.py \
  --preflight-json _workspace/checkpoints/renderer_env_preflight/renderer_env_preflight.json \
  --output-dir _workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/renderer_env_preflight
```

## Checks

- Python executable.
- `gymnasium` import.
- `mani_skill` import.
- `sapien` import.
- `torch` import.
- `lerobot` import.
- `nvidia-smi` availability.
- `vulkaninfo --summary` availability.

## Claim boundary

This preflight does not run simulation, does not create or stop Pods, and does
not prove Tier O. It only decides whether the environment is worth attempting
with `scripts/run_actual_sim_true_oracle_probe_then_policy_cp24.sh`.

