# Actual-Sim True-Oracle Remote Handoff

Date: 2026-06-06

## Purpose

Mac-local execution is blocked by SAPIEN/Vulkan renderer support:

```text
vk::createInstanceUnique: ErrorIncompatibleDriver
```

The next actual Tier O attempt should run in a renderer-capable Linux/GPU
environment, reusing an existing approved pod/environment rather than creating
a new one.

## Two-stage command

```bash
ROOT_OUTPUT_DIR=_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_two_stage \
EPISODES=1 \
STEPS=12 \
MIN_STRICT_STEPS=10 \
sh scripts/run_actual_sim_true_oracle_probe_then_policy_cp24.sh
```

## Why two stages

Stage 1:

- policy: `affordance_oracle_probe`
- action: zero action
- model loading: no SmolVLA
- validates: actual RGB rendering, object pose extraction, camera metadata extraction, projection

Stage 2:

- policy: `smolvla_affordance_oracle`
- model loading: SmolVLA
- validates: policy-input integration after pose/camera capture is known-good

## Expected outputs

Probe manifest:

```text
_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_two_stage/probe/maniskill_rollout/affordance_oracle_probe_true_oracle_steps.json
```

Policy manifest:

```text
_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_two_stage/policy/maniskill_rollout/smolvla_affordance_true_oracle_steps.json
```

Summary:

```text
_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_two_stage/two_stage_summary.json
```

## Pass criteria

```text
probe strict_true_oracle_step_count >= 10
policy strict_true_oracle_step_count >= 10
policy manifest status == passed
```

## Claim boundary

Passing the probe means renderer/env metadata capture works.

Passing the policy stage means SmolVLA can consume the true-oracle overlay
policy input under actual simulation.

Neither stage alone proves final benchmark improvement. Final paper-facing
success still requires environment success flags under the experiment matrix.

