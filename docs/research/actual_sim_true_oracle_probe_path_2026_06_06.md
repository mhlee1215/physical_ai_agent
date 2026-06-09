# Actual-Sim True-Oracle Probe Path

Date: 2026-06-06

## Purpose

The SmolVLA true-oracle path has two independent blockers:

- actual ManiSkill RGB rendering plus object pose/camera metadata capture,
- SmolVLA model loading and policy execution.

The `affordance_oracle_probe` path isolates the first blocker. It uses a
zero-action policy and therefore does not load SmolVLA weights. If this probe
produces 10 strict true-oracle samples, then the remaining blocker is SmolVLA
policy execution rather than pose/camera projection.

## Command

Use only in an already-approved renderer-capable local or remote environment:

```bash
OUTPUT_DIR=_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle_probe \
EPISODES=1 \
STEPS=12 \
sh scripts/run_actual_sim_true_oracle_probe_cp24.sh
```

## Expected manifest

```text
maniskill_rollout/affordance_oracle_probe_true_oracle_steps.json
```

Required fields:

```text
policy == affordance_oracle_probe
source_type == actual_sim_true_oracle_projection
real_sim_episode == true
true_oracle_projection == true
strict_true_oracle_step_count >= 10
status == passed
```

## Interpretation

If the probe passes:

- actual RGB rendering works,
- simulator pose/camera metadata capture works,
- projection codepath works on same-step observations,
- SmolVLA loading remains the next integration step.

If the probe fails:

- `image_center_fallback` means pose/camera metadata did not reach the overlay builder,
- missing frames means RGB rendering failed,
- missing manifest means checkpoint execution failed before records were written.

This probe is not the final policy-input result, but it is the cheapest
renderer/env-metadata gate before running SmolVLA.

