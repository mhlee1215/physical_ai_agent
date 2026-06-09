# Actual Simulation True-Oracle Projection Gate

This gate is the required evidence tier for using Oracle Point Overlay as an
upper-bound diagnostic inside the agentic lightweight VLA study.

After the related-work scan on 2026-06-06, the project should not present
visual overlays as the primary novelty. The true-oracle projection is instead
used to quantify the best-case value of perfect spatial cueing, so later
learned or heuristic affordance modules can be compared against a clearly
bounded target.

## Claim boundary

- Synthetic diagnostic reports validate projection math, rendering, visual
  prompt encodings, and policy-input formatting.
- Actual sim RGB fallback reports validate that overlays can be rendered on
  saved simulator frames, but they do not prove oracle projection.
- Actual sim heuristic reports validate that an image-only cue can be computed
  and rendered on real simulator frames, but they do not prove access to object
  pose or camera geometry.
- Actual sim true-oracle projection requires RGB, object pose, camera metadata,
  overlay frame, episode, and step from the same action-input observation.
- Paper-facing benchmark success must still come from environment success
  flags, not from overlay visibility or internal verifier predicates.

## Required artifact

The CP24 rollout must produce:

```text
maniskill_rollout/smolvla_affordance_true_oracle_steps.json
```

The manifest must report:

```text
strict_true_oracle_step_count >= 10
status == passed
source_type == actual_sim_true_oracle_projection
real_sim_episode == true
true_oracle_projection == true
```

## Execution entrypoint

Use this only inside an already-approved renderer-capable local or remote
environment. It does not create or manage RunPod Pods.

```bash
sh scripts/run_actual_sim_true_oracle_cp24.sh
```

Optional overrides:

```bash
OUTPUT_DIR=_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle \
EPISODES=1 \
STEPS=12 \
ORACLE_OVERLAY_ROOT=_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z \
sh scripts/run_actual_sim_true_oracle_cp24.sh
```

If the manifest already exists, import it directly:

```bash
sh scripts/import_actual_sim_true_oracle_milestone.sh \
  _workspace/checkpoints/checkpoint_24_actual_sim_true_oracle/maniskill_rollout/smolvla_affordance_true_oracle_steps.json
```

## Current status

Current milestone state is blocked because the available artifacts do not yet
include the required true-oracle manifest. Existing actual sim RGB evidence is
fallback-only until the same-step pose/camera metadata is captured.
