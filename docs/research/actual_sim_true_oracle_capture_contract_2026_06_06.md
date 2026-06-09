# Actual-Sim True-Oracle Capture Contract

Date: 2026-06-06

## Purpose

This note defines the exact contract that CP24 must satisfy before a result can
be called Tier O actual-simulation true-oracle projection.

The related-work scan changed the role of Oracle Point Overlay:

- It is not the main novelty.
- It is an upper-bound diagnostic inside the Agentic SmolVLA experiment matrix.
- It is valid only when same-step privileged simulator geometry is captured.

## Code-path update

`src/physical_ai_agent/checkpoints/checkpoint_24.py` now augments the
`smolvla_affordance_oracle` and `affordance_oracle_probe` policy-input
observation with simulator state when available:

- object/cube/object-like pose from the wrapped environment,
- camera parameters from environment sensor maps when available,
- existing observation fields are preserved and not overwritten,
- camera metadata is merged conservatively.

The policy still receives the normal observation path for state extraction, but
the overlay image is built from the augmented observation.

## Required per-step fields

Each strict true-oracle step must have:

- `policy == smolvla_affordance_oracle`
- `raw_frame_path`
- `overlay_frame_path`
- `policy_metadata.oracle_affordance.mode == projected_object_pose`
- `policy_metadata.oracle_affordance.object_pose_xyz`
- `policy_metadata.oracle_affordance.camera_metadata_keys`
- `episode`
- `step`
- `success`

The manifest must report:

```text
source_type == actual_sim_true_oracle_projection
real_sim_episode == true
true_oracle_projection == true
strict_true_oracle_step_count >= 10
status == passed
```

## Execution command

Use only inside an already-approved local or remote environment that can render
ManiSkill RGB observations and load SmolVLA:

```bash
OUTPUT_DIR=_workspace/checkpoints/checkpoint_24_actual_sim_true_oracle \
EPISODES=1 \
STEPS=12 \
sh scripts/run_actual_sim_true_oracle_cp24.sh
```

This command does not create or manage RunPod Pods.

## Interpretation

If this passes:

- Tier O upper-bound diagnostic is available.
- It can be imported into the milestone dashboard.
- It still does not prove a deployable method, because object pose and camera
  metadata are privileged simulator state.

If this fails:

- `mode == image_center_fallback` means object pose or camera projection was
  unavailable.
- missing `object_pose_xyz` means env object/cube pose extraction failed.
- missing `camera_metadata_keys` means camera metadata extraction failed.
- missing `raw_frame_path` or `overlay_frame_path` means same-step visual
  evidence was not saved.

