# Oracle Point Overlay Implementation Plan

Date: 2026-06-06

## Goal

Test whether explicit zero-parameter spatial affordance hints improve SmolVLA
before training any affordance predictor.

Core question:

```text
Does SmolVLA act better when the image already marks where to grasp or place?
```

This is the first implementation step for the larger "Active Affordance
Grounding for Lightweight VLA" direction.

## Design Constraints

- Keep SmolVLA frozen.
- Add 0M runtime parameters in the first experiment.
- Do not use Grounding DINO, SAM, or any heavy perception foundation model at
  runtime.
- Produce visual artifacts so point calculation can be inspected locally.
- Keep the baseline `smolvla_real` path unchanged.

## Pipeline

Training / labeling time:

```text
ManiSkill RGB observation + simulator state
  -> OracleAffordanceLabeler
  -> 2D point / bbox / confidence
  -> overlay image artifact
  -> optional training labels for later tiny predictor
```

Inference / rollout time:

```text
Task instruction + current RGB observation
  -> OracleAffordanceLabeler
  -> OverlayRenderer
  -> SmolVLA batch with overlay image
  -> SmolVLA select_action()
  -> clipped action
  -> env.step()
  -> metrics + overlay artifacts
```

## Implementation

New module:

- `src/physical_ai_agent/perception/affordance_overlay.py`

Main functions:

- `build_oracle_affordance_overlay(obs, output_path=None)`
- `extract_rgb_images(obs)`

Current oracle behavior:

- If common object/target pose and camera calibration fields are available,
  project the 3D point into the selected camera image.
- If pose/calibration is unavailable, use a deterministic image-center fallback.
- Draw a green point/cross marker and write an overlay PNG when requested.

CP24 integration:

- New policy: `smolvla_affordance_oracle`
- Runtime path:

```text
_policy_action()
  -> _smolvla_affordance_oracle_action()
  -> build_oracle_affordance_overlay()
  -> _build_maniskill_smolvla_batch(..., override_camera_pixels=overlay)
  -> policy.select_action()
```

Artifacts:

- `maniskill_rollout/smolvla_affordance_oracle_manifest.json`
- `maniskill_rollout/smolvla_affordance_oracle_frames/*.png`
- `maniskill_rollout/smolvla_affordance_oracle_rollout.gif`

## Local Mac Feasibility

Oracle point calculation and overlay visualization are Mac-local feasible even
if full ManiSkill rendering is blocked:

- The point projection and fallback overlay path use `numpy` and `PIL`.
- Unit tests can synthesize a ManiSkill-like RGB observation, camera intrinsic,
  extrinsic, and object pose.
- This verifies the core point calculation and rendered overlay without Vulkan.

Full CP24 rollout feasibility still depends on the current ManiSkill/Vulkan
state:

- `Empty-v1` fallback can validate pipeline wiring.
- `PickCube-v1` real RGB rollout may still hit macOS/SAPIEN/Vulkan blockers.
- If that happens, preserve the blocker and use unit-level overlay artifacts as
  proof of oracle calculation.

## First Experiment

Run baseline:

```bash
sh scripts/checkpoint_24.sh --require-maniskill --episodes 1 --steps 1 \
  --policy smolvla_real --allow-download --real-images \
  --output-dir _workspace/checkpoints/checkpoint_24_smolvla_real_raw_1step
```

Run oracle overlay:

```bash
sh scripts/checkpoint_24.sh --require-maniskill --episodes 1 --steps 1 \
  --policy smolvla_affordance_oracle --allow-download --real-images \
  --output-dir _workspace/checkpoints/checkpoint_24_smolvla_affordance_oracle_1step
```

Recommended ablation sequence:

```text
A0: smolvla_real, raw RGB
A1: smolvla_affordance_oracle, oracle point overlay
A2: tiny learned affordance head, later
```

## Success Criteria

Implementation success:

- Overlay point calculation produces a visible PNG.
- SmolVLA batch accepts overlay images through existing image feature path.
- CP24 records manifest and frame artifacts.

Research success:

- Oracle overlay improves first-step grasp/contact behavior or reduces
  missed-object failures compared with raw RGB.
- If not, inspect whether SmolVLA ignores marker conventions or the oracle point
  is not aligned with the relevant contact affordance.

