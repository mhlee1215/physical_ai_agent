---
name: robot-photoreal-dataset-rendering
description: Use for deterministic photoreal rendering or rerendering of robot datasets, especially SO101 LeRobot image replacement, render-replay state recovery, camera-contract preservation, Blender Cycles Metal quality tuning, JSON robot-part materials, black-table visual clutter, training-ready dataset conversion, Robot Experiment Manager registration, and MyCobot adaptive-gripper renders.
---

# Robot Photoreal Dataset Rendering

Use this skill to change image quality, materials, lighting, or visual-only
scene content without silently changing the source trajectory, camera contract,
task prompt, or grasp dynamics.

## Read First

- Read `references/deterministic-render-contract.md` when replaying an existing
  dataset, designing render sidecars, or claiming deterministic output.
- Read `references/so101-workflow.md` for SO101 commands, LeRobot conversion,
  camera requirements, registry gates, and known `grip_the_cube_v2_5` blockers.
- Read `references/rendering-quality-and-materials.md` when changing Cycles
  quality, temporal noise, scene props, lighting, or robot-part colors.
- Read `references/mycobot-workflow.md` for MyCobot. MyCobot visual evidence
  defaults to the official adaptive gripper.

## Workflow

1. **Identify the immutable source.** Record the exact dataset root, manifest,
   episode/frame counts, camera keys, prompt semantics, source report, and
   checksums. Existing SO101 datasets are append-only; never overwrite one.
2. **Preflight deterministic replay.** Recreate the report-declared environment,
   validate snapshot dimensions and target-slot identity, restore collision and
   target bookkeeping, and replay every action without rendering. Block the full
   render on any unexplained state, grasp, lift, or object-pose mismatch.
3. **Render a canary.** Render at least start, grasp/contact, and final frames
   from one representative episode using the source camera matrices. Verify
   viewpoint, target color, gripper shape, contact continuity, and temporal
   stability from artifacts rather than visual approximation.
4. **Freeze the render profile.** Persist samples, denoise, Cycles seed, camera
   settings, material profile, scene profile, asset hashes, Blender version,
   and device report. Visual props must stay outside the manipulation zone and
   must never enter MuJoCo collision geometry.
5. **Render every required frame.** A training derivative requires every source
   episode/frame and every policy camera. A two-frame probe or sidecar is review
   evidence, not a dataset. Resume only with report-backed `--skip-existing`.
6. **Build an image-only derivative.** Copy the source LeRobot contract and
   replace camera bytes while preserving state, action, timestamps, episode
   boundaries, and approved task prompts. Source and output roots must differ.
7. **Validate and publish.** Run targeted tests, strict frame/camera checks,
   dataset registry validation, then the mandatory
   `scripts/verify_so101_dataset_completion.py` gate, which restarts the
   launchctl viewer and checks `/api/datasets` plus one `/api/frame` per split.
   Finish with mobile viewer inspection on the existing orchestrated dashboard
   port. Commit only code, config, docs, and small review images; keep raw
   renders in `_workspace`.

## Hard Gates

- Do not infer camera pose by looking at an image. Use stored camera metadata or
  per-frame matrices from the source simulator contract.
- Seed-only replay is insufficient when object state, RNG state, active slots,
  or environment configuration can drift. Prefer full frame state or visual
  geom transforms.
- Do not start a full render when snapshot dimensions differ from the recreated
  model, target slots map to different objects, or a snapshot is not frame 0.
- Do not reset the environment independently for every frame. Reset once per
  episode, restore the start state, render the pre-action frame, then step the
  recorded action.
- Keep `camera1=egocentric_cam`, `camera2=wrist_cam`, and camera3 as the declared
  wrist duplicate unless the source contract explicitly differs.
- Use concrete target prompts such as `green cube` or `red cube`; reject generic
  `visible cube` prompts in color-grounded datasets.
- Never use `--overwrite` with identical source and output paths.
- Do not start another dataset viewer on a fallback port. Reuse the active
  launchctl-managed Robot Experiment Manager instance and port.

## Completion Evidence

Report the source root, derivative root, recipe/config, render profile hash,
episode/frame/camera counts, replay gate metrics, render timing, validation
commands, viewer API checks, and small contact-sheet evidence. Label canaries,
partial renders, missing cameras, reconstructed snapshots, and approximate
replays as non-training-ready.
