# SO101 Teacher-Data Generalization Plan

Date: 2026-06-26

## Purpose

This document explains how to generalize the current SO101/SmolVLA teacher-data
pipeline beyond the cube-edge-specific `move_and_align_cube_edge` skill.

The goal is not to add many more object-specific scripts. The goal is to move
from a single handcrafted teacher to a reusable data-generation abstraction:

- object-centric affordance regions,
- grasp candidate generation,
- primitive-local correction data,
- structured dataset-generation augmentation,
- shape-family curriculum,
- and closed-loop recovery coverage.

This is a design document. It does not propose immediate code changes.

## Current Starting Point

The current pipeline already has the right first principle:

- train-time augmentation is separate from dataset-generation augmentation;
- `move_and_align_cube_edge_train_v2` changes the teacher trajectory
  distribution by adding generated teacher trajectories, terminal hold, and
  near-target correction;
- the teacher/student gap is already acknowledged, and new datasets are
  expected to include recovery or off-nominal states.

Relevant source documents:

- [SO101 SmolVLA Training Pipeline](/Users/minhaeng/workspace/physical_ai_agent/docs/so101_smolvla_training_pipeline.md)
- [SO101 Fixed-Jaw Edge Chain Dataset](/Users/minhaeng/workspace/physical_ai_agent/docs/so101_fixed_jaw_edge_chain_dataset.md)

## High-Level Direction

Replace:

- one teacher specialized to one geometry target such as a cube edge

With:

- one reusable teacher-data engine that generates primitive demonstrations from
  object affordances, geometric candidates, and correction starts.

The key shift is:

- from `asset-specific scripted target`
- to `object-centric primitive with affordance metadata`

## Design Principles

- [ ] Keep runtime policy inputs lightweight and policy-compatible.
- [ ] Allow the teacher to use privileged simulator state and geometry.
- [ ] Do not confuse training-image augmentation with teacher-data distribution design.
- [ ] Optimize for swappable lightweight VLA actors, not one tightly coupled pipeline.
- [ ] Cover learner-induced failure states, not only nominal teacher success states.
- [ ] Generalize first across shape families, then across individual assets.
- [ ] Preserve reproducibility through manifest fields, dataset recipes, and fixed split contracts.

## Target Abstraction

The current primitive should be generalized from:

- `move_and_align_cube_edge`

To a family like:

- `approach_affordance_region`
- `align_for_grasp`
- `close_and_verify_grasp`
- `lift_and_stabilize`
- `transport_to_relation`
- `place_and_release`

Each primitive should be parameterized by structured metadata rather than by
object-specific script names.

Example parameter packet:

```json
{
  "primitive": "align_for_grasp",
  "object_id": "target_object",
  "object_family": "cuboid",
  "affordance_region": "top_edge_long",
  "approach_axis": "negative_y",
  "grasp_mode": "pinch_side",
  "target_relation": null
}
```

## Phase 1: Rename the Problem Correctly

- [ ] Stop treating `cube edge` as the core abstraction.
- [ ] Reframe the current skill as one instance of `approach + align + graspable-region execution`.
- [ ] Define the first generalized primitive taxonomy.
- [ ] Keep the original cube-edge teacher as one checked baseline, not as the future architecture.

Recommended first taxonomy:

- [ ] `approach_pregrasp_region`
- [ ] `align_gripper_to_region`
- [ ] `grasp_from_region`
- [ ] `lift_object`
- [ ] `move_to_place_relation`
- [ ] `place_object`

## Phase 2: Build Object Metadata Instead of New Scripts

Every object or shape family should expose metadata that the teacher-data
generator can use.

- [ ] Define `object_family`.
- [ ] Define `graspable_regions`.
- [ ] Define `avoid_regions`.
- [ ] Define `support_regions`.
- [ ] Define `placeable_surfaces`.
- [ ] Define `preferred_approach_axes`.
- [ ] Define `default_grasp_modes`.
- [ ] Define `release-safe regions`.

Recommended first family set:

- [ ] `cuboid`
- [ ] `cylinder`
- [ ] `sphere_like`
- [ ] `bowl_like`
- [ ] `mug_like`
- [ ] `bottle_like`
- [ ] `tool_like_long`

Recommended first metadata source:

- [ ] simulator oracle state
- [ ] mesh or asset metadata
- [ ] manually authored family-level affordance tags

Do not start with a learned affordance detector. For teacher-data generation,
use privileged simulator information first.

## Phase 3: Move from One Target Pose to Candidate Generation

The generalized teacher should not assume one hand-authored target pose.
Instead, it should:

- [ ] generate multiple grasp or approach candidates,
- [ ] filter them by reachability and collision,
- [ ] score them by primitive compatibility,
- [ ] select one,
- [ ] execute the teacher rollout from that candidate.

Candidate generation plan:

- [ ] Generate `K` approach candidates per object.
- [ ] Generate `K` grasp candidates per affordance region.
- [ ] Filter candidates that violate joint or workspace limits.
- [ ] Reject candidates with obvious collision or self-intersection.
- [ ] Reject candidates inconsistent with the primitive goal.
- [ ] Prefer candidates that preserve visibility in egocentric or wrist cameras.
- [ ] Save chosen candidate metadata in the export report.

This makes the data engine shape-aware instead of cube-edge-script-aware.

## Phase 4: Generalize `near-target correction` into Primitive-Local Recovery Data

The current v2 dataset direction is good, but it is too task-specific.
Generalize it into primitive-local correction buckets.

For each primitive, explicitly generate:

- [ ] nominal successful trajectories
- [ ] near-success correction trajectories
- [ ] post-success terminal hold trajectories
- [ ] mild off-nominal recovery trajectories

Recommended correction buckets:

- [ ] `approach_too_left`
- [ ] `approach_too_right`
- [ ] `approach_too_high`
- [ ] `approach_too_low`
- [ ] `grasp_partial_contact`
- [ ] `grasp_misaligned_yaw`
- [ ] `lift_unstable`
- [ ] `transport_drift`
- [ ] `place_off_center`
- [ ] `release_sticky_or_delayed`

For each bucket:

- [ ] define a simulator start-state perturbation
- [ ] define teacher correction horizon
- [ ] define success predicate
- [ ] define terminal hold length
- [ ] define export ratio in the final dataset

## Phase 5: Keep Two Augmentation Lanes Separate

This repo already does this correctly. Preserve it.

### Dataset-Generation Augmentation

Changes the teacher trajectory distribution:

- [ ] shape family variation
- [ ] object size variation
- [ ] object mass and friction variation
- [ ] object orientation variation
- [ ] initial pose variation
- [ ] clutter variation
- [ ] target relation variation
- [ ] privileged correction starts
- [ ] candidate diversity
- [ ] terminal hold

### Train-Time Augmentation

Changes observations seen during learning:

- [ ] color jitter
- [ ] sharpness jitter
- [ ] affine jitter
- [ ] image masking
- [ ] camera dropout
- [ ] state jitter
- [ ] state dropout

Policy generalization to new objects will depend more on dataset-generation
augmentation than on image-only train-time augmentation. Do not use train-time
augmentation as a substitute for recovery coverage or shape diversity.

## Phase 6: Add Shape Curriculum Instead of Asset Explosion

Do not jump directly from one cube to a giant mixed asset set.
Use a curriculum.

### Stage A: Primitive Solids

- [ ] cuboids
- [ ] cylinders
- [ ] spheres or sphere-like objects

### Stage B: Simple Composite Shapes

- [ ] bowl-like
- [ ] mug-like
- [ ] bottle-like
- [ ] elongated tool-like objects

### Stage C: Same Affordance, Different Geometry

- [ ] multiple shapes with the same grasp semantics
- [ ] multiple sizes per family
- [ ] multiple textures and colors per family

### Stage D: Same Geometry, Different Task Relations

- [ ] pick-only
- [ ] pick-and-lift
- [ ] pick-and-place
- [ ] pick-and-align
- [ ] pick-and-insert preconditions

### Stage E: Clutter and Distractors

- [ ] non-target objects near the target
- [ ] partial occlusion
- [ ] viewpoint degradation
- [ ] tabletop variation

Success criterion for a curriculum stage:

- [ ] the teacher still produces stable successful trajectories
- [ ] the student closed-loop success does not collapse
- [ ] correction buckets remain reachable and useful

## Phase 7: Add DAgger-Like Recovery Collection in Simulation

Nominal teacher data is not enough.

The next collection lane should come from learner-induced failures.

- [ ] Run the current student in closed loop.
- [ ] Detect primitive-local failures.
- [ ] Snapshot failure-near states.
- [ ] Re-run the privileged teacher from those states.
- [ ] Export only short corrective snippets when possible.
- [ ] Add those snippets as a separate recovery lane, not mixed invisibly into nominal data.

Recommended recovery dataset split:

- [ ] `nominal_teacher`
- [ ] `synthetic_correction`
- [ ] `student_failure_recovery`
- [ ] `terminal_hold`

This is the most important step for closing the teacher/student gap on new
objects.

## Phase 8: Define the General Teacher Data Contract

Each exported episode should record more than image, state, and action.

- [ ] `primitive_name`
- [ ] `object_family`
- [ ] `object_instance`
- [ ] `affordance_region`
- [ ] `grasp_mode`
- [ ] `approach_axis`
- [ ] `candidate_count`
- [ ] `selected_candidate_id`
- [ ] `candidate_rejection_summary`
- [ ] `start_distribution`
- [ ] `dataset_lane`
- [ ] `recovery_bucket`
- [ ] `terminal_hold_frames`
- [ ] `teacher_success`
- [ ] `teacher_success_reason`

Recommended `dataset_lane` values:

- [ ] `nominal_teacher`
- [ ] `near_target_correction`
- [ ] `recovery_from_student_failure`
- [ ] `terminal_hold`
- [ ] `shape_curriculum_stage_a`
- [ ] `shape_curriculum_stage_b`

## Phase 9: Define Manifest and Recipe Expansion

The current export recipe and manifest pattern should be preserved and expanded.

- [ ] Add generalized dataset recipes by primitive family, not by one object script.
- [ ] Encode the curriculum stage in the recipe name.
- [ ] Encode the lane mix in the recipe metadata.
- [ ] Keep train, validation, and loop-validation splits separate.
- [ ] Keep validation unaugmented and semantically stable.
- [ ] Validate new manifests in CI after every exporter change.

Recommended naming direction:

- [ ] `approach_affordance_region_train_stageA_v1`
- [ ] `align_for_grasp_train_stageA_v1`
- [ ] `grasp_from_region_train_stageB_v1`
- [ ] `pick_place_relation_train_stageC_v1`

## Phase 10: Evaluation Should Test Generalization, Not Only Memorization

A generalized teacher-data pipeline needs matching evaluation splits.

- [ ] same-family held-out poses
- [ ] same primitive, new object instances
- [ ] same primitive, new shape within family
- [ ] same primitive, new clutter layouts
- [ ] chained primitive execution
- [ ] recovery after learner-induced failure

Recommended evaluation taxonomy:

- [ ] `in_family_nominal`
- [ ] `in_family_off_nominal`
- [ ] `cross_instance`
- [ ] `cross_shape_family`
- [ ] `primitive_chain`
- [ ] `primitive_chain_with_recovery`

Do not treat lower supervised loss on a narrow asset distribution as evidence of
generalization.

## Phase 11: Concrete Short-Term Plan for This Repo

### Immediate Step 1

- [ ] Keep `move_and_align_cube_edge_train_v2` as the checked control.
- [ ] Do not overwrite or rename existing baseline exports yet.

### Immediate Step 2

- [ ] Introduce one generalized abstraction beside it:
  - [ ] `approach_pregrasp_region`
  - [ ] or `align_for_grasp_region`
- [ ] Start with `cuboid` and `cylinder` only.
- [ ] Use simulator-oracle affordance tags.

### Immediate Step 3

- [ ] For the new primitive, generate:
  - [ ] nominal teacher episodes
  - [ ] near-target correction episodes
  - [ ] 20 terminal hold frames
  - [ ] mild recovery starts

### Immediate Step 4

- [ ] Add metadata fields for `object_family`, `affordance_region`, and `dataset_lane`.
- [ ] Keep validation fixed and simple.

### Immediate Step 5

- [ ] Train one generalized primitive checkpoint.
- [ ] Compare against the cube-edge-specific checkpoint on:
  - [ ] original cube edge setup
  - [ ] held-out cuboid sizes
  - [ ] simple cylinders
  - [ ] mild clutter

### Immediate Step 6

- [ ] If the generalized primitive degrades sharply on cube-edge baseline, inspect whether:
  - [ ] candidate quality is too low
  - [ ] correction mix is too aggressive
  - [ ] shape curriculum jumped too quickly
  - [ ] language prompts became too heterogeneous

## What Not To Do

- [ ] Do not create a separate hand-scripted teacher for every new object.
- [ ] Do not rely only on image jitter to claim generalization.
- [ ] Do not remove the privileged teacher just because the student is visual.
- [ ] Do not mix nominal and correction data without lane metadata.
- [ ] Do not evaluate only on the same object geometry used during export.
- [ ] Do not collapse all grasp behaviors into one undifferentiated `pick` label too early.

## Success Criteria

This plan is working only if most of these become true:

- [ ] one primitive generator covers multiple shape families
- [ ] candidate-based teacher rollouts stay stable across families
- [ ] recovery data improves off-nominal closed-loop behavior
- [ ] generalized primitive checkpoints do not overfit one asset geometry
- [ ] chained primitives remain compatible at boundaries
- [ ] new datasets remain reproducible through manifest and recipe validation

## Bottom Line

The current pipeline should generalize by moving upward in abstraction:

- from one asset-specific edge teacher
- to affordance-tagged primitive data generation
- with candidate-based teacher actions
- recovery-aware dataset lanes
- shape-family curriculum
- and evaluation that measures off-nominal primitive robustness

The right next step is not "add more cube-edge variants." It is "promote the
teacher from edge script to object-centric primitive data engine."
