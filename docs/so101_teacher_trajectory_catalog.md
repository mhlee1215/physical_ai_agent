# SO101 Teacher Trajectory Catalog

Last verified: 2026-08-02

This is the human-readable index of the SO101 teacher trajectories that are
currently implemented. It separates production trajectories from supported
primitive experiments and legacy compatibility paths.

For a concrete dataset, the schema-v2 recipe and its generated report remain
authoritative. This catalog explains which trajectory to choose and what each
one does.

## Status Definitions

- **Production**: recipe-backed, exported locally, audited, and training-ready.
- **Supported**: executable and useful for controlled experiments, but not the
  current default data lane.
- **Canary**: deliberately small verification data; do not treat it as a
  training corpus.
- **Legacy**: retained for reproduction or debugging only.

## Shared Contract

Unless a versioned recipe explicitly says otherwise, current SO101 teachers
use this contract:

- `observation.images.camera1`: egocentric camera, `256x256`
- `observation.images.camera2`: wrist camera, `256x256`
- `observation.state`: six SO101 motor/joint positions
- `action`: six absolute target joint positions in the same order
- joint order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
  `wrist_roll`, `gripper`
- prompt: generated from episode object metadata, for example
  `grip the green cube and lift`
- privileged simulator object state: teacher construction and audit only; it is
  not part of the student policy observation
- only episodes that pass the selected teacher success and inspection gates
  are exported

`camera3`, when present in an old export, is a duplicate compatibility image.
It is not a canonical policy input for current datasets.

## Range Taxonomy

- **Continuous-range teacher**: `grip_the_cube_continuous_v1` is the unified
  generation path for requested radii from 10 cm through the verified outer
  envelope. With `trajectory_variant=auto`, it selects the near-contact or
  mid fixed-jaw solver from the requested target radius and records the
  accepted `solver_profile` in episode metadata. The overlap band is 18-22 cm,
  where both IK families are attempted in a deterministic priority order.
- **Mid-range teacher**: `grip_the_cube_v1`, currently verified over the
  recipe-declared 22-28 cm workspace. This includes both `grip_the_cube_v4_3`
  and its correction-start companion `grip_the_cube_v4_3_at`.
- **Near-range teacher**: `grip_the_cube_near_v1`, production-verified over
  10-18 cm. It uses folded-arm, contact-centric IK and measures jaw/face
  alignment at the realized cube-contact width during closing. Production
  placements expose the cube in camera1 or camera2 at frame 0.

The legacy near and mid modes remain separate executable contracts. Do not
silently route either legacy recipe through the other solver. New datasets
that intentionally span both ranges should declare the continuous mode rather
than relabeling one of the legacy modes.

Verified continuous-range envelope on 2026-08-02:

| Radius band | Routing | Verification result | Use |
| --- | --- | --- | --- |
| 10-18 cm | near-contact solver | 15/15 physical grasp/lift; 14/15 strict teacher geometry | Supported |
| 18-22 cm | both solvers, radius-ordered | 50/50 physical and strict teacher success | Supported overlap |
| >22-30 cm | mid fixed-jaw solver | repeated strict successes through 30 cm | Supported |
| 30.5 cm | mid fixed-jaw solver | 3 strict successes in the yaw/bearing boundary probe | Boundary canary only |
| 31-32 cm | mid fixed-jaw solver | 0/27 strict successes | Unsupported by the current hardware/teacher contract |

This is a radius envelope, not a claim that every XY/yaw combination within it
is feasible. Production generation still samples from teacher-feasible
bearing/yaw candidates and applies all recipe inspection gates.

## Recommended Production Trajectories

| Purpose | Range | Exporter settings | Motion phases | Current dataset | Status |
| --- | --- | --- | --- | --- | --- |
| Unified full grasp from the hardware-aligned home pose | Continuous, 10-30 cm stable; 30.5 cm boundary | `skill_mode=grip_the_cube_continuous_v1`, `trajectory_variant=auto`, `grip_the_cube_start_profile=home` | solver-selected move/alignment, descend, settle, close, lift, hold | `grip_the_cube_continuous_v1_canary` | Canary, training-ready |
| Full grasp from the hardware-aligned home pose | Mid, 22-28 cm | `skill_mode=grip_the_cube_v1`, `trajectory_variant=standard`, `grip_the_cube_start_profile=home` | move to cube, wrist-roll alignment, descend, settle, close, lift, hold | `grip_the_cube_v4_3` | Production |
| Local alignment and grasp correction | Mid, 22-28 cm | `skill_mode=grip_the_cube_v1`, `trajectory_variant=direct_align`, `grip_the_cube_start_profile=correction` | correct move/alignment, descend, settle, close, lift, hold | `grip_the_cube_v4_3_at` | Production |
| Folded-arm grasp from the hardware-aligned home pose | Near, 10-18 cm | `skill_mode=grip_the_cube_near_v1`, `trajectory_variant=direct_align`, `grip_the_cube_start_profile=home` | open from hardware pose, contact-centric move/alignment, descend, settle, close, lift, hold | `grip_the_cube_near_v1_train200` | Production |
| Exact real-hardware start-pose check | Diagnostic | `skill_mode=grip_the_cube_v1`, exact initial pose enabled | open from hardware pose, then the configured full grasp path | `grip_the_cube_real_pose_canary_v1` | Canary |

Registry verification on 2026-08-01:

| Dataset | Episodes | Frames | Registry state |
| --- | ---: | ---: | --- |
| `grip_the_cube_continuous_v1_canary` | 8 | 1,519 | available, training-ready |
| `grip_the_cube_v4_3` | 500 | 87,808 | available, training-ready |
| `grip_the_cube_v4_3_at` | 300 | 53,168 | available, training-ready |
| `grip_the_cube_near_v1_train200` | 200 | 34,777 | available, training-ready |
| `grip_the_cube_near_v1_canary` | 5 | 870 | available, training-ready |
| `grip_the_cube_real_pose_canary_v1` | 5 | 888 | available, training-ready |

The production recipes use successful grasp and lift as the terminal task
criterion, a 12-frame terminal hold, geometry/contact alignment inspection,
camera2 pre-close/early-close inspection, and a minimum gripper floor
clearance. Read the recipe rather than copying raw exporter defaults:

- `configs/so101/dataset_generation/grip_the_cube_v4_3.json`
- `configs/so101/dataset_generation/grip_the_cube_v4_3_at.json`
- `configs/so101/dataset_generation/grip_the_cube_continuous_v1_canary.json`
- `configs/so101/dataset_generation/grip_the_cube_near_v1_train200.json`
- `configs/so101/dataset_generation/grip_the_cube_near_v1_canary.json`
- `configs/so101/dataset_generation/grip_the_cube_real_pose_canary_v1.json`

Older audited full-task datasets remain available for comparison:

- `grip_the_cube_v3`: 300 train plus 50 validation episodes
- `grip_the_cube_v3_align`: 200 near-target alignment episodes

They are superseded by `v4_3` and `v4_3_at` for new generation work.

## Agentic Primitive Trajectories

These modes are implemented in
`scripts/export_so101_teacher_rollouts_lerobot.py`. They are useful when a
planner chains short skills, but they are not the default full-task corpus.

| Skill mode | Start state | Teacher motion | Completion condition | Status |
| --- | --- | --- | --- | --- |
| `move_over_cube_edge` | home, gripper closed | move above the selected cube edge, settle | edge-above pose, cube visible and centered | Supported |
| `align_fixed_jaw_cube_edge` | edge-above pose, gripper closed | open and align jaws at the contact edge, settle | fixed-jaw edge contact | Supported |
| `move_and_align_cube_edge` | home or configured near-target perturbation | move and align in one trajectory, settle | contact XY and parallel-angle gates | Supported |
| `grip_from_edge_cube` | aligned edge-contact pose, gripper open | settle, close, lift | grasped and lifted | Supported with hold caveat |
| `grip_from_above_edge_cube` | perturbed pose above the edge | descend/correct, settle, close, lift | grasped and lifted | Supported |
| `move_over_cube` | randomized arm pose | move to an overhead pre-grasp pose, settle | TCP/object distance and height gates | Supported, older primitive |
| `pick_from_top_cube` | perturbed overhead pose | correct to grasp pose, close, lift | grasped and lifted | Supported, older primitive |

The intended three-stage chain is:

```text
move_over_cube_edge
  -> align_fixed_jaw_cube_edge
  -> grip_from_edge_cube
```

The handoff contract is pose compatibility between each preceding final state
and the next primitive's start distribution. Exact dataset-backed closed-loop
tests should still restore the recorded start snapshot; a matching name alone
does not prove start-state equality.

Important implementation detail: `grip_from_edge_cube` currently does not
append `terminal_hold`, even when the exporter hold flag is nonzero. Use
`grip_the_cube_v1` or `grip_from_above_edge_cube` when terminal hold supervision
is required.

### Above-Edge Variants

`grip_from_above_edge_cube` supports these deterministic variants:

- `standard`: descend and align together
- `two_stage_xy_z`: correct XY/wrist roll, then descend
- `roll_first`: correct wrist roll before descending
- `near_miss_correction`: begin from a generated local miss and recover

### Full-Trajectory Variants

`grip_the_cube_v1` supports:

- `standard`: move, then explicit wrist-roll alignment, then descend
- `roll_first`: align wrist roll before the long move
- `direct_align`: combine the move and alignment phase

It also supports `home`, `mid`, `correction`, and `mixed` start profiles. New
datasets should declare one profile explicitly in a schema-v2 recipe instead
of relying on `mixed` exporter defaults.

`grip_the_cube_near_v1` intentionally reuses the same full-task phases and
prompt, but not the same grasp solver. The moving SO101 jaw rotates as it
closes, so the line between fully-open pads is not the authoritative near-range
alignment measurement. The near-range contract therefore:

1. solves both pads at the sampled cube-contact width,
2. checks the complete open-to-close floor-clearance sweep,
3. keeps the open cube center inside the jaw capture corridor, and
4. accepts the trajectory only when the realized jaw line at contact remains
   within the recipe's face-normal threshold.

`grip_the_cube_continuous_v1` preserves those solver-specific contracts rather
than averaging them. Radius routing is:

1. below 18 cm: near-contact only,
2. 18-20 cm: near-contact first, then mid fixed-jaw fallback,
3. above 20 through 22 cm: mid fixed-jaw first, then near-contact fallback,
4. above 22 cm: mid fixed-jaw only.

`trajectory_variant=auto` resolves to `direct_align` for an accepted
near-contact candidate and `standard` for an accepted mid fixed-jaw candidate.
The workspace probe and dataset exporter call the same candidate factory, so a
probe cannot silently validate a different solver path from the exporter.

## Separate Pick-and-Place Teacher

`scripts/export_so101_pickplace_teacher_rollouts_lerobot.py` implements a
separate pick-and-place lane:

```text
optional recovery -> approach -> settle -> close -> lift
  -> transport -> lower -> release -> settle after release
```

It supports home or near-gripper starts and joint or side-slide approaches.
This is a **supported but older separate task lane**, not the current
`grip_the_cube_v4_3` production path. Physical-valid exports must leave
`sticky_grasp` disabled; sticky attachment is a simulation shortcut, not a
valid grasp teacher.

## Legacy Paths

- `pick_cube` (`skill_mode=pick_cube`) uses the older generic staged pick
  teacher.
- `teacher_style=legacy` bypasses the staged skill implementations.

Keep these only for historical comparisons and targeted debugging. Do not use
them as the starting point for a new production dataset without an explicit
reason and a new versioned recipe.

## Inspection Gates

The trajectory name does not define data quality by itself. Production recipes
must explicitly select their gates. Current gate families include:

- jaw line versus contacted cube-face normal through the cube center
- camera2 top-contact visual alignment with constructive wrist-roll refinement
  and a pre-close/early-close probe
- minimum gripper-to-floor clearance across the close sweep
- initial target segmentation area in camera1 and/or camera2 at frame 0
- successful grasp, lift height, and terminal hold
- unique seeds, camera visibility, workspace distribution, and object-yaw
  distribution in the completion report

Available low-level close-alignment modes are `geometry_only`,
`preclose_and_early_trace`, and `strict_image_trace`. Their numeric thresholds
belong in the recipe and must not be inferred from this document.

## Known Limits

- The current fixed-jaw teacher is validated only inside recipe-declared
  feasible spawn catalogs. A cube being visible or kinematically reachable is
  not enough to guarantee a valid grasp trajectory.
- The continuous teacher's stable production cap is 30 cm from the physical
  shoulder-pan axis. A few 30.5 cm configurations pass and are retained as an
  outer-boundary canary; 31 cm and farther require a new trajectory or hardware
  contract rather than looser post-export filtering.
- Physical teacher success and the policy visibility contract are separate
  gates. The bridge probe physically succeeded 50/50, while 10 cases failed
  the stricter camera1-at-home visibility requirement. A production catalog
  must keep only placements that pass both its trajectory and declared camera
  gates.
- The five-episode canary remains useful only as a historical solver check. The
  production near-range corpus spans 10-18 cm and approximately -75 to +75
  degrees, with 200 unique successful positions and seeds.
- Production near-range generation first builds a seed-free placement catalog
  using range, distribution, spacing, and frame-0 segmentation. The exporter
  then applies the authoritative IK, alignment, floor-clearance, grasp, lift,
  and terminal-hold gates to every retained episode. Catalog preflight is not a
  substitute for exporter success validation.
- A correction-start dataset covers local recovery states; it does not replace
  home-start coverage.
- Old primitive datasets and historical documents may contain retired prompt
  wording. The prompt stored in each exported episode and its schema-v2 recipe
  are authoritative.

## Reproduction Workflow

Preview the recipe plan before writing data:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python \
  scripts/generate_so101_dataset_recipe.py \
  --recipe configs/so101/dataset_generation/grip_the_cube_v4_3.json \
  --split all --workers 3 --dry-run
```

Generate a new append-only version by copying the recipe to a new name/root and
then running the same command without `--dry-run`. Never overwrite an existing
SO101 dataset unless the user explicitly requests destructive replacement in
the current task.

Dataset generation is complete only after the recipe's audit, distribution
report, render-replay requirements, registry gate, and viewer gate pass:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/so101_dataset_registry.py validate --require-training-ready
```

Continuous-range evidence is stored under:

- `_workspace/so101_workspace_probes/grip_the_cube_continuous_v1_near_regression_probe/summary.json`
- `_workspace/so101_workspace_probes/grip_the_cube_continuous_v1_bridge_probe/summary.json`
- `_workspace/so101_workspace_probes/grip_the_cube_continuous_v1_outer_boundary_probe/summary.json`
- `_workspace/so101_workspace_probes/grip_the_cube_continuous_v1_outer_yaw_boundary_probe/summary.json`
- `_workspace/so101_lerobot/grip_the_cube_continuous_v1_canary/meta/distribution/distribution.html`

## Source-of-Truth Order

When two descriptions disagree, use this order:

1. generated episode report and audit for what is actually in a dataset
2. schema-v2 recipe for the intended dataset contract
3. exporter implementation for executable trajectory behavior
4. this catalog for selection guidance
5. historical dataset-contract documents for reproduction context only
