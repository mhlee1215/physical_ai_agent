# SO101 Deterministic Teacher Generation

## Goal

Generate a requested SO101 grasp shape from explicit camera-bin and world-spawn
targets instead of repeatedly sampling random trajectories and accepting the
ones that happen to pass.

## Pipeline

1. Build a `camera1` egocentric world-XY lookup at a fixed grid resolution.
2. Calibrate the lookup with teacher-feasible candidates.
3. Run a small calibration export and record only full-trajectory-successful
   `(bin, world_xy, seed)` entries in a trajectory-feasible cache.
4. For production/preview export, consume those entries in fixed bin order.
5. Use the fixed-jaw IK target and a constructive roll solution for the target
   pose. The authoritative alignment gate is geometric: the line joining the
   two jaw pads must be parallel to the contacted cube-face normal translated
   through the cube center. The camera2 image measurement remains diagnostic,
   because projection and jaw occlusion can make it ambiguous during close.
   No random seed fallback is enabled in deterministic mode.

The implementation is in `scripts/export_so101_teacher_rollouts_lerobot.py`.
The cache builder is `scripts/build_so101_trajectory_feasible_lookup_cache.py`.

## Preview Evidence

The paired previews use the same cached entries for bins 5, 6, and 10:

- standard path: `_workspace/so101_lerobot/grip_the_cube_v1_5_deterministic_cached4_v2`
- roll-first path: `_workspace/so101_lerobot/grip_the_cube_v1_5_deterministic_cached_rollfirst4_v2`

For every paired bin, seed, forced world XY, and q_start are identical. Only
the intermediate phase ordering differs (`standard` vs `roll_first`). Both
preview sets have 3/3 grasp-and-lift successes, 342 frames, and 256x256
camera1/camera2 inputs. The maximum measured top-edge parallel error is
0.119 degrees in both sets. The deterministic cached run took about 6.95
seconds per exported episode with four requested bins and three successful
entries.

The failed fourth bin is retained as a calibration result rather than silently
replaced: it shows that a camera-bin candidate can pass the inexpensive
teacher-feasibility probe but fail the complete trajectory contract. That is
why the trajectory-feasible cache is required for the final deterministic
generation stage.

## Constructive Parallel Contract

The exporter records `cube_face_normal_parallel_error_deg` and uses it for
candidate filtering and episode success. It is an unoriented line angle, so a
180-degree reversal of either line is still parallel. The legacy
`finger_axis_parallel_angle_deg` field remains only as a compatibility alias
for older reports. Fixed-jaw IK constructs the pose around the cube-face
normal first; the trajectory-feasible cache then removes only candidates that
fail the full simulation contract.

## Verification

Targeted tests cover:

- strict vs early-contact camera2 close gate behavior,
- empty export audit without an HF network lookup,
- versioned camera1 lookup cache validation,
- preservation of calibrated trajectory-feasible seeds.
