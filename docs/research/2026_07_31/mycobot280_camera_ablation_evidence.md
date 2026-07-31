# myCobot 280 SmolVLA Matched-Camera Ablation

**Date:** 2026-07-31 PDT

**Branch:** `codex/mycobot280-camera-ablation`

**Parent evidence:** `docs/research/2026_07_31/mycobot280_controlled_closed_loop_pilot.md`

## Question

Was the historical deterministic dataset's wide camera a meaningful defect, and
does a deterministic close-camera control improve the 100-step SmolVLA result?

The answer is nuanced:

- the wide framing is a real observation-contract defect because the cube is
  only a few pixels across;
- close framing alone does not improve the unfine-tuned base model;
- matched 100-step fine-tuning changes behavior strongly under both cameras;
- the close-camera checkpoint reaches 5/11 strict successes versus 3/11 for the
  wide-camera checkpoint on identical seeds;
- one training seed is not enough to claim that close-camera training is
  generally superior.

![Matched camera ablation](mycobot280_camera_ablation.png)

## Camera Contract

Both conditions use 256 x 256 RGB. Resolution was not the defect.

The legacy `full_robot` camera places the robot and cube in a much wider scene.
The generic phase-balanced audit found the red cube at only about 0.02-0.07% of
the image and failed the target-visibility gate in every phase.

The new `ground_pickup_closeup` camera is frozen as:

- mode: MuJoCo free camera
- target: initial cube position plus 35 mm in Z
- distance: 0.24 m
- azimuth: 215 degrees
- elevation: -10 degrees
- resolution: 256 x 256

Across 900 phase-balanced samples from 60 episodes, the cube occupied about
3.0-3.7% of each close-camera frame and was visible in 100% of samples.

## Matched Deterministic Control

The close-camera source is a camera-only regeneration of the deterministic
control:

- 50 training episodes and 10 held-out validation episodes
- 31,800 all-frame observations
- fixed object physics
- 60/60 successful teacher episodes
- zero rejected attempts
- 60/60 exact matches for attempt index, seed, yaw, initial cube pose, and
  trajectory hash against the wide source
- no teacher attachment
- no pickup/lift object teleport
- maximum pad-cube penetration: 2.826 mm
- minimum final lift: 54.136 mm
- minimum post-hold lift: 45.237 mm
- 300-step two-pad post-lift hold

Lossless PNG reduced the source from about 6.1 GB of BMP files to 1.2 GB. The
intermediate train/validation adapters hard-link the source images. A sampled
source/intermediate pair had the same device and inode, so no second physical
image copy was written.

Native LeRobot outputs:

| Split | Episodes | Frames | Size |
| --- | ---: | ---: | ---: |
| Train | 50 | 26,500 | 742 MB |
| Validation | 10 | 5,300 | 150 MB |

Both native splits preserve one 256 x 256 camera and exact 7D state/action
features. The wide and close native train statistics for state and action are
exactly equal, so the behavioral comparison does not contain a non-image
normalization difference.

## Training

The close-camera checkpoint used the recipe already used for the wide-camera
checkpoint:

- base policy: `lerobot/smolvla_base`
- optimizer steps: 100
- batch size: 1
- learning rate: `1e-5`
- seed: `20260731`
- device: CUDA
- duration: 127.65 seconds
- local cached weights only
- exact 7D state/action processor
- constant-joint normalization safeguard enabled

The checkpoint contains 907 MB of model weights, 413 MB of optimizer state,
pre/postprocessor configs, exact-7D feature marker, training state, train log,
and TensorBoard events. Total checkpoint size is about 1.3 GB.

The training loss is noisy: 0.674 initially, 2.011 on the final sampled batch,
0.012 minimum, 0.378 median, and 0.432 over the final ten batches. No convergence
claim is made from this trace.

## Held-Out Supervised Check

Both policies were evaluated on the same first 20 held-out validation batches.
Crucially, both used train-split normalization statistics.

| Policy | Mean loss | Mean action RMSE | Step-0 RMSE |
| --- | ---: | ---: | ---: |
| Base | 5.8279 | 0.10309 | 0.11636 |
| Close-camera 100-step | 4.9720 | 0.10889 | 0.12277 |

Fine-tuning reduced mean forward loss by about 14.7% but increased
postprocessed action RMSE by about 5.6%. This mixed diagnostic does not predict
task success by itself.

## Closed-Loop Protocol

The 2 x 2 matrix crosses policy weights with the matching observation camera:

1. base policy plus wide camera;
2. wide-camera 100-step checkpoint plus wide camera;
3. base policy plus close camera;
4. close-camera 100-step checkpoint plus close camera.

Every row uses:

- fixed object physics
- seeds 91000 through 91010
- 11 yaws from -0.20 to +0.20 radians in 0.04-radian increments
- matched Torch seeds
- 530 policy-controlled simulation steps
- exact 7D state/action
- the same prompt, initial scene construction, gravity schedule, and strict
  contact/lift/hold/penetration verifier
- no cube pose, contact metrics, or MuJoCo state as policy inputs
- no teacher attachment or object teleport

This is 44 episodes and 23,320 policy-controlled simulation steps.

## Results

| Policy and camera | Strict success | Mean final lift | Mean post-hold minimum | Mean max penetration | Mean clipped values |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base + wide | 0/11 | 40.34 mm | 26.96 mm | 3.117 mm | 63.8 |
| Fine-tuned + wide | 3/11 | 60.55 mm | 54.74 mm | 3.142 mm | 34.4 |
| Base + close | 0/11 | 26.22 mm | 20.28 mm | 3.181 mm | 78.6 |
| Fine-tuned + close | 5/11 | 62.22 mm | 55.52 mm | 3.157 mm | 43.5 |

Paired results:

- fine-tuning under the wide camera increases mean lift by 20.21 mm, improves
  all 11 paired seeds, and adds three strict successes;
- fine-tuning under the close camera increases mean lift by 36.00 mm, improves
  all 11 paired seeds, and adds five strict successes;
- changing only the base policy's camera from wide to close decreases mean lift
  by 14.12 mm and improves only one of 11 paired lifts;
- among fine-tuned policies, close camera increases mean lift by 1.67 mm in
  aggregate, improves nine of 11 paired lifts, and adds strict successes at
  seeds 91005 and 91008 without losing a wide-camera success.

All 11 fine-tuned-wide and all 11 fine-tuned-close episodes meet the functional
lift, final contact, sustained two-pad lift, and 300-step hold gates. The
remaining strict failures are solely due to maximum pad-cube penetration above
the unchanged 3.0 mm cap:

- fine-tuned wide: 3 passes, 8 penetration-only failures;
- fine-tuned close: 5 passes, 6 penetration-only failures.

The penetration cap remains unchanged. Future reports should show both
functional success and strict penetration-gated success rather than silently
relaxing the standard.

## What This Proves

- The legacy deterministic observation was framed too widely for reliable
  object visibility; the defect is camera framing, not 256 x 256 resolution.
- A deterministic close-camera control can be generated without changing
  trajectories, object physics, split sizes, or success criteria.
- Camera provenance survives source, intermediate, and native LeRobot
  conversion.
- A reloadable exact-7D SmolVLA checkpoint can be trained on the close-camera
  native dataset.
- Under one fixed 100-step training seed, fine-tuning changes closed-loop
  behavior strongly under both camera contracts.
- The close-camera checkpoint reaches 5/11 strict successes and all 11
  functional lift/contact/hold successes on the matched sweep.

## What This Does Not Prove

- a statistically reliable close-camera advantage over wide-camera training;
- convergence or optimality at 100 optimizer steps;
- robustness to randomized pose, mass, size, friction, or camera perturbation;
- transfer to the real myCobot 280;
- benefit from agentic verification or retry;
- superiority over SO101, LIBERO, or another policy;
- a publication-level effect estimate from one training seed.

## Randomized-Task Handoff

The randomized-source task owns the existing 50-train/10-validation randomized
corpus and its source-level QA. Its camera contract matches this close-camera
control. The randomized task reported 31,800 rows, 600 strict camera samples,
2.571% minimum sampled cube occupancy, zero border-touch samples, 13 rejection
records, and zero train/validation overlap.

This branch does not regenerate, convert, or train on that randomized corpus.
The next fair randomized experiment should consume the existing audited source
and compare deterministic-close versus randomized-close policies under the same
exact-7D and evaluator contracts.

## Evidence Paths

Committed:

- `configs/mycobot280/training_datasets/ground_pickup_pose_diverse_closecam_v1.json`
- `scripts/audit_mycobot_280_camera_contract.py`
- `scripts/summarize_mycobot280_camera_ablation.py`
- `docs/research/2026_07_31/mycobot280_camera_ablation.png`

Local generated artifacts:

- source: `_workspace/mycobot_teacher_datasets/mycobot_280_ground_pickup_pose_diverse_closecam_v1_50train_10val_png_20260731/`
- camera audit: `_workspace/mycobot280_runs/pose_diverse_closecam_v1_20260731/camera_contract_audit.json`
- native train/validation: `_workspace/mycobot280_lerobot/ground_pickup_pose_diverse_closecam_v1_*_native/`
- checkpoint: `_workspace/mycobot280_training/ground_pickup_pose_diverse_closecam_v1_100step_seed20260731/`
- four evaluation rows: `_workspace/mycobot280_eval/camera_ablation_20260731/`
- machine-readable summary: `_workspace/mycobot280_runs/pose_diverse_closecam_v1_20260731/camera_ablation_summary.json`

## Recommended Next Step

Open the camera-ablation PR after review. Then use the existing randomized
close-camera corpus without regenerating source data.

For publication-quality evidence, repeat the key deterministic-close and
randomized-close training conditions with at least three training seeds and
evaluate each checkpoint on 20-30 matched environment seeds. Keep the strict
3.0 mm penetration result and functional success as separate reported metrics.
