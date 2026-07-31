# myCobot 280 SmolVLA Controlled Closed-Loop Pilot

**Date:** 2026-07-31 PDT

**Branch:** `codex/mycobot280-controlled-closed-loop-baseline`

**Readiness baseline:** `4938d7c Fix myCobot SmolVLA 7D feature adaptation`

## Question

Is it worth fine-tuning SmolVLA on the existing 50 fixed-physics teacher
episodes before the randomized dataset is ready?

This pilot tests whether the existing data and exact-7D training adapter can
produce any closed-loop task signal. It is a controlled engineering baseline,
not a publication-level comparison.

## Training

- Base policy: `lerobot/smolvla_base`
- Dataset: 50 episodes, 26,500 frames, 30 FPS
- Inputs: one RGB camera, the task prompt, and exact 7D robot state
- Actions: exact 7D arm-plus-gripper action
- Optimizer steps: 100
- Batch size: 1
- Learning rate: `1e-5`
- Seed: `20260731`
- Device: CUDA
- Duration: 215.612 seconds
- Checkpoint: reloadable weights, processors, optimizer state, training state,
  TensorBoard events, and the feature-contract marker

The per-step training loss is noisy: 0.7260 initially, 2.0952 on the final
sample, 0.0107 minimum, 0.3475 median, and 0.4578 over the last ten samples.
No convergence claim is made from this trace.

## Held-Out Supervised Check

Five matched validation batches gave mixed results:

| Policy | Mean loss | Mean action RMSE | First-step RMSE |
| --- | ---: | ---: | ---: |
| Base | 4.3320 | 0.10618 | 0.11807 |
| 100-step fine-tuned | 3.9365 | 0.11468 | 0.12083 |

The fine-tuned checkpoint has about 9.1% lower supervised loss but about 8.0%
higher mean action RMSE. This does not establish a generic imitation-learning
improvement.

## Closed-Loop Protocol

Both policies receive the same initial scene, camera, prompt, robot state,
fixed object physics, 530-step horizon, seeds, and yaw schedule. Policy inputs
exclude cube pose, contact metrics, and MuJoCo state. Teacher attachment and
object teleport are disabled.

Strict success requires:

- valid cube, pad, and gripper-to-mat clearance
- zero pad-cube contacts in the untouched pre-policy setup
- at least 50 mm final cube lift
- two-pad final contact
- at least 60 sustained two-pad lift steps
- 300 sustained two-pad post-lift steps
- at least 45 mm lift throughout the post-lift window
- at most 3.0 mm pad-cube penetration

## Matched Results

| Yaw | Policy | Final lift | Post-hold min | Max penetration | Strict result |
| ---: | --- | ---: | ---: | ---: | --- |
| -0.2 rad | Base | 41.57 mm | 29.99 mm | 3.853 mm | Fail: lift, hold height, penetration |
| -0.2 rad | Fine-tuned | 58.88 mm | 53.23 mm | 3.749 mm | Fail: penetration only |
| 0.0 rad | Base | 42.60 mm | 30.08 mm | 3.025 mm | Fail: lift, hold height, penetration |
| 0.0 rad | Fine-tuned | 58.78 mm | 57.20 mm | 3.102 mm | Fail: penetration only |
| +0.2 rad | Base | 31.36 mm | 22.79 mm | 2.812 mm | Fail: lift and hold height |
| +0.2 rad | Fine-tuned | 62.70 mm | 57.22 mm | 2.994 mm | **Pass** |

Aggregate pilot results:

- strict success: base 0/3; fine-tuned 1/3
- mean final lift: base 38.51 mm; fine-tuned 60.12 mm
- paired mean lift delta: +21.61 mm for fine-tuned
- mean maximum penetration: base 3.230 mm; fine-tuned 3.282 mm
- clipped action values: base 264; fine-tuned 133
- all three fine-tuned runs meet the lift, contact, and hold gates
- two fine-tuned runs miss only the unchanged penetration gate

## Verifier Correction

The first evaluator version reused the teacher verifier's first post-action
record as its no-pregrasp check. A learned policy can close during its first
action, so that record is not an untouched setup measurement.

The evaluator now records a separate pre-policy snapshot and derives success
from one complete list of explicit gates. A three-pose setup audit measured
zero initial contacts at yaw -0.2, 0.0, and +0.2 radians. The decisive +0.2
fine-tuned run was repeated with the corrected evaluator and reproduced the
same trajectory metrics as a strict pass with no failed gates.

The older two-endpoint report retains the pre-correction `verifier_failed`
label for that episode. The corrected single-seed rerun is authoritative.

## Local Evidence

- Training: `_workspace/mycobot280_training/ground_pickup_pose_diverse_v1_100step_seed20260731/`
- Base center: `_workspace/mycobot280_eval/controlled_base_1seed_20260731/`
- Fine-tuned center: `_workspace/mycobot280_eval/controlled_100step_1seed_20260731/`
- Base endpoints: `_workspace/mycobot280_eval/controlled_base_yaw_endpoints_2seed_20260731/`
- Fine-tuned endpoints: `_workspace/mycobot280_eval/controlled_100step_yaw_endpoints_2seed_20260731/`
- Setup audit: `_workspace/mycobot280_eval/controlled_setup_audit_3pose_20260731/`
- Corrected strict pass: `_workspace/mycobot280_eval/controlled_100step_seed91002_corrected_20260731/`

The training directory uses about 1.3 GB. Free host storage was about 4.6 GB
before the final evaluation-only runs, so another checkpoint sweep should wait
until more space is available.

## What This Proves

- The 50-episode dataset can drive a real SmolVLA optimizer run and reloadable
  exact-7D checkpoint.
- Base and fine-tuned policies can be compared end to end in policy-only
  closed-loop MuJoCo with contact, lift, hold, and penetration evidence.
- The 100-step checkpoint changes task behavior in a promising direction and
  can achieve at least one strict fixed-physics success.

## What This Does Not Prove

- statistically reliable superiority over the base model
- robustness to randomized cube pose, size, mass, friction, or camera changes
- generalization to unseen objects or tasks
- real-robot transfer
- convergence or optimality of the 100-step checkpoint
- benefit from an agentic verifier/retry wrapper

## Next Decision

Keep the 3.0 mm penetration threshold unchanged. Treat this as evidence that
the line of work is worth continuing, then move to the randomized dataset and
evaluate base versus multiple fine-tuned checkpoints on at least 20-30 matched
seeds. Report functional success and strict penetration-gated success
separately, with policy-only results established before agentic retries.
