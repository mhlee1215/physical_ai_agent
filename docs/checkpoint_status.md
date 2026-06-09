# Checkpoint Status

This document is the current checkpoint ledger through CP24. It separates
executable readiness, visual evidence, and benchmark claims so paper claims do
not overstate what each checkpoint proves.

## Status Legend

- `passed`: executable evidence exists and the checkpoint claim is supported.
- `partial`: core wiring exists, but the claim is narrower than the final paper
  benchmark claim.
- `blocked`: the expected gate did not execute because of an external or
  environment blocker.
- `planned`: not implemented yet.

## Checkpoints Through CP24

| Checkpoint | Status | Evidence | Claim Boundary |
| --- | --- | --- | --- |
| CP01 | passed/partial | `_workspace/checkpoints/checkpoint_01_*` | Mac-local scaffold and MuJoCo smoke are separate from full Linux LIBERO readiness. |
| CP02-04 | passed | `_workspace/checkpoints/checkpoint_02_04/` | Random baseline evaluator, metrics, traces, and summary are wired. |
| CP05-06 | passed/partial | `_workspace/checkpoints/checkpoint_05_06*/` | Policy adapter and SmolVLA import/readiness path; not task-quality SmolVLA behavior. |
| CP07-13 | passed | `_workspace/checkpoints/checkpoint_07_13/` | SO101-Nexus sim, dry SmolVLA bridge, demo dataset, and visualization wiring. |
| CP14-15 | passed | `_workspace/checkpoints/checkpoint_14_15/` | SO101 3D render plus pretrained `lerobot/smolvla_base` action execution on SO101. |
| CP16 | passed | `_workspace/checkpoints/checkpoint_16/` | SO101 single-camera policy-input preview. |
| CP17 | passed | `_workspace/checkpoints/checkpoint_17/` | SO101 multi-camera visual-input bundle. |
| CP18-19 | passed | `_workspace/checkpoints/checkpoint_18/`, `_workspace/checkpoints/checkpoint_19/` | SmolVLA receives real SO101 RGB frames, not zero-image tensors. |
| CP20 | passed | `sh scripts/checkpoint_20.sh` | Rule-based SO101 planner path. |
| CP21 | passed | `sh scripts/checkpoint_21.sh` | Simulation-state verifier path. |
| CP22 | passed | `sh scripts/checkpoint_22.sh` | Retry loop trace with verifier failure/retry event. |
| CP23 | passed | `sh scripts/checkpoint_23.sh` | First `policy_only` vs `agentic_retry` comparison report. |
| CP24 | partial/pass | `_workspace/checkpoints/checkpoint_24_*` | ManiSkill / ManiSkill-HAB evaluation plumbing, baselines, and real-image SmolVLA bridge. This is not yet a strong task-success result. |
| CP24B | passed | `_workspace/runpod_results/libero_mujoco_broad_diverse_oracle_20260607T013257Z_figure_fixed/` | LIBERO real-sim oracle affordance overlay readiness with curated, visually inspected, paper-facing figure orientation. |

## CP24 Evidence Summary

CP24 currently has two lanes:

1. ManiSkill / ManiSkill-HAB evaluation lane.
2. LIBERO oracle affordance overlay readiness lane.

The ManiSkill lane has executable evidence that `PickCube-v1` can run on the
Mac with real-image SmolVLA input bridging:

- `_workspace/checkpoints/checkpoint_24_pickcube_smolvla_real_images_1ep_1step/checkpoint_report.json`
- `_workspace/checkpoints/checkpoint_24_pickcube_smolvla_real_images_10ep_50step/checkpoint_report.json`

The 10-episode real-image probe reports:

- `executed_env_id=PickCube-v1`
- `real_images=true`
- `smolvla_ready=true`
- `rollout_steps=1000`
- `rollout_success_count=0`

Interpretation: the bridge and rollout plumbing pass, but the policy does not
yet show task success on this probe. Do not use this as a performance
improvement claim.

The LIBERO oracle overlay lane has stronger visual evidence for policy-input
readiness:

- Corrected contact sheet:
  `_workspace/runpod_results/libero_mujoco_broad_diverse_oracle_20260607T013257Z_figure_fixed/libero_mujoco_oracle_curated_diverse_contact_sheet.jpg`
- Corrected manifest:
  `_workspace/runpod_results/libero_mujoco_broad_diverse_oracle_20260607T013257Z_figure_fixed/libero_mujoco_oracle_curated_manifest.json`
- Corrected report:
  `_workspace/runpod_results/libero_mujoco_broad_diverse_oracle_20260607T013257Z_figure_fixed/verification_report_final.md`

This evidence contains 13 curated real LIBERO/MuJoCo samples selected from a
19-sample broad pool. It includes semantic object masks and drawer-handle
targets from raw MuJoCo segmentation. The paper-facing visualization applies a
vertical flip and transforms overlay points as `y_fixed = H - 1 - y_native`.
Native simulator point provenance is retained in the manifest.

## Image Orientation Gate

Any future paper-facing image evidence must pass an explicit orientation gate:

- Compare the figure against official LIBERO figure orientation, not only
  against raw observation tensor shape.
- Do not decide orientation from text labels alone.
- Check robot/table/object layout semantics visually.
- Preserve native simulator coordinates separately from displayed coordinates
  when any visualization transform is applied.

## Next Checkpoint

The next recommended checkpoint is CP24C:

> SmolVLA LIBERO Overlay Ablation Probe

Required comparison:

- `smolvla_rgb_only`
- `smolvla_rgb_oracle_point`
- optionally `smolvla_rgb_oracle_point_agentic_retry`

CP24C should reuse the CP24B overlay generator but measure whether the overlay
changes policy behavior or success. Until CP24C runs, CP24B should be described
only as policy-input readiness evidence.

## CP24 policy-input visualization report

- Status: completed
- Scope: policy-input readiness visualization from fixed real LIBERO/MuJoCo oracle overlay evidence; this is not a success-rate or behavior-improvement benchmark.
- Source evidence: `_workspace/runpod_results/libero_mujoco_broad_diverse_oracle_20260607T013257Z_figure_fixed/libero_mujoco_oracle_curated_manifest.json`
- Output directory: `_workspace/checkpoints/checkpoint_24_libero_oracle_policy_input_report/`
- Contact sheet: `_workspace/checkpoints/checkpoint_24_libero_oracle_policy_input_report/cp24_libero_oracle_policy_input_contact_sheet.jpg`
- Markdown report: `_workspace/checkpoints/checkpoint_24_libero_oracle_policy_input_report/cp24_libero_oracle_policy_input_report.md`
- HTML report: `_workspace/checkpoints/checkpoint_24_libero_oracle_policy_input_report/cp24_libero_oracle_policy_input_report.html`
- Manifest: `_workspace/checkpoints/checkpoint_24_libero_oracle_policy_input_report/cp24_libero_oracle_policy_input_manifest.json`
- Visual QA note: contact sheet was inspected after generation; orientation follows the fixed LIBERO figure convention, and oracle points remain aligned across raw RGB, overlay, and SmolVLA-style 224x224 tensor previews.
