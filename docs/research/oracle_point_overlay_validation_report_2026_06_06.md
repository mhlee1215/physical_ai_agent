# Oracle Point Overlay Validation Report - 2026-06-06

## Summary

- Goal: validate oracle point projection, overlay rendering, and visualization on existing simulation frames.
- Probe pod: `ixtiucz8wr0ock`
- Pod GPU: `NVIDIA RTX 4000 Ada Generation`
- Pod cost: `$0.26/hr`
- Baseline experiment safety: existing LIBERO baseline pod/repo was not touched.
- Result: pass for the implemented lightweight overlay path.

## Validation Results

| Section | Environment | Cases | Result |
| --- | --- | ---: | --- |
| Oracle point projection | RunPod probe pod | 3 | 3/3 pass |
| Overlay rendering | RunPod probe pod | 4 | 4/4 pass |
| Oracle point projection | Mac local HTML milestone | 13 | 13/13 pass |
| Oracle point trajectory | Mac local trajectory milestone | 20 | 20/20 pass |
| Simulator-style pose dict trajectory | Mac local pose-dict milestone | 20 | 20/20 pass |
| Sensor-data camera parameter trajectory | Mac local sensor-camera milestone | 20 | 20/20 pass |
| Multi-camera preferred/fallback trajectory | Mac local multi-camera milestone | 20 | 20/20 pass |
| Projection edge-case fallback safety | Mac local edge-case milestone | 20 | 20/20 pass |
| Raw-vs-oracle-overlay input comparison | Mac local comparison milestone | 30 | 30/30 generated |
| Zoomed raw-vs-overlay representative figures | Mac local zoom milestone | 24 | generated |
| Diverse non-center object projection | Mac local diverse-object milestone | 30 | 30/30 pass |
| Center-bias audit for diverse objects | Mac local center-bias milestone | 30 | pass |
| Paper-facing figure pack | Mac local figure-pack milestone | 9 sections | generated |
| Overlay rendering | Mac local | 4 | 4/4 pass |
| Existing sim frame overlay | Mac local | 20 | 20/20 pass |

## Artifact Paths

- RunPod fetched report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/pod_validation/validation_report.md`
- RunPod fetched JSON: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/pod_validation/validation_report.json`
- Local full report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_full_validation/validation_report.md`
- Local full JSON: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_full_validation/validation_report.json`
- Local HTML milestone report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_html_validation/validation_report.html`
- Local trajectory milestone report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_trajectory_validation/validation_report.html`
- Milestone index HTML: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/milestone_index.html`
- Audited milestone dashboard: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/milestone_dashboard.html`
- Projection contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_html_validation/contact_sheet_projection.png`
- Trajectory contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_trajectory_validation/contact_sheet_projection_trajectory.png`
- Trajectory GIF: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_trajectory_validation/projection_trajectory.gif`
- Pose-dict milestone report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_pose_dict_validation/validation_report.html`
- Pose-dict contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_pose_dict_validation/contact_sheet_dict_pose_trajectory.png`
- Pose-dict GIF: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_pose_dict_validation/dict_pose_trajectory.gif`
- Sensor-data camera milestone report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_sensor_camera_validation/validation_report.html`
- Sensor-data camera contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_sensor_camera_validation/contact_sheet_sensor_data_camera_trajectory.png`
- Sensor-data camera GIF: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_sensor_camera_validation/sensor_data_camera_trajectory.gif`
- Multi-camera milestone report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_multi_camera_validation/validation_report.html`
- Multi-camera contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_multi_camera_validation/contact_sheet_multi_camera_trajectory.png`
- Multi-camera GIF: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_multi_camera_validation/multi_camera_trajectory.gif`
- Edge-case milestone report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_edge_case_validation/validation_report.html`
- Edge-case contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_edge_case_validation/contact_sheet_projection_edge_cases.png`
- Edge-case GIF: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_edge_case_validation/projection_edge_cases.gif`
- Raw-vs-overlay comparison report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/raw_vs_overlay_comparison/comparison_report.html`
- Raw-vs-overlay contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/raw_vs_overlay_comparison/comparison_contact_sheet.png`
- Raw-vs-overlay manifest: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/raw_vs_overlay_comparison/comparison_manifest.json`
- Zoomed raw-vs-overlay report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/zoomed_raw_vs_overlay/zoom_report.html`
- Zoomed raw-vs-overlay contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/zoomed_raw_vs_overlay/zoom_contact_sheet.png`
- Zoomed raw-vs-overlay manifest: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/zoomed_raw_vs_overlay/zoom_manifest.json`
- Diverse object projection report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/diverse_object_projection/diverse_object_report.html`
- Diverse object full-frame contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/diverse_object_projection/diverse_object_full_contact_sheet.png`
- Diverse object zoom contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/diverse_object_projection/diverse_object_zoom_contact_sheet.png`
- Diverse object manifest: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/diverse_object_projection/diverse_object_manifest.json`
- Center-bias audit report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/center_bias_audit/center_bias_audit.html`
- Center-bias distance distribution: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/center_bias_audit/center_distance_distribution.png`
- Center-bias object distribution: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/center_bias_audit/object_distribution.png`
- Center-bias audit JSON: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/center_bias_audit/center_bias_audit.json`
- Paper-facing figure pack: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/paper_figure_pack/figure_pack.html`
- Paper-facing figure pack manifest: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/paper_figure_pack/figure_pack_manifest.json`
- Reusable gallery builder proof: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/gallery_builder_probe/overlay_gallery.html`
- Shared gallery module proof: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/gallery_module_probe/overlay_gallery.html`
- Shared gallery module contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/gallery_module_probe/contact_sheet.png`
- Rendering contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_html_validation/contact_sheet_rendering.png`
- Sim-frame contact sheet: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_html_validation/contact_sheet_sim_frame_overlay.png`
- Live RunPod blocker report: `_workspace/runpod_results/live_oracle_probe_20260606T2008Z/maniskill_blocker.md`
- Live RunPod checkpoint report: `_workspace/runpod_results/live_oracle_probe_20260606T2008Z/checkpoint_report.json`
- Live oracle audit report: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/live_audit_blocked_probe/live_oracle_audit.html`
- Live oracle audit JSON: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/live_audit_blocked_probe/live_oracle_audit.json`
- Representative projected point image: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_full_validation/projection/projected_right_up.png`
- Representative sim overlay image: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/local_full_validation/sim_frames/000_random_episode_000_reset_0000_overlay.png`

## Iteration Notes

Two validation-harness issues were found and fixed during the run:

- The overlay builder returns `(overlay_images, OracleAffordanceOverlay)`, not a single metadata object.
- The current oracle projection implementation reads camera intrinsics/extrinsics from `sensor_param`, and synthetic pose validation should pass `obj_pose` as a flat pose vector.

After these fixes, projection moved from image-center fallback to true `projected_object_pose` mode in synthetic projected cases.

The validation script was then upgraded to generate milestone review artifacts:

- 12 diverse projected-point scenes plus 1 no-pose fallback scene.
- 4 overlay rendering stress cases.
- 20 existing sim-frame overlay samples.
- Section-level contact sheets.
- A self-contained HTML report for visual review.

## Safety Constraint

Uploading local simulation PNG artifacts to the external RunPod host was blocked by the approval policy because it would export workspace-generated image artifacts to a third-party host. To keep the validation safe:

- RunPod validated synthetic projection and rendering samples.
- Mac local validated existing sim frame overlays using the already available local rollout images.

This preserves the baseline pod and avoids exporting local sim image artifacts.

## Interpretation

The implemented overlay path is ready for the next integration step:

- `projected_object_pose` works when object pose and camera parameters are available.
- fallback center overlay works for image-only paths.
- overlay rendering is visible on dark, bright, small, wide, and real sim-frame images.

The current sim-frame overlay validation uses center fallback because saved PNG files do not carry simulator pose/camera metadata. True oracle projection on real rollout frames should be validated inside the live ManiSkill observation loop when a pod exposes working GPU Vulkan/SAPIEN rendering.

## Current Milestone Status

This milestone is complete for static overlay validation:

- `37/37` cases passed in the local HTML milestone run.
- `57/57` cases passed in the local trajectory milestone run.
- `77/77` cases passed in the local pose-dict milestone run.
- `97/97` cases passed in the local sensor-data camera milestone run.
- `117/117` cases passed in the local multi-camera milestone run.
- `137/137` cases passed in the local edge-case milestone run.
- `30` raw-vs-oracle-overlay side-by-side pairs were generated.
- `24` marker-centered zoomed raw-vs-overlay panels were generated after the first representative contact sheet was judged too small for visual review.
- `30/30` diverse-object episodes passed, with `28` explicitly non-center target locations.
- Paper-facing figure pack generated with pass/limited/blocked claim labels.
- At least 10 representative projection samples were generated.
- 20 trajectory samples were generated, with projected oracle points matching expected synthetic object centers.
- 20 simulator-style dict-pose samples were generated, alternating `{p: xyz}` and `{position: xyz}` pose schemas.
- 20 sensor-data camera parameter samples were generated, where intrinsics/extrinsics live under `sensor_data[cam]`.
- 20 multi-camera samples were generated, alternating preferred `base_camera` and fallback `aux_camera` selection.
- 20 projection edge-case samples were generated, covering behind-camera, out-of-frame, missing camera parameters, malformed matrices, and invalid pose fallback.
- 30 baseline-vs-overlay comparison samples were generated: 20 synthetic true projection pairs and 10 saved sim-frame fallback pairs.
- 24 enlarged marker-centered panels make the overlay visible enough for human review and paper/slide drafting.
- 30 diverse-object full-frame episodes show different target objects, distractors, and off-center positions; this addresses the concern that previous figures looked like center markers.
- Figure pack separates supported claims from limited saved-frame evidence and blocked live evidence.
- The paper-facing figure pack now uses diverse non-center object projection as the primary representative evidence, replacing earlier center-looking contact sheets as the preferred review figure.
- At least 10 existing sim-frame overlay samples were generated.
- HTML and contact-sheet artifacts were generated for visual review.

This milestone is not yet complete for paper-quality live simulator evidence:

- The sim-frame section still proves rendering on saved frames, not true pose-aware projection.
- The next required milestone is a live ManiSkill observation capture where RGB, object pose, and camera parameters are saved together before overlay rendering.

## Live ManiSkill Attempt

A separate RunPod live-capture pod was created and stopped:

- Pod id: `ixrv40nim222a7`
- GPU: `NVIDIA L4`
- Cost: `$0.39/hr`
- Status after attempt: `EXITED`
- Command intent: run `affordance_oracle_probe` for at least 10 live frames.

Result: blocked before frame capture.

The blocker was:

```text
RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver
svulkan2: Your GPU driver does not support Vulkan. You may not use the renderer for rendering.
```

This happened for both `PickCube-v1` and fallback `Empty-v1`, so the failure is renderer/driver-level, not task-specific.

Strict live-output audit on the fetched blocker artifact:

```text
status: blocked
frame_count: 0
```

This confirms the live gate does not false-pass when no live overlay frames are captured.

## RunPod Reuse Rule

RunPod operations for this line of work now use a reuse-only default:

- Reuse existing stopped probe pods first.
- Do not create a new pod automatically after a restart or availability failure.
- If existing pods cannot restart, stop and ask before creating any new pod.
- Never touch the active baseline pod/repo for visualization probes.
- Always stop any reused probe pod after the probe attempt.

The reuse-only runner is:

```bash
RUNPOD_SSH='root@host' RUNPOD_SSH_PORT='44000' \
  sh scripts/runpod_reuse_live_oracle_probe.sh
```

Runner constraints:

- It never creates a pod.
- It never starts a stopped pod.
- It never uploads local workspace files to RunPod.
- It only runs against an already-running reused pod and existing remote clone.
- It fetches generated artifacts back to a local `_workspace/runpod_results/...` directory.

Local smoke evidence:

```text
sh scripts/runpod_reuse_live_oracle_probe.sh --help
# exit 0

sh scripts/runpod_reuse_live_oracle_probe.sh
# exit 2
# RUNPOD_SSH is required. Refusing to create or start a Pod.
```

## Live Gallery Builder

The reusable postprocessor is available as both a Python module and a CLI wrapper:

- Module: `physical_ai_agent.perception.overlay_gallery.build_overlay_gallery`
- CLI: `scripts/build_oracle_overlay_gallery.py`

```bash
PYTHONPATH=src .venv/bin/python -B scripts/build_oracle_overlay_gallery.py \
  --image-root <live_output>/affordance_oracle_probe_frames \
  --output-dir <live_output>/affordance_oracle_probe_gallery \
  --title "Live ManiSkill Oracle Overlay Probe" \
  --min-frames 10 \
  --limit 40
```

Expected live milestone artifacts:

- `<live_output>/affordance_oracle_probe_gallery/overlay_gallery.html`
- `<live_output>/affordance_oracle_probe_gallery/contact_sheet.png`
- `<live_output>/affordance_oracle_probe_gallery/overlay_sequence.gif`
- `<live_output>/affordance_oracle_probe_gallery/overlay_gallery_manifest.json`

The gallery builder was validated locally on 20 trajectory frames.

CP24 now calls the same module automatically for oracle overlay policies:

- `smolvla_affordance_oracle` writes `smolvla_affordance_oracle_gallery/`.
- `affordance_oracle_probe` writes `affordance_oracle_probe_gallery/`.
- If fewer than 10 frames exist, the gallery manifest records `status: skipped` rather than failing short debug runs.

## Milestone Dashboard

The audited dashboard builder is:

```bash
PYTHONPATH=src .venv/bin/python -B scripts/build_oracle_overlay_milestone_dashboard.py \
  --root _workspace/runpod_results/affordance_overlay_validation_20260606T1955Z \
  --live-root _workspace/runpod_results/live_oracle_probe_20260606T2008Z \
  --output _workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/milestone_dashboard.html
```

Current audited status:

- Static validation: `PASS`
- Trajectory validation: `PASS`
- Gallery postprocessor: `PASS`
- Simulator-style pose dict projection: `PASS`
- Sensor-data camera parameter projection: `PASS`
- Multi-camera preferred/fallback projection: `PASS`
- Projection edge-case fallback safety: `PASS`
- Raw-vs-oracle-overlay input comparison: `PASS`
- Zoomed raw-vs-overlay representative figures: `PASS`
- Diverse non-center object projection: `PASS`
- Center-bias audit for diverse objects: `PASS`
- Paper-facing figure pack: `PASS`
- Live ManiSkill true oracle projection: `BLOCKED`

The dashboard now audits the paper-facing figure pack as a first-class milestone:

- Figure pack HTML exists.
- Figure pack manifest exists.
- At least 9 claim-labeled evidence sections are present.
- Static supported claims, limited saved-frame evidence, and blocked live evidence remain separated.

The dashboard now also audits diverse non-center object projection:

- At least 30 episodes.
- At least 20 target objects whose projected center is far from the image center.
- Full-frame and zoom contact sheets.
- Manifest status `passed`.

The dashboard now also audits center-bias explicitly:

- `28/30` target objects are non-center by the 45px threshold.
- Object categories are balanced at 5 samples each.
- Max projection error is <= 1px.

## Preferred Representative Figures

Use these first when reviewing or drafting figures:

- Primary full-frame evidence: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/diverse_object_projection/diverse_object_full_contact_sheet.png`
- Primary zoom evidence: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/diverse_object_projection/diverse_object_zoom_contact_sheet.png`
- Paper figure pack: `_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/paper_figure_pack/figure_pack.html`

Earlier center-looking contact sheets remain useful as regression/debug artifacts, but should not be used as the main paper-facing representative figure.

The paper figure pack is reproducible with:

```bash
PYTHONPATH=src .venv/bin/python -B scripts/build_oracle_overlay_figure_pack.py \
  --root _workspace/runpod_results/affordance_overlay_validation_20260606T1955Z \
  --output-dir _workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/paper_figure_pack
```

## Canonical Local Rebuild Command

Use this command to regenerate all non-RunPod overlay milestones, HTML reports, contact sheets, figure artifacts, live blocker audit, and the milestone dashboard:

```bash
sh scripts/rebuild_oracle_overlay_milestones.sh
```

Latest rebuild evidence:

```text
oracle_overlay_milestones_rebuilt=true
dashboard=_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/milestone_dashboard.html
preferred_full_frame=_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/diverse_object_projection/diverse_object_full_contact_sheet.png
preferred_zoom=_workspace/runpod_results/affordance_overlay_validation_20260606T1955Z/diverse_object_projection/diverse_object_zoom_contact_sheet.png
```

The rebuild now also regenerates:

- center-bias audit
- paper-facing figure pack
- live-output blocker audit

The rebuild command does not create, start, stop, or contact RunPod pods.

## Regression Test Evidence

Targeted regression command:

```bash
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_checkpoint_24.py
```

Result:

```text
Ran 15 tests in 11.410s
OK
```

Coverage added for oracle overlay regressions:

- flat pose projection and image writing
- simulator-style pose dict parsing
- camera intrinsics/extrinsics under `sensor_data[cam]`
- multi-camera preferred/fallback selection
- invalid projection fallback
- SmolVLA batch construction with oracle overlay images
- CP24 import/signature compatibility after auto-gallery wiring

The test run still emits local ManiSkill/Vulkan warnings on macOS, but these warnings did not fail the overlay regression path.
