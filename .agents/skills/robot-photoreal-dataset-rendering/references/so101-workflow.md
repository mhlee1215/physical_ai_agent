# SO101 Photoreal Dataset Workflow

## Canonical Paths

- Source LeRobot roots: `_workspace/so101_lerobot/`
- Raw rendered frames: `_workspace/so101_photoreal_renders/`
- Training-ready derivatives: `_workspace/so101_photoreal_lerobot/`
- Dataset recipes: `configs/so101/dataset_generation/*.json`
- Material profiles: `configs/so101/render_profiles/*.json`
- Renderer: `scripts/render_so101_dataset_blender_preview.py`
- LeRobot image replacement: `scripts/build_so101_photoreal_lerobot_dataset.py`
- Viewer: `scripts/serve_so101_dataset_viewer.py`

Use the repo `.venv` with `PYTHONPATH=src:.:scripts`.

## Source Audit

Confirm the source has:

- `data/chunk-*/file-*.parquet` and complete LeRobot metadata;
- `so101_lerobot_export_report.json` with one episode entry per source episode;
- concrete task prompts and target object metadata;
- `observation.state` and `action` as the expected 6D joint contract;
- 256x256 `camera1` and `camera2`, plus the declared camera3 duplicate;
- valid episode frame counts and an audit report.

Never substitute an image-derived camera guess for source camera metadata.

## Deterministic Replay Preflight

Recreate the environment from report-backed factory arguments, not the current
factory defaults. Restore the full start snapshot, target slot, target geom,
active collision masks, lift baseline, and RNG state where available. Replay
all episode actions without Blender and compare state and terminal metrics.

Known `grip_the_cube_v2_5` status as audited on 2026-07-16:

- 300 episodes, 50,456 frames;
- every report entry contains `sim_snapshot`;
- generation used three green cube sizes from `teacher_timing`, while the
  current renderer default creates a nine-slot RGB pool;
- the current default model has `nq=69`, `nv=60`, but source snapshots have
  `qpos=27`, `qvel=24`;
- report-aware three-slot reconstruction makes the dimensions match;
- 296 snapshots replay exactly;
- episodes `2`, `41`, `96`, and `295` contain a lifted/final object pose instead
  of a valid frame-0 pose;
- three damaged seeds have older usable start evidence; episode `295`, seed
  `31010278`, still requires a reviewed frame-0 pose recovery.

Do not launch a full `v2_5` render until report-aware environment construction,
snapshot restoration, damaged-snapshot repair, and an all-episode preflight
gate are implemented and passing.

## Canary Render

Example command shape:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python \
  scripts/render_so101_dataset_blender_preview.py \
  --dataset-root <source-root> \
  --output-dir <append-only-render-root> \
  --env-source high_contrast_picklift \
  --episode 0 \
  --frames start,grip,final \
  --camera-keys observation.images.camera1,observation.images.camera2 \
  --width 256 --height 256 --samples 256 \
  --robot-material-config \
    configs/so101/render_profiles/black_arm_green_white_gripper.json \
  --scene-profile black_table_clutter \
  --asset-root _workspace/photoreal_assets \
  --blender-batch-size 6
```

Review camera1, camera2, target color, start/contact/final continuity, gripper
closure, lift, part materials, visual props, and temporal noise. The renderer
must reset once per episode and step source actions sequentially.

## Full Render

Use `--frames all` and the complete episode list. Keep camera3 as the exact
camera2 duplicate when declared. Use an append-only output name containing the
source id, render profile/version, samples, and denoise state. Use
`--skip-existing` only after validating the existing render report and files.

The full render is incomplete until every source episode/frame has camera1 and
camera2, camera3 duplication is complete, and the render report contains no
missing frame or replay failure.

## Build the LeRobot Derivative

```bash
PYTHONPATH=src:.:scripts .venv/bin/python \
  scripts/build_so101_photoreal_lerobot_dataset.py \
  --source-dataset-root <source-root> \
  --rendered-dir <render-root> \
  --output-root <new-versioned-derivative-root> \
  --repo-id <new-photoreal-repo-id> \
  --rewrite-color-task-prompts \
  --task-skill-mode <source-skill-mode>
```

This copies the source LeRobot root and replaces embedded camera image bytes.
It must preserve state, action, timestamps, episode boundaries, and approved
task semantics. Source and output roots must never be equal. The builder must
fail closed when any rendered frame or camera is missing.

## Training-Ready and Viewer Gates

Every derivative needs a declaration under
`configs/so101/dataset_generation/*.json`. Run:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/so101_dataset_registry.py validate --require-training-ready
```

If `scripts/so101_dataset_registry.py` is absent from the working branch, do
not replace this gate with an ad hoc directory check. Bring the canonical
registry tool into the branch or validate from a compatible checkout before
claiming `training_ready`.

Then refresh the existing launchctl-managed viewer, do not start another port:

```bash
sh scripts/launch_so101_dataset_viewer.sh restart
sh scripts/launch_so101_dataset_viewer.sh status
curl -fsS http://127.0.0.1:8769/api/datasets
curl -fsS 'http://127.0.0.1:8769/api/frame?split=<split>&episode=0&frame=0'
```

If the active configured viewer port differs, reuse that port. Inspect the
mobile layout and play at least one complete episode to confirm there is no
frame jump and the grasp/lift sequence exists.

## PR and Artifact Policy

Commit scripts, tests, recipes, material/scene configs, skill/docs, and small
representative images. Do not commit `_workspace` datasets or raw frame trees.
PR evidence must state whether the result is a canary, partial render, full
derivative, or training-ready dataset.
