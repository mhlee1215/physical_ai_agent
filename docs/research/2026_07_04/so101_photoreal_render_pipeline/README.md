# SO101 Photoreal Render Preview Pipeline

This note records the optional high-fidelity SO101 render lane for dataset
generation. It is a sidecar preview path: the canonical LeRobot policy images
remain MuJoCo camera renders, while Blender/Cycles produces paper or inspection
renders under a separate `photoreal_preview/` directory.

## Example Output

Matte PLA material:

![SO101 matte PLA render](./matte_pla_example.png)

Material comparison:

![SO101 material comparison](./material_compare.png)

SO101 `pick_cube_train` five-episode start/grip preview:

![SO101 pick-cube five-episode photoreal preview](./so101_pick_cube_train5episodes/contact_sheet.png)

Procedural versus HDRI/PBR assets:

![SO101 procedural versus HDRI PBR](./procedural_vs_hdri_pbr.png)

MyCobot adaptive gripper matte PLA material:

![MyCobot adaptive gripper matte PLA render](./mycobot_matte_pla_example.png)

## One-Frame Render

Install Blender and fetch the optional CC0 assets:

```bash
brew install --cask blender
PYTHONPATH=src .venv/bin/python scripts/download_so101_photoreal_assets.py
```

Render one frame with Blender Cycles on Apple Metal:

```bash
PYTHONPATH=src .venv/bin/python scripts/render_so101_blender_probe.py \
  --output-dir _workspace/so101_blender_probe_matte_pla \
  --seed 7 \
  --warmup-steps 8 \
  --width 640 \
  --height 480 \
  --samples 512 \
  --denoise \
  --robot-material matte_pla
```

The measured local example on this Mac was:

- renderer: Blender Cycles
- acceleration: Metal, `Apple M5 Pro (GPU - 20 cores)`
- size: `640x480`
- samples: `512`
- denoise: enabled
- material: `matte_pla`
- render time: `7.85s` for the timed run

## Dataset Export Hook

Generate the normal dataset and add a preview sidecar:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_so101_training_datasets.py \
  --only move_and_align_cube_edge_train_v2 \
  --overwrite \
  --photoreal-preview \
  --photoreal-robot-material matte_pla \
  --photoreal-samples 512
```

The hook writes the preview under:

```text
<recipe root>/photoreal_preview/
```

This does not replace `observation.images.camera1/camera2/camera3` in the
LeRobot dataset. It is intended for dataset QA, paper figures, and visually
checking simulation states with more realistic lighting/materials.

## SO101 Dataset Episode Preview

Five episodes were rendered from the existing SO101 `pick_cube_train` LeRobot
training set:

```text
_workspace/so101_lerobot/pick_cube_train50_ego_wrist_256_seed98200
```

This is the recipe-backed home-start cube-grip training set:

- recipe: `pick_cube_train`
- export script: `scripts/export_so101_teacher_rollouts_lerobot.py`
- task: `pick_cube`
- seed: `98200`
- start mode: `home`
- teacher style: `staged`
- phase settings: `approach_steps=34`, `settle_steps=10`,
  `close_steps=42`, `lift_steps=58`

The preview uses episodes `0,1,2,3,4` and renders each episode's `start` frame
plus the computed `grip` frame. The renderer reads `observation.state` and
`action` from the parquet rows, resets the same high-contrast PickLift env
factory used by the dataset exporter, uses the actual per-episode seed recorded
in `so101_lerobot_export_report.json`, injects the robot qpos, and renders the
frame with Blender Cycles:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python scripts/render_so101_dataset_blender_preview.py \
  --dataset-root _workspace/so101_lerobot/pick_cube_train50_ego_wrist_256_seed98200 \
  --output-dir _workspace/so101_dataset_photoreal_pick_cube_train5episodes \
  --env-source high_contrast_picklift \
  --episodes 0,1,2,3,4 \
  --frames start,grip \
  --asset-root _workspace/photoreal_assets \
  --width 640 \
  --height 480 \
  --samples 192 \
  --denoise \
  --robot-material matte_pla \
  --camera-lens 35
```

This is a sidecar visual preview, not an in-place mutation of the LeRobot
dataset. The robot qpos/action are row-derived; object pose is recreated by
resetting the export-compatible env with the report-backed per-episode seed
because the LeRobot parquet rows do not store full object qpos.

## SO101 Photoreal Dataset

To make the rendered frames visible as an actual dataset, build a compact
photoreal JSONL dataset root:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python scripts/build_so101_photoreal_dataset.py \
  --source-dataset-root _workspace/so101_lerobot/pick_cube_train50_ego_wrist_256_seed98200 \
  --rendered-dir docs/research/2026_07_04/so101_photoreal_render_pipeline/so101_pick_cube_train5episodes \
  --output-root _workspace/so101_photoreal_datasets/pick_cube_train5episodes_start_grip_640_seed98200 \
  --overwrite
```

The generated dataset format is `so101_photoreal_jsonl_v1`. Its stored image
feature is `observation.images.camera1`, and that image is the photoreal render
itself. This differs from the sidecar comparison path above: the Data Viewer
discovers this root under `_workspace/so101_photoreal_datasets/` and lists it
as a separate `Photoreal datasets` entry.

The current compact dataset contains episodes `0..4` from `pick_cube_train`,
with `start` and `grip` frames for each episode. It is intended for visual
dataset QA and dashboard inspection. A full replacement training set would
require rendering every frame for the selected episodes.

## MyCobot Render

The same sidecar approach also works for the local MyCobot Nexus scene. MyCobot
visual evidence defaults to the adaptive gripper: `mycobot_ros2`,
`320-m5-2022-adaptive-gripper`, and the `adaptive-table` pose preset. Synthetic
or parallel-gripper renders should only be used for explicit legacy/debug
checks, not as the default MyCobot visual. The renderer exports MuJoCo mesh
geoms plus visible box primitives such as the cube and work mat, then
path-traces the static state in Blender:

```bash
PYTHONPATH=src .venv/bin/python scripts/render_mycobot_blender_probe.py \
  --official-gripper-root _workspace/_vendor/mycobot_ros2 \
  --render-asset-root _workspace/photoreal_assets \
  --output-dir _workspace/mycobot_blender_probe \
  --seed 7 \
  --warmup-steps 0 \
  --width 640 \
  --height 480 \
  --samples 256 \
  --denoise \
  --robot-material matte_pla
```

The measured local example on this Mac was:

- renderer: Blender Cycles
- acceleration: Metal, `Apple M5 Pro (GPU - 20 cores)`
- size: `640x480`
- samples: `256`
- denoise: enabled
- material: `matte_pla`
- model profile: `320-m5-2022-adaptive-gripper` with official `mycobot_ros2`
  adaptive gripper meshes
- pose preset: `adaptive-table`
- render time: `6.55s`

The adaptive asset/pose path was also checked with:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_mycobot_320_adaptive_visual_pose.py \
  --official-gripper-root _workspace/_vendor/mycobot_ros2
PYTHONPATH=src .venv/bin/python scripts/verify_mycobot_320_adaptive_mimic_motion.py \
  --official-gripper-root _workspace/_vendor/mycobot_ros2
```

## Assets

- HDRI: Poly Haven `studio_small_08`, CC0.
- Table PBR: ambientCG `Wood008`, CC0.
- Plastic normal/roughness source: ambientCG `Plastic013A`, CC0.

If assets are missing, the Blender probe still runs with procedural fallbacks,
but the HDRI/PBR output is more realistic.
