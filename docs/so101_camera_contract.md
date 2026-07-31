# SO101 Camera Contract

## Official Reference

The official LeRobot dataset `lerobot/svla_so100_pickplace` exposes policy image
features as:

- `observation.images.top`
- `observation.images.wrist`

Cached SmolVLA SO101 preprocessing artifacts on this workstation record this
rename map:

```json
{
  "observation.images.top": "observation.images.camera1",
  "observation.images.wrist": "observation.images.camera2"
}
```

## Local SO101 Pick-Place Export Contract

For SO101 pick-place datasets generated in this repo, use:

- `observation.images.camera1` = `egocentric_cam`
- `observation.images.camera2` = `wrist_cam`
- `observation.images.camera3` = `wrist_cam duplicate` only when a three-camera
  SmolVLA base config requires a third image feature.

Current approved `camera1` egocentric pose for dataset generation:

```json
{
  "type": "free",
  "lookat": [0.245, 0.11, 0.035],
  "distance": 0.63,
  "azimuth": 270,
  "elevation": -82,
  "rotation_degrees": 90
}
```

This pose is part of the local real-hardware-aligned dataset contract. Do not
change it, or replace `camera1` with `top_down`, without explicit user approval.
The executable source of truth is
`physical_ai_agent.sim.so101_camera_input.EGOCENTRIC_CAMERA1_POSE`; export
recipes and checksum manifests must agree with that constant.

The reviewed `camera2` candidate preset `over_fixed_jaw_rear_3cm` places the
camera 3 cm above the measured top surface at the rear of the fixed jaw and
preserves the original wrist-camera quaternion so the gripper framing remains
familiar. Only the physical camera position changes; there is no image
postprocessing rotation. Pass `wrist_camera_mount_preset` explicitly when
creating the environment. The default remains unchanged for compatibility
with existing datasets. Adopting the preset as a dataset contract requires
regenerating training/evaluation inputs and using the same extrinsic in
closed-loop and renderer-independent replay.

The explicit candidate preset `integrated_32x32_uvc` models the official
`Wrist_Cam_Mount_32x32_UVC_Module_SO101.stl` fixed-jaw replacement. The source
STL is pinned by SHA-256, converted from ASCII to MuJoCo-compatible binary STL,
and used for the fixed-jaw visual geom only; the original collision mesh stays
unchanged. The camera remains attached to the `gripper` body. The center of the
four 32x32 camera-board M2 holes defines the mount reference. The reviewed
camera assembly is constrained physically: the 32x32 PCB front face is flush
with the rear face of the printed square mount, the 10 mm lens barrel passes
through the center opening, and the optical pinhole is at the barrel tip. The
PCB center is therefore 11.5 mm behind the pinhole (10 mm barrel plus the
1.5 mm PCB half-thickness), while the lens center is 5 mm behind it. The lens
axis is the printed mount's 65-degree face normal; independently tilting the
optical axis is rejected by the Pydantic config because it would describe a
camera that is not seated against the mount. A live 1920x1080 capture from the
installed U20CAM, center-cropped to the policy's square input, is used to
review framing and the visible fixed-jaw hole sequence. The
physical camera mount roll is 180 degrees so the closed home pose
keeps the white moving jaw on the left and the green fixed jaw on the right.
This is an extrinsic rotation, not image postprocessing, so camera2 still has
zero pixel postprocessing rotation. The generated asset manifest is
`_workspace/so101_camera_mount_assets/generated/integrated_32x32_uvc_manifest.json`.
Live 16:9 frames are center-cropped to a square and then resized to 256x256,
matching the square simulator training inputs without stretching geometry or
introducing letterbox bands.
As with every candidate camera preset, existing datasets and the default
factory remain unchanged until the user explicitly adopts it for a new dataset
contract.

The B0CNCSFQC1 listing specifies a 130-degree diagonal FOV (`D`) and a
103-degree horizontal FOV (`H`); 103 degrees is not the vertical FOV. The
16:9 rectilinear vertical equivalent is 70.533 degrees. Camera1 and camera2
share these intrinsics; their framing differs only through the physical
extrinsics and occlusion. Preview rendering first produces a 16:9 frame matching the sensor,
then center-crops it to a square and resizes it to 256x256. It must not render a
square source directly or apply an independent per-camera zoom. Exact
real-camera intrinsics and lens distortion still require checkerboard
calibration. Preview rendering uses the OpenCV Brown-Conrady model with weak
candidate coefficients `[-0.08, 0.01, 0, 0, 0]` and an overscanned pinhole
source to avoid synthetic borders. These coefficients are explicitly marked
`uncalibrated_candidate`; replace them with measured coefficients before
claiming hardware calibration. See the
[OpenCV calibration model](https://docs.opencv.org/master/d9/d0c/group__calib3d.html)
and the [U20CAM-1080P specification](https://www.inno-maker.com/product/u20cam-1080p/).

The explicit rig preset `official_overhead_and_integrated_32x32_uvc` combines
that wrist camera with the official SO-ARM100
`Overhead_Cam_Mount_32x32_UVC_Module` assembly. The four source STLs are pinned
by SHA-256 and assembled as visual-only geometry: `arm_base`, `bottom`,
`middle`, then `top`. The yellow `arm_base` and `bottom` remain at the bottom;
the black `middle` and `top` share the yellow lower mast's connector axis at
CAD `(X=18.7 mm, Z=36.5125 mm)`. Both are translated 188.1 mm in CAD Y so each
joint retains the source design's 7.85 mm insertion overlap; aligning only the
Y bounding planes leaves the mast visibly disconnected and is invalid. The
`arm_base` and lower mast retain their common source-CAD orientation. The lower
mast is attached through the arm base's CAD-south socket and translated
`[-93.9209, -5.0, -205.9750]` mm: X aligns the lower floor's 37.4 mm keyed tab
to the matching 37.4 mm arm-base socket, Z fully inserts the matching 10 mm
tooth profiles, and the -5 mm CAD-Y offset makes both printed floor plates
coplanar. The rig-frame rotation maps CAD south
to world/screen right, placing the mast beside the base rather than behind it.
The SO101 root is translated 19.835 mm along world -X so the robot base-shell
front edge exactly matches the arm-base front edge; this moves the robot away
from the external preview camera without changing its Y alignment or height.
All mast sections retain their source orientation, so the top STL's +X-facing
camera-board plane still points toward the home gripper and workspace. The board centre is
derived from the four STL M2 holes. As with camera2, the 32x32 PCB front face
must sit flush against the rear of the printed square plate, the 10 mm lens
barrel passes through its center hole, and the pinhole is at the barrel tip.
The camera1 pinhole therefore sits exactly 10 mm in front of the mount face; a
larger offset would leave both square plates visibly disconnected. A static named
`egocentric_cam` follows this transformed top-mount lens position
above the simulated table. Its optical axis follows the printed face normal at
65 degrees downward; the Pydantic camera-rig schema rejects an independently
tilted optical axis because the lens could no longer pass through the mount
hole. A segmentation gate requires zero pixels from the camera PCB and lens in
their own camera1 image. With the physical 65-degree axis and the measured wide
FOV, part of the mast can enter the lower edge; removing it by tilting only the
optical axis would describe an impossible assembly and is not allowed.
The official-rig preview rotates the `studio_small_08` HDRI to 90 degrees so
its black softbox is not mistaken for a visible camera tower.
It uses the module's 70.533 degree vertical FOV and no pixel rotation. Camera1
and camera2 share the current U20CAM sensor estimate but retain independent
extrinsic contracts. The
two printed mounts include the same low-detail camera-module envelope: a
32 x 32 mm square board and central cylindrical lens aligned to each camera's
optical axis. These visual-only geoms do not change camera poses or collisions.
The
rig-specific Cycles material profile is
`configs/so101/render_profiles/black_arm_green_white_gripper_official_camera_rig.json`;
it preserves the approved black arm, green fixed jaw, and white moving jaw,
while adding the official yellow/black overhead stand. This remains a review
candidate and does not modify existing dataset roots or the default camera
factory.

The complete reproducible preview contract is stored in
`configs/so101/camera_rigs/official_32x32_uvc_photoreal_v1.json`. It is the
single source of truth for the reviewed STL paths and checksums, final assembly
transforms, camera1/camera2 extrinsics, U20CAM sensor approximation, distortion
candidate, home pose, object seed and geometry, Cycles/Metal material and
lighting settings, evidence cameras, and 256x256 center-crop preprocessing.
The strict Pydantic schema rejects unknown fields. Reproduce the canary from
the repository root with one command:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python scripts/render_so101_official_32x32_camera_rig_preview.py --config configs/so101/camera_rigs/official_32x32_uvc_photoreal_v1.json
```

The output report records the absolute config path, SHA-256, and complete
validated config snapshot. The loaded config is passed through the environment
factory into both camera-mount XML builders, so this command does not depend on
duplicated renderer constants.

The opt-in V4 quality profile is stored separately at
`configs/so101/camera_rigs/official_32x32_uvc_photoreal_v4.json`. It preserves
the V1 robot home pose, environment, sensor, and camera1/camera2 extrinsics
exactly. V4 renders at 912x512, center-crops the 16:9 frame to 512x512, and
downsamples it to the 256x256 policy input,
uses 256 Cycles samples with denoising disabled, applies 0.24-0.32 mm mesh
bevels, hashed PBR roughness/normal/wear/fingerprint textures, real Poly Haven
thermos, screwdriver, and workshop shelf assets at the side and rear of the
work surface, a neutral workshop wall,
directional key/fill/rim lights, and AgX color management. Reproduce the V4
review canary with:

```bash
PYTHONPATH=src:.:scripts .venv/bin/python scripts/render_so101_official_32x32_camera_rig_preview.py --config configs/so101/camera_rigs/official_32x32_uvc_photoreal_v4.json
```

V4 is a review-only render derivative and does not rewrite existing datasets
or replace the V1 default.

`top_down` is debug/teacher evidence only. Do not feed `top_down` to SmolVLA
for the local SO101/real-hardware-aligned training lane.
