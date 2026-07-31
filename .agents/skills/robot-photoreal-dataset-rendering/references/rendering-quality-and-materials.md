# Rendering Quality and Materials

## Engine and Mac Acceleration

Use Blender Cycles with the Metal device on Apple Silicon. MPS is a PyTorch
compute backend, not Blender's renderer API. Record the Blender device report
and verify `compute_device_type=METAL` plus a Metal GPU device. Mitsuba does not
provide the desired native MPS path in this repo.

## Stable Training Profile

For temporally stable 256x256 training frames, start with:

- Cycles samples: 256;
- denoise disabled unless a fixed denoiser is validated across a full episode;
- fixed Cycles seed and animated seed disabled;
- depth of field disabled;
- fixed camera exposure and color management;
- bounded indirect sample clamp;
- stable tabletop and lighting profile.

Increasing samples reduces Monte Carlo noise but increases render time. Denoise
can improve a single still while producing frame-to-frame texture changes. Test
quality settings on consecutive frames, not only a contact sheet.

## Material Profile Schema v2

`configs/so101/render_profiles/black_arm_green_white_gripper.json` separates
reusable material presets from 19 independent SO101 visual parts. Each part
assigns a material and one or more selector rules.

```json
{
  "schema_version": 2,
  "materials": {
    "white_matte_pla": {
      "base_color": [0.82, 0.84, 0.80],
      "roughness": 0.76,
      "metallic": 0.0
    }
  },
  "parts": {
    "wrist_motor_holder": {
      "material": "white_matte_pla",
      "selectors": [
        {
          "body_names": ["lower_arm"],
          "mesh_names": ["motor_holder_so101_wrist_v1"]
        }
      ]
    }
  }
}
```

Rules in a part are OR alternatives. Fields inside one rule are AND conditions.
This is required because servo meshes are reused under different bodies.
Visual/collision duplicates at identical transforms are one editable visual
part, not two independent color parts.

The current profile exposes base mounting plate/shell/motor holder/servo,
shoulder holder/rotation bracket/servos, upper-arm link/servos, forearm
link/servos, wrist motor holder/servo/roll bracket, gripper servo, fixed jaw and
pad, and moving jaw and pad. The approved sample uses a black matte arm, white
perforated wrist motor holder, green fixed jaw/pad, and white moving jaw/pad.

## Scene Profile

`black_table_clutter` uses a matte black workbench plus deterministic visual
props such as a mug, bottle, masking tape, and screwdriver. These are Blender
objects only. Keep them outside the protected manipulation zone, fixed across
all frames and cameras in an episode, and absent from MuJoCo collision state.

Hash the scene profile and every asset. Changing prop placement per frame is a
dataset corruption, not augmentation.

## Quality Review

Verify:

- materials are assigned to the intended mesh, not a neighboring link;
- robot/object edges do not shimmer across consecutive frames;
- camera1 and wrist camera match the source dataset contract;
- cube color and prompt agree;
- contact shadows and object lift are continuous;
- no visual prop occludes the policy manipulation area;
- image dimensions and color mode match source LeRobot metadata.

Record total wall time, seconds per source frame for all rendered cameras,
samples, denoise, Blender version, device, scene/material profile hashes, and
asset hashes in the render report.
