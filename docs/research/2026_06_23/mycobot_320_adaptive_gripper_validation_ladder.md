# myCobot 320 M5 2022 Adaptive Gripper Validation Ladder

This note freezes the validation order for the myCobot 320 M5 2022 adaptive
gripper POC. The goal is to avoid random MuJoCo coordinate tuning. Each step
must produce evidence before the next step changes code.

## Scope

Target upstream source:

- ROS2 Humble robot model:
  `mycobot_description/urdf/mycobot_320_m5_2022/mycobot_320_m5_2022_adaptive_gripper.urdf`
- Arm meshes:
  `mycobot_description/urdf/mycobot_320_m5_2022/*.dae`
- Adaptive gripper meshes:
  `mycobot_description/urdf/pro_adaptive_gripper/*.dae`

Known adjacent but non-target sources:

- ROS1 320 gripper:
  `mycobot_description/urdf/mycobot_320_m5_2022/new_mycobot_pro_320_m5_2022_gripper.urdf`
- ROS1 generic adaptive gripper:
  `mycobot_description/urdf/adaptive_gripper/mycobot_adaptive_gripper.urdf`

The ROS1 files are useful references, but they are not the primary source for
the 320 M5 2022 adaptive gripper profile.

## Ladder

### Gate 1: Source Routing

Question: are we importing the intended upstream files?

Pass evidence:

- The scene builder uses the ROS2 Humble 320 adaptive URDF.
- The generated MuJoCo XML references converted OBJ files for all seven arm
  links and all seven `pro_adaptive_gripper` links.
- The XML contains these mimic-linkage joints:
  `gripper_controller`,
  `gripper_base_to_gripper_left2`,
  `gripper_left3_to_gripper_left1`,
  `gripper_base_to_gripper_right3`,
  `gripper_base_to_gripper_right2`,
  `gripper_right3_to_gripper_right1`.

Current status: passed by XML structure check in PR #20. This does not prove
visual assembly or grasp physics.

### Gate 2: URDF Kinematic Tree Parity

Question: does the converted MuJoCo body tree match the upstream URDF tree?

Pass evidence:

- A script emits a parent-child table from the upstream URDF.
- The generated MuJoCo XML emits the corresponding body/joint table.
- The two tables match for every arm and gripper link, including fixed
  `joint6output_to_gripper_base`.
- Joint origin, axis, lower, upper, and mimic multiplier are compared numerically.

Stop condition:

- Do not tune collision pads, cube placement, friction, or arm trajectory if
  this table does not match.

Current status: passed by
`scripts/verify_mycobot_320_adaptive_kinematic_tree.py` against the ROS2 Humble
adaptive gripper source. Evidence: 13/13 arm and adaptive-gripper joints passed
parent-child, origin, axis, range, and mimic-multiplier comparison.

![Gate 2 kinematic tree evidence](./mycobot_320_adaptive_kinematic_tree_gate.png)

### Gate 3: Mesh Transform Parity

Question: are DAE geometry transforms and URDF visual origins applied exactly
once?

Pass evidence:

- For each link, record whether the OBJ conversion uses raw geometry vertices
  or baked `visual_scene` transforms.
- For the arm, raw geometry is currently required because baking DAE scene
  transforms and then applying URDF origins separated the arm parts.
- For the adaptive gripper, test raw-vs-baked conversion per mesh family and
  compare against a reference render before choosing.

Stop condition:

- Do not infer pad locations from a visually suspect mesh assembly.

### Gate 4: Reference Visual Pose

Question: does zero pose look like the official model?

Pass evidence:

- Render the upstream ROS2 adaptive URDF in a reference tool first. Acceptable
  references are RViz, Gazebo, or an independently generated URDF kinematic
  renderer.
- Render the MuJoCo conversion from front, side, top, and wrist-close views.
- Compare link order and finger orientation against the reference. A screenshot
  alone is not enough if the view hides the finger linkage.

Stop condition:

- If the middle linkage looks flipped, floating, or disconnected, return to
  Gate 2 or Gate 3. Do not proceed to grasp.

### Gate 5: Mimic Motion Parity

Question: does one gripper command move all adaptive gripper links in the
expected direction?

Pass evidence:

- Sample at least five `gripper_controller` values across the upstream range.
- For each sample, record follower joint values after mimic expansion.
- Render or export link poses for open, middle, and closed poses.
- The jaw gap must change monotonically in the intended close direction.

Stop condition:

- If closing increases the jaw gap, do not compensate with cube placement.
  Fix mimic mapping or command convention first.

### Gate 6: Collision Proxy Design

Question: are contact geoms attached to the correct finger link frames?

Pass evidence:

- Only after Gates 1-5 pass, derive simple contact geoms from the validated
  finger link frames.
- Use boxes or capsules, not raw visual meshes, for grasp contact.
- The proxy geoms must be invisible or separately colored for debugging.
- Record local positions, sizes, friction, condim, and the link frame they are
  attached to.

Stop condition:

- Do not move contact pads based only on failed cube traces. Pad placement must
  come from the validated finger-link coordinate frames.

### Gate 7: Static Contact Smoke

Question: can the gripper close on a pre-positioned cube without arm motion?

Pass evidence:

- Start with the cube between validated finger contact geoms.
- Keep the arm fixed.
- Close the gripper slowly.
- Require sustained contacts from both finger sides for a fixed window.

Stop condition:

- If the cube is pushed out before both sides contact, return to Gate 5 or
  Gate 6.

### Gate 8: Natural Arm And Gripper Motion

Question: can the arm and adaptive gripper move together without abrupt jumps?

Pass evidence:

- Use a timed trajectory with separate phases:
  pregrasp, close, lift.
- Interpolate arm joints and gripper command with a smooth curve.
- The cube must remain in contact during close and lift.
- The final render/video must be visually inspected from multiple views.

Stop condition:

- If the arm sweeps the cube before close, reduce or remove pre-close arm
  motion. Do not solve this by moving the cube to an implausible place.

## Current Next Step

The next implementation should be Gate 3: mesh transform parity. This must
decide raw-vs-baked Collada conversion for the adaptive gripper meshes using
reference visual evidence before any contact-pad, cube-placement, friction, or
trajectory tuning.
