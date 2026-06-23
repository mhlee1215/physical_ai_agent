# myCobot ROS/Gazebo/MoveIt Teacher Dataset POC

This POC records the shortest path from the myCobot ROS/Gazebo/MoveIt stack to
a local teacher-data artifact that can later be converted into a full
LeRobotDataset.

## Source Check

- Official ROS1 repo: `https://github.com/elephantrobotics/mycobot_ros`
- Official myCobot 280 gripper source:
  `mycobot_description/urdf/mycobot_280_jn/mycobot_280_jn_parallel_gripper.urdf`
- Official myCobot 320 M5 2022 gripper source:
  `mycobot_description/urdf/mycobot_320_m5_2022/new_mycobot_pro_320_m5_2022_gripper.urdf`
- Official Pro 450 force-gripper source:
  `mycobot_description/urdf/mycobot_pro_450/mycobot_pro_450_force_gripper.urdf`
- Official ROS1 MoveIt doc: `https://docs.elephantrobotics.com/docs/gitbook-en/12-ApplicationBaseROS/12.1-ROS1/12.1.5-Moveit/myCobot-280.html`
- Candidate ROS1 launch: `roslaunch mycobot_280_gripper_moveit demo_gazebo.launch gazebo_gui:=false`
- Smaller unofficial table-world candidate: `roslaunch mycobot_move_it_config demo_gazebo.launch gazebo_gui:=false`

The official ROS1 path has a myCobot 280 gripper MoveIt/Gazebo package with
trajectory-controller configuration for six arm joints plus a gripper joint.
That makes it a better teacher-data starting point than the thin
`mycobot_mujoco` model-only repo.

## POC Artifact

Mac-local one-command smoke:

```bash
sh scripts/run_mycobot_ros_teacher_poc_mac.sh
```

This writes `_workspace/mycobot_ros_teacher_poc_mac/report.json` and checks
that the frame rows, placeholder images, and `viewer.html` exist. It uses only
the Python standard library and does not require ROS, Gazebo, MoveIt, MuJoCo, or
LeRobot on the Mac.

To open the generated viewer with the macOS default browser:

```bash
OPEN_UI=1 sh scripts/run_mycobot_ros_teacher_poc_mac.sh
```

To generate real robot-arm render frames for the viewer, clone the official
myCobot MuJoCo asset repo and enable the renderer:

```bash
git clone https://github.com/elephantrobotics/mycobot_mujoco.git _vendor/mycobot_mujoco
RENDER_3D=1 sh scripts/run_mycobot_ros_teacher_poc_mac.sh
```

To run the proper myCobot-in-Nexus-style simulation POC instead of only
rendering dataset frames:

```bash
PYTHONPATH=src python3 scripts/mycobot_nexus_smoke.py \
  --asset-root _vendor/mycobot_mujoco \
  --output-dir _workspace/mycobot_nexus_smoke \
  --policy cube-approach
```

This creates a real `MyCobotNexusEnv` with MuJoCo `MjModel`/`MjData`, calls
`reset(seed)`, steps teacher-style 7D actions through a qpos-target controller,
uses the `cube-approach` Jacobian policy to move the TCP proxy toward the task
cube, renders the resulting scene, and writes a trace/report with initial,
final, and minimum TCP-to-cube distance. For dependency-light CI or code review,
the dry contract path records the env surface without importing MuJoCo:

The verified Mac-local cube-approach smoke reduced TCP-to-cube distance from
`0.518` to `0.237` over 16 steps with `approach_improved=true`.

To move from pure approach to contact-oriented simulation with the official
parallel gripper visual geometry, clone the official ROS description repo and
run the gripper/cube lift smoke:

```bash
git clone https://github.com/elephantrobotics/mycobot_ros.git _vendor/mycobot_ros

PYTHONPATH=src python3 scripts/mycobot_nexus_smoke.py \
  --asset-root _vendor/mycobot_mujoco \
  --official-gripper-root _vendor/mycobot_ros \
  --output-dir _workspace/mycobot_nexus_parallel_gripper_grasp_lift \
  --policy grasp-lift
```

This follows the official `mycobot_ros` 280 JN parallel-gripper assembly
documented in `mycobot_description/urdf/mycobot_280_jn/
mycobot_280_jn_parallel_gripper.urdf` and converts the official
`mycobot_description/urdf/parallel_gripper/*.dae` meshes to OBJ files under the
smoke output directory so MuJoCo can load them on macOS. The rendered gripper
visual geometry comes from the official parallel-gripper meshes; contact still
uses small transparent proxy pads attached to the official mimic-joint gripper
bodies. The task cube is smaller and closer to the reachable pre-grasp zone.

The verified Mac-local teacher grasp/lift smoke reached `grasp_success=true`,
`cube_lifted=true`, `final_cube_z=0.075`, `min_tcp_to_cube_dist=0.0018`, and
`gripper_cube_contacts=4` in 42 steps. The success label is
`teacher_grasp_lift_success`: after gripper closure near/contacting the cube,
the cube is explicitly attached to the midpoint of the finger contact pads so a
teacher dataset can record grasp/lift transitions before calibrated actuator
and force-closure modeling exists. This is simulation-state teacher supervision,
not a claim of physical force-closure.

Representative verified frames:

![myCobot official-gripper teacher grasp wide frame](./mycobot_nexus_official_gripper_frame.png)
![myCobot official-gripper teacher grasp front close-up](./mycobot_nexus_official_gripper_front_close.png)
![myCobot official-gripper teacher grasp side close-up](./mycobot_nexus_official_gripper_side_close.png)
![myCobot official-gripper teacher grasp wrist close-up](./mycobot_nexus_official_gripper_wrist_close.png)

### Official 320 M5 2022 Gripper Path

The official 320 M5 2022 gripper source is
`mycobot_320_m5_2022/new_mycobot_pro_320_m5_2022_gripper.urdf`, because the
upstream ROS1 README explicitly documents it as `mycobot 320 m5 2022 gripper`
and ships a matching RViz screenshot. That URDF includes the full 320 M5 2022
arm links, the flange-to-gripper joint, and the gripper mimic-joint tree.

Official RViz reference copied from the upstream repo:

![Official 320 M5 2022 gripper reference](./mycobot_320_m5_2022_official_gripper_reference.png)

The implemented path does not graft the 320 gripper onto the 280 MuJoCo arm.
Instead, `--model-profile 320-m5-2022-gripper` imports the official 320 arm
URDF tree and converts the official Collada meshes to OBJ for MuJoCo on macOS.
For the 320 arm links, the converter intentionally uses raw Collada geometry
vertices instead of baking the Collada `visual_scene` transforms; baking those
scene transforms and then applying URDF origins again visibly separated the arm
parts.

The raw upstream mimic-linkage gripper is replaced with a MuJoCo functional
friction-contact gripper at the official 320 flange: short slide jaws with
high-friction finger pads, MuJoCo elliptic friction cones, a higher-iteration
contact solver, and a 5 g dynamic cube. The 320 success condition requires both
finger pads to contact the cube while lifting it, without using the teacher
attachment proxy:

```bash
PYTHONPATH=src python3 scripts/mycobot_nexus_smoke.py \
  --model-profile 320-m5-2022-gripper \
  --official-gripper-root _vendor/mycobot_ros \
  --asset-root _vendor/mycobot_mujoco \
  --output-dir _workspace/mycobot_nexus_320_friction_grasp \
  --steps 220 \
  --policy grasp-lift
```

The verified 320 M5 2022 friction-contact gripper smoke reached
`success_label=contact_grasp_lift_success`, `grasp_success=true`,
`cube_lifted=true`, `grasp_attached=false`, `final_cube_z=0.0585`,
`min_tcp_to_cube_dist=0.0369`, `gripper_cube_contacts=7`, and
`gripper_cube_contact_pads=2` in 88 steps. This is still a simplified MuJoCo
functional gripper, not the raw official mimic-linkage gripper and not
calibrated hardware force closure. The arm assembly no longer has the previous
large part-separation artifact, but the end-effector remains a functional
stand-in rather than a finished official gripper assembly. The cube lift is now
produced by finger-pad contact friction rather than by directly attaching the
cube to the gripper or by a surrounding form-closure cage.

Representative 320 M5 2022 friction-contact gripper frames:

![myCobot 320 M5 2022 official-gripper zero pose](./mycobot_nexus_320_m5_2022_gripper_zero_pose.png)
![myCobot 320 M5 2022 friction-contact wide frame](./mycobot_nexus_320_m5_2022_gripper_frame.png)
![myCobot 320 M5 2022 friction-contact top view](./mycobot_nexus_320_m5_2022_gripper_front_close.png)
![myCobot 320 M5 2022 friction-contact lift side view](./mycobot_nexus_320_m5_2022_gripper_side_close.png)
![myCobot 320 M5 2022 friction-contact lift jaw view](./mycobot_nexus_320_m5_2022_gripper_wrist_close.png)

### Pro 450 Reference Boundary

The Pro 450 product photos use a different end-effector family than the 280 JN
parallel gripper above. The official `mycobot_ros` reference for that assembly
is the full `mycobot_pro_450_force_gripper.urdf`, including the Pro 450 arm,
`gripper_connection`, and force-gripper links. Do not graft Pro 450 gripper
meshes onto the 280 MuJoCo arm by hand; that produces a non-official hybrid
assembly and visually invalid screenshots.

Official RViz reference copied from the upstream repo:

![Official Pro 450 force gripper reference](./mycobot_pro450_official_force_gripper_reference.png)

The correct next implementation path is a dedicated Pro 450 official-URDF
renderer/importer that preserves the entire URDF tree and Collada semantics.
Until that importer is verified visually against the upstream RViz reference,
the Pro 450 gripper should remain a reference image, not a claimed MuJoCo
simulation artifact.

```bash
python3 scripts/mycobot_nexus_smoke.py \
  --dry-contract \
  --output-dir _workspace/mycobot_nexus_contract
```

Use `REQUIRE_3D_RENDER=1` when the run should fail unless MuJoCo produces real
RGB frames:

```bash
MYCOBOT_MUJOCO_ROOT=_vendor/mycobot_mujoco \
REQUIRE_3D_RENDER=1 sh scripts/run_mycobot_ros_teacher_poc_mac.sh
```

To override the output path or size:

```bash
ROOT=_workspace/mycobot_ros_teacher_poc_mac_small \
FRAMES=4 WIDTH=32 HEIGHT=24 \
sh scripts/run_mycobot_ros_teacher_poc_mac.sh
```

Direct exporter run:

```bash
PYTHONPATH=. python3 scripts/export_mycobot_ros_teacher_poc.py \
  --root _workspace/mycobot_ros_teacher_poc \
  --overwrite
```

When a ROS/Gazebo JSONL trace is available:

```bash
INPUT_TRACE=_workspace/mycobot_ros_trace/joint_and_action_trace.jsonl \
sh scripts/run_mycobot_ros_teacher_poc_mac.sh
```

The script writes:

- `meta/info.json`: feature schema, joint order, source links, and claim boundary.
- `data/frames.jsonl`: one JSONL row per teacher frame.
- `data/episodes.jsonl`: one episode row with `success` intentionally unset.
- `images/top/*.ppm` and `images/wrist/*.ppm`: deterministic placeholder images
  so downstream viewers/converters can test image paths without ROS.
- `render/scene/*.bmp`: optional real myCobot MuJoCo robot-arm render frames
  generated from `elephantrobotics/mycobot_mujoco` assets in a nexus-style
  stage with a work mat, lighting, skybox, and task cube.
- `render/render_report.json` and `render/render_blocker.md`: renderer status
  and dependency/asset blocker details.
- `_workspace/mycobot_nexus_smoke/mycobot_nexus_trace.jsonl`: optional actual
  `MyCobotNexusEnv` reset/step trace.
- `_workspace/mycobot_nexus_smoke/mycobot_nexus_report.json`: optional actual
  simulation smoke report with observation/action dimensions and scene path.
- `_workspace/mycobot_nexus_parallel_gripper_grasp_lift/mycobot_nexus_report.json`:
  optional official-gripper visual grasp/lift smoke report with contact and
  cube-lift fields.
- `viewer.html`: standalone local UI with playback controls, real MuJoCo render
  frame slots, and state/action visualizations.
- `report.json`: POC status and next steps.

## Boundary

This is not yet a task-success dataset. It is an offline adapter POC for
MoveIt/Gazebo traces. A training-quality dataset still needs:

- real ROS topic capture for `/joint_states`, FollowJointTrajectory goals, and
  Gazebo camera images;
- Gazebo model-state object pose and gripper/contact success oracle;
- replacement of placeholder PPM images with decoded ROS image messages;
- calibrated MuJoCo actuators and contact-based grasp success in
  `MyCobotNexusEnv`;
- native Gazebo/MuJoCo RGB/depth camera streams if task training needs multiple
  rendered policy observations beyond the current scene render;
- fresh rollout filtering before any `save_episode()`-equivalent claim.
