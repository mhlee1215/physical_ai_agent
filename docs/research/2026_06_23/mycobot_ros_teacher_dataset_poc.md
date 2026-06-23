# myCobot ROS/Gazebo/MoveIt Teacher Dataset POC

This POC records the shortest path from the myCobot ROS/Gazebo/MoveIt stack to
a local teacher-data artifact that can later be converted into a full
LeRobotDataset.

## Source Check

- Official ROS1 repo: `https://github.com/elephantrobotics/mycobot_ros`
- Official ROS2 repo: `https://github.com/elephantrobotics/mycobot_ros2`
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
- native Gazebo/MuJoCo RGB/depth camera streams if task training needs rendered
  observations beyond the current real robot-arm render preview;
- fresh rollout filtering before any `save_episode()`-equivalent claim.
