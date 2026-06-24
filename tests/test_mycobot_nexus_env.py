import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from physical_ai_agent.sim.mycobot_nexus_env import (
    MYCOBOT_TEACHER_JOINT_NAMES,
    OFFICIAL_GRIPPER_MESH_NAMES,
    build_mycobot_nexus_scene_model,
    mycobot_nexus_contract,
    sample_mycobot_nexus_action,
    sanitize_teacher_action,
    write_dry_contract,
)
from scripts.mycobot_nexus_smoke import build_parser
from scripts.verify_mycobot_320_adaptive_kinematic_tree import (
    verify_adaptive_kinematic_tree,
)
from scripts.verify_mycobot_320_adaptive_collision_proxy import (
    verify_adaptive_collision_proxy,
)
from scripts.verify_mycobot_320_adaptive_mesh_transform import (
    verify_adaptive_mesh_transform,
)
from scripts.verify_mycobot_320_adaptive_mimic_motion import (
    verify_adaptive_mimic_motion,
)
from scripts.verify_mycobot_320_adaptive_visual_pose import (
    verify_adaptive_visual_pose,
)


class MyCobotNexusEnvTest(unittest.TestCase):
    def test_contract_declares_reset_step_render_surface(self) -> None:
        contract = mycobot_nexus_contract()

        self.assertEqual(contract["env"], "MyCobotNexusEnv")
        self.assertEqual(
            contract["surface"],
            ["reset(seed)", "step(action)", "render()", "close()"],
        )
        self.assertEqual(contract["joint_order"], MYCOBOT_TEACHER_JOINT_NAMES)
        self.assertIn("cube-approach", contract["policies"])
        self.assertIn("grasp-lift", contract["policies"])
        self.assertIn("320-m5-2022-gripper", contract["model_profiles"])
        self.assertIn("320-m5-2022-adaptive-gripper", contract["model_profiles"])
        self.assertIn("official_parallel_gripper", contract["task_objects"])
        self.assertIn("official_320_m5_2022_gripper", contract["task_objects"])
        self.assertIn("official_320_m5_2022_adaptive_gripper", contract["task_objects"])
        self.assertIn("synthetic_parallel_gripper_fallback", contract["task_objects"])
        self.assertIn("teacher_grasp_attachment_proxy", contract["task_objects"])
        self.assertEqual(contract["action_dim"], 7)
        self.assertEqual(contract["real_robot_execution"], "disabled")

    def test_dry_contract_writes_artifact_without_mujoco_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_dry_contract(Path(tmp))
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["env"], "MyCobotNexusEnv")
        self.assertIn("task_cube", payload["task_objects"])

    def test_parser_accepts_asset_and_dry_contract_options(self) -> None:
        args = build_parser().parse_args(
            [
                "--output-dir",
                "_workspace/test_mycobot",
                "--asset-root",
                "_vendor/mycobot_mujoco",
                "--official-gripper-root",
                "_vendor/mycobot_ros",
                "--steps",
                "3",
                "--seed",
                "9",
                "--width",
                "320",
                "--height",
                "180",
                "--policy",
                "grasp-lift",
                "--model-profile",
                "320-m5-2022-adaptive-gripper",
                "--dry-contract",
            ]
        )

        self.assertEqual(str(args.output_dir), "_workspace/test_mycobot")
        self.assertEqual(str(args.asset_root), "_vendor/mycobot_mujoco")
        self.assertEqual(str(args.official_gripper_root), "_vendor/mycobot_ros")
        self.assertEqual(args.steps, 3)
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.width, 320)
        self.assertEqual(args.height, 180)
        self.assertEqual(args.policy, "grasp-lift")
        self.assertEqual(args.model_profile, "320-m5-2022-adaptive-gripper")
        self.assertTrue(args.dry_contract)

    def test_scene_builder_injects_nexus_cube_world(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "xml" / "mycobot.xml"
            scene_path = tmp_path / "render" / "scene.xml"
            model_path.parent.mkdir()
            model_path.write_text(
                """
<mujoco model="tiny_mycobot">
  <compiler angle="radian" meshdir="../meshes_mujoco/" />
  <asset />
  <worldbody>
    <body name="joint2">
      <joint name="joint2_to_joint1" axis="0 0 1" range="-1 1" limited="true" />
      <geom type="sphere" size="0.02" />
      <body name="joint6_flange" pos="0 0 0.05" />
    </body>
  </worldbody>
</mujoco>
""".strip(),
                encoding="utf-8",
            )

            build_mycobot_nexus_scene_model(model_path=model_path, scene_path=scene_path)
            scene = ET.parse(scene_path).getroot()
            names = {element.attrib.get("name") for element in scene.iter()}

        self.assertIn("task_cube", names)
        self.assertIn("task_cube_body", names)
        self.assertIn("task_cube_freejoint", names)
        self.assertIn("synthetic_parallel_gripper", names)
        self.assertIn("left_finger_slide", names)
        self.assertIn("right_finger_slide", names)
        self.assertIn("mycobot_tcp_site", names)
        self.assertIn("nexus_work_mat", names)
        self.assertIn("nexus_skybox", names)
        self.assertIn("nexus_key_light", names)

    def test_scene_builder_can_use_official_ros1_parallel_gripper_meshes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "xml" / "mycobot.xml"
            scene_path = tmp_path / "render" / "scene.xml"
            gripper_root = tmp_path / "mycobot_ros"
            mesh_dir = gripper_root / "mycobot_description" / "urdf" / "parallel_gripper"
            mesh_dir.mkdir(parents=True)
            for name in OFFICIAL_GRIPPER_MESH_NAMES:
                (mesh_dir / f"{name}.dae").write_text(_tiny_collada_triangle(), encoding="utf-8")
            model_path.parent.mkdir()
            model_path.write_text(
                """
<mujoco model="tiny_mycobot">
  <compiler angle="radian" meshdir="../meshes_mujoco/" />
  <asset />
  <worldbody>
    <body name="joint2">
      <joint name="joint2_to_joint1" axis="0 0 1" range="-1 1" limited="true" />
      <geom type="sphere" size="0.02" />
      <body name="joint6_flange" pos="0 0 0.05" />
    </body>
  </worldbody>
</mujoco>
""".strip(),
                encoding="utf-8",
            )

            build_mycobot_nexus_scene_model(
                model_path=model_path,
                scene_path=scene_path,
                official_gripper_root=gripper_root,
            )
            scene = ET.parse(scene_path).getroot()
            names = {element.attrib.get("name") for element in scene.iter()}

        self.assertIn("official_parallel_gripper", names)
        self.assertIn("official_gripper_base", names)
        self.assertIn("gripper_controller", names)
        self.assertIn("gripper_base_to_gripper_left", names)
        self.assertIn("left_finger_pad", names)
        self.assertIn("right_finger_pad", names)

    def test_scene_builder_can_use_ros2_320_adaptive_gripper_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scene_path = tmp_path / "render" / "scene.xml"
            ros_root = _write_minimal_320_adaptive_ros2_tree(tmp_path)

            build_mycobot_nexus_scene_model(
                model_path=Path(""),
                scene_path=scene_path,
                official_gripper_root=ros_root,
                model_profile="320-m5-2022-adaptive-gripper",
            )
            scene = ET.parse(scene_path).getroot()
            compiler = scene.find("compiler")
            names = {element.attrib.get("name") for element in scene.iter()}
            mesh_files = {
                Path(element.attrib["file"]).name
                for element in scene.findall(".//mesh")
                if "file" in element.attrib
            }
            equality_names = {
                element.attrib.get("name")
                for element in scene.findall(".//equality/connect")
            }

        self.assertIsNotNone(compiler)
        self.assertEqual(compiler.attrib.get("eulerseq"), "XYZ")
        self.assertIn("gripper_controller", names)
        self.assertIn("gripper_base_to_gripper_left2", names)
        self.assertIn("gripper_right3_to_gripper_right1", names)
        self.assertIn("left_finger_pad", names)
        self.assertIn("right_finger_pad", names)
        self.assertIn("left2_loop_site", names)
        self.assertIn("left1_loop_site", names)
        self.assertIn("right2_loop_site", names)
        self.assertIn("right1_loop_site", names)
        self.assertIn("left_adaptive_fourbar_loop", equality_names)
        self.assertIn("right_adaptive_fourbar_loop", equality_names)
        self.assertIn("gripper_base.obj", mesh_files)
        self.assertIn("link6.obj", mesh_files)

    def test_320_adaptive_kinematic_tree_verifier_writes_visual_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ros_root = _write_minimal_320_adaptive_ros2_tree(tmp_path)
            report = verify_adaptive_kinematic_tree(
                official_gripper_root=ros_root,
                output_dir=tmp_path / "verify",
            )

            self.assertEqual(report.status, "passed")
            self.assertEqual(report.failed_joint_count, 0)
            self.assertEqual(report.compared_joint_count, 13)
            self.assertTrue(Path(report.artifacts["json"]).exists())
            self.assertTrue(Path(report.artifacts["markdown"]).exists())
            self.assertTrue(Path(report.artifacts["svg"]).exists())

    def test_320_adaptive_mesh_transform_verifier_writes_visual_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ros_root = _write_minimal_320_adaptive_ros2_tree(tmp_path)
            report = verify_adaptive_mesh_transform(
                official_gripper_root=ros_root,
                output_dir=tmp_path / "verify_meshes",
            )

            self.assertEqual(report.status, "passed")
            self.assertEqual(report.failed_mesh_count, 0)
            self.assertEqual(report.compared_mesh_count, 14)
            self.assertEqual(report.selected_transform_mode, "raw_geometry")
            self.assertTrue(Path(report.artifacts["json"]).exists())
            self.assertTrue(Path(report.artifacts["markdown"]).exists())
            self.assertTrue(Path(report.artifacts["svg"]).exists())

    def test_320_adaptive_visual_pose_verifier_writes_visual_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ros_root = _write_minimal_320_adaptive_ros2_tree(tmp_path)
            report = verify_adaptive_visual_pose(
                official_gripper_root=ros_root,
                output_dir=tmp_path / "verify_visual_pose",
            )

            self.assertEqual(report.status, "passed")
            self.assertEqual(report.failed_link_count, 0)
            self.assertEqual(report.compared_link_count, 14)
            self.assertTrue(Path(report.artifacts["json"]).exists())
            self.assertTrue(Path(report.artifacts["markdown"]).exists())
            self.assertTrue(Path(report.artifacts["svg"]).exists())

    def test_320_adaptive_mimic_motion_verifier_writes_visual_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ros_root = _write_minimal_320_adaptive_ros2_tree(tmp_path)
            report = verify_adaptive_mimic_motion(
                official_gripper_root=ros_root,
                output_dir=tmp_path / "verify_mimic_motion",
            )

            self.assertEqual(report.status, "passed")
            self.assertTrue(report.controller_increase_opens)
            self.assertLess(report.closed_jaw_gap_xy, report.open_jaw_gap_xy)
            self.assertEqual(report.sample_count, 5)
            self.assertTrue(Path(report.artifacts["json"]).exists())
            self.assertTrue(Path(report.artifacts["markdown"]).exists())
            self.assertTrue(Path(report.artifacts["svg"]).exists())

    def test_320_adaptive_collision_proxy_verifier_writes_visual_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ros_root = _write_minimal_320_adaptive_ros2_tree(tmp_path)
            report = verify_adaptive_collision_proxy(
                official_gripper_root=ros_root,
                output_dir=tmp_path / "verify_collision_proxy",
            )

            self.assertEqual(report.compared_proxy_count, 2)
            self.assertTrue(Path(report.artifacts["json"]).exists())
            self.assertTrue(Path(report.artifacts["markdown"]).exists())
            self.assertTrue(Path(report.artifacts["svg"]).exists())

    def test_sample_and_sanitize_teacher_action_keep_seven_dim_contract(self) -> None:
        action = sample_mycobot_nexus_action(step=1, total_steps=4)
        self.assertEqual(len(action), 7)

        sanitized = sanitize_teacher_action([1.0, float("nan"), 2.0])
        self.assertEqual(len(sanitized), 7)
        self.assertEqual(sanitized[0], 1.0)
        self.assertEqual(sanitized[1], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in sanitized))


def _write_minimal_320_adaptive_ros2_tree(tmp_path: Path) -> Path:
    ros_root = tmp_path / "mycobot_ros2"
    arm_mesh_dir = ros_root / "mycobot_description" / "urdf" / "mycobot_320_m5_2022"
    adaptive_mesh_dir = ros_root / "mycobot_description" / "urdf" / "pro_adaptive_gripper"
    arm_mesh_dir.mkdir(parents=True)
    adaptive_mesh_dir.mkdir(parents=True)
    for name in ("base", "link1", "link2", "link3", "link4", "link5", "link6"):
        (arm_mesh_dir / f"{name}.dae").write_text(
            _tiny_collada_triangle(),
            encoding="utf-8",
        )
    for name in (
        "gripper_base",
        "gripper_left1",
        "gripper_left2",
        "gripper_left3",
        "gripper_right1",
        "gripper_right2",
        "gripper_right3",
    ):
        (adaptive_mesh_dir / f"{name}.dae").write_text(
            _tiny_collada_triangle(),
            encoding="utf-8",
        )
    (arm_mesh_dir / "mycobot_320_m5_2022_adaptive_gripper.urdf").write_text(
        _minimal_320_adaptive_urdf(),
        encoding="utf-8",
    )
    return ros_root


def _tiny_collada_triangle() -> str:
    return """
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit meter="1.0" name="meter" /></asset>
  <library_geometries>
    <geometry id="mesh">
      <mesh>
        <source id="mesh-position" name="position">
          <float_array id="mesh-position-array" count="9">0 0 0 0.01 0 0 0 0.01 0</float_array>
        </source>
        <vertices id="mesh-vertices">
          <input semantic="POSITION" source="#mesh-position" />
        </vertices>
        <triangles count="1">
          <input semantic="VERTEX" source="#mesh-vertices" offset="0" />
          <p>0 1 2</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
</COLLADA>
""".strip()


def _minimal_320_adaptive_urdf() -> str:
    links = "\n".join(
        f"""
  <link name="{name}">
    <visual>
      <geometry>
        <mesh filename="package://mycobot_description/urdf/{mesh_dir}/{name}.dae"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </visual>
  </link>
"""
        for mesh_dir, names in (
            (
                "mycobot_320_m5_2022",
                ("base", "link1", "link2", "link3", "link4", "link5", "link6"),
            ),
            (
                "pro_adaptive_gripper",
                (
                    "gripper_base",
                    "gripper_left1",
                    "gripper_left2",
                    "gripper_left3",
                    "gripper_right1",
                    "gripper_right2",
                    "gripper_right3",
                ),
            ),
        )
        for name in names
    )
    joints = """
  <joint name="joint2_to_joint1" type="revolute">
    <parent link="base"/><child link="link1"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="joint3_to_joint2" type="revolute">
    <parent link="link1"/><child link="link2"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="joint4_to_joint3" type="revolute">
    <parent link="link2"/><child link="link3"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="joint5_to_joint4" type="revolute">
    <parent link="link3"/><child link="link4"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="joint6_to_joint5" type="revolute">
    <parent link="link4"/><child link="link5"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="joint6output_to_joint6" type="revolute">
    <parent link="link5"/><child link="link6"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="joint6output_to_gripper_base" type="fixed">
    <parent link="link6"/><child link="gripper_base"/><origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>
  <joint name="gripper_controller" type="revolute">
    <parent link="gripper_base"/><child link="gripper_left3"/>
    <origin xyz="-0.018 0.015 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-1.11" upper="0" effort="1" velocity="1"/>
  </joint>
  <joint name="gripper_base_to_gripper_left2" type="revolute">
    <parent link="gripper_base"/><child link="gripper_left2"/>
    <origin xyz="-0.047 -0.01 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-0.8" upper="0.5" effort="1" velocity="1"/>
    <mimic joint="gripper_controller" multiplier="1.0" offset="0"/>
  </joint>
  <joint name="gripper_left3_to_gripper_left1" type="revolute">
    <parent link="gripper_left3"/><child link="gripper_left1"/>
    <origin xyz="-0.05 0.035 -0.015" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-0.5" upper="0.5" effort="1" velocity="1"/>
    <mimic joint="gripper_controller" multiplier="-1.0" offset="0"/>
  </joint>
  <joint name="gripper_base_to_gripper_right3" type="revolute">
    <parent link="gripper_base"/><child link="gripper_right3"/>
    <origin xyz="0.016 0.014 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-0.3" upper="0.7" effort="1" velocity="1"/>
    <mimic joint="gripper_controller" multiplier="-1.0" offset="0"/>
  </joint>
  <joint name="gripper_base_to_gripper_right2" type="revolute">
    <parent link="gripper_base"/><child link="gripper_right2"/>
    <origin xyz="0.044 -0.01 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-0.5" upper="0.8" effort="1" velocity="1"/>
    <mimic joint="gripper_controller" multiplier="-1.0" offset="0"/>
  </joint>
  <joint name="gripper_right3_to_gripper_right1" type="revolute">
    <parent link="gripper_right3"/><child link="gripper_right1"/>
    <origin xyz="0.052 0.035 -0.015" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-0.5" upper="0.5" effort="1" velocity="1"/>
    <mimic joint="gripper_controller" multiplier="1.0" offset="0"/>
  </joint>
"""
    return f"<robot name=\"minimal_320_adaptive\">{links}{joints}</robot>"


if __name__ == "__main__":
    unittest.main()
