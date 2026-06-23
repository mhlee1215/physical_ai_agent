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
        self.assertIn("official_parallel_gripper", contract["task_objects"])
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

    def test_sample_and_sanitize_teacher_action_keep_seven_dim_contract(self) -> None:
        action = sample_mycobot_nexus_action(step=1, total_steps=4)
        self.assertEqual(len(action), 7)

        sanitized = sanitize_teacher_action([1.0, float("nan"), 2.0])
        self.assertEqual(len(sanitized), 7)
        self.assertEqual(sanitized[0], 1.0)
        self.assertEqual(sanitized[1], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in sanitized))

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


if __name__ == "__main__":
    unittest.main()
