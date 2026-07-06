from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from physical_ai_agent.sim.mycobot_nexus_env import (
    MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
    MODEL_PROFILE_320_ADAPTIVE_GRIPPER,
)
from scripts import export_mycobot_adaptive_teacher_dataset as teacher_export
from scripts.mycobot_adaptive_grasp_lift_smoke import build_parser as grasp_lift_parser
from scripts.mycobot_adaptive_static_contact_smoke import build_parser as static_contact_parser
from scripts.mycobot_280_pi_adaptive_grasp_lift_smoke import build_parser as pi_grasp_lift_parser
from scripts.mycobot_280_pi_adaptive_static_contact_smoke import build_parser as pi_static_contact_parser


class MyCobot280PiGate8ParityTest(unittest.TestCase):
    def test_shared_gate_scripts_accept_280_pi_profile_without_changing_320_default(self) -> None:
        static_args = static_contact_parser().parse_args([])
        grasp_args = grasp_lift_parser().parse_args([])

        self.assertEqual(static_args.model_profile, MODEL_PROFILE_320_ADAPTIVE_GRIPPER)
        self.assertEqual(grasp_args.model_profile, MODEL_PROFILE_320_ADAPTIVE_GRIPPER)

        static_280 = static_contact_parser().parse_args(
            ["--model-profile", MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER]
        )
        grasp_280 = grasp_lift_parser().parse_args(
            ["--model-profile", MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER]
        )

        self.assertEqual(static_280.model_profile, MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
        self.assertEqual(grasp_280.model_profile, MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)

    def test_280_wrapper_parsers_default_to_280_pi_profile(self) -> None:
        static_args = pi_static_contact_parser().parse_args([])
        grasp_args = pi_grasp_lift_parser().parse_args([])

        self.assertEqual(static_args.model_profile, MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
        self.assertEqual(grasp_args.model_profile, MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
        self.assertFalse(grasp_args.disable_teacher_attachment)

        raw_grasp_args = pi_grasp_lift_parser().parse_args(["--disable-teacher-attachment"])
        self.assertTrue(raw_grasp_args.disable_teacher_attachment)

    def test_280_teacher_export_manifest_uses_280_joint_order_and_robot_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "dataset"

            def fake_episode(**kwargs):
                self.assertEqual(kwargs["model_profile"], MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
                return {
                    "episode_index": kwargs["episode_index"],
                    "path": f"episodes/episode_{kwargs['episode_index']:04d}.jsonl",
                    "frames": 2,
                    "rendered_frames": 1,
                    "success": True,
                    "close_best_sustained_contact_steps": 15,
                    "lift_best_sustained_contact_steps": 25,
                    "final_cube_lift": 0.03,
                    "final_gripper_cube_contact_pads": 2,
                    "final_gripper_cube_contacts": 4,
                }

            with patch.object(teacher_export, "_export_episode", side_effect=fake_episode):
                manifest = teacher_export.export_dataset(
                    output_dir=output_dir,
                    episodes=2,
                    seed=10,
                    asset_root=Path("_vendor/mycobot_mujoco"),
                    official_gripper_root=Path("_vendor/mycobot_ros"),
                    model_profile=MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER,
                    width=32,
                    height=24,
                    fps=12,
                    render_every=2,
                    pregrasp_steps=3,
                    close_steps=4,
                    lift_steps=5,
                    placement_gripper_command=0.25,
                    close_gripper_command=-0.7,
                    cube_half_size=0.02,
                )

            written = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["robot"], "myCobot 280 Pi + adaptive gripper")
        self.assertEqual(written["model_profile"], MODEL_PROFILE_280_PI_ADAPTIVE_GRIPPER)
        self.assertIn("joint7_to_joint6", written["joint_names"])
        self.assertNotIn("joint6output_to_joint6", written["joint_names"])
        self.assertEqual(written["failed_episodes"], [])
        self.assertEqual(written["frames"], 4)


if __name__ == "__main__":
    unittest.main()
