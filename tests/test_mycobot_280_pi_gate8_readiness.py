from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_mycobot_280_pi_gate8_readiness import check_readiness


class MyCobot280PiGate8ReadinessTest(unittest.TestCase):
    def test_missing_runtime_and_assets_report_blocked_with_next_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = check_readiness(
                asset_root=root / "missing_mujoco",
                official_gripper_root=root / "missing_ros",
            )

        self.assertEqual(report.status, "blocked")
        self.assertGreaterEqual(len([check for check in report.checks if check.status == "failed"]), 3)
        self.assertIn("gate7_static_contact", report.next_commands)
        self.assertIn("gate8_grasp_lift", report.next_commands)
        self.assertIn("Gate 7/8 physics success", report.claim_boundary)

    def test_present_assets_and_mujoco_spec_report_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset_root = root / "mycobot_mujoco"
            ros_root = root / "mycobot_ros"
            (asset_root / "xml").mkdir(parents=True)
            (asset_root / "xml" / "mycobot_280jn_mujoco.xml").write_text("<mujoco />", encoding="utf-8")
            arm_dir = ros_root / "mycobot_description" / "urdf" / "mycobot_280_pi"
            gripper_dir = ros_root / "mycobot_description" / "urdf" / "adaptive_gripper"
            arm_dir.mkdir(parents=True)
            gripper_dir.mkdir(parents=True)
            (arm_dir / "mycobot_280_pi.urdf").write_text("<robot />", encoding="utf-8")
            (gripper_dir / "mycobot_adaptive_gripper.urdf").write_text("<robot />", encoding="utf-8")

            with patch("importlib.util.find_spec", return_value=object()):
                report = check_readiness(asset_root=asset_root, official_gripper_root=ros_root)

        self.assertEqual(report.status, "passed")
        self.assertTrue(all(check.status == "passed" for check in report.checks))
        self.assertIn(str(asset_root), report.next_commands["gate8_teacher_dataset"])
        self.assertIn(str(ros_root), report.next_commands["gate8_teacher_dataset"])


if __name__ == "__main__":
    unittest.main()
