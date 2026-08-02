import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from scripts.run_real_so101_smolvla_chunk import (
    _calibration_range_report,
    _is_fresh_inference_step,
    _parse_sim_qpos,
    _policy_image,
    _start_state_contract_report,
)


JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class RealSO101HardwareRolloutTest(TestCase):
    def test_requeries_every_n_action_steps(self) -> None:
        self.assertEqual([step for step in range(45) if _is_fresh_inference_step(step, 15)], [0, 15, 30])

    def test_start_range_tolerance_blocks_only_large_violations(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            path.write_text(
                json.dumps({joint: {"range_min": 100, "range_max": 200} for joint in JOINTS}),
                encoding="utf-8",
            )
            positions = {joint: 150.0 for joint in JOINTS}
            positions["elbow_flex"] = 206.0
            tolerated = _calibration_range_report(path, positions, tolerance_raw=10.0)
            positions["elbow_flex"] = 211.0
            blocked = _calibration_range_report(path, positions, tolerance_raw=10.0)

        self.assertTrue(tolerated["safe"])
        self.assertFalse(blocked["safe"])
        self.assertEqual(blocked["violations"], ["elbow_flex"])

    def test_policy_image_center_crops_widescreen_without_stretching(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :50] = 255

        center_cropped = np.asarray(_policy_image(image, resize_mode="center_crop"))
        stretched = np.asarray(_policy_image(image, resize_mode="stretch"))

        self.assertEqual(center_cropped.shape, (256, 256, 3))
        self.assertEqual(int(center_cropped.max()), 0)
        self.assertGreater(int(stretched.max()), 0)

    def test_start_state_contract_checks_arm_and_gripper(self) -> None:
        expected = [0.0, -1.0, 1.0, 0.5, 0.0, 1.7453292519943295]
        actual = {
            "shoulder_pan": 0.0,
            "shoulder_lift": -57.3,
            "elbow_flex": 57.3,
            "wrist_flex": 28.6,
            "wrist_roll": 0.0,
            "gripper": 100.0,
        }

        passed = _start_state_contract_report(
            actual,
            expected,
            max_arm_error_degrees=1.0,
            max_gripper_error_percent=1.0,
        )
        actual["gripper"] = 0.0
        failed = _start_state_contract_report(
            actual,
            expected,
            max_arm_error_degrees=1.0,
            max_gripper_error_percent=1.0,
        )

        self.assertTrue(passed["passed"])
        self.assertEqual(failed["violations"], ["gripper"])

    def test_parses_exact_six_axis_start_qpos(self) -> None:
        self.assertEqual(_parse_sim_qpos("[0, 1, 2, 3, 4, 5]"), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
