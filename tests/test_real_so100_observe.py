from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts.real_so100_observe import record_observation


class RealSO100ObserveTest(TestCase):
    def test_camera_only_mode_preserves_no_actuation_when_robot_connection_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus = _FailingBus()
            with patch("scripts.real_so100_observe._make_so100_bus", return_value=(bus, {"gripper": object()})), patch(
                "scripts.real_so100_observe._open_cameras", return_value={}
            ):
                report = record_observation(
                    port="/dev/fake",
                    camera_indexes=[0, 1],
                    output_dir=tmp,
                    duration_seconds=0.0,
                    fps=2.0,
                    task="camera-only test",
                    calibration_file=None,
                    policy_camera_indexes=[0, 1],
                    observer_camera_indexes=[],
                    allow_camera_only_without_robot=True,
                )
                episode_exists = Path(report["episode_jsonl"]).exists()

        self.assertTrue(report["ok"])
        self.assertEqual(report["mode"], "camera_only_without_robot")
        self.assertFalse(report["robot_connected"])
        self.assertFalse(report["robot_state_available"])
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["policy_actions_executed"])
        self.assertEqual(report["policy_camera_indexes"], [0, 1])
        self.assertEqual(report["observer_camera_indexes"], [])
        self.assertTrue(episode_exists)

    def test_robot_connection_failure_still_blocks_by_default(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bus = _FailingBus()
            with patch("scripts.real_so100_observe._make_so100_bus", return_value=(bus, {"gripper": object()})):
                report = record_observation(
                    port="/dev/fake",
                    camera_indexes=[0, 1],
                    output_dir=tmp,
                    duration_seconds=0.0,
                    fps=2.0,
                    task="default failure test",
                    calibration_file=None,
                    policy_camera_indexes=[0, 1],
                    observer_camera_indexes=[],
                )

        self.assertFalse(report["ok"])
        self.assertIn("robot unavailable", report["error"])
        self.assertFalse(report["send_action_called"])


class _FailingBus:
    is_connected = False

    def connect(self, *, handshake: bool) -> None:
        raise ConnectionError("robot unavailable")
