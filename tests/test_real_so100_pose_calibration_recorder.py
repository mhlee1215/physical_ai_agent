from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.real_so100_pose_calibration_recorder import (
    RecorderConfig,
    run_pose_calibration_recorder,
)


class RealSO100PoseCalibrationRecorderTest(unittest.TestCase):
    def test_synthetic_recording_writes_aligned_diverse_dataset_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "shoulder_pan": {"range_min": 1200, "range_max": 2900},
                        "shoulder_lift": {"range_min": 2000, "range_max": 3800},
                        "elbow_flex": {"range_min": 40, "range_max": 2050},
                        "wrist_flex": {"range_min": 500, "range_max": 2200},
                        "wrist_roll": {"range_min": 0, "range_max": 4095},
                        "gripper": {"range_min": 2000, "range_max": 3300},
                    }
                ),
                encoding="utf-8",
            )

            result = run_pose_calibration_recorder(
                RecorderConfig(
                    port="/dev/null",
                    output_dir=root / "sessions",
                    calibration=calibration,
                    camera_indexes=[],
                    discover_cameras=False,
                    max_camera_index=2,
                    fps=6.0,
                    motor_hz=18.0,
                    duration_seconds=2.0,
                    primary_dedupe_camera=0,
                    hash_distance_threshold=4,
                    motor_distance_threshold=0.01,
                    max_selected_samples=20,
                    tts=False,
                    start_phrase="start",
                    stop_phrase="stop",
                    stop_file=None,
                    interactive=False,
                    synthetic=True,
                    synthetic_cameras=2,
                    execute_random_motion=False,
                    motion_strategy="random_micro_step",
                    human_confirmed=False,
                    workspace_clear_confirmed=False,
                    random_motion_period_seconds=1.0,
                    frame_after_motion_delay_seconds=0.0,
                    random_step_fraction=0.035,
                    sweep_max_delta_raw=60.0,
                    random_seed=3,
                )
            )

            self.assertEqual(result["status"], "passed")
            self.assertGreater(result["aligned_samples"], 0)
            self.assertGreater(result["selected_samples"], 0)
            self.assertFalse(result["send_action_called"])
            html_path = Path(result["html"])
            self.assertTrue(html_path.exists())
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("SO-100 Pose Calibration Dataset", html)
            self.assertTrue((html_path.parent / "aligned_samples.jsonl").exists())
            self.assertTrue((html_path.parent / "selected" / "selected_samples.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
