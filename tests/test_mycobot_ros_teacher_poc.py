from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.export_mycobot_ros_teacher_poc import (
    JOINT_NAMES,
    export_mycobot_ros_teacher_poc,
    extract_action_vector,
    extract_joint_vector,
)


class MyCobotRosTeacherPocTest(unittest.TestCase):
    def test_synthetic_poc_writes_schema_and_placeholder_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            report = export_mycobot_ros_teacher_poc(
                root=root,
                input_trace=None,
                episode_index=0,
                frames=5,
                fps=10,
                width=32,
                height=24,
                repo_id="physical-ai-agent/mycobot-ros-teacher-poc",
                overwrite=False,
            )

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["frames"], 5)
            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["joint_names"], JOINT_NAMES)
            self.assertEqual(info["features"]["observation.state"]["shape"], [7])
            self.assertIn("does not claim Gazebo task success", info["poc_boundary"])

            frame_rows = [
                json.loads(line)
                for line in (root / "data" / "frames.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(frame_rows), 5)
            self.assertEqual(len(frame_rows[0]["observation_state"]), 7)
            self.assertEqual(len(frame_rows[0]["action"]), 7)
            self.assertTrue((root / frame_rows[0]["top_image"]).exists())
            self.assertTrue((root / frame_rows[0]["wrist_image"]).exists())
            self.assertTrue((root / "viewer.html").exists())

            episode = json.loads((root / "data" / "episodes.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(episode["success_label"], "not_claimed_poc_trace_only")

    def test_custom_trace_uses_joint_names_and_trajectory_point_positions(self) -> None:
        record = {
            "joint_state": {
                "name": list(reversed(JOINT_NAMES)),
                "position": [float(index) for index in reversed(range(7))],
            },
            "trajectory_point": {
                "joint_names": JOINT_NAMES,
                "positions": [0.1 * index for index in range(7)],
            },
        }

        self.assertEqual(extract_joint_vector(record), [float(index) for index in range(7)])
        self.assertEqual(
            extract_action_vector(record, fallback=[0.0] * 7),
            [0.1 * i for i in range(7)],
        )

    def test_mac_runner_script_creates_checked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mac_poc"
            result = subprocess.run(
                [
                    "sh",
                    "scripts/run_mycobot_ros_teacher_poc_mac.sh",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                env={
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                    "PYTHON": "/usr/bin/python3",
                    "ROOT": str(root),
                    "FRAMES": "3",
                    "WIDTH": "24",
                    "HEIGHT": "24",
                },
                text=True,
                capture_output=True,
            )

            self.assertIn("mac_poc_status=passed", result.stdout)
            self.assertIn("mac_poc_viewer=", result.stdout)
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["frames"], 3)
            viewer = (root / "viewer.html").read_text(encoding="utf-8")
            self.assertIn("myCobot ROS Teacher POC", viewer)
            self.assertIn("frameSlider", viewer)
            self.assertIn("renderImage", viewer)
            self.assertIn("MuJoCo Robot Render", viewer)

    def test_mac_runner_records_render_blocker_when_assets_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mac_poc"
            result = subprocess.run(
                [
                    "sh",
                    "scripts/run_mycobot_ros_teacher_poc_mac.sh",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                env={
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                    "PYTHON": "/usr/bin/python3",
                    "ROOT": str(root),
                    "FRAMES": "2",
                    "WIDTH": "24",
                    "HEIGHT": "24",
                    "RENDER_3D": "1",
                    "MYCOBOT_MUJOCO_ROOT": str(Path(tmp) / "missing_assets"),
                },
                text=True,
                capture_output=True,
            )

            self.assertIn("mac_poc_render_status=blocked", result.stdout)
            report = json.loads(
                (root / "render" / "render_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "blocked")
            blocker = (root / "render" / "render_blocker.md").read_text(encoding="utf-8")
            self.assertIn("myCobot MuJoCo Render Blocker", blocker)


if __name__ == "__main__":
    unittest.main()
