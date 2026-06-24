from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_mycobot_280_pi_adaptive_lerobot_dataset import (
    JOINT_NAMES,
    compute_object_contact_oracle,
    export_mycobot_280_pi_adaptive_lerobot_dataset,
    extract_action_vector,
    extract_joint_vector,
)


class MyCobot280PiDatasetPipelineTest(unittest.TestCase):
    def test_real_frame_trace_exports_lerobot_style_dataset_with_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trace_path = tmp_path / "trace.jsonl"
            manifest_path = tmp_path / "camera_manifest.json"
            camera_root = tmp_path / "camera"
            camera_root.mkdir()
            frames = []
            for index in range(3):
                top = camera_root / f"top_{index}.ppm"
                wrist = camera_root / f"wrist_{index}.ppm"
                _write_tiny_ppm(top, tint=20 + index)
                _write_tiny_ppm(wrist, tint=70 + index)
                frames.append({"top": str(top), "wrist": str(wrist), "timestamp": index / 12.0})
            manifest_path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
            trace_records = [
                _trace_record(0, z=0.020, contacts=0),
                _trace_record(1, z=0.030, contacts=1),
                _trace_record(2, z=0.061, contacts=2),
            ]
            trace_path.write_text(
                "\n".join(json.dumps(record, sort_keys=True) for record in trace_records) + "\n",
                encoding="utf-8",
            )
            root = tmp_path / "dataset"

            report = export_mycobot_280_pi_adaptive_lerobot_dataset(
                root=root,
                input_trace=trace_path,
                camera_manifest=manifest_path,
                episode_index=2,
                fps=12,
                repo_id="physical-ai-agent/mycobot-280pi-adaptive-test",
                task="Pick up the test cube.",
                overwrite=False,
            )

            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["real_camera_frames"])
            self.assertTrue(report["oracle"]["success"])
            self.assertEqual(report["oracle"]["success_label"], "object_contact_lift_success")
            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["robot_type"], "mycobot_280_pi_adaptive_gripper")
            self.assertEqual(info["joint_names"], JOINT_NAMES)
            self.assertEqual(info["features"]["observation.state"]["shape"], [7])
            rows = [
                json.loads(line)
                for line in (root / "data" / "frames.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["episode_index"], 2)
            self.assertTrue((root / rows[0]["top_image"]).exists())
            self.assertTrue((root / rows[0]["wrist_image"]).exists())
            self.assertEqual(rows[2]["contact_count"], 2)
            episode = json.loads((root / "data" / "episodes.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(episode["success"])
            smoke = json.loads(
                (root / "meta" / "smolvla_tiny_smoke_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(smoke["status"], "blocked_until_lerobot_smolvla_env_available")
            self.assertIn("smolvla", smoke["minimum_command_shape"])

    def test_missing_real_camera_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trace_path = tmp_path / "trace.jsonl"
            manifest_path = tmp_path / "camera_manifest.json"
            trace_path.write_text(
                "\n".join(json.dumps(_trace_record(index, z=0.02, contacts=0)) for index in range(2))
                + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "frames": [
                            {"top": str(tmp_path / "missing_top.ppm"), "wrist": str(tmp_path / "missing_wrist.ppm")},
                            {"top": str(tmp_path / "missing_top2.ppm"), "wrist": str(tmp_path / "missing_wrist2.ppm")},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                export_mycobot_280_pi_adaptive_lerobot_dataset(
                    root=tmp_path / "dataset",
                    input_trace=trace_path,
                    camera_manifest=manifest_path,
                    episode_index=0,
                    fps=12,
                    repo_id="physical-ai-agent/mycobot-280pi-adaptive-test",
                    task="Pick up the test cube.",
                    overwrite=False,
                )

    def test_joint_and_action_extraction_follow_280_pi_joint_order(self) -> None:
        record = {
            "joint_state": {
                "name": list(reversed(JOINT_NAMES)),
                "position": [float(index) for index in reversed(range(7))],
            },
            "trajectory_point": {
                "joint_names": JOINT_NAMES,
                "positions": [0.2 * index for index in range(7)],
            },
        }

        self.assertEqual(extract_joint_vector(record), [float(index) for index in range(7)])
        self.assertEqual(
            extract_action_vector(record, fallback=[0.0] * 7),
            [0.2 * index for index in range(7)],
        )

    def test_oracle_requires_object_pose_evidence(self) -> None:
        with self.assertRaises(ValueError):
            compute_object_contact_oracle([])


def _trace_record(index: int, *, z: float, contacts: int) -> dict[str, object]:
    state = [0.01 * (index + offset) for offset in range(7)]
    action = [value + 0.05 for value in state]
    return {
        "timestamp": index / 12.0,
        "joint_state": {"name": JOINT_NAMES, "position": state},
        "trajectory_point": {"joint_names": JOINT_NAMES, "positions": action},
        "object_pose": {"position": [0.12, -0.04, z]},
        "contacts": {"left_finger_pad": contacts >= 1, "right_finger_pad": contacts >= 2},
        "task": "Pick up the test cube.",
    }


def _write_tiny_ppm(path: Path, *, tint: int) -> None:
    pixels = bytes((tint, 30, 90, tint, 40, 100, tint, 50, 110, tint, 60, 120))
    path.write_bytes(b"P6\n2 2\n255\n" + pixels)


if __name__ == "__main__":
    unittest.main()
