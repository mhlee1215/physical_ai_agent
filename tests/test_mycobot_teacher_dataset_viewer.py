import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.serve_so101_dataset_viewer import (
    _mycobot_dataset_summary,
    _mycobot_frame_payload,
    _mycobot_dataset,
)


class MyCobotTeacherDatasetViewerTest(TestCase):
    def test_mycobot_jsonl_dataset_loads_frame_with_render_fallback(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "mycobot_320_adaptive_gate8_10eps"
            (dataset / "episodes").mkdir(parents=True)
            (dataset / "frames" / "episode_0000").mkdir(parents=True)
            manifest = {
                "format": "mycobot_jsonl_v1",
                "dataset_id": "mycobot_320_adaptive_gate8_10eps",
                "episodes": 1,
                "frames": 2,
                "fps": 20,
                "failed_episodes": [],
                "robot_model": "mycobot_320",
                "gripper": "adaptive",
                "gate": "gate8",
                "episode_summaries": [
                    {
                        "episode_index": 0,
                        "frames": 2,
                        "rendered_frames": 1,
                        "path": "episodes/episode_0000.jsonl",
                    }
                ],
            }
            (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (dataset / "frames" / "episode_0000" / "frame_0000.bmp").write_bytes(b"BMfake")
            rows = [
                _row(0, image="frames/episode_0000/frame_0000.bmp"),
                _row(1, image=""),
            ]
            (dataset / "episodes" / "episode_0000.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            loaded = _mycobot_dataset(dataset)
            summary = _mycobot_dataset_summary("mycobot_demo", loaded)
            frame = _mycobot_frame_payload(dataset, "mycobot_demo", 0, 1)

        self.assertEqual(summary["type"], "mycobot_jsonl")
        self.assertEqual(summary["dataset_format"], "mycobot_jsonl_v1")
        self.assertEqual(summary["platform"], "mycobot")
        self.assertEqual(summary["platform_label"], "MyCobot")
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["episode_lengths"], [2])
        self.assertEqual(summary["rendered_frames"], 1)
        self.assertEqual(summary["failed_episodes"], [])
        self.assertFalse(summary["training_ready"])
        self.assertEqual(frame["frame"], 1)
        self.assertEqual(frame["phase"], "lift")
        self.assertIn("render", frame["images"])
        self.assertEqual(frame["state"]["gripper_controller"], 0.7)


def _row(index: int, *, image: str) -> dict:
    return {
        "episode_index": 0,
        "frame_index": index,
        "timestamp": index / 20,
        "phase": "lift",
        "task": "short_grasp_lift_red_cube",
        "observation": {
            "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "images": {"render": image} if image else {},
        },
        "action": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.7],
        "reward": 0.0,
        "done": False,
        "info": {"gripper_cube_contact_pads": 2},
    }
