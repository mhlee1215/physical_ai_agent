import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_checkpoint_26_gate import run_checkpoint_26_gate


class RealSO100Checkpoint26GateTest(TestCase):
    def test_reuses_existing_episode_and_blocks_when_camera_zero_is_edge_clipped(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            camera_0 = tmp / "camera_0.jpg"
            camera_1 = tmp / "camera_1.jpg"
            observer = tmp / "camera_3.jpg"
            _write_camera_zero_edge_clipped(camera_0)
            _write_camera_one_usable(camera_1)
            _write_camera_one_usable(observer)
            episode = tmp / "episode.jsonl"
            episode.write_text(
                json.dumps(
                    {
                        "frame_index": 1,
                        "task": "test",
                        "observation": {
                            "state": {"gripper": 1864},
                            "images": {"0": str(camera_0), "1": str(camera_1), "3": str(observer)},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            grasp = _write_json(tmp / "grasp.json", {"status": "passed", "grasp_outcome": "grasp_failed_object_stationary"})

            result = run_checkpoint_26_gate(
                output_dir=tmp / "gate",
                port=None,
                episode=episode,
                frame_index=1,
                grasp_outcome=grasp,
                calibration_file=None,
                duration_seconds=1.0,
                fps=2.0,
                task="test gate",
                policy_camera_indexes=[0, 1],
                observer_camera_indexes=[3],
            )
            manifest_exists = Path(result["manifest_path"]).exists()

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["recommended_action"], "reframe_camera_0_or_camera_1_or_object")
        self.assertFalse(result["send_action_called"])
        self.assertEqual(result["policy_camera_indexes"], [0, 1])
        self.assertEqual(result["observer_camera_indexes"], [3])
        self.assertEqual(result["camera_roles"]["1"], "egocentric_cam")
        self.assertTrue(manifest_exists)

    def test_observer_camera_can_be_temporarily_unavailable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            camera_0 = tmp / "camera_0.jpg"
            camera_1 = tmp / "camera_1.jpg"
            _write_camera_zero_edge_clipped(camera_0)
            _write_camera_one_usable(camera_1)
            episode = tmp / "episode.jsonl"
            episode.write_text(
                json.dumps(
                    {
                        "frame_index": 1,
                        "task": "test",
                        "observation": {
                            "state": {"gripper": 1864},
                            "images": {"0": str(camera_0), "1": str(camera_1)},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            grasp = _write_json(tmp / "grasp.json", {"status": "passed", "grasp_outcome": "grasp_failed_object_stationary"})

            result = run_checkpoint_26_gate(
                output_dir=tmp / "gate",
                port=None,
                episode=episode,
                frame_index=1,
                grasp_outcome=grasp,
                calibration_file=None,
                duration_seconds=1.0,
                fps=2.0,
                task="test gate",
                policy_camera_indexes=[0, 1],
                observer_camera_indexes=[],
            )

        self.assertEqual(result["camera_indexes"], [0, 1])
        self.assertEqual(result["observer_camera_indexes"], [])
        self.assertEqual(result["observer_camera_status"], "temporarily_unavailable")
        self.assertFalse(result["send_action_called"])


def _write_camera_zero_edge_clipped(path: Path) -> None:
    import cv2
    import numpy as np

    image = np.full((160, 220, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (-10, 40), (35, 115), (0, 190, 55), thickness=-1)
    cv2.rectangle(image, (130, 90), (180, 135), (140, 60, 20), thickness=-1)
    cv2.imwrite(str(path), image)


def _write_camera_one_usable(path: Path) -> None:
    import cv2
    import numpy as np

    image = np.full((160, 220, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (105, 125), (0, 190, 55), thickness=-1)
    cv2.imwrite(str(path), image)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
