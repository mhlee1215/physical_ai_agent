import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase


def _load_probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "real_so100_pregrasp_probe.py"
    spec = importlib.util.spec_from_file_location("real_so100_pregrasp_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RealSo100PregraspProbeTest(TestCase):
    def test_rejects_edge_clipped_camera_and_selects_usable_primary(self) -> None:
        import cv2
        import numpy as np

        probe = _load_probe_module()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            edge_image = tmp / "camera_0.jpg"
            usable_image = tmp / "camera_2.jpg"

            edge = np.zeros((80, 120, 3), dtype=np.uint8)
            edge[55:80, 70:115] = (0, 220, 0)
            cv2.imwrite(str(edge_image), edge)

            usable = np.zeros((80, 120, 3), dtype=np.uint8)
            usable[20:60, 20:80] = (0, 220, 0)
            cv2.imwrite(str(usable_image), usable)

            episode = tmp / "episode.jsonl"
            episode.write_text(
                json.dumps(
                    {
                        "frame_index": 3,
                        "task": "pregrasp",
                        "observation": {
                            "state": {"gripper": 1717},
                            "images": {"0": str(edge_image), "2": str(usable_image)},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = probe.assess_episode_frame(
                episode=episode,
                frame_index=3,
                output=tmp / "result.json",
                min_area_px=300,
                edge_margin_px=0,
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["primary_camera"], "2")
        self.assertEqual(result["usable_cameras"], ["2"])
        by_camera = {item["camera"]: item for item in result["assessments"]}
        self.assertTrue(by_camera["0"]["edge_clipped"])
        self.assertFalse(by_camera["0"]["usable_for_pregrasp"])
        self.assertTrue(by_camera["2"]["usable_for_pregrasp"])
