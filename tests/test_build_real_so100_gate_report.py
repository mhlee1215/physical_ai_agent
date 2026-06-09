from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_real_so100_gate_report import build_gate_report


class BuildRealSO100GateReportTest(unittest.TestCase):
    def test_builds_blocked_gate_visual_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera_0 = root / "camera_0.jpg"
            camera_2 = root / "camera_2.jpg"
            _write_image(camera_0, edge_clipped=True)
            _write_image(camera_2, edge_clipped=False)
            pregrasp = _write_json(
                root / "pregrasp.json",
                {
                    "status": "passed",
                    "assessments": [
                        {
                            "camera": "0",
                            "image_path": str(camera_0),
                            "object_visible": True,
                            "bbox_xyxy": [0, 40, 45, 110],
                            "edge_clipped": True,
                            "usable_for_pregrasp": False,
                            "notes": ["green detection touches image boundary"],
                        },
                        {
                            "camera": "2",
                            "image_path": str(camera_2),
                            "object_visible": True,
                            "bbox_xyxy": [70, 40, 135, 125],
                            "edge_clipped": False,
                            "usable_for_pregrasp": True,
                            "notes": [],
                        },
                    ],
                },
            )
            jaw = _write_json(
                root / "jaw.json",
                {
                    "status": "blocked",
                    "blockers": ["green object touches image boundary"],
                    "object_candidate": {"bbox_xyxy": [0, 40, 45, 110]},
                    "jaw_marker_candidate": {"bbox_xyxy": [120, 80, 170, 135]},
                },
            )
            next_action = _write_json(
                root / "next_action.json",
                {
                    "status": "blocked",
                    "notes": ["Next-action gate only; it does not execute robot actions."],
                },
            )
            manifest = _write_json(
                root / "checkpoint_26_gate_manifest.json",
                {
                    "status": "blocked",
                    "recommended_action": "reframe_camera_0_or_object",
                    "allowed_physical_action": None,
                    "blockers": ["camera 0 jaw/object framing gate is not ready"],
                    "pregrasp_status": "passed",
                    "jaw_status": "blocked",
                    "pregrasp_probe": str(pregrasp),
                    "jaw_readiness": str(jaw),
                    "next_action_gate": str(next_action),
                },
            )
            output = root / "gate_report.html"

            result = build_gate_report(gate_manifest=manifest, output=output)

            html = output.read_text(encoding="utf-8")
            self.assertEqual(result["current_gate_status"], "blocked")
            self.assertEqual(result["allowed_physical_action"], None)
            self.assertIn("agentic-layer pre-stage", html)
            self.assertIn("reframe_camera_0_or_object", html)
            self.assertIn("camera_0_pregrasp_overlay.jpg", html)
            self.assertTrue((root / "camera_0_pregrasp_overlay.jpg").exists())
            self.assertTrue((root / "camera_2_pregrasp_overlay.jpg").exists())


def _write_image(path: Path, *, edge_clipped: bool) -> None:
    import cv2
    import numpy as np

    image = np.full((160, 220, 3), 235, dtype=np.uint8)
    if edge_clipped:
        cv2.rectangle(image, (-10, 40), (45, 110), (0, 190, 55), thickness=-1)
    else:
        cv2.rectangle(image, (70, 40), (135, 125), (0, 190, 55), thickness=-1)
    cv2.imwrite(str(path), image)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
