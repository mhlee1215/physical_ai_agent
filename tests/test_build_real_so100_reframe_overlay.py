from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_real_so100_reframe_overlay import build_reframe_overlay


class BuildRealSO100ReframeOverlayTest(TestCase):
    def test_builds_external_setup_diagnostic_overlay(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image = tmp / "camera_0.jpg"
            _write_image(image)
            advice = _write_json(
                tmp / "advice.json",
                {
                    "actions": [
                        {
                            "camera": "0",
                            "agent_actionable": False,
                            "external_setup_required": True,
                            "diagnostic_summary": "camera 0 target detection is clipped; external setup blocker",
                            "image_space_nudge": {
                                "current_bbox_xyxy": [0.0, 20.0, 60.0, 90.0],
                                "current_center_px": [30.0, 55.0],
                                "desired_center_px": [62.0, 55.0],
                                "recommended_shift_px": [32.0, 0.0],
                                "target_margin_px": 32,
                                "instruction": "shift target appearance about 32px right",
                            },
                        }
                    ]
                },
            )

            manifest = build_reframe_overlay(
                image=image,
                advice=advice,
                output=tmp / "overlay.jpg",
            )
            output_exists = Path(manifest["output_image"]).exists()

        self.assertEqual(manifest["status"], "passed")
        self.assertFalse(manifest["agent_actionable"])
        self.assertTrue(manifest["external_setup_required"])
        self.assertEqual(manifest["recommended_shift_px"], [32.0, 0.0])
        self.assertTrue(output_exists)


def _write_image(path: Path) -> None:
    import cv2
    import numpy as np

    image = np.full((140, 220, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (-5, 20), (60, 90), (0, 190, 55), thickness=-1)
    cv2.imwrite(str(path), image)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
