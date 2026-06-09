from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_reframe_advisor import build_reframe_advice


class RealSO100ReframeAdvisorTest(TestCase):
    def test_advises_right_and_down_when_jaw_object_is_left_top_clipped(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pregrasp = _write_json(
                tmp / "pregrasp.json",
                {
                    "edge_margin_px": 8,
                    "assessments": [
                        {
                            "camera": "0",
                            "bbox_xyxy": [0, 0, 60, 80],
                            "image_shape": [180, 260, 3],
                            "usable_for_pregrasp": False,
                        },
                        {
                            "camera": "1",
                            "bbox_xyxy": [100, 80, 140, 130],
                            "image_shape": [180, 260, 3],
                            "usable_for_pregrasp": True,
                        },
                    ],
                },
            )
            jaw = _write_json(
                tmp / "jaw.json",
                {
                    "status": "blocked",
                    "blockers": ["green object touches image boundary"],
                    "edge_margin_px": 8,
                    "image_shape": [180, 260, 3],
                    "object_candidate": {"bbox_xyxy": [0, 0, 60, 80]},
                },
            )

            result = build_reframe_advice(pregrasp_probe=pregrasp, jaw_readiness=jaw)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["jaw_object_clipped_sides"], ["left", "top"])
        self.assertEqual(result["actions"][0]["type"], "repair_jaw_camera_framing")
        self.assertFalse(result["actions"][0]["agent_actionable"])
        self.assertTrue(result["actions"][0]["external_setup_required"])
        self.assertIn("external setup blocker", result["actions"][0]["diagnostic_summary"])
        self.assertIn("left, top edge", result["actions"][0]["diagnostic_summary"])
        self.assertIn("image-space diagnostic", result["actions"][0]["image_space_goal"])
        self.assertEqual(result["actions"][0]["image_space_nudge"]["recommended_shift_px"], [32.0, 32.0])
        self.assertIn("32px too far left", result["actions"][0]["image_space_nudge"]["instruction"])
        self.assertIn("32px too high", result["actions"][0]["image_space_nudge"]["instruction"])

    def test_no_action_when_jaw_and_object_view_are_ready(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pregrasp = _write_json(
                tmp / "pregrasp.json",
                {
                    "assessments": [
                        {"camera": "0", "image_shape": [180, 260, 3], "usable_for_pregrasp": True},
                        {"camera": "1", "image_shape": [180, 260, 3], "usable_for_pregrasp": True},
                    ],
                },
            )
            jaw = _write_json(
                tmp / "jaw.json",
                {
                    "status": "ready",
                    "image_shape": [180, 260, 3],
                    "object_candidate": {"bbox_xyxy": [80, 60, 130, 120]},
                },
            )

            result = build_reframe_advice(pregrasp_probe=pregrasp, jaw_readiness=jaw)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["actions"], [])


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
