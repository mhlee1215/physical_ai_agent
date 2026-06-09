from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_real_so100_prestage_bundle import build_prestage_bundle


class BuildRealSO100PrestageBundleTest(unittest.TestCase):
    def test_builds_full_video_backed_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera_0 = root / "camera_0.jpg"
            camera_2 = root / "camera_2.jpg"
            before = root / "before.jpg"
            after = root / "after.jpg"
            video = root / "motion.mp4"
            _write_image(camera_0, edge_clipped=True)
            _write_image(camera_2, edge_clipped=False)
            _write_image(before, edge_clipped=False)
            _write_image(after, edge_clipped=False)
            _write_video(video)
            movement_report = _write_json(
                root / "micro_report.json",
                {
                    "status": "passed",
                    "joint": "shoulder_pan",
                    "observed_delta_raw": -28,
                    "visual_check": {
                        "before": {"image_path": str(before)},
                        "after": {"image_path": str(after), "mean_absdiff": 3.0, "visual_motion_detected": True},
                    },
                    "motion_video": {
                        "path": str(video),
                        "frames_recorded": 19,
                        "actual_codec": "mp4v",
                        "actual_frame_count": 4,
                        "first_frame_readable": True,
                        "browser_preview_recommended": False,
                    },
                },
            )
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
                    "recommended_action": "reframe_camera_0_or_object",
                    "allowed_physical_action": None,
                    "blockers": ["camera 0 jaw/object framing gate is not ready"],
                    "notes": ["Next-action gate only; it does not execute robot actions."],
                },
            )
            gate_manifest = _write_json(
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
            grasp = _write_json(root / "grasp.json", {"grasp_outcome": "grasp_failed_object_stationary"})

            manifest = build_prestage_bundle(
                reports=[movement_report],
                gate_manifest=gate_manifest,
                grasp_outcome=grasp,
                output_dir=root / "reports",
                label="bundle_test",
            )

            self.assertEqual(manifest["status"], "passed")
            self.assertEqual(manifest["video_count"], 1)
            self.assertEqual(manifest["allowed_physical_action"], None)
            for key in [
                "movement_report_html",
                "gate_report_html",
                "pre_stage_pack",
                "runbook_markdown",
                "audit_manifest",
                "dashboard_html",
            ]:
                self.assertTrue(Path(manifest[key]).exists(), key)
            dashboard = Path(manifest["dashboard_html"]).read_text(encoding="utf-8")
            self.assertIn("motion.mp4", dashboard)
            self.assertIn("motion_preview_0.gif", dashboard)
            self.assertIn("not a benchmark success claim", dashboard)


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


def _write_video(path: Path) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (80, 60))
    if not writer.isOpened():
        raise RuntimeError("failed to open test video writer")
    for index in range(4):
        frame = np.full((60, 80, 3), 240, dtype=np.uint8)
        cv2.rectangle(frame, (10 + index * 5, 20), (35 + index * 5, 45), (0, 190, 55), thickness=-1)
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    unittest.main()
