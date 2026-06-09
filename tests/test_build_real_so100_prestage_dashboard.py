from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_real_so100_prestage_dashboard import build_prestage_dashboard


class BuildRealSO100PrestageDashboardTest(unittest.TestCase):
    def test_builds_single_entry_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "motion.mp4"
            _write_video(video)
            movement_html = root / "movement.html"
            movement_html.write_text("movement", encoding="utf-8")
            gate_html = root / "gate.html"
            gate_html.write_text("gate", encoding="utf-8")
            runbook_md = root / "runbook.md"
            runbook_md.write_text("runbook", encoding="utf-8")
            movement_report = _write_json(
                root / "movement_report.json",
                {
                    "joint": "shoulder_pan",
                    "observed_delta_raw": -28,
                    "motion_video": {
                        "path": str(video),
                        "frames_recorded": 19,
                        "actual_codec": "FMP4",
                        "actual_frame_count": 4,
                        "actual_fps": 5.0,
                        "first_frame_readable": True,
                        "browser_preview_recommended": True,
                    },
                },
            )
            movement = _write_json(
                root / "movement_manifest.json",
                {
                    "output_html": str(movement_html),
                    "reports": [str(movement_report)],
                    "video_count": 1,
                    "legacy_without_video_count": 0,
                },
            )
            gate = _write_json(
                root / "gate_manifest.json",
                {
                    "output_html": str(gate_html),
                    "current_gate_status": "blocked",
                    "recommended_action": "reframe_camera_0_or_object",
                },
            )
            pack = _write_json(
                root / "pack.json",
                {
                    "movement_report_manifest": str(movement),
                    "movement_report_html": str(movement_html),
                    "gate_report_manifest": str(gate),
                    "gate_report_html": str(gate_html),
                    "current_gate_status": "blocked",
                    "recommended_action": "reframe_camera_0_or_object",
                    "allowed_physical_action": None,
                    "agentic_lessons": [
                        {
                            "observation": "Camera 0 is clipped.",
                            "agentic_update": "Reframe before contact.",
                        }
                    ],
                },
            )
            runbook = _write_json(root / "runbook.json", {"output_markdown": str(runbook_md)})
            audit = _write_json(
                root / "audit.json",
                {
                    "status": "passed",
                    "failed_check_count": 0,
                    "pre_stage_pack": str(pack),
                    "runbook_manifest": str(runbook),
                    "manifest_path": str(root / "audit.json"),
                    "checks": [],
                },
            )
            output = root / "dashboard.html"

            manifest = build_prestage_dashboard(audit_manifest=audit, output=output)

            html = output.read_text(encoding="utf-8")
            self.assertEqual(manifest["status"], "passed")
            self.assertEqual(manifest["current_gate_status"], "blocked")
            self.assertIn("agentic-layer pre-stage", html)
            self.assertIn("movement.html", html)
            self.assertIn("gate.html", html)
            self.assertIn("motion.mp4", html)
            self.assertIn("motion_preview_0.gif", html)
            self.assertIn("codec=FMP4", html)
            self.assertIn("actual_frames=4", html)
            self.assertIn("GIF preview is recommended", html)
            self.assertIn("<video", html)
            self.assertTrue((root / "motion_preview_0.gif").exists())
            self.assertEqual(manifest["motion_video_previews"], [str(root / "motion_preview_0.gif")])


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
