import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_real_so100_movement_report import build_movement_report


class BuildRealSO100MovementReportTest(TestCase):
    def test_builds_html_with_video_and_legacy_missing_video_note(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            visual = tmp / "visual"
            visual.mkdir()
            for name in ["before.jpg", "after.jpg", "diff_heatmap.jpg"]:
                (visual / name).write_bytes(b"fake")
            _write_video(visual / "motion.mp4")
            with_video = _write_report(tmp / "with_video.json", visual=visual, include_video=True)
            legacy = _write_report(tmp / "legacy.json", visual=visual, include_video=False)
            output = tmp / "movement_report.html"

            manifest = build_movement_report(reports=[with_video, legacy], output=output)
            html = output.read_text(encoding="utf-8")

        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(manifest["video_count"], 1)
        self.assertEqual(manifest["video_preview_count"], 1)
        self.assertEqual(manifest["legacy_without_video_count"], 1)
        self.assertIn("motion.mp4", html)
        self.assertIn("motion_report_preview_0.gif", html)
        self.assertIn("Video codec", html)
        self.assertIn("FMP4", html)
        self.assertIn("GIF preview recommended", html)
        self.assertIn("Motion video missing", html)
        self.assertIn("agentic-layer pre-stage", html)


def _write_report(path: Path, *, visual: Path, include_video: bool) -> Path:
    payload = {
        "status": "passed",
        "timestamp": "2026-06-06T00:00:00-0700",
        "joint": "gripper",
        "manual_delta_raw": -20,
        "observed_delta_raw": -18,
        "send_action_called": True,
        "contact_probe_allowed": True,
        "visual_check": {
            "before": {"image_path": str(visual / "before.jpg")},
            "after": {
                "image_path": str(visual / "after.jpg"),
                "mean_absdiff": 2.5,
                "visual_motion_detected": True,
            },
        },
    }
    if include_video:
        payload["motion_video"] = {
            "path": str(visual / "motion.mp4"),
            "frames_recorded": 8,
            "fps": 12,
            "actual_codec": "FMP4",
            "actual_frame_count": 4,
            "actual_fps": 5.0,
            "first_frame_readable": True,
            "browser_preview_recommended": True,
        }
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
