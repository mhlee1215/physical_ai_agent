import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_real_so100_agentic_prestage_pack import build_agentic_prestage_pack


class BuildRealSO100AgenticPrestagePackTest(TestCase):
    def test_pack_derives_agentic_lessons_from_failed_grasp_and_blocked_gate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            movement = _write_json(
                tmp / "movement.json",
                {
                    "output_html": "movement.html",
                    "video_count": 0,
                    "legacy_without_video_count": 2,
                },
            )
            gate_report = _write_json(
                tmp / "gate_report.json",
                {
                    "output_html": "gate_report.html",
                    "current_gate_status": "blocked",
                },
            )
            gate = _write_json(
                tmp / "gate.json",
                {
                    "status": "blocked",
                    "recommended_action": "reframe_camera_0_or_object",
                    "allowed_physical_action": None,
                },
            )
            grasp = _write_json(tmp / "grasp.json", {"grasp_outcome": "grasp_failed_object_stationary"})
            output = tmp / "pack.json"

            pack = build_agentic_prestage_pack(
                output=output,
                movement_report_manifest=movement,
                gate_report_manifest=gate_report,
                next_action_gate=gate,
                grasp_outcome=grasp,
            )

        self.assertEqual(pack["status"], "passed")
        self.assertEqual(pack["purpose"], "pre-stage evidence for improving the agentic layer, not benchmark success")
        self.assertEqual(pack["gate_report_html"], "gate_report.html")
        self.assertEqual(pack["recommended_action"], "reframe_camera_0_or_object")
        lesson_text = " ".join(item["agentic_update"] for item in pack["agentic_lessons"])
        self.assertIn("better camera-0/end-effector view", lesson_text)
        self.assertIn("--record-video", lesson_text)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
