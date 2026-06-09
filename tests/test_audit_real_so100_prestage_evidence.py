from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_real_so100_prestage_evidence import audit_prestage_evidence


class AuditRealSO100PrestageEvidenceTest(unittest.TestCase):
    def test_audit_passes_for_video_backed_blocked_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "motion.mp4"
            video.write_bytes(b"fake video bytes")
            preview = root / "motion_preview.gif"
            preview.write_bytes(b"gif")
            movement_html = root / "movement.html"
            movement_html.write_text("pre-stage movement", encoding="utf-8")
            gate_html = root / "gate.html"
            gate_html.write_text("gate pre-stage", encoding="utf-8")
            overlay = root / "camera_0_overlay.jpg"
            overlay.write_bytes(b"jpg")
            next_action = _write_json(root / "next_action.json", {"status": "blocked"})
            grasp = _write_json(root / "grasp.json", {"grasp_outcome": "grasp_failed_object_stationary"})
            report = _write_json(
                root / "movement_report.json",
                {
                    "motion_video": {
                        "path": str(video),
                        "actual_codec": "FMP4",
                        "actual_frame_count": 4,
                        "first_frame_readable": True,
                        "browser_preview_recommended": True,
                    }
                },
            )
            movement = _write_json(
                root / "movement_manifest.json",
                {
                    "output_html": str(movement_html),
                    "video_count": 1,
                    "video_preview_count": 1,
                    "video_previews": [str(preview)],
                    "reports": [str(report)],
                },
            )
            gate = _write_json(
                root / "gate_manifest.json",
                {
                    "output_html": str(gate_html),
                    "current_gate_status": "blocked",
                    "overlays": {"0": str(overlay)},
                },
            )
            pack = _write_json(
                root / "pack.json",
                {
                    "purpose": "pre-stage evidence for improving the agentic layer, not benchmark success",
                    "movement_report_manifest": str(movement),
                    "movement_report_html": str(movement_html),
                    "gate_report_manifest": str(gate),
                    "gate_report_html": str(gate_html),
                    "next_action_gate": str(next_action),
                    "grasp_outcome": str(grasp),
                    "current_gate_status": "blocked",
                    "video_count": 1,
                },
            )
            runbook_md = root / "runbook.md"
            runbook_md.write_text("do not close the gripper", encoding="utf-8")
            runbook = _write_json(
                root / "runbook_manifest.json",
                {
                    "output_markdown": str(runbook_md),
                    "contains_physical_command": False,
                    "gate_report_html": str(gate_html),
                },
            )

            result = audit_prestage_evidence(pre_stage_pack=pack, runbook_manifest=runbook)

            self.assertEqual(result["status"], "passed")
            names = {item["name"]: item["status"] for item in result["checks"]}
            self.assertEqual(names["motion_video_0"], "passed")
            self.assertEqual(names["motion_video_0_has_probe_metadata"], "passed")
            self.assertEqual(names["motion_video_0_first_frame_readable"], "passed")
            self.assertEqual(names["motion_video_0_frame_count_positive"], "passed")
            self.assertEqual(names["motion_video_preview_0"], "passed")
            self.assertEqual(names["movement_has_preview_for_each_video"], "passed")
            self.assertEqual(names["blocked_runbook_has_no_physical_command"], "passed")

    def test_audit_fails_when_video_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            movement_html = root / "movement.html"
            movement_html.write_text("pre-stage movement", encoding="utf-8")
            report = _write_json(root / "movement_report.json", {"motion_video": {"path": str(root / "missing.mp4")}})
            movement = _write_json(
                root / "movement_manifest.json",
                {"output_html": str(movement_html), "video_count": 1, "video_preview_count": 0, "video_previews": [], "reports": [str(report)]},
            )
            gate_html = root / "gate.html"
            gate_html.write_text("gate", encoding="utf-8")
            gate = _write_json(root / "gate_manifest.json", {"output_html": str(gate_html), "current_gate_status": "blocked"})
            next_action = _write_json(root / "next_action.json", {"status": "blocked"})
            grasp = _write_json(root / "grasp.json", {})
            pack = _write_json(
                root / "pack.json",
                {
                    "purpose": "pre-stage evidence for improving the agentic layer, not benchmark success",
                    "movement_report_manifest": str(movement),
                    "movement_report_html": str(movement_html),
                    "gate_report_manifest": str(gate),
                    "gate_report_html": str(gate_html),
                    "next_action_gate": str(next_action),
                    "grasp_outcome": str(grasp),
                    "current_gate_status": "blocked",
                    "video_count": 1,
                },
            )

            result = audit_prestage_evidence(pre_stage_pack=pack)

            self.assertEqual(result["status"], "failed")
            failed_names = {item["name"] for item in result["checks"] if item["status"] == "failed"}
            self.assertIn("motion_video_0", failed_names)
            self.assertIn("movement_has_preview_for_each_video", failed_names)

    def test_audit_fails_when_motion_video_probe_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "motion.mp4"
            video.write_bytes(b"fake video bytes")
            preview = root / "motion_preview.gif"
            preview.write_bytes(b"gif")
            movement_html = root / "movement.html"
            movement_html.write_text("pre-stage movement", encoding="utf-8")
            report = _write_json(root / "movement_report.json", {"motion_video": {"path": str(video)}})
            movement = _write_json(
                root / "movement_manifest.json",
                {
                    "output_html": str(movement_html),
                    "video_count": 1,
                    "video_preview_count": 1,
                    "video_previews": [str(preview)],
                    "reports": [str(report)],
                },
            )
            gate_html = root / "gate.html"
            gate_html.write_text("gate", encoding="utf-8")
            gate = _write_json(root / "gate_manifest.json", {"output_html": str(gate_html), "current_gate_status": "blocked"})
            next_action = _write_json(root / "next_action.json", {"status": "blocked"})
            grasp = _write_json(root / "grasp.json", {})
            pack = _write_json(
                root / "pack.json",
                {
                    "purpose": "pre-stage evidence for improving the agentic layer, not benchmark success",
                    "movement_report_manifest": str(movement),
                    "movement_report_html": str(movement_html),
                    "gate_report_manifest": str(gate),
                    "gate_report_html": str(gate_html),
                    "next_action_gate": str(next_action),
                    "grasp_outcome": str(grasp),
                    "current_gate_status": "blocked",
                    "video_count": 1,
                },
            )

            result = audit_prestage_evidence(pre_stage_pack=pack)

            self.assertEqual(result["status"], "failed")
            failed_names = {item["name"] for item in result["checks"] if item["status"] == "failed"}
            self.assertIn("motion_video_0_has_probe_metadata", failed_names)
            self.assertIn("motion_video_0_first_frame_readable", failed_names)
            self.assertIn("motion_video_0_frame_count_positive", failed_names)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
