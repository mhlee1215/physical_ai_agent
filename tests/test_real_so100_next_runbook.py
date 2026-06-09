from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.real_so100_next_runbook import build_next_runbook


class RealSO100NextRunbookTest(unittest.TestCase):
    def test_blocked_runbook_holds_physical_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = root / "next_action_gate.json"
            movement = root / "movement_report.json"
            pack = root / "pack.json"
            output = root / "runbook.md"
            gate.write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "recommended_action": "reframe_camera_0_or_object",
                        "allowed_physical_action": None,
                        "blockers": ["camera 0 jaw/object framing gate is not ready"],
                    }
                ),
                encoding="utf-8",
            )
            movement.write_text(
                json.dumps(
                    {
                        "output_html": str(root / "movement.html"),
                        "video_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            pack.write_text(
                json.dumps(
                    {
                        "current_gate_status": "blocked",
                        "recommended_action": "reframe_camera_0_or_object",
                        "allowed_physical_action": None,
                        "movement_report_manifest": str(movement),
                        "movement_report_html": str(root / "movement.html"),
                        "gate_report_html": str(root / "gate.html"),
                        "next_action_gate": str(gate),
                        "grasp_outcome": str(root / "grasp.json"),
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_next_runbook(pre_stage_pack=pack, output=output)

            text = output.read_text(encoding="utf-8")
            self.assertEqual(manifest["contains_physical_command"], False)
            self.assertIn("do not close the gripper yet", text)
            self.assertIn("move the target appearance rightward/downward", text)
            self.assertIn("Gate report", text)
            self.assertIn("gate.html", text)
            self.assertIn("real_so100_checkpoint_26_gate.py", text)
            self.assertNotIn("real_so100_micro_step.py", text)

    def test_ready_runbook_requires_video_backed_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = root / "next_action_gate.json"
            movement = root / "movement_report.json"
            pack = root / "pack.json"
            output = root / "runbook.md"
            gate.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "recommended_action": "contact_probe_allowed",
                        "allowed_physical_action": {"joint": "gripper"},
                        "blockers": [],
                    }
                ),
                encoding="utf-8",
            )
            movement.write_text(json.dumps({"output_html": "movement.html", "video_count": 1}), encoding="utf-8")
            pack.write_text(
                json.dumps(
                    {
                        "current_gate_status": "ready",
                        "recommended_action": "contact_probe_allowed",
                        "allowed_physical_action": {"joint": "gripper"},
                        "movement_report_manifest": str(movement),
                        "movement_report_html": "movement.html",
                        "gate_report_html": "gate.html",
                        "next_action_gate": str(gate),
                        "grasp_outcome": str(root / "grasp.json"),
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_next_runbook(pre_stage_pack=pack, output=output)

            text = output.read_text(encoding="utf-8")
            self.assertEqual(manifest["contains_physical_command"], True)
            self.assertIn("real_so100_micro_step.py", text)
            self.assertIn("--record-video", text)
            self.assertIn("--camera-index 3", text)
            self.assertIn("--policy-camera-index 0", text)
            self.assertIn("--policy-camera-index 1", text)
            self.assertIn("--observer-camera-index 3", text)
            self.assertIn("--contact-ok-for-gripper", text)
            self.assertIn("--visual-output-dir _workspace/real_so100/gripper_contact_probe_next/visual", text)
            self.assertIn("build_real_so100_relocation_verifier_packet.py", text)
            self.assertIn("--execution-report _workspace/real_so100/gripper_contact_probe_next/report.json", text)
            self.assertIn("real_so100_object_relocation.py", text)
            self.assertIn("--target-direction right", text)


if __name__ == "__main__":
    unittest.main()
