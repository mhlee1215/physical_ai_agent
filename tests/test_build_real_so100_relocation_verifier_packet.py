from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_real_so100_relocation_verifier_packet import build_relocation_verifier_packet


class BuildRealSO100RelocationVerifierPacketTest(TestCase):
    def test_builds_waiting_packet_from_vla_prompt_packet(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            vla_packet = _write_json(tmp / "vla_packet.json", _vla_packet())

            packet = build_relocation_verifier_packet(
                vla_prompt_packet=vla_packet,
                output=tmp / "verifier_packet.json",
            )

        self.assertEqual(packet["status"], "waiting_for_before_after_images")
        self.assertFalse(packet["can_run"])
        self.assertTrue(packet["cannot_run_without_before_after_images"])
        self.assertFalse(packet["task_success_claim_allowed_without_this"])
        self.assertEqual(packet["vla_prompt_target"], "SmolVLA")
        self.assertEqual(packet["observer_camera_index"], "3")
        self.assertTrue(packet["policy_inputs_are_not_verifier_frame"])
        self.assertEqual(packet["target_direction"], "right")
        self.assertIn("after_object_center.image_x", packet["success_predicate"])
        self.assertIsNone(packet["command"])
        self.assertIn("${before_observer_image}", packet["command_template"])
        self.assertIn("right", packet["command_template"])
        self.assertIn("robot arm moves left", packet["coordinate_guardrails"]["not_equivalent_to"])

    def test_builds_executable_command_when_images_are_available(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            vla_packet = _write_json(tmp / "vla_packet.json", _vla_packet())
            before = _write_text(tmp / "before.jpg", "before")
            after = _write_text(tmp / "after.jpg", "after")
            result = tmp / "relocation.json"

            packet = build_relocation_verifier_packet(
                vla_prompt_packet=vla_packet,
                before_image=before,
                after_image=after,
                output=tmp / "verifier_packet.json",
                relocation_output=result,
            )

        self.assertEqual(packet["status"], "ready")
        self.assertTrue(packet["can_run"])
        self.assertFalse(packet["cannot_run_without_before_after_images"])
        self.assertEqual(packet["before_image"], str(before))
        self.assertEqual(packet["after_image"], str(after))
        self.assertEqual(packet["relocation_output"], str(result))
        self.assertIn("scripts/real_so100_object_relocation.py", packet["command"])
        self.assertEqual(packet["command"][packet["command"].index("--target-direction") + 1], "right")
        self.assertEqual(packet["command"][packet["command"].index("--min-delta-px") + 1], "40")
        self.assertNotIn("robot arm moves right", " ".join(packet["command"]))

    def test_materializes_before_after_from_observer_execution_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            vla_packet = _write_json(tmp / "vla_packet.json", _vla_packet())
            before = _write_text(tmp / "visual" / "before.jpg", "before")
            after = _write_text(tmp / "visual" / "after.jpg", "after")
            report = _write_json(
                tmp / "report.json",
                {
                    "status": "passed",
                    "send_action_called": True,
                    "camera_index": 3,
                    "visual_check": {
                        "before": {"camera_index": 3, "image_path": str(before)},
                        "after": {"camera_index": 3, "image_path": str(after), "before_path": str(before)},
                    },
                },
            )

            packet = build_relocation_verifier_packet(
                vla_prompt_packet=vla_packet,
                execution_report=report,
                output=tmp / "verifier_packet.json",
            )

        self.assertEqual(packet["status"], "ready")
        self.assertTrue(packet["can_run"])
        self.assertEqual(packet["source_execution_report"], str(report))
        self.assertEqual(packet["source_execution_camera_index"], "3")
        self.assertTrue(packet["observer_camera_matches_report"])
        self.assertEqual(packet["before_image"], str(before))
        self.assertEqual(packet["after_image"], str(after))
        self.assertEqual(packet["command"][packet["command"].index("--before") + 1], str(before))

    def test_blocks_legacy_non_observer_report_even_when_images_exist(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            vla_packet = _write_json(tmp / "vla_packet.json", _vla_packet())
            before = _write_text(tmp / "visual" / "before.jpg", "before")
            after = _write_text(tmp / "visual" / "after.jpg", "after")
            report = _write_json(
                tmp / "report.json",
                {
                    "status": "passed",
                    "send_action_called": True,
                    "camera_index": 2,
                    "visual_check": {
                        "before": {"camera_index": 2, "image_path": str(before)},
                        "after": {"camera_index": 2, "image_path": str(after), "before_path": str(before)},
                    },
                },
            )

            packet = build_relocation_verifier_packet(
                vla_prompt_packet=vla_packet,
                execution_report=report,
                output=tmp / "verifier_packet.json",
            )

        self.assertEqual(packet["status"], "observer_camera_mismatch")
        self.assertFalse(packet["can_run"])
        self.assertFalse(packet["observer_camera_matches_report"])
        self.assertEqual(packet["observer_camera_index"], "3")
        self.assertEqual(packet["source_execution_camera_index"], "2")
        self.assertIsNone(packet["command"])


def _vla_packet() -> dict:
    return {
        "vla_prompt": {
            "target": "SmolVLA",
            "policy_camera_indexes": ["0", "1"],
            "observer_camera_indexes_excluded_from_policy": ["3"],
        },
        "agentic_layer_contract": {"does_not_prompt_operator": True},
        "success_verifier": {
            "type": "object_relocation_image_space",
            "target_object": "green Android figure",
            "target_direction": "right",
            "success_predicate": "after_object_center.image_x - before_object_center.image_x >= min_delta_px",
            "min_delta_px_default": 40.0,
            "do_not_translate_goal_to_fixed_robot_direction": True,
            "frame": {
                "name": "observer_image_frame",
                "camera_indexes": ["3"],
                "primary_camera_index": "3",
                "axis": "image_x",
                "positive_direction": "right",
            },
        },
        "coordinate_semantics": {
            "not_equivalent_to": ["robot arm moves left", "robot arm moves right", "fixed joint sign"]
        },
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
