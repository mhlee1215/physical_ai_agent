import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_next_action_gate import decide_next_action


class RealSO100NextActionGateTest(TestCase):
    def test_blocks_close_probe_when_jaw_readiness_is_blocked(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pregrasp = _write_json(
                tmp / "pregrasp.json",
                {"status": "passed", "primary_camera": "1", "usable_cameras": ["1"]},
            )
            jaw = _write_json(
                tmp / "jaw.json",
                {"status": "blocked", "blockers": ["green object touches image boundary"]},
            )
            grasp = _write_json(
                tmp / "grasp.json",
                {"status": "passed", "grasp_outcome": "grasp_failed_object_stationary"},
            )

            result = decide_next_action(pregrasp_probe=pregrasp, jaw_readiness=jaw, grasp_outcome=grasp)

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["recommended_action"], "reframe_camera_0_or_camera_1_or_object")
        self.assertIsNone(result["allowed_physical_action"])
        self.assertFalse(result["vla_prompt_allowed"])
        self.assertIn("camera 0 jaw/object framing gate is not ready", result["blockers"])

    def test_allows_vla_proposal_but_blocks_physical_action_when_wrist_object_is_clipped(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pregrasp = _write_json(
                tmp / "pregrasp.json",
                {"status": "passed", "primary_camera": "3", "usable_cameras": ["1", "3"]},
            )
            jaw = _write_json(
                tmp / "jaw.json",
                {
                    "status": "blocked",
                    "blockers": ["green object touches image boundary"],
                    "jaw_marker_candidate": {"bbox_xyxy": [100, 120, 180, 220]},
                    "object_candidate": {"bbox_xyxy": [0, 0, 240, 280]},
                },
            )
            grasp = _write_json(
                tmp / "grasp.json",
                {"status": "passed", "grasp_outcome": "grasp_failed_object_stationary"},
            )

            result = decide_next_action(pregrasp_probe=pregrasp, jaw_readiness=jaw, grasp_outcome=grasp)

        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["vla_prompt_allowed"])
        self.assertEqual(result["vla_prompt_gate"]["status"], "ready")
        self.assertIsNone(result["allowed_physical_action"])
        self.assertEqual(result["physical_execution_gate"]["status"], "blocked")

    def test_allows_contact_probe_when_pregrasp_and_jaw_are_ready(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pregrasp = _write_json(
                tmp / "pregrasp.json",
                {"status": "passed", "primary_camera": "1", "usable_cameras": ["1"]},
            )
            jaw = _write_json(
                tmp / "jaw.json",
                {"status": "ready", "blockers": [], "jaw_marker_candidate": {"bbox_xyxy": [100, 120, 180, 220]}},
            )
            grasp = _write_json(
                tmp / "grasp.json",
                {"status": "passed", "grasp_outcome": "grasp_failed_object_stationary"},
            )

            result = decide_next_action(pregrasp_probe=pregrasp, jaw_readiness=jaw, grasp_outcome=grasp)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["recommended_action"], "contact_probe_allowed_after_reframe")
        self.assertTrue(result["vla_prompt_allowed"])
        self.assertEqual(result["allowed_physical_action"]["joint"], "gripper")
        self.assertEqual(result["allowed_physical_action"]["object_view_camera"], "1")
        self.assertTrue(result["allowed_physical_action"]["requires_grasp_outcome_verifier"])


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
