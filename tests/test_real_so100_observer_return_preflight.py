from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_observer_return_preflight import build_observer_return_preflight


class RealSO100ObserverReturnPreflightTest(TestCase):
    def test_blocks_while_observer_camera_is_unavailable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gate = tmp / "gate.json"
            gate.write_text(json.dumps(_transition_gate()), encoding="utf-8")

            report = build_observer_return_preflight(
                transition_gate=gate,
                output=tmp / "preflight.json",
                observer_camera_status="temporarily_unavailable",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["execution_ready_with_observer"])
        self.assertTrue(any("Observer camera 3 must be available" in blocker for blocker in report["blockers"]))
        self.assertIn("Transition must be regenerated from live SO-100 readback before execution.", report["blockers"])
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertFalse(report["task_success_claim_allowed"])

    def test_ready_only_when_observer_live_readback_and_confirmations_pass(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gate = tmp / "gate.json"
            gate.write_text(json.dumps(_transition_gate()), encoding="utf-8")

            report = build_observer_return_preflight(
                transition_gate=gate,
                output=tmp / "preflight.json",
                observer_camera_status="available",
                live_readback_regenerated=True,
                user_confirmed=True,
                workspace_clear_confirmed=True,
            )

        self.assertEqual(report["status"], "ready_for_observer_backed_execution_gate")
        self.assertTrue(report["execution_ready_with_observer"])
        self.assertEqual(report["observer_camera_indexes"], [3])
        self.assertEqual(report["next_agentic_layer_step"]["type"], "open_observer_backed_physical_execution_gate")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])

    def test_blocks_if_observer_camera_is_policy_input(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gate = tmp / "gate.json"
            payload = _transition_gate()
            payload["policy_camera_indexes"] = ["0", "3"]
            gate.write_text(json.dumps(payload), encoding="utf-8")

            report = build_observer_return_preflight(
                transition_gate=gate,
                output=tmp / "preflight.json",
                observer_camera_status="available",
                live_readback_regenerated=True,
                user_confirmed=True,
                workspace_clear_confirmed=True,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("must not be a SmolVLA policy input" in blocker for blocker in report["blockers"]))


def _transition_gate() -> dict:
    return {
        "operation": "real_so100_transition_candidate_gate",
        "status": "passed",
        "policy_camera_indexes": ["0", "1"],
        "observer_camera_indexes": [],
        "observer_camera_status": "temporarily_unavailable",
        "camera_3_status": "off",
        "send_action_called": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "transition_chunk_count": 2,
        "transition_step_count": 20,
    }
