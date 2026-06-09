from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_transition_execution_feedback import build_transition_execution_feedback


class RealSO100TransitionExecutionFeedbackTest(TestCase):
    def test_blocked_dry_run_preserves_candidate_without_prompt_mutation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            packet = _write_packet(tmp, ready=False)
            execution = _write_execution(tmp, packet=packet, executed=False, status="dry_run")

            report = build_transition_execution_feedback(
                execution_report=execution,
                output=tmp / "feedback.json",
            )

        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["execution_outcome"]["send_action_called"])
        self.assertFalse(report["execution_outcome"]["physical_robot_motion"])
        self.assertFalse(report["prompt_mutation_allowed"])
        self.assertFalse(report["task_success_claim_allowed"])
        self.assertIn("execution_packet_not_ready", report["failure_modes"])
        self.assertIn("observer_or_live_readback_preflight_incomplete", report["failure_modes"])
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "rerun_observer_return_refresh_live_readonly_when_camera_3_available",
        )

    def test_executed_report_without_verifiers_requests_task_verifiers(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            packet = _write_packet(tmp, ready=True)
            execution = _write_execution(tmp, packet=packet, executed=True, status="passed")

            report = build_transition_execution_feedback(
                execution_report=execution,
                output=tmp / "feedback.json",
            )

        self.assertTrue(report["execution_outcome"]["send_action_called"])
        self.assertTrue(report["execution_outcome"]["physical_robot_motion"])
        self.assertTrue(report["prompt_mutation_allowed"])
        self.assertFalse(report["task_success_claim_allowed"])
        self.assertIn("grasp_outcome_not_verified", report["failure_modes"])
        self.assertIn("task_success_not_verified", report["failure_modes"])
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "run_grasp_and_relocation_verifiers_before_prompt_mutation",
        )

    def test_grasp_and_relocation_pass_make_task_success_candidate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            packet = _write_packet(tmp, ready=True)
            execution = _write_execution(tmp, packet=packet, executed=True, status="passed")
            grasp = tmp / "grasp.json"
            grasp.write_text(json.dumps({"status": "passed", "grasp_outcome": "object_moved_or_occluded_candidate"}), encoding="utf-8")
            relocation = tmp / "relocation.json"
            relocation.write_text(
                json.dumps({"status": "passed", "relocation_outcome": "object_moved_right", "task_success_candidate": True}),
                encoding="utf-8",
            )

            report = build_transition_execution_feedback(
                execution_report=execution,
                output=tmp / "feedback.json",
                grasp_outcome=grasp,
                relocation_outcome=relocation,
            )

        self.assertTrue(report["task_success_candidate"])
        self.assertTrue(report["task_success_claim_allowed"])
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "record_task_success_candidate_and_prepare_repro_run",
        )


def _write_packet(root: Path, *, ready: bool) -> Path:
    packet = root / "packet.json"
    packet.write_text(
        json.dumps(
            {
                "status": "ready_for_observer_backed_execution" if ready else "blocked",
                "execution_ready": ready,
                "live_readback_regenerated": ready,
                "observer_camera_status": "available" if ready else "off",
                "observer_camera_indexes": [3] if ready else [],
                "policy_camera_indexes": ["0", "1"],
                "transition_chunk_count": 2,
                "transition_step_count": 20,
            }
        ),
        encoding="utf-8",
    )
    return packet


def _write_execution(root: Path, *, packet: Path, executed: bool, status: str) -> Path:
    execution = root / "execution.json"
    payload = {
        "status": status,
        "packet": str(packet),
        "packet_status": "ready_for_observer_backed_execution" if executed else "blocked",
        "packet_execution_ready": executed,
        "observer_camera_index": 3,
        "send_action_called": executed,
        "policy_actions_executed": executed,
        "physical_robot_motion": executed,
        "transition_chunk_count": 2,
        "transition_step_count": 20,
        "blockers": [] if executed else ["Transition execution packet is not ready."],
    }
    if executed:
        payload.update(
            {
                "execute_requested": True,
                "executed_action_steps": 20,
                "motion_video": {"path": str(root / "motion.mp4"), "exists": True},
                "visual_check": {
                    "before": {"image_path": str(root / "before.jpg")},
                    "after": {"image_path": str(root / "after.jpg")},
                },
                "readback_before_raw": {"shoulder_pan": 1},
                "readback_after_raw": {"shoulder_pan": 2},
                "observed_delta_raw": {"shoulder_pan": 1},
            }
        )
    execution.write_text(json.dumps(payload), encoding="utf-8")
    return execution
