from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_agentic_feedback_router import route_agentic_feedback


class RealSO100AgenticFeedbackRouterTest(TestCase):
    def test_routes_execution_preflight_without_prompt_mutation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            feedback = tmp / "feedback.json"
            feedback.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_transition_execution_feedback",
                        "status": "passed",
                        "camera_contract": {
                            "policy_camera_indexes": [0, 1],
                            "observer_camera_indexes": [],
                            "observer_camera_status": "off",
                        },
                        "execution_outcome": {
                            "send_action_called": False,
                            "policy_actions_executed": False,
                            "physical_robot_motion": False,
                        },
                        "failure_modes": [
                            "execution_packet_not_ready",
                            "observer_or_live_readback_preflight_incomplete",
                        ],
                        "prompt_mutation_allowed": False,
                        "task_success_claim_allowed": False,
                    }
                ),
                encoding="utf-8",
            )

            report = route_agentic_feedback(feedback_reports=[feedback], output=tmp / "router.json")

        self.assertEqual(report["selected_route"]["type"], "resolve_execution_preflight")
        self.assertFalse(report["selected_route"]["prompt_mutation_allowed"])
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "rerun_observer_return_refresh_live_readonly_when_camera_3_available",
        )
        self.assertEqual(report["feedback_items"][0]["observer_camera_indexes"], [])
        self.assertFalse(report["physical_robot_motion"])
        self.assertFalse(report["task_success_claim_allowed"])

    def test_routes_executed_missing_verifiers_to_task_verifier(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            feedback = tmp / "feedback.json"
            feedback.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_transition_execution_feedback",
                        "status": "passed",
                        "camera_contract": {
                            "policy_camera_indexes": [0, 1],
                            "observer_camera_indexes": [3],
                            "observer_camera_status": "available",
                        },
                        "execution_outcome": {
                            "send_action_called": True,
                            "policy_actions_executed": True,
                            "physical_robot_motion": True,
                        },
                        "failure_modes": ["grasp_outcome_not_verified", "task_success_not_verified"],
                        "prompt_mutation_allowed": True,
                        "task_success_claim_allowed": False,
                    }
                ),
                encoding="utf-8",
            )

            report = route_agentic_feedback(feedback_reports=[feedback], output=tmp / "router.json")

        self.assertEqual(report["selected_route"]["type"], "run_task_verifiers")
        self.assertFalse(report["selected_route"]["prompt_mutation_allowed"])
        self.assertEqual(report["next_agentic_layer_step"]["type"], "run_grasp_and_relocation_verifiers")
        self.assertTrue(report["physical_robot_motion"])

    def test_routes_candidate_memory_to_prompt_mutation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            memory = tmp / "memory.json"
            memory.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_agentic_candidate_memory",
                        "status": "passed",
                        "policy_camera_indexes": [0, 1],
                        "observer_camera_indexes": [],
                        "observer_camera_status": "temporarily_unavailable",
                        "next_agentic_layer_step": {
                            "type": "reuse_best_historical_prompt_family",
                        },
                        "task_success_claim_allowed": False,
                    }
                ),
                encoding="utf-8",
            )

            report = route_agentic_feedback(feedback_reports=[memory], output=tmp / "router.json")

        self.assertEqual(report["selected_route"]["type"], "mutate_smolvla_prompt_or_plan")
        self.assertTrue(report["selected_route"]["prompt_mutation_allowed"])
        self.assertEqual(report["next_agentic_layer_step"]["type"], "run_no_actuation_proposal_sweep")

    def test_regressed_candidate_memory_preserves_best_candidate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            memory = tmp / "memory.json"
            memory.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_agentic_candidate_memory",
                        "status": "passed",
                        "policy_camera_indexes": [0, 1],
                        "observer_camera_indexes": [],
                        "observer_camera_status": "temporarily_unavailable",
                        "regression_from_best": {
                            "is_regression": True,
                            "penalty_delta": 16303.1839,
                        },
                        "next_agentic_layer_step": {
                            "type": "reuse_best_historical_prompt_family",
                        },
                        "task_success_claim_allowed": False,
                    }
                ),
                encoding="utf-8",
            )

            report = route_agentic_feedback(feedback_reports=[memory], output=tmp / "router.json")

        self.assertEqual(report["selected_route"]["type"], "preserve_best_historical_candidate")
        self.assertFalse(report["selected_route"]["prompt_mutation_allowed"])
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "preserve_best_transition_candidate_until_observer_gate",
        )
        self.assertFalse(report["physical_robot_motion"])

    def test_existing_await_observer_router_artifact_stays_await_observer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            router = tmp / "router_feedback.json"
            router.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_agentic_feedback_router",
                        "status": "passed",
                        "policy_camera_indexes": [0, 1],
                        "observer_camera_indexes": [],
                        "observer_camera_status": "off",
                        "selected_route": {
                            "type": "await_observer_camera_3",
                            "prompt_mutation_allowed": False,
                        },
                        "next_agentic_layer_step": {
                            "type": "wait_for_camera_3_then_run_live_readonly_refresh",
                        },
                        "send_action_called": False,
                        "policy_actions_executed": False,
                        "physical_robot_motion": False,
                        "task_success_claim_allowed": False,
                    }
                ),
                encoding="utf-8",
            )

            report = route_agentic_feedback(feedback_reports=[router], output=tmp / "router.json")

        self.assertEqual(report["selected_route"]["type"], "await_observer_camera_3")
        self.assertFalse(report["selected_route"]["prompt_mutation_allowed"])
        self.assertEqual(report["next_agentic_layer_step"]["type"], "wait_for_camera_3_then_run_live_readonly_refresh")

    def test_passed_runbook_audit_waits_for_observer_camera_3(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            old_blocker = tmp / "old_blocker.json"
            old_blocker.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_transition_execution_feedback",
                        "status": "passed",
                        "camera_contract": {
                            "policy_camera_indexes": [0, 1],
                            "observer_camera_indexes": [],
                            "observer_camera_status": "temporarily_unavailable",
                        },
                        "execution_outcome": {
                            "send_action_called": False,
                            "policy_actions_executed": False,
                            "physical_robot_motion": False,
                        },
                        "failure_modes": ["observer_or_live_readback_preflight_incomplete"],
                        "prompt_mutation_allowed": False,
                        "task_success_claim_allowed": False,
                    }
                ),
                encoding="utf-8",
            )
            audit = tmp / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_execution_preflight_runbook_audit",
                        "status": "passed",
                        "failed_check_count": 0,
                        "policy_camera_indexes": [0, 1],
                        "observer_camera_indexes": [],
                        "observer_camera_status": "temporarily_unavailable",
                        "send_action_called": False,
                        "policy_actions_executed": False,
                        "physical_robot_motion": False,
                        "task_success_claim_allowed": False,
                        "next_agentic_layer_step": {
                            "type": "safe_to_run_live_readonly_refresh_when_camera_3_returns",
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = route_agentic_feedback(feedback_reports=[old_blocker, audit], output=tmp / "router.json")

        self.assertEqual(report["selected_route"]["type"], "await_observer_camera_3")
        self.assertFalse(report["selected_route"]["prompt_mutation_allowed"])
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "wait_for_camera_3_then_run_live_readonly_refresh",
        )
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertFalse(report["task_success_claim_allowed"])

    def test_failed_runbook_audit_repairs_preflight_without_prompt_mutation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            audit = tmp / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_execution_preflight_runbook_audit",
                        "status": "blocked",
                        "failed_check_count": 1,
                        "policy_camera_indexes": [0, 1],
                        "observer_camera_indexes": [],
                        "observer_camera_status": "temporarily_unavailable",
                        "send_action_called": False,
                        "policy_actions_executed": False,
                        "physical_robot_motion": False,
                        "task_success_claim_allowed": False,
                    }
                ),
                encoding="utf-8",
            )

            report = route_agentic_feedback(feedback_reports=[audit], output=tmp / "router.json")

        self.assertEqual(report["selected_route"]["type"], "fix_execution_preflight_runbook")
        self.assertFalse(report["selected_route"]["prompt_mutation_allowed"])
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "repair_and_reaudit_execution_preflight_runbook",
        )
