from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_agentic_loop_state import build_agentic_loop_state


class RealSO100AgenticLoopStateTest(TestCase):
    def test_builds_authoritative_observer_wait_state_with_regression_block(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            router = tmp / "router.json"
            memory = tmp / "memory.json"
            observation = tmp / "observation.json"
            router.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "policy_camera_indexes": [0, 1],
                        "observer_camera_indexes": [],
                        "observer_camera_status": "off",
                        "send_action_called": False,
                        "policy_actions_executed": False,
                        "physical_robot_motion": False,
                        "task_success_claim_allowed": False,
                        "selected_route": {
                            "type": "await_observer_camera_3",
                            "prompt_mutation_allowed": False,
                        },
                        "next_agentic_layer_step": {
                            "type": "wait_for_camera_3_then_run_live_readonly_refresh",
                            "reason": "Camera 3 is required.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            memory.write_text(
                json.dumps(
                    {
                        "best_candidate": {
                            "source_report": "best.json",
                            "candidate_index": 2,
                            "prompt": "best prompt",
                            "action_path": "action.json",
                            "execute_gate_path": "gate.json",
                            "score": {
                                "penalty_score": 4622.1257,
                                "range_violation_count": 11,
                                "ready_for_execution": False,
                            },
                        },
                        "regression_from_best": {
                            "is_regression": True,
                            "penalty_delta": 16303.1839,
                        },
                    }
                ),
                encoding="utf-8",
            )
            observation.write_text(json.dumps({"task": "Pick up the green figure and move it right."}), encoding="utf-8")

            state = build_agentic_loop_state(
                router_report=router,
                candidate_memory=memory,
                observation_manifest=observation,
                output=tmp / "state.json",
            )

        self.assertEqual(state["selected_route"]["type"], "await_observer_camera_3")
        self.assertEqual(state["allowed_next_actions"][0]["type"], "wait_for_camera_3_then_run_live_readonly_refresh")
        self.assertFalse(state["allowed_next_actions"][0]["physical_robot_motion"])
        self.assertEqual(state["best_historical_candidate"]["candidate_index"], 2)
        blocked_types = {item["type"] for item in state["blocked_actions"]}
        self.assertIn("physical_execution", blocked_types)
        self.assertIn("task_success_claim", blocked_types)
        self.assertIn("rerun_regressed_policy_camera_prompt", blocked_types)
        self.assertIn("prompt_mutation_before_observer_refresh", blocked_types)
        self.assertFalse(state["execution_flags"]["physical_robot_motion"])
        self.assertFalse(state["execution_flags"]["task_success_claim_allowed"])
