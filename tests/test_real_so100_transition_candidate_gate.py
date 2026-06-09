from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_transition_candidate_gate import gate_transition_candidate


class RealSO100TransitionCandidateGateTest(TestCase):
    def test_passes_two_bounded_ten_step_chunks(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            transition = tmp / "transition.json"
            transition.write_text(json.dumps(_transition_plan()), encoding="utf-8")

            report = gate_transition_candidate(
                transition_plan=transition,
                output=tmp / "gate.json",
                expected_chunk_size=10,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["transition_chunk_count"], 2)
        self.assertEqual(report["transition_step_count"], 20)
        self.assertTrue(all(chunk["all_targets_in_range"] for chunk in report["chunks"]))
        self.assertEqual(report["next_agentic_layer_step"]["type"], "wait_for_observer_camera_3_before_physical_execution_gate")
        self.assertFalse(report["execution_ready_with_observer"])
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertFalse(report["task_success_claim_allowed"])

    def test_blocks_when_chunk_size_is_not_ten(self) -> None:
        plan = _transition_plan()
        plan["transition_steps"] = plan["transition_steps"][:-1]
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            transition = tmp / "transition.json"
            transition.write_text(json.dumps(plan), encoding="utf-8")

            report = gate_transition_candidate(
                transition_plan=transition,
                output=tmp / "gate.json",
                expected_chunk_size=10,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("expected 10" in blocker for blocker in report["blockers"]))
        self.assertFalse(report["send_action_called"])

    def test_blocks_when_observer_off_contract_is_broken(self) -> None:
        plan = _transition_plan()
        plan["observer_camera_indexes"] = [3]
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            transition = tmp / "transition.json"
            transition.write_text(json.dumps(plan), encoding="utf-8")

            report = gate_transition_candidate(
                transition_plan=transition,
                output=tmp / "gate.json",
                expected_chunk_size=10,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("Observer cameras must be empty", report["blockers"][0])
        self.assertFalse(report["send_action_called"])


def _transition_plan() -> dict:
    steps = []
    for step_index in range(20):
        chunk_index = step_index // 10
        step_index_in_chunk = step_index % 10
        raw = 1000.0 + step_index * 10.0
        steps.append(
            {
                "step_index": step_index,
                "chunk_index": chunk_index,
                "step_index_in_chunk": step_index_in_chunk,
                "joint_targets": [
                    {
                        "joint": joint,
                        "target_raw": raw,
                        "target_command_value": raw / 10.0,
                        "range_min": 0.0,
                        "range_max": 4095.0,
                        "raw_target_in_calibrated_range": True,
                    }
                    for joint in [
                        "shoulder_pan",
                        "shoulder_lift",
                        "elbow_flex",
                        "wrist_flex",
                        "wrist_roll",
                        "gripper",
                    ]
                ],
            }
        )
    return {
        "operation": "real_so100_bridge_transition_plan",
        "status": "passed",
        "policy_camera_indexes": ["0", "1"],
        "observer_camera_indexes": [],
        "observer_camera_status": "temporarily_unavailable",
        "camera_3_status": "off",
        "send_action_called": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "max_abs_raw_delta_per_step": 80.0,
        "transition_steps": copy.deepcopy(steps),
    }
