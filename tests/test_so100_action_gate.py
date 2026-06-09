import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from physical_ai_agent.safety.so100_action_gate import evaluate_so100_action_safety, load_action_chunk_payload


CURRENT_STATE = {
    "shoulder_pan": 2082,
    "shoulder_lift": 2049,
    "elbow_flex": 2047,
    "wrist_flex": 1966,
    "wrist_roll": 2050,
    "gripper": 1711,
}

CALIBRATION = {
    "shoulder_pan": {"range_min": 1363, "range_max": 2443},
    "shoulder_lift": {"range_min": 2001, "range_max": 3695},
    "elbow_flex": {"range_min": 468, "range_max": 2048},
    "wrist_flex": {"range_min": 1315, "range_max": 2047},
    "wrist_roll": {"range_min": 0, "range_max": 4095},
    "gripper": {"range_min": 1658, "range_max": 2747},
}


class SO100ActionGateTest(TestCase):
    def test_candidate_can_pass_checks_while_execution_stays_blocked(self) -> None:
        report = evaluate_so100_action_safety(
            action=[-0.1, 0.1, -1.0, 0.5, 1.0, -0.8],
            current_state=CURRENT_STATE,
            calibration=CALIBRATION,
            human_confirmed=False,
        )

        self.assertEqual(report.status, "passed")
        self.assertTrue(report.candidate_safe)
        self.assertFalse(report.execution_allowed)
        self.assertFalse(report.send_action_called)
        self.assertFalse(report.policy_actions_executed)
        self.assertIn("Human confirmation is required", " ".join(report.blockers))
        self.assertIn("action semantics", " ".join(report.blockers))

    def test_large_delta_blocks_candidate(self) -> None:
        report = evaluate_so100_action_safety(
            action=[-0.1, 9.0, -1.0, 0.5, 1.0, -0.8],
            current_state=CURRENT_STATE,
            calibration=CALIBRATION,
            human_confirmed=True,
            require_known_action_semantics=False,
        )

        self.assertEqual(report.status, "blocked")
        self.assertFalse(report.candidate_safe)
        self.assertFalse(report.execution_allowed)
        self.assertIn("action_delta_too_large", report.checks[1].reasons)

    def test_wrong_action_dimension_blocks_candidate(self) -> None:
        report = evaluate_so100_action_safety(
            action=[0.0, 0.0],
            current_state=CURRENT_STATE,
            calibration=CALIBRATION,
            human_confirmed=True,
            require_known_action_semantics=False,
        )

        self.assertEqual(report.status, "blocked")
        self.assertFalse(report.candidate_safe)
        self.assertEqual(report.action_dim, 2)
        self.assertEqual(report.expected_action_dim, 6)

    def test_load_action_chunk_payload_prefers_chunk_steps(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "action.json"
            path.write_text(
                json.dumps(
                    {
                        "raw_action": [99, 99, 99, 99, 99, 99],
                        "raw_action_chunk": [[step + joint for joint in range(6)] for step in range(12)],
                    }
                ),
                encoding="utf-8",
            )

            chunk = load_action_chunk_payload(path, action_steps=10)

        self.assertEqual(len(chunk), 10)
        self.assertEqual(chunk[0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(chunk[-1], [9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
