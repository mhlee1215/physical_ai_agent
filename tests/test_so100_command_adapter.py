from unittest import TestCase

from physical_ai_agent.safety.so100_command_adapter import build_so100_command_chunk_plan, build_so100_command_plan


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


class SO100CommandAdapterTest(TestCase):
    def test_builds_no_actuation_command_plan(self) -> None:
        plan = build_so100_command_plan(
            action=[-0.1, 0.1, -1.0, 0.5, 1.0, -0.8],
            current_state=CURRENT_STATE,
            calibration=CALIBRATION,
        )

        self.assertEqual(plan.status, "passed")
        self.assertFalse(plan.ready_for_execution)
        self.assertFalse(plan.send_action_called)
        self.assertFalse(plan.policy_actions_executed)
        self.assertEqual(plan.command_units, "feetech_raw_ticks")
        self.assertEqual(len(plan.joint_plans), 6)
        self.assertAlmostEqual(plan.joint_plans[0].target_raw, 2081.9)
        self.assertIn("Human confirmation", " ".join(plan.blockers))
        self.assertIn("Adapter semantics", " ".join(plan.blockers))

    def test_clips_delta_and_calibrated_range(self) -> None:
        plan = build_so100_command_plan(
            action=[99.0, 99.0, 99.0, 99.0, 99.0, -99.0],
            current_state=CURRENT_STATE,
            calibration=CALIBRATION,
            human_confirmed=True,
            adapter_semantics_confirmed=True,
        )

        self.assertTrue(plan.ready_for_execution)
        self.assertTrue(all(abs(joint.clipped_delta_raw) <= 2.0 for joint in plan.joint_plans))
        self.assertTrue(plan.joint_plans[2].clipped_by_calibrated_range)
        self.assertEqual(plan.joint_plans[2].target_raw, 2048)
        self.assertTrue(plan.joint_plans[5].target_raw >= CALIBRATION["gripper"]["range_min"])

    def test_wrong_action_dimension_blocks_plan(self) -> None:
        plan = build_so100_command_plan(
            action=[0.0],
            current_state=CURRENT_STATE,
            calibration=CALIBRATION,
            human_confirmed=True,
            adapter_semantics_confirmed=True,
        )

        self.assertEqual(plan.status, "blocked")
        self.assertFalse(plan.ready_for_execution)
        self.assertEqual(plan.action_dim, 1)

    def test_builds_no_actuation_command_chunk_plan(self) -> None:
        chunk = [[0.5, 0.25, -0.5, 0.25, 0.0, -0.25] for _index in range(10)]

        plan = build_so100_command_chunk_plan(
            action_chunk=chunk,
            current_state=CURRENT_STATE,
            calibration=CALIBRATION,
        )

        self.assertEqual(plan.status, "passed")
        self.assertFalse(plan.ready_for_execution)
        self.assertFalse(plan.send_action_called)
        self.assertFalse(plan.policy_actions_executed)
        self.assertEqual(plan.action_chunk_steps, 10)
        self.assertEqual(plan.action_dim, 6)
        self.assertEqual(len(plan.step_plans), 10)
        self.assertAlmostEqual(plan.step_plans[0].joint_plans[0].target_raw, 2082.5)
        self.assertAlmostEqual(plan.step_plans[-1].joint_plans[0].target_raw, 2087.0)
        self.assertIn("Human confirmation", " ".join(plan.blockers))
        self.assertIn("Adapter semantics", " ".join(plan.blockers))
