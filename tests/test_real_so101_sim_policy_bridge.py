from __future__ import annotations

import math
import unittest

from physical_ai_agent.real_so100.sim_policy_bridge import (
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    JOINT_ORDER,
    clamp_hardware_positions,
    hardware_position_limits_from_calibration,
    hardware_positions_to_sim_qpos,
    raw_hardware_positions_to_lerobot_positions,
    raw_hardware_positions_to_sim_qpos,
    sim_qpos_to_hardware_positions,
)


class RealSO101SimPolicyBridgeTest(unittest.TestCase):
    def test_raw_readback_uses_calibration_before_sim_conversion(self) -> None:
        calibration = {
            joint: {"range_min": 1000, "range_max": 3000} for joint in JOINT_ORDER
        }
        raw = {joint: 2000.0 for joint in JOINT_ORDER}
        normalized = raw_hardware_positions_to_lerobot_positions(raw, calibration)
        qpos = raw_hardware_positions_to_sim_qpos(raw, calibration)

        self.assertEqual([normalized[joint] for joint in JOINT_ORDER], [0.0] * 5 + [50.0])
        self.assertEqual(qpos[:5], [0.0] * 5)
        self.assertAlmostEqual(qpos[-1], (GRIPPER_CLOSED_RAD + GRIPPER_OPEN_RAD) / 2.0)

    def test_round_trip_hardware_positions(self) -> None:
        hardware = {
            "shoulder_pan": -15.0,
            "shoulder_lift": -75.0,
            "elbow_flex": 70.0,
            "wrist_flex": 65.0,
            "wrist_roll": 5.0,
            "gripper": 37.0,
        }

        restored = sim_qpos_to_hardware_positions(hardware_positions_to_sim_qpos(hardware))

        for joint in JOINT_ORDER:
            self.assertAlmostEqual(restored[joint], hardware[joint], places=5)

    def test_gripper_endpoints_match_training_qpos(self) -> None:
        closed = {joint: 0.0 for joint in JOINT_ORDER}
        opened = dict(closed, gripper=100.0)

        self.assertAlmostEqual(hardware_positions_to_sim_qpos(closed)[-1], GRIPPER_CLOSED_RAD)
        self.assertAlmostEqual(hardware_positions_to_sim_qpos(opened)[-1], GRIPPER_OPEN_RAD)
        self.assertEqual(sim_qpos_to_hardware_positions([0.0] * 5 + [math.radians(-20.0)])["gripper"], 0.0)
        self.assertEqual(sim_qpos_to_hardware_positions([0.0] * 5 + [math.radians(110.0)])["gripper"], 100.0)

    def test_calibration_limits_match_lerobot_degree_normalization(self) -> None:
        calibration = {
            joint: {"range_min": 1000, "range_max": 3000} for joint in JOINT_ORDER
        }
        limits = hardware_position_limits_from_calibration(calibration)

        expected_degrees = 1000 * 360 / 4095
        self.assertAlmostEqual(limits["shoulder_pan"][0], -expected_degrees)
        self.assertAlmostEqual(limits["shoulder_pan"][1], expected_degrees)
        self.assertEqual(limits["gripper"], (0.0, 100.0))

    def test_clamps_hardware_targets_to_calibration(self) -> None:
        limits = {joint: (-70.0, 70.0) for joint in JOINT_ORDER}
        limits["gripper"] = (0.0, 100.0)
        requested = {joint: 0.0 for joint in JOINT_ORDER}
        requested.update(elbow_flex=90.0, wrist_flex=-90.0, gripper=105.0)

        clamped, changes = clamp_hardware_positions(requested, limits)

        self.assertEqual(clamped["elbow_flex"], 70.0)
        self.assertEqual(clamped["wrist_flex"], -70.0)
        self.assertEqual(clamped["gripper"], 100.0)
        self.assertEqual(set(changes), {"elbow_flex", "wrist_flex", "gripper"})


if __name__ == "__main__":
    unittest.main()
