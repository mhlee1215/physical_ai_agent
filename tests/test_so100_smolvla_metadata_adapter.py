from unittest import TestCase

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from physical_ai_agent.safety.so100_smolvla_metadata_adapter import (
    build_so100_smolvla_metadata_command_chunk_plan,
    extract_policy_postprocessor_action_stats,
    inspect_smolvla_action_metadata,
)


CONFIG = {
    "output_features": {"action": {"type": "ACTION", "shape": [6]}},
    "normalization_mapping": {"ACTION": "MEAN_STD"},
    "chunk_size": 50,
    "n_action_steps": 50,
}

STATS = {
    "action": {
        "mean": [2000, 2100, 1900, 1800, 2050, 1700],
        "std": [100, 100, 50, 25, 10, 80],
    }
}

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


class SO100SmolVLAMetadataAdapterTest(TestCase):
    def test_mean_std_without_stats_blocks(self) -> None:
        metadata = inspect_smolvla_action_metadata(
            config=CONFIG,
            model_id="lerobot/smolvla_base",
            stats=None,
            action_semantics="absolute_joint_position",
            joint_order=SO100_JOINT_ORDER,
            gripper_semantics="higher_raw_opens",
            command_units="feetech_raw_ticks",
        )

        self.assertTrue(metadata.output_is_normalized)
        self.assertFalse(metadata.action_stats_available)
        self.assertIn("mean/std stats", " ".join(metadata.blockers))

    def test_missing_semantics_blocks(self) -> None:
        metadata = inspect_smolvla_action_metadata(
            config=CONFIG,
            model_id="lerobot/smolvla_base",
            stats=STATS,
            action_semantics=None,
            joint_order=SO100_JOINT_ORDER,
            gripper_semantics="higher_raw_opens",
            command_units="feetech_raw_ticks",
        )

        self.assertIn("Action semantics", " ".join(metadata.blockers))

    def test_joint_order_mismatch_blocks(self) -> None:
        metadata = inspect_smolvla_action_metadata(
            config=CONFIG,
            model_id="lerobot/smolvla_base",
            stats=STATS,
            action_semantics="absolute_joint_position",
            joint_order=list(reversed(SO100_JOINT_ORDER)),
            gripper_semantics="higher_raw_opens",
            command_units="feetech_raw_ticks",
        )

        self.assertIn("Joint order", " ".join(metadata.blockers))

    def test_missing_gripper_semantics_blocks(self) -> None:
        metadata = inspect_smolvla_action_metadata(
            config=CONFIG,
            model_id="lerobot/smolvla_base",
            stats=STATS,
            action_semantics="absolute_joint_position",
            joint_order=SO100_JOINT_ORDER,
            gripper_semantics=None,
            command_units="feetech_raw_ticks",
        )

        self.assertIn("Gripper semantics", " ".join(metadata.blockers))

    def test_missing_command_units_blocks(self) -> None:
        metadata = inspect_smolvla_action_metadata(
            config=CONFIG,
            model_id="lerobot/smolvla_base",
            stats=STATS,
            action_semantics="absolute_joint_position",
            joint_order=SO100_JOINT_ORDER,
            gripper_semantics="higher_raw_opens",
        )

        self.assertIn("Command units", " ".join(metadata.blockers))

    def test_unnormalizes_absolute_positions_and_clips_range(self) -> None:
        plan = build_so100_smolvla_metadata_command_chunk_plan(
            action_chunk=[[0.0, 0.0, 10.0, 0.0, 0.0, -1.0] for _index in range(10)],
            current_state=CURRENT_STATE,
            calibration=CALIBRATION,
            config=CONFIG,
            model_id="lerobot/smolvla_base",
            stats=STATS,
            action_semantics="absolute_joint_position",
            joint_order=SO100_JOINT_ORDER,
            gripper_semantics="higher_raw_opens",
            command_units="feetech_raw_ticks",
        )

        self.assertEqual(plan.status, "passed")
        self.assertTrue(plan.ready_for_execution)
        self.assertEqual(plan.action_chunk_steps, 10)
        first = plan.step_plans[0].joint_targets
        self.assertEqual(first[0].target_raw, 2000)
        self.assertEqual(first[2].unnormalized_action_value, 2400)
        self.assertEqual(first[2].target_raw, 2048)
        self.assertTrue(first[2].clipped_by_calibrated_range)
        self.assertEqual(first[5].target_raw, 1658)

    def test_unnormalizes_joint_delta_from_current_state(self) -> None:
        plan = build_so100_smolvla_metadata_command_chunk_plan(
            action_chunk=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            current_state=CURRENT_STATE,
            calibration=CALIBRATION,
            config=CONFIG,
            model_id="lerobot/smolvla_base",
            stats={"action": {"mean": [0, 0, 0, 0, 0, 0], "std": [10, 1, 1, 1, 1, 1]}},
            action_semantics="joint_delta",
            joint_order=SO100_JOINT_ORDER,
            gripper_semantics="higher_raw_opens",
            command_units="feetech_raw_ticks",
        )

        self.assertTrue(plan.ready_for_execution)
        self.assertEqual(plan.step_plans[0].joint_targets[0].target_raw, CURRENT_STATE["shoulder_pan"] + 10)

    def test_lerobot_so100_position_units_map_to_raw_estimate(self) -> None:
        plan = build_so100_smolvla_metadata_command_chunk_plan(
            action_chunk=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            current_state=CURRENT_STATE,
            calibration=CALIBRATION,
            config=CONFIG,
            model_id="lerobot/smolvla_base",
            stats={"action": {"mean": [0, 0, 0, 0, 0, 50], "std": [1, 1, 1, 1, 1, 1]}},
            action_semantics="absolute_joint_position",
            joint_order=SO100_JOINT_ORDER,
            gripper_semantics="higher_raw_opens",
            command_units="lerobot_so100_position",
        )

        self.assertTrue(plan.ready_for_execution)
        self.assertEqual(plan.command_units, "lerobot_so100_position")
        first = plan.step_plans[0].joint_targets
        self.assertTrue(first[0].write_normalize)
        self.assertEqual(first[0].target_command_value, 0)
        self.assertEqual(first[0].target_raw, (CALIBRATION["shoulder_pan"]["range_min"] + CALIBRATION["shoulder_pan"]["range_max"]) / 2)
        self.assertEqual(first[5].target_command_value, 50)
        self.assertEqual(first[5].target_raw, (CALIBRATION["gripper"]["range_min"] + CALIBRATION["gripper"]["range_max"]) / 2)

    def test_lerobot_so100_position_units_block_raw_range_violation(self) -> None:
        plan = build_so100_smolvla_metadata_command_chunk_plan(
            action_chunk=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            current_state=CURRENT_STATE,
            calibration=CALIBRATION,
            config=CONFIG,
            model_id="lerobot/smolvla_base",
            stats={"action": {"mean": [0, 170, 0, 0, 0, 50], "std": [1, 1, 1, 1, 1, 1]}},
            action_semantics="absolute_joint_position",
            joint_order=SO100_JOINT_ORDER,
            gripper_semantics="higher_raw_opens",
            command_units="lerobot_so100_position",
        )

        self.assertFalse(plan.ready_for_execution)
        self.assertIn("outside calibrated range", " ".join(plan.blockers))

    def test_extracts_policy_postprocessor_action_stats_from_local_artifacts(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import torch
        from safetensors.torch import save_file

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "policy_postprocessor.json").write_text(
                """
                {
                  "name": "policy_postprocessor",
                  "steps": [
                    {
                      "registry_name": "unnormalizer_processor",
                      "config": {},
                      "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            save_file(
                {
                    "so100.buffer.action.mean": torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32),
                    "so100.buffer.action.std": torch.tensor([7, 8, 9, 10, 11, 12], dtype=torch.float32),
                },
                tmp / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            )

            report = extract_policy_postprocessor_action_stats(
                model_id_or_path=str(tmp),
                output=tmp / "stats.json",
                local_files_only=True,
            )

        self.assertEqual(report["selected_action_stats_key"], "so100.buffer")
        self.assertEqual(report["action"]["mean"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(report["_source"], "lerobot_policy_postprocessor")
