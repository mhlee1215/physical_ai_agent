from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_real_so100_vla_prompt_packet import build_vla_prompt_packet


class BuildRealSO100VLAPromptPacketTest(TestCase):
    def test_builds_smolvla_prompt_and_observer_frame_verifier(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            contract = _write_json(
                tmp / "contract.json",
                {
                    "policy": {
                        "instruction": "Pick up the green Android figure and move it to the right.",
                        "model_id": "lerobot/smolvla_base",
                        "policy_camera_indexes": ["0", "1"],
                        "observer_camera_indexes": ["3"],
                        "camera_source_mapping": {"0": "wrist_cam", "1": "egocentric_cam"},
                        "send_action_called": False,
                    },
                    "task_goal": {
                        "target_object": "green Android figure",
                        "transport_direction": "right",
                        "final_success_verifier": "object_relocation_image_space",
                    },
                },
            )
            iteration = _write_json(
                tmp / "iteration.json",
                {
                    "next_iteration": {
                        "external_setup_blocker": {"agent_actionable": False},
                        "autonomous_next_steps": [],
                    }
                },
            )

            packet = build_vla_prompt_packet(
                contract=contract,
                prompt_iteration=iteration,
                output=tmp / "packet.json",
            )

        self.assertEqual(packet["vla_prompt"]["target"], "SmolVLA")
        self.assertEqual(packet["vla_prompt"]["policy_camera_indexes"], ["0", "1"])
        self.assertEqual(packet["vla_prompt"]["observer_camera_indexes_excluded_from_policy"], ["3"])
        self.assertTrue(packet["agentic_layer_contract"]["does_not_prompt_operator"])
        self.assertTrue(packet["agentic_layer_contract"]["external_setup_blocked"])
        self.assertEqual(packet["success_verifier"]["frame"]["primary_camera_index"], "3")
        self.assertIn("after_object_center.image_x", packet["success_verifier"]["success_predicate"])
        self.assertTrue(packet["success_verifier"]["do_not_translate_goal_to_fixed_robot_direction"])
        self.assertIn("robot arm moves left", packet["coordinate_semantics"]["not_equivalent_to"])

    def test_observer_off_uses_policy_context_frame_without_success_claim(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            contract = _write_json(
                tmp / "contract.json",
                {
                    "policy": {
                        "instruction": "Pick up the green Android figure and move it to the right.",
                        "model_id": "lerobot/smolvla_base",
                        "policy_camera_indexes": ["0", "1"],
                        "observer_camera_indexes": [],
                        "camera_source_mapping": {"0": "wrist_cam", "1": "egocentric_cam"},
                        "send_action_called": False,
                    },
                    "task_goal": {
                        "target_object": "green Android figure",
                        "transport_direction": "right",
                        "final_success_verifier": "object_relocation_image_space",
                    },
                },
            )

            packet = build_vla_prompt_packet(contract=contract)

        self.assertEqual(packet["vla_prompt"]["observer_camera_indexes_excluded_from_policy"], [])
        self.assertEqual(packet["success_verifier"]["frame"]["name"], "policy_context_image_frame")
        self.assertEqual(packet["success_verifier"]["frame"]["primary_camera_index"], "1")
        self.assertFalse(packet["success_verifier"]["frame"]["task_success_claim_allowed"])

    def test_prompt_iteration_observer_off_overrides_stale_contract_observer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            contract = _write_json(
                tmp / "contract.json",
                {
                    "policy": {
                        "instruction": "Pick up the green Android figure and move it to the right.",
                        "model_id": "lerobot/smolvla_base",
                        "policy_camera_indexes": ["0", "1"],
                        "observer_camera_indexes": ["3"],
                        "camera_source_mapping": {"0": "wrist_cam", "1": "egocentric_cam"},
                        "send_action_called": False,
                    },
                    "task_goal": {
                        "target_object": "green Android figure",
                        "transport_direction": "right",
                    },
                },
            )
            iteration = _write_json(
                tmp / "iteration.json",
                {
                    "camera_contract": {
                        "smolvla_policy_inputs": ["0", "1"],
                        "observer_inputs": [],
                        "observer_camera_status": "temporarily_unavailable",
                        "camera_source_mapping": {"0": "wrist_cam", "1": "egocentric_cam"},
                    }
                },
            )

            packet = build_vla_prompt_packet(contract=contract, prompt_iteration=iteration)

        self.assertEqual(packet["vla_prompt"]["observer_camera_indexes_excluded_from_policy"], [])
        self.assertEqual(packet["success_verifier"]["frame"]["name"], "policy_context_image_frame")
        self.assertEqual(packet["success_verifier"]["frame"]["primary_camera_index"], "1")


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
