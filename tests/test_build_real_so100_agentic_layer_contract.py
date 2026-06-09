from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_real_so100_agentic_layer_contract import build_agentic_layer_contract


class BuildRealSO100AgenticLayerContractTest(TestCase):
    def test_contract_turns_smolvla_proposal_into_reframe_decision(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            smolvla_report = _write_json(
                tmp / "smolvla_report.json",
                {
                    "model_id": "lerobot/smolvla_base",
                    "instruction": "Pick up the green Android figure and move it to the right.",
                    "instruction_tokenized": True,
                    "language_token_count": 9,
                    "raw_action_dim": 6,
                    "raw_action_chunk_steps": 10,
                    "predicted_chunk_size": 12,
                    "planned_action_steps": 10,
                    "executed_action_steps": 10,
                    "actuation_enabled": False,
                    "send_action_called": False,
                    "camera_source_mapping": {"0": "wrist_cam", "1": "egocentric_cam"},
                    "policy_camera_indexes": ["0", "1"],
                    "observer_camera_indexes": ["3"],
                    "observer_camera_role": "codex_debug_only_not_smolvla_input",
                },
            )
            _write_json(
                tmp / "action_metadata" / "smolvla_action_metadata_report.json",
                {
                    "status": "blocked",
                    "model_id": "lerobot/smolvla_base",
                    "metadata": {
                        "action_normalization": "MEAN_STD",
                        "output_is_normalized": True,
                        "action_stats_available": False,
                        "blockers": [
                            "Action normalization is MEAN_STD, but action mean/std stats are unavailable."
                        ],
                    },
                    "required_next_steps": [
                        "Find or provide authoritative action mean/std stats for this SmolVLA checkpoint."
                    ],
                },
            )
            smolvla_action = _write_json(
                tmp / "smolvla_action.json",
                {
                    "instruction_tokenized": True,
                    "language_token_count": 9,
                    "raw_action": [0, 1, 2, 3, 4, 5],
                    "raw_action_chunk": [[0, 1, 2, 3, 4, 5] for _index in range(10)],
                    "action_chunk_semantics": "SmolVLA predicts an action chunk; real execution must consume chunk steps, not one isolated action.",
                    "safe_to_execute": False,
                },
            )
            safety = _write_json(
                tmp / "safety.json",
                {
                    "status": "blocked",
                    "execution_allowed": False,
                    "human_confirmed": False,
                    "blockers": ["unknown raw SmolVLA action semantics"],
                },
            )
            command = _write_json(
                tmp / "command.json",
                {
                    "ready_for_execution": False,
                    "adapter_semantics_confirmed": False,
                    "human_confirmed": False,
                    "blockers": ["adapter semantics are not confirmed"],
                },
            )
            gate = _write_json(
                tmp / "gate.json",
                {
                    "status": "blocked",
                    "recommended_action": "reframe_camera_0_or_object",
                    "allowed_physical_action": None,
                    "blockers": ["camera 0 jaw/object framing gate is not ready"],
                    "evidence": {
                        "pregrasp_status": "passed",
                        "jaw_status": "blocked",
                        "object_view_camera": "1",
                        "jaw_camera": "0",
                    },
                },
            )
            grasp = _write_json(tmp / "grasp.json", {"grasp_outcome": "grasp_failed_object_stationary"})
            pack = _write_json(
                tmp / "pack.json",
                {
                    "movement_report_html": "movement.html",
                    "gate_report_html": "gate.html",
                    "agentic_lessons": [
                        {
                            "observation": "The object stayed stationary.",
                            "agentic_update": "Do not repeat close from same pose.",
                        }
                    ],
                },
            )
            output = tmp / "contract.json"
            output_md = tmp / "contract.md"

            contract = build_agentic_layer_contract(
                smolvla_report=smolvla_report,
                smolvla_action=smolvla_action,
                safety_report=safety,
                command_plan=command,
                next_action_gate=gate,
                grasp_outcome=grasp,
                pre_stage_pack=pack,
                output=output,
                output_markdown=output_md,
            )
            markdown = output_md.read_text(encoding="utf-8")

        self.assertEqual(contract["status"], "passed")
        self.assertEqual(contract["output_markdown"], str(output_md))
        self.assertFalse(contract["agentic_success_claim"])
        self.assertEqual(contract["agentic_layer"]["decision"], "blocked_reframe_before_retry")
        self.assertEqual(contract["policy"]["policy_camera_indexes"], ["0", "1"])
        self.assertEqual(contract["policy"]["observer_camera_indexes"], ["3"])
        self.assertEqual(contract["policy"]["observer_camera_status"], "available")
        self.assertEqual(contract["policy"]["observer_camera_role"], "codex_debug_only_not_smolvla_input")
        self.assertEqual(contract["policy"]["raw_action_chunk_steps"], 10)
        self.assertEqual(contract["policy"]["predicted_chunk_size"], 12)
        self.assertEqual(contract["policy"]["planned_action_steps"], 10)
        self.assertEqual(contract["policy"]["executed_action_steps"], 10)
        self.assertEqual(contract["task_goal"]["transport_direction"], "right")
        self.assertTrue(contract["task_goal"]["requires_transport"])
        self.assertEqual(contract["agentic_layer"]["next_agentic_action"]["type"], "observe_reframe")
        self.assertFalse(contract["agentic_layer"]["next_agentic_action"]["physical_robot_motion"])
        self.assertIn("camera_1_object_view", contract["agentic_layer"]["next_agentic_action"]["required_observations"])
        self.assertEqual(contract["agentic_layer"]["verifier_contract"]["final_success_source"], "none_in_real_so100_prestage")
        self.assertEqual(contract["agentic_layer"]["verifier_contract"]["relocation_verifier_status"], "not_run")
        self.assertTrue(contract["agentic_layer"]["verifier_contract"]["final_task_success_requires_relocation_verifier"])
        self.assertIn("unknown raw SmolVLA action semantics", contract["agentic_layer"]["blockers"])
        self.assertIn(
            "Action normalization is MEAN_STD, but action mean/std stats are unavailable.",
            contract["agentic_layer"]["blockers"],
        )
        self.assertEqual(contract["adapter_and_safety"]["action_metadata_status"], "blocked")
        self.assertFalse(contract["adapter_and_safety"]["action_metadata"]["action_stats_available"])
        self.assertIn("debug_and_human_feedback_only", contract["evidence"]["video_evidence_role"])
        self.assertIn("Real SO-100 SmolVLA Agentic Layer Contract", markdown)
        self.assertIn("Policy cameras: `['0', '1']`", markdown)
        self.assertIn("Raw action chunk steps: `10`", markdown)
        self.assertIn("Observer status: `available`", markdown)
        self.assertIn("Action metadata status: `blocked`", markdown)
        self.assertIn("Transport direction: `right`", markdown)
        self.assertIn("blocked_reframe_before_retry", markdown)
        self.assertIn("Internal verifier success is a retry signal", markdown)

    def test_contract_separates_vla_proposal_from_blocked_physical_execution(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            smolvla_report = _write_json(
                tmp / "smolvla_report.json",
                {
                    "model_id": "lerobot/smolvla_base",
                    "instruction": "Pick up the green Android figure and move it to the right.",
                    "instruction_tokenized": True,
                    "language_token_count": 9,
                    "raw_action_dim": 6,
                    "raw_action_chunk_steps": 10,
                    "predicted_chunk_size": 10,
                    "actuation_enabled": False,
                    "send_action_called": False,
                    "camera_source_mapping": {"0": "wrist_cam", "1": "egocentric_cam"},
                    "policy_camera_indexes": ["0", "1"],
                    "observer_camera_indexes": [],
                    "observer_camera_role": "codex_debug_only_not_smolvla_input",
                },
            )
            smolvla_action = _write_json(
                tmp / "smolvla_action.json",
                {
                    "instruction_tokenized": True,
                    "raw_action": [0, 1, 2, 3, 4, 5],
                    "raw_action_chunk": [[0, 1, 2, 3, 4, 5] for _index in range(10)],
                    "safe_to_execute": False,
                },
            )
            safety = _write_json(tmp / "safety.json", {"status": "blocked", "execution_allowed": False})
            command = _write_json(tmp / "command.json", {"ready_for_execution": False})
            gate = _write_json(
                tmp / "gate.json",
                {
                    "status": "blocked",
                    "recommended_action": "reframe_camera_0_or_camera_1_or_object",
                    "vla_prompt_allowed": True,
                    "vla_prompt_gate": {
                        "status": "ready",
                        "reason": "camera 1 has usable object context and camera 0 has jaw-marker evidence",
                    },
                    "physical_execution_gate": {"status": "blocked", "allowed_physical_action": None},
                    "allowed_physical_action": None,
                    "blockers": ["camera 0 jaw/object framing gate is not ready"],
                    "evidence": {"pregrasp_status": "passed", "jaw_status": "blocked"},
                },
            )
            grasp = _write_json(tmp / "grasp.json", {"grasp_outcome": "grasp_failed_object_stationary"})
            pack = _write_json(tmp / "pack.json", {})

            contract = build_agentic_layer_contract(
                smolvla_report=smolvla_report,
                smolvla_action=smolvla_action,
                safety_report=safety,
                command_plan=command,
                next_action_gate=gate,
                grasp_outcome=grasp,
                pre_stage_pack=pack,
                output=tmp / "contract.json",
            )

        self.assertEqual(contract["agentic_layer"]["decision"], "ready_for_smolvla_proposal_physical_blocked")
        self.assertTrue(contract["agentic_layer"]["vla_prompt_allowed"])
        self.assertEqual(contract["agentic_layer"]["next_agentic_action"]["type"], "smolvla_proposal_only")
        self.assertFalse(contract["agentic_layer"]["next_agentic_action"]["physical_robot_motion"])
        self.assertTrue(contract["agentic_layer"]["next_agentic_action"]["physical_execution_blocked"])
        self.assertEqual(contract["policy"]["raw_action_chunk_steps"], 10)
        self.assertEqual(contract["policy"]["observer_camera_indexes"], [])
        self.assertEqual(contract["policy"]["observer_camera_status"], "temporarily_unavailable")
        self.assertIn("camera 3 is temporarily off", contract["policy"]["observer_camera_note"])


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
