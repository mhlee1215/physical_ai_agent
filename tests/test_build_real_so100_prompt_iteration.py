from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_real_so100_prompt_iteration import build_prompt_iteration


class BuildRealSO100PromptIterationTest(TestCase):
    def test_builds_prompt_iteration_without_success_claim(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            smolvla = _write_json(
                tmp / "smolvla.json",
                {
                    "instruction_tokenized": True,
                    "language_token_count": 14,
                    "policy_camera_indexes": ["0", "1"],
                    "observer_camera_indexes": ["3"],
                    "observer_camera_role": "codex_debug_only_not_smolvla_input",
                    "camera_source_mapping": {
                        "0": "wrist_cam",
                        "1": "egocentric_cam",
                        "3": "codex_observer",
                    },
                    "send_action_called": False,
                    "policy_actions_executed": False,
                    "action_preview": [0.1, -0.2],
                },
            )
            refresh = _write_json(
                tmp / "refresh.json",
                {
                    "task": "Pick up the green Android figure and move it to the right.",
                    "gate_status": "blocked",
                    "agentic_decision": "blocked_reframe_before_retry",
                    "physical_robot_motion": False,
                    "send_action_called": False,
                },
            )
            analysis = _write_json(
                tmp / "analysis.json",
                {
                    "failure_modes": [
                        {"type": "jaw_object_framing_not_ready"},
                        {"type": "task_success_not_verified"},
                    ],
                    "agentic_layer_improvements": [
                        {"target": "policy_input_quality_gate"},
                        {"target": "success_criteria"},
                    ],
                    "loop_continuation": {
                        "repeat_prompt_after_repair": True,
                    },
                },
            )
            state = _write_json(
                tmp / "state.json",
                {
                    "active_constraints": [
                        "external_setup_ready_before_contact",
                        "relocation_verifier_required_for_transport_success",
                    ],
                    "failure_memory": {"jaw_object_framing_not_ready": {"count": 1}},
                    "policy_updates": [
                        {
                            "target": "success_criteria",
                            "generalization": "separates grasp success from task-level transport success",
                        }
                    ],
                },
            )
            next_plan = _write_json(
                tmp / "next_plan.json",
                {
                    "stage": "external_setup_blocked",
                    "physical_robot_motion": False,
                    "repair_escalation": {"type": "repeated_observation_repair_blocker"},
                    "next_steps": [],
                    "autonomous_next_steps": [],
                    "external_setup_blocker": {
                        "type": "external_setup_blocker",
                        "agent_actionable": False,
                        "diagnostics": [
                            {
                                "image_space_nudge": {
                                    "recommended_shift_px": [32.0, 0.0],
                                }
                            }
                        ],
                    },
                    "post_external_setup_verification": [
                        {"type": "rerun_no_actuation_gate_after_external_setup_change"}
                    ],
                },
            )

            result = build_prompt_iteration(
                prompt="녹색 인형을 집어서 오른쪽으로 옮겨줘",
                smolvla_report=smolvla,
                refresh_manifest=refresh,
                analysis=analysis,
                agentic_state=state,
                next_plan=next_plan,
                output_json=tmp / "iteration.json",
                output_md=tmp / "iteration.md",
                iteration_index=1,
                vla_prompt_packet=tmp / "vla_prompt_packet.json",
                agentic_policy_patch=tmp / "policy_patch.json",
            )

        self.assertEqual(result["camera_contract"]["smolvla_policy_inputs"], ["0", "1"])
        self.assertEqual(result["camera_contract"]["observer_inputs"], ["3"])
        self.assertEqual(result["policy_proposal"]["vla_prompt_packet"], str(Path(tmpdir) / "vla_prompt_packet.json"))
        self.assertEqual(result["agentic_policy_patch"], str(Path(tmpdir) / "policy_patch.json"))
        self.assertFalse(result["policy_proposal"]["send_action_called"])
        self.assertFalse(result["success_accounting"]["task_success_claim_allowed"])
        self.assertEqual(result["next_iteration"]["stage"], "external_setup_blocked")
        self.assertEqual(
            result["next_iteration"]["external_setup_blocker"]["diagnostics"][0]["image_space_nudge"][
                "recommended_shift_px"
            ],
            [32.0, 0.0],
        )
        self.assertEqual(result["next_iteration"]["autonomous_next_steps"], [])
        self.assertEqual(
            result["next_iteration"]["post_external_setup_verification"][0]["type"],
            "rerun_no_actuation_gate_after_external_setup_change",
        )
        self.assertEqual(
            result["next_iteration"]["repair_escalation"]["type"],
            "repeated_observation_repair_blocker",
        )
        self.assertIsNone(result["next_iteration"]["next_step_type"])
        self.assertIn("jaw_object_framing_not_ready", result["analysis"]["failure_modes"])


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
