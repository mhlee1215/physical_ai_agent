from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_agentic_controller import build_agentic_next_plan


class RealSO100AgenticControllerTest(TestCase):
    def test_blocked_contract_plans_observation_repair_without_motion(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            contract = _write_json(tmp / "contract.json", _contract_payload(next_type="observe_reframe"))
            advice = _write_json(
                tmp / "advice.json",
                {
                    "manifest_path": str(tmp / "advice.json"),
                    "actions": [
                        {
                            "type": "repair_jaw_camera_framing",
                            "camera": "0",
                            "agent_actionable": False,
                            "external_setup_required": True,
                            "diagnostic_summary": "camera 0 target detection is clipped; external setup blocker",
                        }
                    ],
                },
            )
            state = _write_json(
                tmp / "state.json",
                {
                    "active_constraints": ["external_setup_ready_before_contact"],
                    "failure_memory": {"jaw_object_framing_not_ready": {"count": 1}},
                },
            )

            plan = build_agentic_next_plan(
                contract=contract,
                reframe_advice=advice,
                agentic_state=state,
                output=tmp / "plan.json",
            )

        self.assertEqual(plan["stage"], "external_setup_blocked")
        self.assertFalse(plan["physical_robot_motion"])
        self.assertEqual(plan["next_steps"], [])
        self.assertEqual(plan["autonomous_next_steps"], [])
        self.assertEqual(plan["external_setup_blocker"]["type"], "external_setup_blocker")
        self.assertFalse(plan["external_setup_blocker"]["agent_actionable"])
        self.assertFalse(plan["external_setup_blocker"]["vla_prompt_allowed"])
        self.assertEqual(plan["external_setup_blocker"]["diagnostics"][0]["camera"], "0")
        self.assertEqual(plan["post_external_setup_verification"][0]["type"], "rerun_no_actuation_gate_after_external_setup_change")
        self.assertIn("--policy-camera-index", plan["post_external_setup_verification"][0]["command"])
        self.assertIn("object_relocation_verifier", plan["required_evidence_before_success_claim"])
        self.assertIn("external_setup_ready_before_contact", plan["active_constraints"])
        self.assertEqual(plan["failure_memory"]["jaw_object_framing_not_ready"]["count"], 1)

    def test_ready_contract_plans_contact_then_relocation_verifier(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            contract = _write_json(
                tmp / "contract.json",
                _contract_payload(
                    next_type="minimal_contact_probe",
                    decision="ready_for_reframed_contact_probe",
                    command_ready=True,
                ),
            )

            vla_prompt_packet = tmp / "vla_prompt_packet.json"
            vla_prompt_packet.write_text("{}", encoding="utf-8")

            plan = build_agentic_next_plan(
                contract=contract,
                output=tmp / "plan.json",
                vla_prompt_packet=vla_prompt_packet,
            )

        self.assertEqual(plan["stage"], "minimal_contact_probe")
        self.assertTrue(plan["physical_robot_motion"])
        self.assertEqual(plan["next_steps"][0]["type"], "execute_video_backed_contact_probe")
        self.assertIn("_workspace/real_so100/contact_probe_next/visual", plan["next_steps"][0]["command"])
        self.assertEqual(plan["next_steps"][1]["type"], "materialize_relocation_verifier_packet")
        self.assertIn("scripts/build_real_so100_relocation_verifier_packet.py", plan["next_steps"][1]["command"])
        self.assertIn("--execution-report", plan["next_steps"][1]["command"])
        self.assertEqual(plan["next_steps"][2]["type"], "run_relocation_verifier")
        self.assertIn("--target-direction", plan["next_steps"][2]["command"])
        self.assertIn("right", plan["next_steps"][2]["command"])
        self.assertIn("_workspace/real_so100/contact_probe_next/visual/before.jpg", plan["next_steps"][2]["command"])

    def test_vla_proposal_only_contract_plans_smolvla_dry_without_motion(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            contract = _write_json(
                tmp / "contract.json",
                _contract_payload(
                    next_type="smolvla_proposal_only",
                    decision="ready_for_smolvla_proposal_physical_blocked",
                ),
            )

            plan = build_agentic_next_plan(contract=contract, output=tmp / "plan.json")

        self.assertEqual(plan["stage"], "smolvla_proposal_only")
        self.assertFalse(plan["physical_robot_motion"])
        self.assertTrue(plan["vla_prompt_allowed"])
        self.assertTrue(plan["physical_execution_blocked"])
        self.assertEqual(plan["next_steps"][0]["type"], "rerun_smolvla_dry")
        self.assertEqual(plan["next_steps"][0]["policy_camera_indexes"], ["0", "1"])
        self.assertEqual(plan["next_steps"][0]["observer_camera_indexes_excluded_from_policy"], ["3"])

    def test_repeated_observation_blocker_escalates_reframe_requirement(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            contract = _write_json(tmp / "contract.json", _contract_payload(next_type="observe_reframe"))
            state = _write_json(
                tmp / "state.json",
                {
                    "active_constraints": ["external_setup_ready_before_contact"],
                    "failure_memory": {
                        "jaw_object_framing_not_ready": {"count": 3},
                    },
                },
            )

            plan = build_agentic_next_plan(
                contract=contract,
                agentic_state=state,
                output=tmp / "plan.json",
            )

        self.assertEqual(plan["repair_escalation"]["type"], "repeated_observation_repair_blocker")
        self.assertEqual(plan["repair_escalation"]["count"], 3)
        self.assertTrue(plan["external_setup_blocker"]["requires_external_setup_change_before_rerun"])
        self.assertIn("external camera/object reframe", plan["external_setup_blocker"]["escalation_reason"])


def _contract_payload(
    *,
    next_type: str,
    decision: str = "blocked_reframe_before_retry",
    command_ready: bool = False,
) -> dict:
    return {
        "manifest_path": "contract.json",
        "policy": {
            "instruction": "Pick up the green Android figure and move it to the right.",
            "instruction_tokenized": True,
            "policy_camera_indexes": ["0", "1"],
            "observer_camera_indexes": ["3"],
            "observer_camera_role": "codex_debug_only_not_smolvla_input",
            "camera_source_mapping": {"0": "wrist_cam", "1": "egocentric_cam"},
        },
        "task_goal": {
            "instruction": "Pick up the green Android figure and move it to the right.",
            "target_object": "green Android figure",
            "transport_direction": "right",
            "requires_grasp": True,
            "requires_transport": True,
            "final_success_verifier": "object_relocation_image_space",
        },
        "agentic_layer": {
            "decision": decision,
            "next_agentic_action": {
                "type": next_type,
                "joint": "gripper",
                "reason": "reframe_camera_0_or_camera_1_or_object",
                "required_observations": ["camera_0_jaw_object_framing", "camera_1_object_view"],
                "required_before_execution": ["adapter_semantics_confirmed"],
            },
            "verifier_contract": {
                "relocation_task_success_candidate": False,
            },
        },
        "adapter_and_safety": {
            "command_plan_ready_for_execution": command_ready,
        },
        "evidence": {
            "grasp_outcome": "grasp.json",
        },
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
