from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.update_real_so100_agentic_state import normalize_agentic_state, update_agentic_state


class UpdateRealSO100AgenticStateTest(TestCase):
    def test_accumulates_failures_and_active_constraints(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            analysis = _write_json(
                tmp / "analysis.json",
                {
                    "task": "Pick up the green Android figure and move it to the right.",
                    "stage": "observation_repair",
                    "gate_status": "blocked",
                    "physical_robot_motion": False,
                    "failure_modes": [
                        {"type": "jaw_object_framing_not_ready", "camera": "0", "clipped_sides": ["left"]},
                        {"type": "adapter_semantics_not_executable", "command_plan": "command.json"},
                        {"type": "previous_contact_failed_stationary_object", "evidence": "grasp.json"},
                        {"type": "task_success_not_verified", "required_verifier": "object_relocation_image_space"},
                    ],
                    "agentic_layer_improvements": [
                        {
                            "target": "policy_input_quality_gate",
                            "change": "prioritize camera-role-specific reframe advice before any contact probe",
                            "generalization": "applies across camera-role failures",
                        },
                        {
                            "target": "success_criteria",
                            "change": "require object relocation verifier after grasp/contact attempts",
                            "generalization": "separates grasp success from transport success",
                        },
                    ],
                    "loop_continuation": {"next_stage": "observation_repair"},
                },
            )

            state = update_agentic_state(analysis=analysis, state=tmp / "state.json")

        self.assertEqual(state["failure_memory"]["jaw_object_framing_not_ready"]["count"], 1)
        self.assertIn("external_setup_ready_before_contact", state["active_constraints"])
        self.assertIn("relocation_verifier_required_for_transport_success", state["active_constraints"])
        targets = {item["target"] for item in state["policy_updates"]}
        self.assertIn("policy_input_quality_gate", targets)
        self.assertIn("success_criteria", targets)

    def test_normalizes_legacy_state_when_merging_new_analysis(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_path = _write_json(
                tmp / "state.json",
                {
                    "status": "passed",
                    "operation": "real_so100_agentic_state",
                    "iterations": [{"stage": "observation_repair"}],
                    "failure_memory": {},
                    "active_constraints": ["observation_repair_before_contact"],
                    "next_loop_hint": {
                        "next_stage": "observation_repair",
                        "next_step_type": "manual_or_fixture_reframe",
                        "repeat_prompt_after_repair": True,
                    },
                    "policy_updates": [
                        {
                            "target": "observation_repair_policy",
                            "change": "legacy",
                            "latest_advice": [{"operator_instruction": "move object"}],
                        }
                    ],
                },
            )
            analysis = _write_json(
                tmp / "analysis.json",
                {
                    "task": "Pick up the green Android figure and move it to the right.",
                    "stage": "external_setup_blocked",
                    "gate_status": "blocked",
                    "physical_robot_motion": False,
                    "failure_modes": [{"type": "jaw_object_framing_not_ready", "camera": "0"}],
                    "agentic_layer_improvements": [
                        {
                            "target": "policy_input_quality_gate",
                            "change": "block VLA prompting/contact execution when required policy cameras do not provide usable target evidence",
                        }
                    ],
                    "loop_continuation": {
                        "next_stage": "external_setup_blocked",
                        "next_step_type": None,
                        "repeat_prompt_after_repair": False,
                    },
                },
            )

            state = update_agentic_state(analysis=analysis, state=state_path)

        self.assertIn("external_setup_ready_before_contact", state["active_constraints"])
        self.assertNotIn("observation_repair_before_contact", state["active_constraints"])
        self.assertEqual(state["iterations"][0]["stage"], "external_setup_blocked")
        self.assertEqual(state["next_loop_hint"]["next_stage"], "external_setup_blocked")
        self.assertIsNone(state["next_loop_hint"]["next_step_type"])
        targets = {item["target"] for item in state["policy_updates"]}
        self.assertIn("policy_input_quality_gate", targets)
        self.assertNotIn("observation_repair_policy", targets)
        self.assertNotIn("operator_instruction", json.dumps(state["policy_updates"]))
        quality_update = [item for item in state["policy_updates"] if item["target"] == "policy_input_quality_gate"][0]
        self.assertIn("block VLA prompting/contact execution", quality_update["change"])

    def test_normalize_only_does_not_append_iterations_or_increment_counts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_path = _write_json(
                tmp / "state.json",
                {
                    "status": "passed",
                    "operation": "real_so100_agentic_state",
                    "iterations": [{"stage": "observation_repair"}],
                    "failure_memory": {"jaw_object_framing_not_ready": {"count": 3, "examples": []}},
                    "active_constraints": ["observation_repair_before_contact"],
                    "next_loop_hint": {"next_stage": "observation_repair", "next_step_type": "manual_or_fixture_reframe"},
                    "policy_updates": [
                        {
                            "target": "observation_repair_policy",
                            "count": 3,
                            "change": "legacy",
                            "latest_advice": [{"operator_instruction": "move object"}],
                        }
                    ],
                },
            )

            state = normalize_agentic_state(state=state_path, output=tmp / "normalized.json")

        self.assertEqual(len(state["iterations"]), 1)
        self.assertEqual(state["failure_memory"]["jaw_object_framing_not_ready"]["count"], 3)
        self.assertEqual(state["iterations"][0]["stage"], "external_setup_blocked")
        self.assertEqual(state["next_loop_hint"]["next_stage"], "external_setup_blocked")
        self.assertIsNone(state["next_loop_hint"]["next_step_type"])
        self.assertIn("external_setup_ready_before_contact", state["active_constraints"])
        self.assertEqual(state["policy_updates"][0]["target"], "policy_input_quality_gate")
        self.assertIn("block VLA prompting/contact execution", state["policy_updates"][0]["change"])
        self.assertNotIn("operator_instruction", json.dumps(state["policy_updates"]))

    def test_update_is_idempotent_for_same_analysis_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            analysis = _write_json(
                tmp / "analysis.json",
                {
                    "task": "Pick up the green Android figure and move it to the right.",
                    "stage": "external_setup_blocked",
                    "gate_status": "blocked",
                    "physical_robot_motion": False,
                    "failure_modes": [{"type": "jaw_object_framing_not_ready", "camera": "0"}],
                    "agentic_layer_improvements": [
                        {
                            "target": "policy_input_quality_gate",
                            "change": "block VLA prompting/contact execution when required policy cameras do not provide usable target evidence",
                        }
                    ],
                    "loop_continuation": {"next_stage": "external_setup_blocked"},
                },
            )
            state_path = tmp / "state.json"

            first = update_agentic_state(analysis=analysis, state=state_path)
            second = update_agentic_state(analysis=analysis, state=state_path)

        self.assertEqual(len(first["iterations"]), 1)
        self.assertEqual(len(second["iterations"]), 1)
        self.assertEqual(second["failure_memory"]["jaw_object_framing_not_ready"]["count"], 1)
        self.assertEqual(second["policy_updates"][0]["count"], 1)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
