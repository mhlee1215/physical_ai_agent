from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_real_so100_agentic_policy_patch import build_agentic_policy_patch


class BuildRealSO100AgenticPolicyPatchTest(TestCase):
    def test_normalizes_legacy_observation_repair_policy(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            analysis = _write_json(
                tmp / "analysis.json",
                {
                    "task": "Pick up the green Android figure and move it to the right.",
                    "stage": "external_setup_blocked",
                    "failure_modes": [
                        {"type": "jaw_object_framing_not_ready"},
                        {"type": "adapter_semantics_not_executable"},
                        {"type": "previous_contact_failed_stationary_object"},
                        {"type": "task_success_not_verified"},
                    ],
                    "agentic_layer_improvements": [
                        {
                            "target": "policy_input_quality_gate",
                            "change": "block VLA prompting/contact execution when required policy cameras do not provide usable target evidence",
                            "generalization": "applies to any task with bad policy inputs",
                        }
                    ],
                    "loop_continuation": {"external_setup_blocked": True},
                },
            )
            state = _write_json(
                tmp / "state.json",
                {
                    "active_constraints": ["observation_repair_before_contact"],
                    "policy_updates": [
                        {
                            "target": "observation_repair_policy",
                            "change": "prioritize camera-role-specific reframe advice before any contact probe",
                            "generalization": "legacy",
                            "latest_advice": [
                                {
                                    "operator_instruction": "move the target appearance rightward",
                                }
                            ],
                        },
                        {
                            "target": "retry_policy",
                            "change": "forbid repeated gripper closes from the same pose after stationary-object failure",
                        },
                    ],
                },
            )
            iteration = _write_json(
                tmp / "prompt_iteration.json",
                {"next_iteration": {"stage": "external_setup_blocked"}},
            )

            patch = build_agentic_policy_patch(
                analysis=analysis,
                agentic_state=state,
                prompt_iteration=iteration,
                output=tmp / "patch.json",
            )

        self.assertEqual(patch["status"], "passed")
        self.assertEqual(patch["stage"], "external_setup_blocked")
        self.assertIn("external_setup_ready_before_contact", patch["normalized_active_constraints"])
        self.assertNotIn("observation_repair_before_contact", patch["normalized_active_constraints"])
        targets = {item["target"] for item in patch["normalized_policy_updates"]}
        self.assertIn("policy_input_quality_gate", targets)
        self.assertNotIn("observation_repair_policy", targets)
        self.assertTrue(patch["prompt_contract"]["does_not_prompt_operator"])
        self.assertFalse(patch["prompt_contract"]["vla_prompt_allowed"])
        rules = {item["id"] for item in patch["rules"]}
        self.assertIn("policy_input_quality_gate", rules)
        self.assertIn("success_criteria", rules)
        normalizations = {(item["from"], item["to"]) for item in patch["legacy_normalizations"]}
        self.assertIn(("observation_repair_policy", "policy_input_quality_gate"), normalizations)
        self.assertIn(("operator_instruction", "external_setup_diagnostic"), normalizations)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
