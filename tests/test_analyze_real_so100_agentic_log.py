from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.analyze_real_so100_agentic_log import analyze_agentic_log


class AnalyzeRealSO100AgenticLogTest(TestCase):
    def test_analyzes_observation_and_success_failures(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            contract = _write_json(
                tmp / "contract.json",
                {
                    "adapter_and_safety": {
                        "command_plan_ready_for_execution": False,
                        "command_plan_path": "command.json",
                    },
                    "agentic_layer": {
                        "verifier_contract": {
                            "jaw_gate_status": "blocked",
                            "last_grasp_outcome": "grasp_failed_object_stationary",
                            "relocation_task_success_candidate": False,
                        }
                    },
                    "evidence": {"grasp_outcome": "grasp.json"},
                    "task_goal": {"final_success_verifier": "object_relocation_image_space"},
                },
            )
            plan = _write_json(
                tmp / "plan.json",
                {
                    "stage": "observation_repair",
                    "physical_robot_motion": False,
                    "next_steps": [{"type": "manual_or_fixture_reframe"}],
                },
            )
            advice = _write_json(
                tmp / "advice.json",
                {
                    "jaw_camera": "0",
                    "jaw_object_clipped_sides": ["left", "top"],
                    "jaw_object_candidate": {"bbox_xyxy": [0, 0, 40, 50]},
                    "actions": [{"reason": "green object touches image boundary"}],
                },
            )
            refresh = _write_json(
                tmp / "refresh.json",
                {
                    "contract": str(contract),
                    "next_plan": str(plan),
                    "reframe_advice": str(advice),
                    "gate_status": "blocked",
                    "gate_manifest": "gate.json",
                    "physical_robot_motion": False,
                    "agentic_decision": "blocked_reframe_before_retry",
                },
            )

            result = analyze_agentic_log(refresh_manifest=refresh, output=tmp / "analysis.json")

        mode_types = {item["type"] for item in result["failure_modes"]}
        targets = {item["target"] for item in result["agentic_layer_improvements"]}
        self.assertIn("jaw_object_framing_not_ready", mode_types)
        self.assertIn("adapter_semantics_not_executable", mode_types)
        self.assertIn("task_success_not_verified", mode_types)
        self.assertIn("policy_input_quality_gate", targets)
        self.assertIn("success_criteria", targets)
        self.assertTrue(result["loop_continuation"]["blocked_by_external_reframe"])


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
