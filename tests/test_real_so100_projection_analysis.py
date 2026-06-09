from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_projection_analysis import (
    analyze_execute_gate_projection,
    analyze_projection_sweep,
    project_joint_target,
)


class RealSO100ProjectionAnalysisTest(TestCase):
    def test_project_joint_target_clips_raw_and_back_converts_command(self) -> None:
        projected = project_joint_target(
            {
                "joint": "shoulder_lift",
                "target_raw": 4400,
                "range_min": 2000,
                "range_max": 3600,
                "target_command_value": 150,
            }
        )

        self.assertTrue(projected["was_out_of_range"])
        self.assertEqual(projected["projected_raw"], 3600)
        self.assertEqual(projected["raw_distortion"], 800)
        self.assertLess(projected["projected_command_value"], projected["target_command_value"])

    def test_analyze_execute_gate_projection_summarizes_distortion(self) -> None:
        report = analyze_execute_gate_projection(
            {
                "dry_plan": {
                    "ready_for_execution": False,
                    "step_plans": [
                        {
                            "step_index": 0,
                            "joint_targets": [
                                _target("shoulder_pan", 1500, 1000, 2000, 0),
                                _target("elbow_flex", 2300, 1000, 2000, 90),
                            ],
                        },
                    ],
                }
            }
        )

        self.assertTrue(report["projected_ready_for_execution_shape_only"])
        self.assertFalse(report["source_ready_for_execution"])
        self.assertEqual(report["range_violation_count"], 1)
        self.assertEqual(report["total_raw_distortion"], 300)
        self.assertEqual(report["max_raw_distortion"], 300)
        self.assertEqual(report["joint_distortion"]["elbow_flex"]["violation_count"], 1)

    def test_projection_sweep_ranks_lowest_projection_penalty(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bad_gate = tmp / "bad.json"
            better_gate = tmp / "better.json"
            bad_gate.write_text(json.dumps(_execute_gate(raw=2600)), encoding="utf-8")
            better_gate.write_text(json.dumps(_execute_gate(raw=2100)), encoding="utf-8")
            sweep = tmp / "sweep.json"
            sweep.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "policy_camera_indexes": ["0", "1"],
                        "observer_camera_indexes": [],
                        "observer_camera_status": "temporarily_unavailable",
                        "camera_3_status": "off",
                        "candidates": [
                            {"candidate_index": 1, "prompt": "bad", "execute_gate_path": str(bad_gate)},
                            {"candidate_index": 2, "prompt": "better", "execute_gate_path": str(better_gate)},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = analyze_projection_sweep(
                sweep_report=sweep,
                output=tmp / "projection.json",
            )

        self.assertEqual(report["best_candidate"]["candidate_index"], 2)
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertFalse(report["task_success_claim_allowed"])


def _target(joint: str, raw: float, range_min: float, range_max: float, command: float) -> dict:
    return {
        "joint": joint,
        "target_raw": raw,
        "range_min": range_min,
        "range_max": range_max,
        "target_command_value": command,
    }


def _execute_gate(*, raw: float) -> dict:
    return {
        "dry_plan": {
            "ready_for_execution": False,
            "step_plans": [
                {
                    "step_index": 0,
                    "joint_targets": [
                        _target("shoulder_pan", raw, 1000, 2000, 0),
                    ],
                }
            ],
        }
    }
