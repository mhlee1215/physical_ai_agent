from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_trajectory_diagnostic import analyze_candidate_trajectory, analyze_projection_trajectory


class RealSO100TrajectoryDiagnosticTest(TestCase):
    def test_candidate_trajectory_finds_unsafe_prefix_and_safe_late_run(self) -> None:
        trajectory = analyze_candidate_trajectory(
            {
                "candidate_index": 2,
                "projection": {
                    "projected_steps": [
                        _step(0, [_target("shoulder_lift", 10), _target("elbow_flex", 5), _target("wrist_flex", 0)]),
                        _step(1, [_target("shoulder_lift", 2)]),
                        _step(2, [_target("elbow_flex", 0), _target("wrist_flex", 0)]),
                        _step(3, [_target("elbow_flex", 0)]),
                    ],
                    "projection_penalty_score": 100,
                    "total_raw_distortion": 17,
                    "max_raw_distortion": 10,
                    "range_violation_count": 3,
                },
            }
        )

        self.assertEqual(trajectory["safe_prefix_length"], 0)
        self.assertEqual(trajectory["violation_step_count"], 2)
        self.assertEqual(trajectory["safe_suffix_after_first_violations"], {"start_step": 2, "length": 2})
        self.assertEqual(trajectory["dominant_violation_joints"][0]["joint"], "shoulder_lift")

    def test_projection_trajectory_report_preserves_no_actuation_contract(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            projection = tmp / "projection.json"
            projection.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_projection_analysis",
                        "policy_camera_indexes": ["0", "1"],
                        "observer_camera_indexes": [],
                        "observer_camera_status": "temporarily_unavailable",
                        "camera_3_status": "off",
                        "best_candidate": {
                            "candidate_index": 2,
                            "prompt": "best",
                            "projection": {
                                "projected_steps": [_step(0, [_target("elbow_flex", 12)]), _step(1, [_target("elbow_flex", 0)])],
                                "projection_penalty_score": 98,
                                "total_raw_distortion": 12,
                                "max_raw_distortion": 12,
                                "range_violation_count": 1,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = analyze_projection_trajectory(
                projection_report=projection,
                output=tmp / "trajectory.json",
            )
            self.assertTrue(Path(report["json_path"]).exists())
            self.assertTrue(Path(report["markdown_path"]).exists())

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["source_candidate_index"], 2)
        self.assertEqual(report["next_agentic_layer_step"]["type"], "do_not_execute_prefix_replan_to_safe_late_pose")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertFalse(report["task_success_claim_allowed"])


def _step(index: int, targets: list[dict]) -> dict:
    return {"step_index": index, "projected_targets": targets}


def _target(joint: str, distortion: float) -> dict:
    return {
        "joint": joint,
        "finite": True,
        "was_out_of_range": distortion > 0,
        "raw_distortion": distortion,
    }
