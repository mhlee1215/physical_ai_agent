from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_late_safe_pose_bridge import build_late_safe_pose_bridge


class RealSO100LateSafePoseBridgeTest(TestCase):
    def test_extracts_late_safe_pose_from_trajectory_start(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            projection = tmp / "projection.json"
            trajectory = tmp / "trajectory.json"
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
                            "prompt": "best prompt",
                            "projection": {
                                "projected_steps": [
                                    _step(0, [_target("shoulder_lift", 4500, 3695, was_out=True)]),
                                    _step(1, [_target("shoulder_lift", 3800, 3695, was_out=True)]),
                                    _step(3, [_target("shoulder_lift", 3453.6, 3695), _target("elbow_flex", 1777.6, 2048)]),
                                ]
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            trajectory.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_trajectory_diagnostic",
                        "source_candidate_index": 2,
                        "trajectory": {
                            "safe_suffix_after_first_violations": {"start_step": 3, "length": 7},
                        },
                        "next_agentic_layer_step": {
                            "type": "do_not_execute_prefix_replan_to_safe_late_pose",
                            "safe_run_start_step": 3,
                            "safe_run_length": 7,
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = build_late_safe_pose_bridge(
                projection_report=projection,
                trajectory_report=trajectory,
                output=tmp / "bridge.json",
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["bridge_target_step_index"], 3)
        self.assertEqual(report["safe_run_length"], 7)
        self.assertTrue(report["all_bridge_targets_in_range"])
        self.assertEqual(report["bridge_target_joints"][0]["joint"], "shoulder_lift")
        self.assertEqual(report["next_agentic_layer_step"]["type"], "generate_transition_to_late_safe_pose_without_executing")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertFalse(report["task_success_claim_allowed"])

    def test_blocks_when_late_pose_is_not_available(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            projection = tmp / "projection.json"
            trajectory = tmp / "trajectory.json"
            projection.write_text(json.dumps({"best_candidate": {"candidate_index": 1, "projection": {"projected_steps": []}}}), encoding="utf-8")
            trajectory.write_text(
                json.dumps(
                    {
                        "source_candidate_index": 1,
                        "trajectory": {"safe_suffix_after_first_violations": {"start_step": -1, "length": 0}},
                    }
                ),
                encoding="utf-8",
            )

            report = build_late_safe_pose_bridge(
                projection_report=projection,
                trajectory_report=trajectory,
                output=tmp / "bridge.json",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("range-safe late run", report["blockers"][0])
        self.assertFalse(report["send_action_called"])


def _step(index: int, targets: list[dict]) -> dict:
    return {"step_index": index, "projected_targets": targets}


def _target(joint: str, projected_raw: float, range_max: float, *, was_out: bool = False) -> dict:
    return {
        "joint": joint,
        "finite": True,
        "target_raw": projected_raw,
        "projected_raw": projected_raw,
        "range_min": 0.0,
        "range_max": range_max,
        "target_command_value": 10.0,
        "projected_command_value": 10.0,
        "raw_distortion": 0.0,
        "command_distortion": 0.0,
        "was_out_of_range": was_out,
    }
