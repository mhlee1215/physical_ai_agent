from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_agentic_candidate_memory import build_candidate_memory


class RealSO100AgenticCandidateMemoryTest(TestCase):
    def test_memory_selects_best_historical_candidate_when_latest_regresses(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            earlier = tmp / "earlier.json"
            latest = tmp / "latest.json"
            output = tmp / "memory.json"
            earlier.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_agentic_proposal_sweep",
                        "status": "passed",
                        "policy_camera_indexes": ["0", "1"],
                        "observer_camera_indexes": [],
                        "observer_camera_status": "temporarily_unavailable",
                        "ranked_candidates": [
                            {
                                "candidate_index": 3,
                                "prompt": "best historical prompt",
                                "score": {
                                    "ready_for_execution": False,
                                    "range_penalty_score": 100.0,
                                    "range_violation_count": 2,
                                    "total_range_excess_raw_ticks": 70.0,
                                    "max_range_excess_raw_ticks": 10.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            latest.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_agentic_proposal_sweep",
                        "status": "passed",
                        "policy_camera_indexes": ["0", "1"],
                        "observer_camera_indexes": [],
                        "observer_camera_status": "temporarily_unavailable",
                        "ranked_candidates": [
                            {
                                "candidate_index": 1,
                                "prompt": "latest regressed prompt",
                                "score": {
                                    "ready_for_execution": False,
                                    "range_penalty_score": 150.0,
                                    "range_violation_count": 3,
                                    "total_range_excess_raw_ticks": 100.0,
                                    "max_range_excess_raw_ticks": 15.0,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            memory = build_candidate_memory(reports=[earlier, latest], output=output)
            output_exists = output.exists()
            markdown_exists = output.with_suffix(".md").exists()

        self.assertEqual(memory["status"], "passed")
        self.assertEqual(memory["best_candidate"]["prompt"], "best historical prompt")
        self.assertEqual(memory["latest_best_candidate"]["prompt"], "latest regressed prompt")
        self.assertTrue(memory["regression_from_best"]["is_regression"])
        self.assertEqual(memory["regression_from_best"]["penalty_delta"], 50.0)
        self.assertEqual(memory["next_agentic_layer_step"]["type"], "reuse_best_historical_prompt_family")
        self.assertFalse(memory["send_action_called"])
        self.assertFalse(memory["physical_robot_motion"])
        self.assertFalse(memory["task_success_claim_allowed"])
        self.assertTrue(output_exists)
        self.assertTrue(markdown_exists)

    def test_memory_can_read_projection_reports(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            projection = tmp / "projection.json"
            output = tmp / "memory.json"
            projection.write_text(
                json.dumps(
                    {
                        "operation": "real_so100_projection_analysis",
                        "status": "passed",
                        "observer_camera_indexes": [],
                        "observer_camera_status": "temporarily_unavailable",
                        "ranked_candidates": [
                            {
                                "candidate_index": 2,
                                "prompt": "projection prompt",
                                "projection": {
                                    "projected_ready_for_execution_shape_only": True,
                                    "projection_penalty_score": 123.0,
                                    "range_violation_count": 4,
                                    "total_raw_distortion": 80.0,
                                    "max_raw_distortion": 20.0,
                                    "joint_distortion": {
                                        "wrist_flex": {"violation_count": 3, "total_raw_distortion": 60.0}
                                    },
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            memory = build_candidate_memory(reports=[projection], output=output)

        self.assertEqual(memory["best_candidate"]["score"]["penalty_score"], 123.0)
        self.assertEqual(memory["best_candidate"]["score"]["joint_excess_raw_ticks"], {"wrist_flex": 60.0})
        self.assertEqual(memory["next_agentic_layer_step"]["type"], "continue_from_latest_best_prompt_family")
