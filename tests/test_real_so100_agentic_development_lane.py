from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_agentic_development_lane import build_development_lane


class RealSO100AgenticDevelopmentLaneTest(TestCase):
    def test_separates_execution_wait_from_policy_camera_development(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = _write_loop_state(tmp)
            launch = _write_launch_packet(tmp)
            feedback = _write_policy_feedback(tmp)
            memory = _write_candidate_memory(tmp, is_regression=True)

            report = build_development_lane(
                loop_state=state,
                launch_packet=launch,
                policy_feedback=feedback,
                candidate_memory=memory,
                output=tmp / "lane.json",
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["execution_lane"]["state"], "waiting_for_observer_camera_3")
        self.assertEqual(report["execution_lane"]["next_command_name"], "observer_return_refresh_live_readonly")
        self.assertEqual(report["policy_camera_development_lane"]["state"], "active_no_actuation")
        self.assertFalse(report["prompt_mutation_allowed"])
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertEqual(report["next_agentic_layer_step"]["type"], "build_policy_camera_task_state_packet")
        blocked = {item["type"] for item in report["policy_camera_development_lane"]["blocked_no_actuation_actions"]}
        self.assertIn("mutate_prompt_from_latest_policy_camera_feedback", blocked)
        allowed = {item["type"] for item in report["policy_camera_development_lane"]["allowed_no_actuation_actions"]}
        self.assertIn("build_policy_camera_task_state_packet", allowed)

    def test_allows_prompt_variant_sweep_when_feedback_has_no_regression(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = _write_loop_state(tmp)
            launch = _write_launch_packet(tmp)
            feedback = _write_policy_feedback(tmp)
            memory = _write_candidate_memory(tmp, is_regression=False)

            report = build_development_lane(
                loop_state=state,
                launch_packet=launch,
                policy_feedback=feedback,
                candidate_memory=memory,
                output=tmp / "lane.json",
            )

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["prompt_mutation_allowed"])
        allowed = {item["type"] for item in report["policy_camera_development_lane"]["allowed_no_actuation_actions"]}
        self.assertIn("run_no_actuation_prompt_variant_sweep", allowed)

    def test_blocks_when_launch_packet_is_execution_authorization(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = _write_loop_state(tmp)
            launch = _write_launch_packet(tmp, authorizes_execution=True)

            report = build_development_lane(
                loop_state=state,
                launch_packet=launch,
                policy_feedback=None,
                candidate_memory=None,
                output=tmp / "lane.json",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["execution_lane"]["state"], "blocked")
        self.assertTrue(report["blockers"])


def _write_loop_state(root: Path) -> Path:
    path = root / "state.json"
    path.write_text(
        json.dumps(
            {
                "operation": "real_so100_agentic_loop_state",
                "status": "passed",
                "allowed_next_actions": [
                    {
                        "type": "wait_for_camera_3_then_run_live_readonly_refresh",
                        "reason": "Camera 3 is required before physical execution.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_launch_packet(root: Path, *, authorizes_execution: bool = False) -> Path:
    path = root / "launch.json"
    path.write_text(
        json.dumps(
            {
                "operation": "real_so100_first_command_launch_packet",
                "status": "passed",
                "launch_command_name": "observer_return_refresh_live_readonly",
                "launch_command": "PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_observer_return_refresh.py --mode live_readonly --observer-camera-index 3 --observer-camera-status available",
                "launch_command_allowed_when": "observer_camera_3_available",
                "not_a_physical_execution_authorization": not authorizes_execution,
                "physical_robot_motion": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_policy_feedback(root: Path) -> Path:
    path = root / "feedback.json"
    path.write_text(
        json.dumps(
            {
                "operation": "real_so100_policy_camera_pseudo_llm_feedback",
                "status": "passed",
                "camera_contract": {
                    "policy_camera_roles": {"0": "wrist_cam", "1": "egocentric_cam"},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_candidate_memory(root: Path, *, is_regression: bool) -> Path:
    path = root / "memory.json"
    path.write_text(
        json.dumps(
            {
                "operation": "real_so100_agentic_candidate_memory",
                "status": "passed",
                "best_candidate": {
                    "source_report": "best.json",
                    "candidate_index": 2,
                    "prompt": "best prompt",
                    "score": {"penalty_score": 42.0, "ready_for_execution": False},
                },
                "regression_from_best": {
                    "is_regression": is_regression,
                    "penalty_delta": 10.0 if is_regression else 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path
