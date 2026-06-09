from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_agentic_state_command_plan import build_state_command_plan


class RealSO100AgenticStateCommandPlanTest(TestCase):
    def test_builds_live_readonly_plan_from_observer_wait_state(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = _write_state(tmp, allowed_action="wait_for_camera_3_then_run_live_readonly_refresh")
            bridge = _write_bridge(tmp)

            report = build_state_command_plan(
                loop_state=state,
                bridge_report=bridge,
                output=tmp / "plan.json",
                port="/dev/cu.fake",
                label="test_058",
            )

        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertTrue(report["requires_observer_camera_available"])
        self.assertEqual([command["name"] for command in report["commands"]], [
            "observer_return_refresh_live_readonly",
            "build_transition_execution_packet",
            "executor_dry_run",
        ])
        joined = "\n".join(command["command"] for command in report["commands"])
        self.assertIn("--mode live_readonly", joined)
        self.assertIn("--observer-camera-status available", joined)
        self.assertIn("--record-video", joined)
        self.assertNotIn("--execute", joined)
        self.assertEqual(report["next_agentic_layer_step"]["type"], "run_first_command_only_when_camera_3_available")

    def test_blocks_when_state_allows_prompt_mutation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = _write_state(tmp, allowed_action="run_no_actuation_proposal_sweep")
            bridge = _write_bridge(tmp)

            report = build_state_command_plan(
                loop_state=state,
                bridge_report=bridge,
                output=tmp / "plan.json",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["commands"], [])
        self.assertTrue(report["blockers"])

    def test_blocks_when_state_records_physical_motion(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = _write_state(tmp, allowed_action="wait_for_camera_3_then_run_live_readonly_refresh", physical_motion=True)
            bridge = _write_bridge(tmp)

            report = build_state_command_plan(
                loop_state=state,
                bridge_report=bridge,
                output=tmp / "plan.json",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("physical robot motion" in blocker for blocker in report["blockers"]))


def _write_state(root: Path, *, allowed_action: str, physical_motion: bool = False) -> Path:
    path = root / "state.json"
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "allowed_next_actions": [{"type": allowed_action, "physical_robot_motion": False}],
                "blocked_actions": [{"type": "physical_execution"}, {"type": "task_success_claim"}],
                "camera_contract": {
                    "policy_camera_indexes": [0, 1],
                    "observer_camera_indexes": [],
                    "observer_camera_status": "off",
                },
                "execution_flags": {
                    "send_action_called": False,
                    "policy_actions_executed": False,
                    "physical_robot_motion": physical_motion,
                    "task_success_claim_allowed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_bridge(root: Path) -> Path:
    path = root / "bridge.json"
    path.write_text(
        json.dumps(
            {
                "source_prompt": "best prompt",
                "bridge_target_step_index": 3,
                "all_bridge_targets_in_range": True,
                "bridge_target_joints": [
                    {"joint": "shoulder_pan", "target_raw": 1787.0, "target_command_value": -10.0},
                    {"joint": "shoulder_lift", "target_raw": 3453.0, "target_command_value": 53.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path
