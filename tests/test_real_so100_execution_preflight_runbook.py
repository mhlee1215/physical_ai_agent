from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_execution_preflight_runbook import build_execution_preflight_runbook


class RealSO100ExecutionPreflightRunbookTest(TestCase):
    def test_builds_live_readonly_commands_without_execute(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            router = _write_router(tmp, route="resolve_execution_preflight")
            bridge = _write_bridge(tmp)

            report = build_execution_preflight_runbook(
                router_report=router,
                bridge_report=bridge,
                output=tmp / "runbook.json",
                port="/dev/cu.fake",
                label="test_live_readonly",
            )

        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertEqual(report["next_agentic_layer_step"]["type"], "wait_for_camera_3_then_run_live_readonly_refresh")
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

    def test_blocks_when_router_does_not_select_execution_preflight(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            router = _write_router(tmp, route="mutate_smolvla_prompt_or_plan")
            bridge = _write_bridge(tmp)

            report = build_execution_preflight_runbook(
                router_report=router,
                bridge_report=bridge,
                output=tmp / "runbook.json",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["blockers"])
        self.assertEqual(report["next_agentic_layer_step"]["type"], "resolve_router_route_before_preflight")


def _write_router(root: Path, *, route: str) -> Path:
    path = root / "router.json"
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "operation": "real_so100_agentic_feedback_router",
                "policy_camera_indexes": [0, 1],
                "observer_camera_indexes": [],
                "observer_camera_status": "off",
                "selected_route": {
                    "type": route,
                    "prompt_mutation_allowed": route != "resolve_execution_preflight",
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
                "status": "passed",
                "source_prompt": "test prompt",
                "bridge_target_step_index": 3,
                "all_bridge_targets_in_range": True,
                "safe_run_start_step": 3,
                "safe_run_length": 7,
                "bridge_target_joints": [
                    {"joint": "shoulder_pan", "target_raw": 1787.0, "target_command_value": -10.0},
                    {"joint": "shoulder_lift", "target_raw": 3453.0, "target_command_value": 53.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path
