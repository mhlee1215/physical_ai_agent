from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.audit_real_so100_agentic_state_command_plan import audit_state_command_plan


class AuditRealSO100AgenticStateCommandPlanTest(TestCase):
    def test_audit_passes_for_live_readonly_no_execute_plan(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            plan = _write_plan(tmp, include_execute=False)

            report = audit_state_command_plan(command_plan=plan, output=tmp / "audit.json")
            audit_exists = (tmp / "audit.json").exists()

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["failed_check_count"], 0)
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "safe_to_run_first_command_when_camera_3_available",
        )
        self.assertFalse(report["physical_robot_motion"])
        self.assertTrue(audit_exists)

    def test_audit_fails_when_execute_flag_is_present(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            plan = _write_plan(tmp, include_execute=True)

            report = audit_state_command_plan(command_plan=plan)

        self.assertEqual(report["status"], "failed")
        failed = {check["name"] for check in report["checks"] if check["status"] == "failed"}
        self.assertIn("commands_have_no_execute_flag", failed)
        self.assertIn("executor_command_is_dry_run_shape", failed)

    def test_audit_fails_when_blocked_actions_are_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            plan = _write_plan(tmp, include_execute=False, include_all_blocked_actions=False)

            report = audit_state_command_plan(command_plan=plan)

        self.assertEqual(report["status"], "failed")
        failed = {check["name"] for check in report["checks"] if check["status"] == "failed"}
        self.assertIn("blocked_actions_carried_forward", failed)


def _write_plan(root: Path, *, include_execute: bool, include_all_blocked_actions: bool = True) -> Path:
    execute_suffix = " --execute" if include_execute else ""
    blocked_actions = [
        {"type": "physical_execution"},
        {"type": "task_success_claim"},
    ]
    if include_all_blocked_actions:
        blocked_actions.extend(
            [
                {"type": "rerun_regressed_policy_camera_prompt"},
                {"type": "prompt_mutation_before_observer_refresh"},
            ]
        )
    path = root / "plan.json"
    path.write_text(
        json.dumps(
            {
                "operation": "real_so100_agentic_state_command_plan",
                "status": "passed",
                "actuation_enabled": False,
                "send_action_called": False,
                "policy_actions_executed": False,
                "physical_robot_motion": False,
                "task_success_claim_allowed": False,
                "policy_camera_indexes": [0, 1],
                "required_observer_camera_index": 3,
                "requires_observer_camera_available": True,
                "bridge_target": {"all_bridge_targets_in_range": True},
                "blocked_actions_carried_forward": blocked_actions,
                "next_agentic_layer_step": {"type": "run_first_command_only_when_camera_3_available"},
                "commands": [
                    {
                        "name": "observer_return_refresh_live_readonly",
                        "command": "PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_observer_return_refresh.py --mode live_readonly --observer-camera-index 3 --observer-camera-status available",
                    },
                    {
                        "name": "build_transition_execution_packet",
                        "command": "PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_transition_execution_packet.py --observer-camera-index 3",
                    },
                    {
                        "name": "executor_dry_run",
                        "command": f"PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_execute_transition_packet.py --observer-camera-index 3 --record-video{execute_suffix}",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path
