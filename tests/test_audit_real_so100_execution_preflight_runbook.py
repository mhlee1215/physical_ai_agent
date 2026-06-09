from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.audit_real_so100_execution_preflight_runbook import audit_execution_preflight_runbook


class AuditRealSO100ExecutionPreflightRunbookTest(TestCase):
    def test_audit_passes_for_no_execute_live_readonly_runbook(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runbook = _write_runbook(tmp, include_execute=False)

            report = audit_execution_preflight_runbook(runbook=runbook, output=tmp / "audit.json")
            audit_exists = (tmp / "audit.json").exists()

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["failed_check_count"], 0)
        self.assertEqual(
            report["next_agentic_layer_step"]["type"],
            "safe_to_run_live_readonly_refresh_when_camera_3_returns",
        )
        self.assertFalse(report["physical_robot_motion"])
        self.assertTrue(audit_exists)

    def test_audit_fails_when_execute_flag_is_present(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            runbook = _write_runbook(tmp, include_execute=True)

            report = audit_execution_preflight_runbook(runbook=runbook)

        self.assertEqual(report["status"], "failed")
        failed_names = {check["name"] for check in report["checks"] if check["status"] == "failed"}
        self.assertIn("commands_have_no_execute_flag", failed_names)
        self.assertIn("executor_is_dry_run_shape", failed_names)
        self.assertEqual(report["next_agentic_layer_step"]["type"], "fix_execution_preflight_runbook_before_use")


def _write_runbook(root: Path, *, include_execute: bool) -> Path:
    execute_suffix = " --execute" if include_execute else ""
    path = root / "runbook.json"
    path.write_text(
        json.dumps(
            {
                "operation": "real_so100_execution_preflight_runbook",
                "status": "passed",
                "actuation_enabled": False,
                "send_action_called": False,
                "policy_actions_executed": False,
                "physical_robot_motion": False,
                "task_success_claim_allowed": False,
                "policy_camera_indexes": [0, 1],
                "required_observer_camera_index": 3,
                "bridge_target": {"all_bridge_targets_in_range": True},
                "next_agentic_layer_step": {"type": "wait_for_camera_3_then_run_live_readonly_refresh"},
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
