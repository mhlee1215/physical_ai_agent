from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.real_so100_first_command_launch_packet import build_first_command_launch_packet


class RealSO100FirstCommandLaunchPacketTest(TestCase):
    def test_extracts_only_live_readonly_first_command(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            plan = _write_plan(tmp)
            audit = _write_audit(tmp)

            report = build_first_command_launch_packet(
                command_plan=plan,
                audit=audit,
                output=tmp / "launch.json",
            )
            persisted = json.loads((tmp / "launch.json").read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["launch_command_name"], "observer_return_refresh_live_readonly")
        self.assertIn("--mode live_readonly", report["launch_command"])
        self.assertIn("--observer-camera-index 3", report["launch_command"])
        self.assertNotIn("--execute", report["launch_command"])
        self.assertFalse(report["send_action_called"])
        self.assertFalse(report["physical_robot_motion"])
        self.assertTrue(report["not_a_physical_execution_authorization"])
        self.assertEqual([item["name"] for item in report["blocked_followup_commands"]], [
            "build_transition_execution_packet",
            "executor_dry_run",
        ])
        self.assertNotIn("real_so100_transition_execution_packet.py", report["launch_command"])
        self.assertEqual(persisted["status"], "passed")

    def test_blocks_when_audit_failed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            plan = _write_plan(tmp)
            audit = _write_audit(tmp, status="failed", failed_check_count=1)

            report = build_first_command_launch_packet(
                command_plan=plan,
                audit=audit,
                output=tmp / "launch.json",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertIsNone(report["launch_command"])
        self.assertTrue(any("Audit status" in blocker for blocker in report["blockers"]))

    def test_blocks_when_first_command_is_not_live_readonly_refresh(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            plan = _write_plan(tmp, first_name="executor_dry_run")
            audit = _write_audit(tmp)

            report = build_first_command_launch_packet(
                command_plan=plan,
                audit=audit,
                output=tmp / "launch.json",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("First command" in blocker for blocker in report["blockers"]))

    def test_blocks_when_first_command_contains_execute(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            plan = _write_plan(tmp, first_suffix=" --execute")
            audit = _write_audit(tmp)

            report = build_first_command_launch_packet(
                command_plan=plan,
                audit=audit,
                output=tmp / "launch.json",
            )

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("--execute" in blocker for blocker in report["blockers"]))


def _write_plan(root: Path, *, first_name: str = "observer_return_refresh_live_readonly", first_suffix: str = "") -> Path:
    path = root / "plan.json"
    path.write_text(
        json.dumps(
            {
                "operation": "real_so100_agentic_state_command_plan",
                "status": "passed",
                "policy_camera_indexes": [0, 1],
                "actuation_enabled": False,
                "send_action_called": False,
                "policy_actions_executed": False,
                "physical_robot_motion": False,
                "task_success_claim_allowed": False,
                "commands": [
                    {
                        "name": first_name,
                        "command": (
                            "PYTHONPATH=src:. .venv/bin/python -B "
                            "scripts/real_so100_observer_return_refresh.py "
                            "--mode live_readonly --observer-camera-index 3 "
                            f"--observer-camera-status available{first_suffix}"
                        ),
                    },
                    {
                        "name": "build_transition_execution_packet",
                        "command": "PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_transition_execution_packet.py --observer-camera-index 3",
                    },
                    {
                        "name": "executor_dry_run",
                        "command": "PYTHONPATH=src:. .venv/bin/python -B scripts/real_so100_execute_transition_packet.py --observer-camera-index 3 --record-video",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_audit(root: Path, *, status: str = "passed", failed_check_count: int = 0) -> Path:
    path = root / "audit.json"
    path.write_text(
        json.dumps(
            {
                "operation": "real_so100_agentic_state_command_plan_audit",
                "status": status,
                "failed_check_count": failed_check_count,
                "actuation_enabled": False,
                "send_action_called": False,
                "policy_actions_executed": False,
                "physical_robot_motion": False,
                "task_success_claim_allowed": False,
                "next_agentic_layer_step": {"type": "safe_to_run_first_command_when_camera_3_available"},
            }
        ),
        encoding="utf-8",
    )
    return path
