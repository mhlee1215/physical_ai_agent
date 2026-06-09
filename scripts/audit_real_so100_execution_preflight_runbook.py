#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_COMMANDS = [
    "observer_return_refresh_live_readonly",
    "build_transition_execution_packet",
    "executor_dry_run",
]


def audit_execution_preflight_runbook(
    *,
    runbook: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    payload = _load_json(runbook)
    commands = payload.get("commands") or []
    checks = [
        _check_bool("runbook_exists", runbook.exists(), {"path": str(runbook)}),
        _check_bool("runbook_status_passed", payload.get("status") == "passed", {"status": payload.get("status")}),
        _check_bool("operation_matches", payload.get("operation") == "real_so100_execution_preflight_runbook", {"operation": payload.get("operation")}),
        _check_bool("no_actuation_flags", _no_actuation(payload), _actuation_details(payload)),
        _check_bool("policy_cameras_are_0_1", [int(index) for index in payload.get("policy_camera_indexes") or []] == [0, 1], {"policy_camera_indexes": payload.get("policy_camera_indexes")}),
        _check_bool("required_observer_camera_is_3", payload.get("required_observer_camera_index") == 3, {"required_observer_camera_index": payload.get("required_observer_camera_index")}),
        _check_bool("command_order", [item.get("name") for item in commands] == EXPECTED_COMMANDS, {"command_names": [item.get("name") for item in commands]}),
        _check_bool("commands_have_no_execute_flag", all("--execute" not in str(item.get("command", "")) for item in commands), {"commands": [item.get("command") for item in commands]}),
        _check_bool("refresh_is_live_readonly", _command_contains(commands, "observer_return_refresh_live_readonly", "--mode live_readonly"), {"command": _command(commands, "observer_return_refresh_live_readonly")}),
        _check_bool("refresh_requires_observer_camera_available", _command_contains(commands, "observer_return_refresh_live_readonly", "--observer-camera-status available"), {"command": _command(commands, "observer_return_refresh_live_readonly")}),
        _check_bool("refresh_uses_camera_3", _command_contains(commands, "observer_return_refresh_live_readonly", "--observer-camera-index 3"), {"command": _command(commands, "observer_return_refresh_live_readonly")}),
        _check_bool("packet_uses_camera_3", _command_contains(commands, "build_transition_execution_packet", "--observer-camera-index 3"), {"command": _command(commands, "build_transition_execution_packet")}),
        _check_bool("executor_is_dry_run_shape", _executor_is_dry_run(commands), {"command": _command(commands, "executor_dry_run")}),
        _check_bool("bridge_target_in_range", bool((payload.get("bridge_target") or {}).get("all_bridge_targets_in_range")), {"bridge_target": payload.get("bridge_target")}),
        _check_bool("next_step_waits_for_camera_3", (payload.get("next_agentic_layer_step") or {}).get("type") == "wait_for_camera_3_then_run_live_readonly_refresh", {"next_agentic_layer_step": payload.get("next_agentic_layer_step")}),
    ]
    failed = [check for check in checks if check["status"] != "passed"]
    result = {
        "operation": "real_so100_execution_preflight_runbook_audit",
        "status": "passed" if not failed else "failed",
        "runbook": str(runbook),
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "checks": checks,
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "next_agentic_layer_step": _next_step(failed),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result["manifest_path"] = str(output)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _no_actuation(payload: dict[str, Any]) -> bool:
    return (
        payload.get("actuation_enabled") is False
        and payload.get("send_action_called") is False
        and payload.get("policy_actions_executed") is False
        and payload.get("physical_robot_motion") is False
        and payload.get("task_success_claim_allowed") is False
    )


def _actuation_details(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "actuation_enabled": payload.get("actuation_enabled"),
        "send_action_called": payload.get("send_action_called"),
        "policy_actions_executed": payload.get("policy_actions_executed"),
        "physical_robot_motion": payload.get("physical_robot_motion"),
        "task_success_claim_allowed": payload.get("task_success_claim_allowed"),
    }


def _executor_is_dry_run(commands: list[dict[str, Any]]) -> bool:
    command = _command(commands, "executor_dry_run")
    return (
        "scripts/real_so100_execute_transition_packet.py" in command
        and "--record-video" in command
        and "--observer-camera-index 3" in command
        and "--execute" not in command
    )


def _command_contains(commands: list[dict[str, Any]], name: str, needle: str) -> bool:
    return needle in _command(commands, name)


def _command(commands: list[dict[str, Any]], name: str) -> str:
    for command in commands:
        if command.get("name") == name:
            return str(command.get("command", ""))
    return ""


def _next_step(failed: list[dict[str, Any]]) -> dict[str, str]:
    if not failed:
        return {
            "type": "safe_to_run_live_readonly_refresh_when_camera_3_returns",
            "reason": "The runbook is a no-actuation live-readonly preflight path; run it only after observer camera 3 is available.",
        }
    return {
        "type": "fix_execution_preflight_runbook_before_use",
        "reason": "One or more runbook safety checks failed.",
    }


def _check_bool(name: str, condition: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed" if condition else "failed",
        "details": details,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a real SO-100 execution preflight runbook before use.")
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_execution_preflight_runbook(runbook=args.runbook, output=args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
