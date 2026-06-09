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
EXPECTED_BLOCKED_ACTIONS = {
    "physical_execution",
    "task_success_claim",
    "rerun_regressed_policy_camera_prompt",
    "prompt_mutation_before_observer_refresh",
}


def audit_state_command_plan(*, command_plan: Path, output: Path | None = None) -> dict[str, Any]:
    payload = _load_json(command_plan)
    commands = payload.get("commands") or []
    checks = [
        _check("plan_exists", command_plan.exists(), {"path": str(command_plan)}),
        _check("plan_status_passed", payload.get("status") == "passed", {"status": payload.get("status")}),
        _check("operation_matches", payload.get("operation") == "real_so100_agentic_state_command_plan", {"operation": payload.get("operation")}),
        _check("no_actuation_flags", _no_actuation(payload), _actuation_details(payload)),
        _check("policy_cameras_are_0_1", [int(index) for index in payload.get("policy_camera_indexes") or []] == [0, 1], {"policy_camera_indexes": payload.get("policy_camera_indexes")}),
        _check("observer_camera_required_available", payload.get("requires_observer_camera_available") is True, {"requires_observer_camera_available": payload.get("requires_observer_camera_available")}),
        _check("required_observer_camera_is_3", payload.get("required_observer_camera_index") == 3, {"required_observer_camera_index": payload.get("required_observer_camera_index")}),
        _check("command_order", [item.get("name") for item in commands] == EXPECTED_COMMANDS, {"command_names": [item.get("name") for item in commands]}),
        _check("commands_have_no_execute_flag", all("--execute" not in str(item.get("command", "")) for item in commands), {"commands": [item.get("command") for item in commands]}),
        _check("first_command_is_live_readonly_refresh", _first_command_is_live_readonly(commands), {"command": _command(commands, "observer_return_refresh_live_readonly")}),
        _check("packet_command_uses_camera_3", _contains(commands, "build_transition_execution_packet", "--observer-camera-index 3"), {"command": _command(commands, "build_transition_execution_packet")}),
        _check("executor_command_is_dry_run_shape", _executor_is_dry_run(commands), {"command": _command(commands, "executor_dry_run")}),
        _check("bridge_target_in_range", bool((payload.get("bridge_target") or {}).get("all_bridge_targets_in_range")), {"bridge_target": payload.get("bridge_target")}),
        _check("next_step_runs_first_command_only", (payload.get("next_agentic_layer_step") or {}).get("type") == "run_first_command_only_when_camera_3_available", {"next_agentic_layer_step": payload.get("next_agentic_layer_step")}),
        _check("blocked_actions_carried_forward", EXPECTED_BLOCKED_ACTIONS.issubset(_blocked_action_types(payload)), {"blocked_actions": list(_blocked_action_types(payload))}),
    ]
    failed = [check for check in checks if check["status"] != "passed"]
    result = {
        "operation": "real_so100_agentic_state_command_plan_audit",
        "status": "passed" if not failed else "failed",
        "command_plan": str(command_plan),
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


def _first_command_is_live_readonly(commands: list[dict[str, Any]]) -> bool:
    if not commands or commands[0].get("name") != "observer_return_refresh_live_readonly":
        return False
    command = str(commands[0].get("command", ""))
    return (
        "scripts/real_so100_observer_return_refresh.py" in command
        and "--mode live_readonly" in command
        and "--observer-camera-index 3" in command
        and "--observer-camera-status available" in command
    )


def _executor_is_dry_run(commands: list[dict[str, Any]]) -> bool:
    command = _command(commands, "executor_dry_run")
    return (
        "scripts/real_so100_execute_transition_packet.py" in command
        and "--observer-camera-index 3" in command
        and "--record-video" in command
        and "--execute" not in command
    )


def _contains(commands: list[dict[str, Any]], name: str, needle: str) -> bool:
    return needle in _command(commands, name)


def _command(commands: list[dict[str, Any]], name: str) -> str:
    for command in commands:
        if command.get("name") == name:
            return str(command.get("command", ""))
    return ""


def _blocked_action_types(payload: dict[str, Any]) -> set[str]:
    return {str(item.get("type")) for item in payload.get("blocked_actions_carried_forward") or []}


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


def _next_step(failed: list[dict[str, Any]]) -> dict[str, str]:
    if failed:
        return {
            "type": "fix_state_command_plan_before_use",
            "reason": "One or more state command-plan safety checks failed.",
        }
    return {
        "type": "safe_to_run_first_command_when_camera_3_available",
        "reason": "The plan is no-actuation and the first command is live-readonly observer refresh.",
    }


def _check(name: str, condition: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": "passed" if condition else "failed", "details": details}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a real SO-100 agentic state command plan before use.")
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_state_command_plan(command_plan=args.command_plan, output=args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
