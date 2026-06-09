#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_FIRST_COMMAND = "observer_return_refresh_live_readonly"
EXPECTED_PLAN_OPERATION = "real_so100_agentic_state_command_plan"
EXPECTED_AUDIT_OPERATION = "real_so100_agentic_state_command_plan_audit"
EXPECTED_AUDIT_NEXT_STEP = "safe_to_run_first_command_when_camera_3_available"


def build_first_command_launch_packet(
    *,
    command_plan: Path,
    audit: Path,
    output: Path,
    markdown: Path | None = None,
) -> dict[str, Any]:
    plan_payload = _load_json(command_plan)
    audit_payload = _load_json(audit)
    commands = plan_payload.get("commands") or []
    first = commands[0] if commands else {}
    blockers = _blockers(plan=plan_payload, audit=audit_payload, first=first)
    passed = not blockers
    report = {
        "operation": "real_so100_first_command_launch_packet",
        "status": "passed" if passed else "blocked",
        "purpose": "extract only the next allowed live-readonly observer refresh command without authorizing actuation",
        "source_command_plan": str(command_plan),
        "source_audit": str(audit),
        "launch_command_name": first.get("name") if passed else None,
        "launch_command": first.get("command") if passed else None,
        "launch_command_allowed_when": "observer_camera_3_available",
        "requires_observer_camera_available": True,
        "required_observer_camera_index": 3,
        "policy_camera_indexes": plan_payload.get("policy_camera_indexes") or [0, 1],
        "observer_camera_indexes": [3],
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "not_a_physical_execution_authorization": True,
        "does_not_run_command": True,
        "blocked_followup_commands": _blocked_followups(commands),
        "blockers": blockers,
        "next_agentic_layer_step": _next_step(passed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path = markdown or output.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report["json_path"] = str(output)
    report["markdown_path"] = str(md_path)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real SO-100 First Command Launch Packet",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Launch command: `{report.get('launch_command_name')}`",
        f"- Allowed when: `{report.get('launch_command_allowed_when')}`",
        f"- Required observer camera: `{report.get('required_observer_camera_index')}`",
        f"- Actuation enabled: `{report.get('actuation_enabled')}`",
        f"- Physical robot motion: `{report.get('physical_robot_motion')}`",
        f"- Not physical execution authorization: `{report.get('not_a_physical_execution_authorization')}`",
        "",
    ]
    if report.get("launch_command"):
        lines.extend(["## First Command Only", "", "```bash", str(report["launch_command"]), "```", ""])
    if report.get("blocked_followup_commands"):
        lines.extend(["## Blocked Followups", ""])
        for item in report["blocked_followup_commands"]:
            lines.append(f"- `{item.get('name')}`: {item.get('reason')}")
        lines.append("")
    if report.get("blockers"):
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
        lines.append("")
    return "\n".join(lines)


def _blockers(*, plan: dict[str, Any], audit: dict[str, Any], first: dict[str, Any]) -> list[str]:
    blockers = []
    if plan.get("operation") != EXPECTED_PLAN_OPERATION:
        blockers.append(f"Command plan operation is {plan.get('operation')!r}.")
    if plan.get("status") != "passed":
        blockers.append(f"Command plan status is {plan.get('status')!r}.")
    if audit.get("operation") != EXPECTED_AUDIT_OPERATION:
        blockers.append(f"Audit operation is {audit.get('operation')!r}.")
    if audit.get("status") != "passed":
        blockers.append(f"Audit status is {audit.get('status')!r}.")
    if audit.get("failed_check_count") not in (0, None):
        blockers.append(f"Audit failed_check_count is {audit.get('failed_check_count')!r}.")
    if (audit.get("next_agentic_layer_step") or {}).get("type") != EXPECTED_AUDIT_NEXT_STEP:
        blockers.append("Audit next step does not allow first-command launch when camera 3 is available.")
    if first.get("name") != EXPECTED_FIRST_COMMAND:
        blockers.append(f"First command is {first.get('name')!r}, not {EXPECTED_FIRST_COMMAND!r}.")
    command = str(first.get("command", ""))
    if "--execute" in command:
        blockers.append("First command contains --execute.")
    if "--mode live_readonly" not in command:
        blockers.append("First command is not live_readonly.")
    if "--observer-camera-index 3" not in command:
        blockers.append("First command does not require observer camera 3.")
    if "--observer-camera-status available" not in command:
        blockers.append("First command does not require observer camera availability.")
    if _has_actuation(plan):
        blockers.append("Command plan records actuation or physical motion.")
    if _has_actuation(audit):
        blockers.append("Audit records actuation or physical motion.")
    return blockers


def _blocked_followups(commands: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "name": str(command.get("name")),
            "reason": "depend_on_refresh_output",
            "command": str(command.get("command", "")),
        }
        for command in commands[1:]
    ]


def _has_actuation(payload: dict[str, Any]) -> bool:
    return any(
        payload.get(key) is not False
        for key in (
            "actuation_enabled",
            "send_action_called",
            "policy_actions_executed",
            "physical_robot_motion",
            "task_success_claim_allowed",
        )
    )


def _next_step(passed: bool) -> dict[str, str]:
    if passed:
        return {
            "type": "run_launch_command_when_camera_3_available_and_capture_output",
            "reason": "The packet contains only the no-actuation live-readonly refresh command; followup commands still depend on refresh output.",
        }
    return {
        "type": "repair_state_command_plan_or_audit_before_launch",
        "reason": "The first-command launch packet could not be safely extracted.",
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the first safe command from a real SO-100 state command plan.")
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_first_command_launch_packet(
                command_plan=args.command_plan,
                audit=args.audit,
                output=args.output,
                markdown=args.markdown,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
