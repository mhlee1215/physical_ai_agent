#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_observer_return_preflight(
    *,
    transition_gate: Path,
    output: Path,
    observer_camera_index: int = 3,
    observer_camera_status: str = "temporarily_unavailable",
    live_readback_regenerated: bool = False,
    user_confirmed: bool = False,
    workspace_clear_confirmed: bool = False,
    markdown: Path | None = None,
) -> dict[str, Any]:
    gate = json.loads(transition_gate.read_text(encoding="utf-8"))
    checks = _checks(
        gate=gate,
        observer_camera_index=observer_camera_index,
        observer_camera_status=observer_camera_status,
        live_readback_regenerated=live_readback_regenerated,
        user_confirmed=user_confirmed,
        workspace_clear_confirmed=workspace_clear_confirmed,
    )
    blockers = [check["blocker"] for check in checks if not check["passed"]]
    status = "ready_for_observer_backed_execution_gate" if not blockers else "blocked"
    result = {
        "operation": "real_so100_observer_return_preflight",
        "status": status,
        "source_transition_gate": str(transition_gate),
        "source_transition_gate_status": gate.get("status"),
        "policy_camera_indexes": gate.get("policy_camera_indexes"),
        "required_observer_camera_index": observer_camera_index,
        "observer_camera_indexes": [observer_camera_index] if observer_camera_status == "available" else [],
        "observer_camera_status": observer_camera_status,
        "camera_3_status": observer_camera_status if observer_camera_index == 3 else "not_required_index",
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "execution_ready_with_observer": not blockers,
        "live_readback_regenerated": live_readback_regenerated,
        "user_confirmed": user_confirmed,
        "workspace_clear_confirmed": workspace_clear_confirmed,
        "checks": checks,
        "blockers": blockers,
        "required_execution_artifacts": [
            f"camera_{observer_camera_index}_before_frame",
            f"camera_{observer_camera_index}_motion_video",
            f"camera_{observer_camera_index}_after_frame",
            "live_readback_before_raw",
            "live_readback_regenerated_transition_plan",
            "per_step_target_commands",
            "readback_after_raw",
            "observed_delta_raw",
            "task_level_grasp_outcome_if_contact_attempted",
            "task_level_object_relocation_if_transport_attempted",
            "agentic_feedback_report",
        ],
        "next_agentic_layer_step": _next_step(blockers),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path = markdown or output.with_suffix(".md")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    result["json_path"] = str(output)
    result["markdown_path"] = str(md_path)
    return result


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real SO-100 Observer Return Preflight",
        "",
        f"- Status: `{report['status']}`",
        f"- Source transition gate: `{report.get('source_transition_gate')}`",
        f"- Required observer camera: `{report.get('required_observer_camera_index')}`",
        f"- Observer camera status: `{report.get('observer_camera_status')}`",
        f"- Execution ready with observer: `{report.get('execution_ready_with_observer', False)}`",
        f"- Actuation enabled: `{report.get('actuation_enabled', False)}`",
        f"- Physical robot motion: `{report.get('physical_robot_motion', False)}`",
        f"- Task success claim allowed: `{report.get('task_success_claim_allowed', False)}`",
        "",
        "## Checks",
        "",
    ]
    for check in report.get("checks", []):
        lines.append(f"- `{check['name']}`: passed=`{check['passed']}`")
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
    lines.extend(
        [
            "",
            "## Required Execution Artifacts",
            "",
        ]
    )
    lines.extend(f"- `{artifact}`" for artifact in report.get("required_execution_artifacts", []))
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            f"- Type: `{report['next_agentic_layer_step']['type']}`",
            f"- Reason: {report['next_agentic_layer_step']['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def _checks(
    *,
    gate: dict[str, Any],
    observer_camera_index: int,
    observer_camera_status: str,
    live_readback_regenerated: bool,
    user_confirmed: bool,
    workspace_clear_confirmed: bool,
) -> list[dict[str, Any]]:
    return [
        _check(
            "transition_gate_passed",
            gate.get("status") == "passed",
            "Transition candidate gate must pass before physical execution preflight.",
        ),
        _check(
            "transition_gate_no_actuation",
            not gate.get("send_action_called") and not gate.get("physical_robot_motion"),
            "Transition candidate gate must be no-actuation evidence only.",
        ),
        _check(
            "observer_camera_available",
            observer_camera_status == "available",
            f"Observer camera {observer_camera_index} must be available for before/during/after evidence.",
        ),
        _check(
            "observer_camera_not_policy_input",
            observer_camera_index not in [int(index) for index in gate.get("policy_camera_indexes") or []],
            f"Observer camera {observer_camera_index} must not be a SmolVLA policy input.",
        ),
        _check(
            "live_readback_regenerated",
            live_readback_regenerated,
            "Transition must be regenerated from live SO-100 readback before execution.",
        ),
        _check(
            "workspace_clear_confirmed",
            workspace_clear_confirmed,
            "Workspace must be confirmed clear before any physical movement.",
        ),
        _check(
            "user_confirmed",
            user_confirmed,
            "User confirmation is required before any physical SO-100 execution.",
        ),
    ]


def _check(name: str, passed: bool, blocker: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "blocker": blocker}


def _next_step(blockers: list[str]) -> dict[str, Any]:
    if not blockers:
        return {
            "type": "open_observer_backed_physical_execution_gate",
            "reason": "All preflight checks passed; the next stage may build an execution report with observer video and live readbacks.",
        }
    return {
        "type": "wait_for_observer_camera_3_and_live_readback_regeneration",
        "reason": "The transition candidate is valid, but physical execution preflight is not complete.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight observer return requirements before SO-100 transition execution.")
    parser.add_argument("--transition-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-camera-index", type=int, default=3)
    parser.add_argument("--observer-camera-status", choices=["available", "temporarily_unavailable", "off"], default="temporarily_unavailable")
    parser.add_argument("--live-readback-regenerated", action="store_true")
    parser.add_argument("--user-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_observer_return_preflight(
                transition_gate=args.transition_gate,
                output=args.output,
                observer_camera_index=args.observer_camera_index,
                observer_camera_status=args.observer_camera_status,
                live_readback_regenerated=args.live_readback_regenerated,
                user_confirmed=args.user_confirmed,
                workspace_clear_confirmed=args.workspace_clear_confirmed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
