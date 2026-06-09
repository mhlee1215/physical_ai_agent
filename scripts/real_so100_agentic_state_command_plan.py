#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_BRIDGE_REPORT = "_workspace/real_so100/agentic_smolvla_late_safe_pose_bridge_move_right_observer_off_036/late_safe_pose_bridge_report.json"


def build_state_command_plan(
    *,
    loop_state: Path,
    output: Path,
    bridge_report: Path = Path(DEFAULT_BRIDGE_REPORT),
    port: str = DEFAULT_PORT,
    label: str = "move_right_live_readonly_058",
    observer_camera_index: int = 3,
    chunk_size: int = 10,
    max_abs_raw_delta_per_step: float = 80.0,
    markdown: Path | None = None,
) -> dict[str, Any]:
    state = _load_json(loop_state)
    bridge = _load_json(bridge_report)
    allowed = state.get("allowed_next_actions") or []
    primary = allowed[0] if allowed else {}
    allowed_type = primary.get("type")
    commands = _commands(
        bridge_report=bridge_report,
        label=label,
        port=port,
        observer_camera_index=observer_camera_index,
        max_abs_raw_delta_per_step=max_abs_raw_delta_per_step,
        chunk_size=chunk_size,
    )
    can_prepare = (
        state.get("status") == "passed"
        and allowed_type == "wait_for_camera_3_then_run_live_readonly_refresh"
        and _is_blocked(state, "physical_execution")
        and not state.get("execution_flags", {}).get("physical_robot_motion")
    )
    report = {
        "operation": "real_so100_agentic_state_command_plan",
        "status": "passed" if can_prepare else "blocked",
        "purpose": "convert the current agentic loop state into the next allowed no-actuation command plan",
        "source_loop_state": str(loop_state),
        "source_bridge_report": str(bridge_report),
        "selected_allowed_action": primary,
        "policy_camera_indexes": (state.get("camera_contract") or {}).get("policy_camera_indexes") or [0, 1],
        "observer_camera_indexes": (state.get("camera_contract") or {}).get("observer_camera_indexes") or [],
        "observer_camera_status": (state.get("camera_contract") or {}).get("observer_camera_status"),
        "required_observer_camera_index": observer_camera_index,
        "requires_observer_camera_available": True,
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "bridge_target": _bridge_target(bridge),
        "commands": commands if can_prepare else [],
        "blocked_actions_carried_forward": state.get("blocked_actions") or [],
        "blockers": [] if can_prepare else _blockers(state=state, allowed_type=allowed_type),
        "next_agentic_layer_step": _next_step(can_prepare),
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
        "# Real SO-100 Agentic State Command Plan",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Allowed action: `{(report.get('selected_allowed_action') or {}).get('type')}`",
        f"- Requires observer camera: `{report.get('required_observer_camera_index')}`",
        f"- Actuation enabled: `{report.get('actuation_enabled')}`",
        f"- Physical robot motion: `{report.get('physical_robot_motion')}`",
        "",
        "## Commands",
        "",
    ]
    for command in report.get("commands", []):
        lines.extend(["```bash", command["command"], "```", ""])
    if report.get("blockers"):
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
        lines.append("")
    return "\n".join(lines)


def _commands(
    *,
    bridge_report: Path,
    label: str,
    port: str,
    observer_camera_index: int,
    max_abs_raw_delta_per_step: float,
    chunk_size: int,
) -> list[dict[str, str]]:
    refresh_dir = Path("_workspace/real_so100") / f"agentic_smolvla_observer_return_refresh_{label}"
    refresh_report = refresh_dir / "observer_return_refresh.json"
    packet = Path("_workspace/real_so100") / f"agentic_smolvla_transition_execution_packet_{label}" / "transition_execution_packet.json"
    dry_run = Path("_workspace/real_so100") / f"agentic_smolvla_execute_transition_packet_{label}_dry_run" / "execute_transition_packet_report.json"
    visual_dir = Path("_workspace/real_so100") / f"agentic_smolvla_execute_transition_packet_{label}" / "observer_camera_3"
    refresh = " ".join(
        [
            "PYTHONPATH=src:.",
            ".venv/bin/python",
            "-B",
            "scripts/real_so100_observer_return_refresh.py",
            f"--bridge-report {bridge_report}",
            f"--output-dir {refresh_dir}",
            f"--port {port}",
            "--mode live_readonly",
            f"--observer-camera-index {observer_camera_index}",
            "--observer-camera-status available",
            "--user-confirmed",
            "--workspace-clear-confirmed",
            f"--max-abs-raw-delta-per-step {max_abs_raw_delta_per_step}",
            f"--chunk-size {chunk_size}",
        ]
    )
    packet_cmd = " ".join(
        [
            "PYTHONPATH=src:.",
            ".venv/bin/python",
            "-B",
            "scripts/real_so100_transition_execution_packet.py",
            f"--refresh-report {refresh_report}",
            f"--output {packet}",
            f"--observer-camera-index {observer_camera_index}",
        ]
    )
    dry = " ".join(
        [
            "PYTHONPATH=src:.",
            ".venv/bin/python",
            "-B",
            "scripts/real_so100_execute_transition_packet.py",
            f"--packet {packet}",
            f"--output {dry_run}",
            f"--port {port}",
            f"--observer-camera-index {observer_camera_index}",
            f"--visual-output-dir {visual_dir}",
            "--record-video",
            "--step-settle-seconds 0.0",
        ]
    )
    return [
        {
            "name": "observer_return_refresh_live_readonly",
            "command": refresh,
            "writes": str(refresh_report),
        },
        {
            "name": "build_transition_execution_packet",
            "command": packet_cmd,
            "writes": str(packet),
        },
        {
            "name": "executor_dry_run",
            "command": dry,
            "writes": str(dry_run),
        },
    ]


def _bridge_target(bridge: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_prompt": bridge.get("source_prompt"),
        "bridge_target_step_index": bridge.get("bridge_target_step_index"),
        "all_bridge_targets_in_range": bool(bridge.get("all_bridge_targets_in_range")),
        "joints": [
            {
                "joint": item.get("joint"),
                "target_raw": item.get("target_raw"),
                "target_command_value": item.get("target_command_value"),
            }
            for item in bridge.get("bridge_target_joints") or []
        ],
    }


def _is_blocked(state: dict[str, Any], action_type: str) -> bool:
    return any(item.get("type") == action_type for item in state.get("blocked_actions", []))


def _blockers(*, state: dict[str, Any], allowed_type: str | None) -> list[str]:
    blockers = []
    if state.get("status") != "passed":
        blockers.append("Loop state is not passed.")
    if allowed_type != "wait_for_camera_3_then_run_live_readonly_refresh":
        blockers.append(f"Loop state allowed action is {allowed_type!r}, not observer-return live-readonly refresh.")
    if state.get("execution_flags", {}).get("physical_robot_motion"):
        blockers.append("Loop state already records physical robot motion; require verifier feedback before planning another command.")
    if not _is_blocked(state, "physical_execution"):
        blockers.append("Loop state does not explicitly block physical execution; refusing to infer no-actuation command context.")
    return blockers or ["Loop state cannot be converted into a command plan."]


def _next_step(can_prepare: bool) -> dict[str, str]:
    if can_prepare:
        return {
            "type": "run_first_command_only_when_camera_3_available",
            "reason": "Only the live-readonly refresh command is the next allowed command; later commands depend on its output.",
        }
    return {
        "type": "repair_loop_state_before_command_planning",
        "reason": "The loop state did not authorize the observer-return live-readonly command plan.",
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the next command plan from the real SO-100 agentic loop state.")
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge-report", type=Path, default=Path(DEFAULT_BRIDGE_REPORT))
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--label", default="move_right_live_readonly_058")
    parser.add_argument("--observer-camera-index", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--max-abs-raw-delta-per-step", type=float, default=80.0)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_state_command_plan(
                loop_state=args.loop_state,
                output=args.output,
                bridge_report=args.bridge_report,
                port=args.port,
                label=args.label,
                observer_camera_index=args.observer_camera_index,
                chunk_size=args.chunk_size,
                max_abs_raw_delta_per_step=args.max_abs_raw_delta_per_step,
                markdown=args.markdown,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
