#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_BRIDGE_REPORT = "_workspace/real_so100/agentic_smolvla_late_safe_pose_bridge_move_right_observer_off_036/late_safe_pose_bridge_report.json"


def build_execution_preflight_runbook(
    *,
    router_report: Path,
    output: Path,
    bridge_report: Path = Path(DEFAULT_BRIDGE_REPORT),
    port: str = DEFAULT_PORT,
    label: str = "move_right_live_readonly_049",
    observer_camera_index: int = 3,
    chunk_size: int = 10,
    max_abs_raw_delta_per_step: float = 80.0,
    markdown: Path | None = None,
) -> dict[str, Any]:
    router = _load_json(router_report)
    bridge = _load_json(bridge_report)
    route = router.get("selected_route") or {}
    output_dir = Path("_workspace/real_so100") / f"agentic_smolvla_observer_return_refresh_{label}"
    refresh_report = output_dir / "observer_return_refresh.json"
    execution_packet = Path("_workspace/real_so100") / f"agentic_smolvla_transition_execution_packet_{label}" / "transition_execution_packet.json"
    executor_dry_run = Path("_workspace/real_so100") / f"agentic_smolvla_execute_transition_packet_{label}_dry_run" / "execute_transition_packet_report.json"
    visual_dir = Path("_workspace/real_so100") / f"agentic_smolvla_execute_transition_packet_{label}" / "observer_camera_3"
    allowed_to_prepare = (
        router.get("status") == "passed"
        and route.get("type") == "resolve_execution_preflight"
        and route.get("prompt_mutation_allowed") is False
    )
    report = {
        "operation": "real_so100_execution_preflight_runbook",
        "status": "passed" if allowed_to_prepare else "blocked",
        "purpose": "prepare the observer-return live-readonly preflight path without moving the SO-100",
        "source_router_report": str(router_report),
        "source_bridge_report": str(bridge_report),
        "selected_route": route,
        "policy_camera_indexes": router.get("policy_camera_indexes") or [0, 1],
        "observer_camera_indexes": [],
        "observer_camera_status": router.get("observer_camera_status", "off"),
        "required_observer_camera_index": observer_camera_index,
        "port": port,
        "label": label,
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "bridge_target": _bridge_target(bridge),
        "commands": _commands(
            bridge_report=bridge_report,
            output_dir=output_dir,
            refresh_report=refresh_report,
            execution_packet=execution_packet,
            executor_dry_run=executor_dry_run,
            visual_dir=visual_dir,
            port=port,
            observer_camera_index=observer_camera_index,
            max_abs_raw_delta_per_step=max_abs_raw_delta_per_step,
            chunk_size=chunk_size,
        ),
        "blockers": [] if allowed_to_prepare else ["Router did not select resolve_execution_preflight with prompt mutation blocked."],
        "next_agentic_layer_step": _next_step(allowed_to_prepare),
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
        "# Real SO-100 Execution Preflight Runbook",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Route: `{(report.get('selected_route') or {}).get('type')}`",
        f"- Policy cameras: `{report.get('policy_camera_indexes')}`",
        f"- Required observer camera: `{report.get('required_observer_camera_index')}`",
        f"- Actuation enabled: `{report.get('actuation_enabled', False)}`",
        f"- Physical robot motion: `{report.get('physical_robot_motion', False)}`",
        "",
        "## Commands",
        "",
    ]
    for command in report.get("commands", []):
        lines.extend(
            [
                f"### {command['name']}",
                "",
                f"- Purpose: {command['purpose']}",
                f"- Writes: `{command['writes']}`",
                "",
                "```bash",
                command["command"],
                "```",
                "",
            ]
        )
    if report.get("blockers"):
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
        lines.append("")
    lines.extend(
        [
            "## Next Step",
            "",
            f"- Type: `{report['next_agentic_layer_step']['type']}`",
            f"- Reason: {report['next_agentic_layer_step']['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def _commands(
    *,
    bridge_report: Path,
    output_dir: Path,
    refresh_report: Path,
    execution_packet: Path,
    executor_dry_run: Path,
    visual_dir: Path,
    port: str,
    observer_camera_index: int,
    max_abs_raw_delta_per_step: float,
    chunk_size: int,
) -> list[dict[str, str]]:
    refresh_command = " ".join(
        [
            "PYTHONPATH=src:.",
            ".venv/bin/python",
            "-B",
            "scripts/real_so100_observer_return_refresh.py",
            f"--bridge-report {bridge_report}",
            f"--output-dir {output_dir}",
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
    packet_command = " ".join(
        [
            "PYTHONPATH=src:.",
            ".venv/bin/python",
            "-B",
            "scripts/real_so100_transition_execution_packet.py",
            f"--refresh-report {refresh_report}",
            f"--output {execution_packet}",
            f"--observer-camera-index {observer_camera_index}",
        ]
    )
    dry_run_command = " ".join(
        [
            "PYTHONPATH=src:.",
            ".venv/bin/python",
            "-B",
            "scripts/real_so100_execute_transition_packet.py",
            f"--packet {execution_packet}",
            f"--output {executor_dry_run}",
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
            "purpose": "Read live SO-100 state, regenerate the transition, gate it, and run observer preflight without motor writes.",
            "writes": str(refresh_report),
            "command": refresh_command,
        },
        {
            "name": "build_transition_execution_packet",
            "purpose": "Convert a ready refresh bundle into a two-chunk execution packet.",
            "writes": str(execution_packet),
            "command": packet_command,
        },
        {
            "name": "executor_dry_run",
            "purpose": "Dry-run the ready packet shape without --execute before any physical movement.",
            "writes": str(executor_dry_run),
            "command": dry_run_command,
        },
    ]


def _bridge_target(bridge: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_prompt": bridge.get("source_prompt"),
        "bridge_target_step_index": bridge.get("bridge_target_step_index"),
        "all_bridge_targets_in_range": bool(bridge.get("all_bridge_targets_in_range")),
        "safe_run_start_step": bridge.get("safe_run_start_step"),
        "safe_run_length": bridge.get("safe_run_length"),
        "joints": [
            {
                "joint": item.get("joint"),
                "target_raw": item.get("target_raw"),
                "target_command_value": item.get("target_command_value"),
            }
            for item in bridge.get("bridge_target_joints") or []
        ],
    }


def _next_step(allowed_to_prepare: bool) -> dict[str, str]:
    if allowed_to_prepare:
        return {
            "type": "wait_for_camera_3_then_run_live_readonly_refresh",
            "reason": "The current candidate is preserved; when observer camera 3 returns, run the live-readonly refresh command before any execution.",
        }
    return {
        "type": "resolve_router_route_before_preflight",
        "reason": "The router did not select execution preflight as the next step.",
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an SO-100 observer-return execution preflight runbook.")
    parser.add_argument("--router-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge-report", type=Path, default=Path(DEFAULT_BRIDGE_REPORT))
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--label", default="move_right_live_readonly_049")
    parser.add_argument("--observer-camera-index", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--max-abs-raw-delta-per-step", type=float, default=80.0)
    args = parser.parse_args()
    print(
        json.dumps(
            build_execution_preflight_runbook(
                router_report=args.router_report,
                output=args.output,
                bridge_report=args.bridge_report,
                port=args.port,
                label=args.label,
                observer_camera_index=args.observer_camera_index,
                chunk_size=args.chunk_size,
                max_abs_raw_delta_per_step=args.max_abs_raw_delta_per_step,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
