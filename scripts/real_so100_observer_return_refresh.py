#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.real_so100_observer_return_preflight import build_observer_return_preflight
from scripts.real_so100_read_only_probe import run_probe
from scripts.real_so100_readback_transition_regenerator import regenerate_transition_from_readback
from scripts.real_so100_transition_candidate_gate import gate_transition_candidate


def run_observer_return_refresh(
    *,
    bridge_report: Path,
    output_dir: Path,
    port: str,
    mode: str = "replay",
    replay_readback: Path | None = None,
    frame_index: int | None = None,
    observer_camera_index: int = 3,
    observer_camera_status: str = "off",
    user_confirmed: bool = False,
    workspace_clear_confirmed: bool = False,
    max_abs_raw_delta_per_step: float = 80.0,
    chunk_size: int = 10,
) -> dict[str, Any]:
    if mode not in {"replay", "live_readonly"}:
        raise ValueError(f"mode must be replay or live_readonly, got {mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    readback_path = output_dir / "live_readback.json"
    readback_source = "live"
    readback_report: dict[str, Any] | None = None
    if mode == "live_readonly":
        readback_report = run_probe(port, readback_path)
    else:
        if replay_readback is None:
            raise ValueError("--replay-readback is required in replay mode")
        readback_path = replay_readback
        readback_source = "replay"

    regen_path = output_dir / "readback_transition_regen.json"
    regen_report = regenerate_transition_from_readback(
        bridge_report=bridge_report,
        readback=readback_path,
        output=regen_path,
        readback_source=readback_source,
        frame_index=frame_index,
        max_abs_raw_delta_per_step=max_abs_raw_delta_per_step,
        chunk_size=chunk_size,
    )
    gate_path = output_dir / "transition_candidate_gate.json"
    gate_report = gate_transition_candidate(
        transition_plan=regen_path,
        output=gate_path,
        expected_chunk_size=chunk_size,
        max_abs_raw_delta_per_step=max_abs_raw_delta_per_step,
    )
    preflight_path = output_dir / "observer_return_preflight.json"
    preflight_report = build_observer_return_preflight(
        transition_gate=gate_path,
        output=preflight_path,
        observer_camera_index=observer_camera_index,
        observer_camera_status=observer_camera_status,
        live_readback_regenerated=bool(regen_report.get("live_readback_regenerated")),
        user_confirmed=user_confirmed,
        workspace_clear_confirmed=workspace_clear_confirmed,
    )
    blockers = []
    if readback_report is not None and not readback_report.get("ok"):
        blockers.append("Live read-only probe did not pass.")
    if regen_report.get("status") != "passed":
        blockers.append("Readback transition regeneration did not pass.")
    if gate_report.get("status") != "passed":
        blockers.append("Transition candidate gate did not pass.")
    if preflight_report.get("status") != "ready_for_observer_backed_execution_gate":
        blockers.append("Observer return preflight is not ready.")
    status = "ready_for_execution_gate" if not blockers else "blocked"
    summary = {
        "operation": "real_so100_observer_return_refresh",
        "status": status,
        "mode": mode,
        "port": port,
        "source_bridge_report": str(bridge_report),
        "readback_path": str(readback_path),
        "readback_source": readback_source,
        "live_readback_regenerated": bool(regen_report.get("live_readback_regenerated")),
        "policy_camera_indexes": regen_report.get("policy_camera_indexes"),
        "observer_camera_indexes": preflight_report.get("observer_camera_indexes", []),
        "observer_camera_status": preflight_report.get("observer_camera_status"),
        "camera_3_status": preflight_report.get("camera_3_status"),
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "workspace_clear_confirmed": workspace_clear_confirmed,
        "user_confirmed": user_confirmed,
        "readback_report_path": str(readback_path) if mode == "live_readonly" else None,
        "regenerated_transition_path": str(regen_path),
        "transition_gate_path": str(gate_path),
        "observer_preflight_path": str(preflight_path),
        "regenerated_transition_status": regen_report.get("status"),
        "transition_gate_status": gate_report.get("status"),
        "observer_preflight_status": preflight_report.get("status"),
        "transition_chunk_count": regen_report.get("transition_chunk_count"),
        "transition_step_count": regen_report.get("transition_step_count"),
        "blockers": blockers,
        "next_agentic_layer_step": _next_step(status, mode),
    }
    summary_path = output_dir / "observer_return_refresh.json"
    summary_md_path = output_dir / "observer_return_refresh.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md_path.write_text(render_markdown(summary), encoding="utf-8")
    summary["json_path"] = str(summary_path)
    summary["markdown_path"] = str(summary_md_path)
    return summary


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real SO-100 Observer Return Refresh",
        "",
        f"- Status: `{report['status']}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Readback source: `{report.get('readback_source')}`",
        f"- Live readback regenerated: `{report.get('live_readback_regenerated', False)}`",
        f"- Observer camera status: `{report.get('observer_camera_status')}`",
        f"- Actuation enabled: `{report.get('actuation_enabled', False)}`",
        f"- Physical robot motion: `{report.get('physical_robot_motion', False)}`",
        "",
        "## Artifacts",
        "",
        f"- Regenerated transition: `{report.get('regenerated_transition_path')}`",
        f"- Transition gate: `{report.get('transition_gate_path')}`",
        f"- Observer preflight: `{report.get('observer_preflight_path')}`",
    ]
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report["blockers"])
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


def _next_step(status: str, mode: str) -> dict[str, Any]:
    if status == "ready_for_execution_gate":
        return {
            "type": "build_observer_backed_execution_report",
            "reason": "Readback regeneration, transition gate, and observer preflight passed; build the actual observer-backed execution report next.",
        }
    if mode == "replay":
        return {
            "type": "rerun_refresh_in_live_readonly_mode_when_camera_3_returns",
            "reason": "Replay refresh validates the orchestration path, but physical execution still needs live readback and observer camera 3.",
        }
    return {
        "type": "resolve_observer_or_confirmation_blockers",
        "reason": "Live readback refresh ran, but observer preflight or confirmation requirements remain incomplete.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the SO-100 observer-return execution preflight chain.")
    parser.add_argument("--bridge-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", default="/dev/cu.usbmodem5AE60824791")
    parser.add_argument("--mode", choices=["replay", "live_readonly"], default="replay")
    parser.add_argument("--replay-readback", type=Path)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--observer-camera-index", type=int, default=3)
    parser.add_argument("--observer-camera-status", choices=["available", "temporarily_unavailable", "off"], default="off")
    parser.add_argument("--user-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    parser.add_argument("--max-abs-raw-delta-per-step", type=float, default=80.0)
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            run_observer_return_refresh(
                bridge_report=args.bridge_report,
                output_dir=args.output_dir,
                port=args.port,
                mode=args.mode,
                replay_readback=args.replay_readback,
                frame_index=args.frame_index,
                observer_camera_index=args.observer_camera_index,
                observer_camera_status=args.observer_camera_status,
                user_confirmed=args.user_confirmed,
                workspace_clear_confirmed=args.workspace_clear_confirmed,
                max_abs_raw_delta_per_step=args.max_abs_raw_delta_per_step,
                chunk_size=args.chunk_size,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
