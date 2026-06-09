#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER


def build_transition_execution_packet(
    *,
    refresh_report: Path,
    output: Path,
    observer_camera_index: int = 3,
    markdown: Path | None = None,
) -> dict[str, Any]:
    refresh = json.loads(refresh_report.read_text(encoding="utf-8"))
    regen_path = _required_path(refresh, "regenerated_transition_path")
    preflight_path = _required_path(refresh, "observer_preflight_path")
    gate_path = _required_path(refresh, "transition_gate_path")
    regen = json.loads(regen_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    chunks = _execution_chunks(regen.get("transition_steps") or [])
    blockers = _blockers(refresh=refresh, regen=regen, gate=gate, preflight=preflight, chunks=chunks)
    status = "ready_for_observer_backed_execution" if not blockers else "blocked"
    result = {
        "operation": "real_so100_transition_execution_packet",
        "status": status,
        "source_refresh_report": str(refresh_report),
        "source_regenerated_transition": str(regen_path),
        "source_transition_gate": str(gate_path),
        "source_observer_preflight": str(preflight_path),
        "policy_camera_indexes": refresh.get("policy_camera_indexes") or regen.get("policy_camera_indexes"),
        "observer_camera_indexes": [observer_camera_index] if status == "ready_for_observer_backed_execution" else preflight.get("observer_camera_indexes", []),
        "observer_camera_status": preflight.get("observer_camera_status", "unknown"),
        "camera_3_status": preflight.get("camera_3_status", "unknown"),
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "execution_ready": status == "ready_for_observer_backed_execution",
        "live_readback_regenerated": bool(regen.get("live_readback_regenerated")),
        "transition_chunk_count": len(chunks),
        "transition_step_count": sum(len(chunk["steps"]) for chunk in chunks),
        "chunks": chunks,
        "required_observer_artifacts": [
            f"camera_{observer_camera_index}_before_frame",
            f"camera_{observer_camera_index}_motion_video",
            f"camera_{observer_camera_index}_after_frame",
        ],
        "required_execution_artifacts": preflight.get("required_execution_artifacts", []),
        "readback_before_raw": None,
        "readback_after_raw": None,
        "observed_delta_raw": None,
        "blockers": blockers,
        "next_agentic_layer_step": _next_step(status),
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
        "# Real SO-100 Transition Execution Packet",
        "",
        f"- Status: `{report['status']}`",
        f"- Source refresh: `{report.get('source_refresh_report')}`",
        f"- Execution ready: `{report.get('execution_ready', False)}`",
        f"- Live readback regenerated: `{report.get('live_readback_regenerated', False)}`",
        f"- Observer cameras: `{report.get('observer_camera_indexes', [])}` (`{report.get('observer_camera_status', 'unknown')}`)",
        f"- Actuation enabled: `{report.get('actuation_enabled', False)}`",
        f"- Physical robot motion: `{report.get('physical_robot_motion', False)}`",
        "",
        "## Chunks",
        "",
    ]
    for chunk in report.get("chunks", []):
        lines.append(f"- Chunk `{chunk['chunk_index']}`: steps=`{len(chunk['steps'])}`")
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


def _required_path(payload: dict[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not value:
        raise ValueError(f"Refresh report is missing {key}")
    return Path(value)


def _execution_chunks(transition_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for step in transition_steps:
        grouped.setdefault(int(step.get("chunk_index", 0)), []).append(step)
    chunks = []
    for chunk_index in sorted(grouped):
        steps = []
        for step in sorted(grouped[chunk_index], key=lambda item: int(item.get("step_index_in_chunk", item.get("step_index", 0)))):
            targets = {str(target["joint"]): target for target in step.get("joint_targets") or []}
            target_command = {
                joint: float(targets[joint]["target_command_value"])
                for joint in SO100_JOINT_ORDER
                if joint in targets
            }
            target_raw_estimate = {
                joint: float(targets[joint]["target_raw"])
                for joint in SO100_JOINT_ORDER
                if joint in targets
            }
            steps.append(
                {
                    "step_index": int(step.get("step_index", 0)),
                    "step_index_in_chunk": int(step.get("step_index_in_chunk", 0)),
                    "target_command": target_command,
                    "write_normalize": True,
                    "target_raw_estimate": target_raw_estimate,
                }
            )
        chunks.append({"chunk_index": chunk_index, "steps": steps})
    return chunks


def _blockers(
    *,
    refresh: dict[str, Any],
    regen: dict[str, Any],
    gate: dict[str, Any],
    preflight: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> list[str]:
    blockers = []
    if refresh.get("status") != "ready_for_execution_gate":
        blockers.append("Observer return refresh is not ready for execution gate.")
    if regen.get("status") != "passed":
        blockers.append("Regenerated transition did not pass.")
    if not regen.get("live_readback_regenerated"):
        blockers.append("Transition was not regenerated from live readback.")
    if gate.get("status") != "passed":
        blockers.append("Transition candidate gate did not pass.")
    if preflight.get("status") != "ready_for_observer_backed_execution_gate":
        blockers.append("Observer preflight did not pass.")
    if not chunks:
        blockers.append("No execution chunks were available.")
    for chunk in chunks:
        if len(chunk["steps"]) != 10:
            blockers.append(f"Execution chunk {chunk['chunk_index']} has {len(chunk['steps'])} steps, expected 10.")
        for step in chunk["steps"]:
            missing = [joint for joint in SO100_JOINT_ORDER if joint not in step["target_command"]]
            if missing:
                blockers.append(f"Execution chunk {chunk['chunk_index']} step {step['step_index']} missing target joints {missing}.")
    return blockers


def _next_step(status: str) -> dict[str, Any]:
    if status == "ready_for_observer_backed_execution":
        return {
            "type": "execute_packet_with_camera_3_recording",
            "reason": "Execution packet is ready; the next stage may record observer evidence and execute chunks.",
        }
    return {
        "type": "resolve_refresh_preflight_before_execution_packet",
        "reason": "Execution packet is blocked until refresh, live readback, transition gate, and observer preflight are all ready.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a gated SO-100 transition execution packet from observer-return refresh artifacts.")
    parser.add_argument("--refresh-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-camera-index", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            build_transition_execution_packet(
                refresh_report=args.refresh_report,
                output=args.output,
                observer_camera_index=args.observer_camera_index,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
