#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from physical_ai_agent.safety.so100_smolvla_metadata_adapter import raw_to_lerobot_so100_position


def regenerate_transition_from_readback(
    *,
    bridge_report: Path,
    readback: Path,
    output: Path,
    readback_source: str = "replay",
    frame_index: int | None = None,
    max_abs_raw_delta_per_step: float = 80.0,
    chunk_size: int = 10,
    markdown: Path | None = None,
) -> dict[str, Any]:
    if max_abs_raw_delta_per_step <= 0:
        raise ValueError(f"max_abs_raw_delta_per_step must be positive, got {max_abs_raw_delta_per_step}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    bridge = json.loads(bridge_report.read_text(encoding="utf-8"))
    readback_payload = _load_readback_payload(readback, frame_index=frame_index)
    current_state = _extract_readback_state(readback_payload)
    target_by_joint = {str(target["joint"]): target for target in bridge.get("bridge_target_joints") or []}
    blockers = _input_blockers(bridge, current_state, target_by_joint)
    transition_steps: list[dict[str, Any]] = []
    delta_summary: dict[str, dict[str, float]] = {}
    transition_chunk_count = 0
    if not blockers:
        transition_chunk_count = _required_chunk_count(
            current_state=current_state,
            target_by_joint=target_by_joint,
            chunk_size=chunk_size,
            max_abs_raw_delta_per_step=max_abs_raw_delta_per_step,
        )
        transition_steps, delta_summary = _transition_steps(
            current_state=current_state,
            target_by_joint=target_by_joint,
            step_count=transition_chunk_count * chunk_size,
            chunk_size=chunk_size,
        )
        blockers.extend(_delta_blockers(delta_summary, max_abs_raw_delta_per_step))
    live_readback_regenerated = readback_source == "live"
    status = "passed" if not blockers else "blocked"
    result = {
        "operation": "real_so100_readback_transition_regenerator",
        "status": status,
        "source_bridge_report": str(bridge_report),
        "source_readback": str(readback),
        "source_frame_index": frame_index,
        "readback_source": readback_source,
        "live_readback_regenerated": live_readback_regenerated,
        "policy_camera_indexes": bridge.get("policy_camera_indexes"),
        "observer_camera_indexes": bridge.get("observer_camera_indexes", []),
        "observer_camera_status": bridge.get("observer_camera_status", "unknown"),
        "camera_3_status": bridge.get("camera_3_status", "unknown"),
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "chunk_size": chunk_size,
        "transition_chunk_count": transition_chunk_count,
        "transition_step_count": len(transition_steps),
        "max_abs_raw_delta_per_step": max_abs_raw_delta_per_step,
        "source_current_raw": {joint: current_state[joint] for joint in SO100_JOINT_ORDER if joint in current_state},
        "bridge_target_raw": {joint: float(target_by_joint[joint]["projected_raw"]) for joint in SO100_JOINT_ORDER if joint in target_by_joint},
        "delta_summary": delta_summary,
        "transition_steps": transition_steps,
        "blockers": blockers,
        "next_agentic_layer_step": _next_step(status, live_readback_regenerated),
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
        "# Real SO-100 Readback Transition Regenerator",
        "",
        f"- Status: `{report['status']}`",
        f"- Source bridge: `{report.get('source_bridge_report')}`",
        f"- Source readback: `{report.get('source_readback')}`",
        f"- Readback source: `{report.get('readback_source')}`",
        f"- Live readback regenerated: `{report.get('live_readback_regenerated', False)}`",
        f"- Transition chunks: `{report.get('transition_chunk_count')}`",
        f"- Transition steps: `{report.get('transition_step_count')}`",
        f"- Actuation enabled: `{report.get('actuation_enabled', False)}`",
        f"- Physical robot motion: `{report.get('physical_robot_motion', False)}`",
        f"- Task success claim allowed: `{report.get('task_success_claim_allowed', False)}`",
        "",
        "## Delta Summary",
        "",
    ]
    for joint, summary in report.get("delta_summary", {}).items():
        lines.append(
            f"- `{joint}`: start=`{summary['start_raw']}`, target=`{summary['target_raw']}`, "
            f"total_delta=`{summary['total_delta_raw']}`, per_step=`{summary['per_step_delta_raw']}`"
        )
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


def _extract_readback_state(payload: dict[str, Any]) -> dict[str, float]:
    for key in ["positions_raw", "readback_before_raw", "source_current_raw", "state"]:
        value = payload.get(key)
        if isinstance(value, dict):
            return {joint: float(value[joint]) for joint in SO100_JOINT_ORDER if joint in value}
    observation = payload.get("observation")
    if isinstance(observation, dict) and isinstance(observation.get("state"), dict):
        state = observation["state"]
        return {joint: float(state[joint]) for joint in SO100_JOINT_ORDER if joint in state}
    raise ValueError("Could not find readback positions in payload.")


def _load_readback_payload(path: Path, *, frame_index: int | None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        selected = None
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if frame_index is None or int(record.get("frame_index", -1)) == int(frame_index):
                selected = record
                break
        if selected is None:
            raise ValueError(f"Could not find frame_index={frame_index} in JSONL readback {path}")
        return selected


def _input_blockers(
    bridge: dict[str, Any],
    current_state: dict[str, float],
    target_by_joint: dict[str, dict[str, Any]],
) -> list[str]:
    blockers = []
    if bridge.get("status") != "passed":
        blockers.append("Source bridge report did not pass.")
    missing_state = [joint for joint in SO100_JOINT_ORDER if joint not in current_state]
    if missing_state:
        blockers.append(f"Readback is missing joints: {missing_state}.")
    missing_targets = [joint for joint in SO100_JOINT_ORDER if joint not in target_by_joint]
    if missing_targets:
        blockers.append(f"Bridge report is missing target joints: {missing_targets}.")
    for joint, target in target_by_joint.items():
        if target.get("was_out_of_range"):
            blockers.append(f"Bridge target joint {joint} was out of calibrated range.")
        if not target.get("finite", True):
            blockers.append(f"Bridge target joint {joint} is nonfinite.")
    return blockers


def _transition_steps(
    *,
    current_state: dict[str, float],
    target_by_joint: dict[str, dict[str, Any]],
    step_count: int,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    delta_summary = {}
    for joint in SO100_JOINT_ORDER:
        start = float(current_state[joint])
        target = float(target_by_joint[joint]["projected_raw"])
        per_step = (target - start) / float(step_count)
        delta_summary[joint] = {
            "start_raw": round(start, 4),
            "target_raw": round(target, 4),
            "total_delta_raw": round(target - start, 4),
            "per_step_delta_raw": round(per_step, 4),
        }
    steps = []
    for step_index in range(step_count):
        fraction = float(step_index + 1) / float(step_count)
        joint_targets = []
        for joint in SO100_JOINT_ORDER:
            target_info = target_by_joint[joint]
            start = float(current_state[joint])
            target_raw = float(target_info["projected_raw"])
            raw = start + (target_raw - start) * fraction
            command = raw_to_lerobot_so100_position(
                joint=joint,
                raw_value=raw,
                calibration={"range_min": target_info["range_min"], "range_max": target_info["range_max"]},
            )
            joint_targets.append(
                {
                    "joint": joint,
                    "target_raw": round(raw, 4),
                    "target_command_value": round(command, 6),
                    "range_min": target_info["range_min"],
                    "range_max": target_info["range_max"],
                    "raw_target_in_calibrated_range": float(target_info["range_min"]) <= raw <= float(target_info["range_max"]),
                    "write_normalize": True,
                    "command_units": "lerobot_so100_position",
                }
            )
        steps.append(
            {
                "step_index": step_index,
                "chunk_index": step_index // chunk_size,
                "step_index_in_chunk": step_index % chunk_size,
                "joint_targets": joint_targets,
            }
        )
    return steps, delta_summary


def _required_chunk_count(
    *,
    current_state: dict[str, float],
    target_by_joint: dict[str, dict[str, Any]],
    chunk_size: int,
    max_abs_raw_delta_per_step: float,
) -> int:
    max_total_delta = max(
        abs(float(target_by_joint[joint]["projected_raw"]) - float(current_state[joint]))
        for joint in SO100_JOINT_ORDER
    )
    required_steps = max(1, math.ceil(max_total_delta / max_abs_raw_delta_per_step))
    return max(1, math.ceil(required_steps / chunk_size))


def _delta_blockers(delta_summary: dict[str, dict[str, float]], max_abs_raw_delta_per_step: float) -> list[str]:
    blockers = []
    for joint, summary in delta_summary.items():
        per_step = abs(float(summary["per_step_delta_raw"]))
        if per_step > max_abs_raw_delta_per_step:
            blockers.append(
                f"Joint {joint} requires {per_step:.4f} raw ticks per step, above limit {max_abs_raw_delta_per_step:.4f}."
            )
    return blockers


def _next_step(status: str, live_readback_regenerated: bool) -> dict[str, Any]:
    if status == "passed" and live_readback_regenerated:
        return {
            "type": "run_transition_candidate_gate_on_live_readback_plan",
            "reason": "A live-readback transition was regenerated; validate chunk/range/delta structure before observer-backed execution.",
        }
    if status == "passed":
        return {
            "type": "rerun_with_live_readback_before_execution",
            "reason": "The transition was regenerated from replay/static readback only; physical execution still requires a live readback source.",
        }
    return {
        "type": "fix_readback_or_bridge_inputs",
        "reason": "Could not regenerate a bounded transition from the supplied readback and bridge target.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate an SO-100 bridge transition from readback positions.")
    parser.add_argument("--bridge-report", type=Path, required=True)
    parser.add_argument("--readback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readback-source", choices=["live", "replay", "static"], default="replay")
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--max-abs-raw-delta-per-step", type=float, default=80.0)
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            regenerate_transition_from_readback(
                bridge_report=args.bridge_report,
                readback=args.readback,
                output=args.output,
                readback_source=args.readback_source,
                frame_index=args.frame_index,
                max_abs_raw_delta_per_step=args.max_abs_raw_delta_per_step,
                chunk_size=args.chunk_size,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
