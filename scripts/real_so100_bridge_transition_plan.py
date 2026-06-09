#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER, load_episode_state
from physical_ai_agent.safety.so100_smolvla_metadata_adapter import raw_to_lerobot_so100_position


def build_bridge_transition_plan(
    *,
    bridge_report: Path,
    episode: Path,
    frame_index: int,
    output: Path,
    step_count: int = 10,
    max_abs_raw_delta_per_step: float = 80.0,
    auto_chunks: bool = False,
    chunk_size: int = 10,
    markdown: Path | None = None,
) -> dict[str, Any]:
    if step_count < 1:
        raise ValueError(f"step_count must be positive, got {step_count}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if max_abs_raw_delta_per_step <= 0:
        raise ValueError(f"max_abs_raw_delta_per_step must be positive, got {max_abs_raw_delta_per_step}")
    bridge = json.loads(bridge_report.read_text(encoding="utf-8"))
    current_state = {joint: float(value) for joint, value in load_episode_state(episode, frame_index).items()}
    target_by_joint = {str(target["joint"]): target for target in bridge.get("bridge_target_joints") or []}
    blockers = _input_blockers(bridge, target_by_joint, current_state)
    transition_steps: list[dict[str, Any]] = []
    delta_summary: dict[str, dict[str, float]] = {}
    resolved_step_count = step_count
    transition_chunk_count = 1
    if not blockers:
        if auto_chunks:
            transition_chunk_count = _required_chunk_count(
                current_state=current_state,
                target_by_joint=target_by_joint,
                chunk_size=chunk_size,
                max_abs_raw_delta_per_step=max_abs_raw_delta_per_step,
            )
            resolved_step_count = transition_chunk_count * chunk_size
        transition_steps, delta_summary = _transition_steps(
            current_state=current_state,
            target_by_joint=target_by_joint,
            step_count=resolved_step_count,
            chunk_size=chunk_size,
        )
        blockers.extend(_delta_blockers(delta_summary, max_abs_raw_delta_per_step))
    status = "passed" if not blockers else "blocked"
    result = {
        "operation": "real_so100_bridge_transition_plan",
        "status": status,
        "source_bridge_report": str(bridge_report),
        "source_episode": str(episode),
        "source_frame_index": frame_index,
        "policy_camera_indexes": bridge.get("policy_camera_indexes"),
        "observer_camera_indexes": bridge.get("observer_camera_indexes", []),
        "observer_camera_status": bridge.get("observer_camera_status", "unknown"),
        "camera_3_status": bridge.get("camera_3_status", "unknown"),
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "requested_step_count": step_count,
        "auto_chunks": auto_chunks,
        "chunk_size": chunk_size,
        "transition_chunk_count": transition_chunk_count,
        "transition_step_count": resolved_step_count,
        "max_abs_raw_delta_per_step": max_abs_raw_delta_per_step,
        "source_current_raw": {joint: current_state[joint] for joint in SO100_JOINT_ORDER},
        "bridge_target_raw": {joint: float(target_by_joint[joint]["projected_raw"]) for joint in SO100_JOINT_ORDER if joint in target_by_joint},
        "delta_summary": delta_summary,
        "transition_steps": transition_steps,
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
        "# Real SO-100 Bridge Transition Plan",
        "",
        f"- Status: `{report['status']}`",
        f"- Source bridge: `{report.get('source_bridge_report')}`",
        f"- Source episode: `{report.get('source_episode')}` frame `{report.get('source_frame_index')}`",
        f"- Observer cameras: `{report.get('observer_camera_indexes', [])}` (`{report.get('observer_camera_status', 'unknown')}`)",
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


def _input_blockers(
    bridge: dict[str, Any],
    target_by_joint: dict[str, dict[str, Any]],
    current_state: dict[str, float],
) -> list[str]:
    blockers: list[str] = []
    if bridge.get("status") != "passed":
        blockers.append("Source bridge report did not pass.")
    missing_targets = [joint for joint in SO100_JOINT_ORDER if joint not in target_by_joint]
    if missing_targets:
        blockers.append(f"Bridge report is missing target joints: {missing_targets}.")
    missing_state = [joint for joint in SO100_JOINT_ORDER if joint not in current_state]
    if missing_state:
        blockers.append(f"Episode state is missing joints: {missing_state}.")
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


def _next_step(status: str) -> dict[str, Any]:
    if status == "passed":
        return {
            "type": "run_projection_and_trajectory_diagnostics_on_transition_candidate",
            "reason": (
                "A bounded transition-to-bridge candidate was generated. Keep it analysis-only, "
                "validate the transition with projection/trajectory diagnostics, and do not execute while camera 3 is off."
            ),
        }
    return {
        "type": "increase_transition_steps_or_regenerate_bridge_target",
        "reason": "The transition candidate is missing required inputs or exceeds the allowed per-step raw delta.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a no-actuation transition plan from current SO-100 state to a bridge pose.")
    parser.add_argument("--bridge-report", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-count", type=int, default=10)
    parser.add_argument("--max-abs-raw-delta-per-step", type=float, default=80.0)
    parser.add_argument("--auto-chunks", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            build_bridge_transition_plan(
                bridge_report=args.bridge_report,
                episode=args.episode,
                frame_index=args.frame_index,
                output=args.output,
                step_count=args.step_count,
                max_abs_raw_delta_per_step=args.max_abs_raw_delta_per_step,
                auto_chunks=args.auto_chunks,
                chunk_size=args.chunk_size,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
