#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_late_safe_pose_bridge(
    *,
    projection_report: Path,
    trajectory_report: Path,
    output: Path,
    candidate_index: int | None = None,
    markdown: Path | None = None,
) -> dict[str, Any]:
    projection = json.loads(projection_report.read_text(encoding="utf-8"))
    trajectory = json.loads(trajectory_report.read_text(encoding="utf-8"))
    candidate = _select_candidate(projection, candidate_index or trajectory.get("source_candidate_index"))
    if candidate is None:
        result = _blocked(
            projection_report=projection_report,
            trajectory_report=trajectory_report,
            blockers=["No matching candidate found in projection report."],
        )
    else:
        target_step = _target_step_index(trajectory)
        projected_step = _select_projected_step(candidate, target_step)
        if target_step is None:
            result = _blocked(
                projection_report=projection_report,
                trajectory_report=trajectory_report,
                blockers=["Trajectory report does not contain a range-safe late run."],
            )
        elif projected_step is None:
            result = _blocked(
                projection_report=projection_report,
                trajectory_report=trajectory_report,
                blockers=[f"Projection report does not contain safe-run start step {target_step}."],
            )
        else:
            bridge_targets = [_bridge_target(target) for target in projected_step.get("projected_targets") or []]
            all_in_range = bool(bridge_targets) and all(
                target["finite"] and not target["was_out_of_range"] for target in bridge_targets
            )
            safe_run = trajectory["trajectory"]["safe_suffix_after_first_violations"]
            result = {
                "operation": "real_so100_late_safe_pose_bridge",
                "status": "passed" if all_in_range else "blocked",
                "source_projection_report": str(projection_report),
                "source_trajectory_report": str(trajectory_report),
                "source_candidate_index": candidate.get("candidate_index"),
                "source_prompt": candidate.get("prompt"),
                "policy_camera_indexes": projection.get("policy_camera_indexes"),
                "observer_camera_indexes": projection.get("observer_camera_indexes", []),
                "observer_camera_status": projection.get("observer_camera_status", "unknown"),
                "camera_3_status": projection.get("camera_3_status", "unknown"),
                "actuation_enabled": False,
                "send_action_called": False,
                "policy_actions_executed": False,
                "physical_robot_motion": False,
                "task_success_claim_allowed": False,
                "safe_run_start_step": safe_run.get("start_step"),
                "safe_run_length": safe_run.get("length"),
                "bridge_target_step_index": target_step,
                "bridge_target_joints": bridge_targets,
                "all_bridge_targets_in_range": all_in_range,
                "next_agentic_layer_step": _next_step(all_in_range, target_step),
            }
            if not all_in_range:
                result["blockers"] = ["Selected bridge target is not fully inside calibrated ranges."]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path = markdown or output.with_suffix(".md")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    result["json_path"] = str(output)
    result["markdown_path"] = str(md_path)
    return result


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real SO-100 Late Safe Pose Bridge",
        "",
        f"- Status: `{report['status']}`",
        f"- Source projection: `{report.get('source_projection_report')}`",
        f"- Source trajectory: `{report.get('source_trajectory_report')}`",
        f"- Candidate: `{report.get('source_candidate_index')}`",
        f"- Observer cameras: `{report.get('observer_camera_indexes', [])}` (`{report.get('observer_camera_status', 'unknown')}`)",
        f"- Actuation enabled: `{report.get('actuation_enabled', False)}`",
        f"- Physical robot motion: `{report.get('physical_robot_motion', False)}`",
        f"- Task success claim allowed: `{report.get('task_success_claim_allowed', False)}`",
        "",
    ]
    if report["status"] != "passed":
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report.get("blockers", []))
        return "\n".join(lines)
    lines.extend(
        [
            "## Bridge Target",
            "",
            f"- Safe run start step: `{report['safe_run_start_step']}`",
            f"- Safe run length: `{report['safe_run_length']}`",
            f"- Bridge target step: `{report['bridge_target_step_index']}`",
            f"- All bridge targets in range: `{report['all_bridge_targets_in_range']}`",
            "",
            "## Joint Targets",
            "",
        ]
    )
    for target in report.get("bridge_target_joints", []):
        lines.append(
            f"- `{target['joint']}`: projected_raw=`{target['projected_raw']}`, "
            f"projected_command_value=`{target['projected_command_value']}`, "
            f"range=`[{target['range_min']}, {target['range_max']}]`"
        )
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


def _blocked(*, projection_report: Path, trajectory_report: Path, blockers: list[str]) -> dict[str, Any]:
    return {
        "operation": "real_so100_late_safe_pose_bridge",
        "status": "blocked",
        "source_projection_report": str(projection_report),
        "source_trajectory_report": str(trajectory_report),
        "blockers": blockers,
        "observer_camera_indexes": [],
        "observer_camera_status": "temporarily_unavailable",
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
    }


def _select_candidate(report: dict[str, Any], candidate_index: int | None) -> dict[str, Any] | None:
    if candidate_index is None:
        return report.get("best_candidate")
    for candidate in report.get("candidates") or report.get("ranked_candidates") or []:
        if int(candidate.get("candidate_index", -1)) == int(candidate_index):
            return candidate
    best = report.get("best_candidate")
    if best and int(best.get("candidate_index", -1)) == int(candidate_index):
        return best
    return None


def _target_step_index(trajectory_report: dict[str, Any]) -> int | None:
    next_step = trajectory_report.get("next_agentic_layer_step") or {}
    if "safe_run_start_step" in next_step:
        return int(next_step["safe_run_start_step"])
    trajectory = trajectory_report.get("trajectory") or {}
    safe_run = trajectory.get("safe_suffix_after_first_violations") or {}
    start = safe_run.get("start_step")
    length = int(safe_run.get("length", 0) or 0)
    if start is None or int(start) < 0 or length <= 0:
        return None
    return int(start)


def _select_projected_step(candidate: dict[str, Any], step_index: int | None) -> dict[str, Any] | None:
    if step_index is None:
        return None
    for step in (candidate.get("projection") or {}).get("projected_steps") or []:
        if int(step.get("step_index", -1)) == int(step_index):
            return step
    return None


def _bridge_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "joint": target.get("joint"),
        "finite": bool(target.get("finite", True)),
        "target_raw": target.get("target_raw"),
        "projected_raw": target.get("projected_raw"),
        "range_min": target.get("range_min"),
        "range_max": target.get("range_max"),
        "target_command_value": target.get("target_command_value"),
        "projected_command_value": target.get("projected_command_value"),
        "raw_distortion": target.get("raw_distortion"),
        "command_distortion": target.get("command_distortion"),
        "was_out_of_range": bool(target.get("was_out_of_range", False)),
    }


def _next_step(all_in_range: bool, step_index: int) -> dict[str, Any]:
    if all_in_range:
        return {
            "type": "generate_transition_to_late_safe_pose_without_executing",
            "reason": (
                f"The source chunk has an unsafe prefix, but step {step_index} is a calibrated in-range "
                "late safe pose. Generate a new transition plan to this pose, rerun the dry gate, and keep "
                "physical execution disabled until observer camera 3 returns."
            ),
            "bridge_target_step_index": step_index,
        }
    return {
        "type": "rerun_projection_or_trajectory_diagnostic",
        "reason": "The selected late pose is not safe enough to become a bridge target.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a late range-safe SO-100 pose from a SmolVLA projection trajectory.")
    parser.add_argument("--projection-report", type=Path, required=True)
    parser.add_argument("--trajectory-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            build_late_safe_pose_bridge(
                projection_report=args.projection_report,
                trajectory_report=args.trajectory_report,
                output=args.output,
                candidate_index=args.candidate_index,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
