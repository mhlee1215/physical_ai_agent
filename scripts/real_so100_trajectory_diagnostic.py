#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def analyze_projection_trajectory(
    *,
    projection_report: Path,
    output: Path,
    candidate_index: int | None = None,
    markdown: Path | None = None,
) -> dict[str, Any]:
    report = json.loads(projection_report.read_text(encoding="utf-8"))
    candidate = _select_candidate(report, candidate_index)
    if candidate is None:
        result = {
            "operation": "real_so100_trajectory_diagnostic",
            "status": "blocked",
            "source_projection_report": str(projection_report),
            "blockers": ["No candidate found in projection report."],
            "actuation_enabled": False,
            "send_action_called": False,
            "policy_actions_executed": False,
            "physical_robot_motion": False,
            "task_success_claim_allowed": False,
        }
    else:
        trajectory = analyze_candidate_trajectory(candidate)
        result = {
            "operation": "real_so100_trajectory_diagnostic",
            "status": "passed",
            "source_projection_report": str(projection_report),
            "source_candidate_index": candidate.get("candidate_index"),
            "source_prompt": candidate.get("prompt"),
            "policy_camera_indexes": report.get("policy_camera_indexes"),
            "observer_camera_indexes": report.get("observer_camera_indexes", []),
            "observer_camera_status": report.get("observer_camera_status", "unknown"),
            "camera_3_status": report.get("camera_3_status", "unknown"),
            "actuation_enabled": False,
            "send_action_called": False,
            "policy_actions_executed": False,
            "physical_robot_motion": False,
            "task_success_claim_allowed": False,
            "trajectory": trajectory,
            "next_agentic_layer_step": _next_step(trajectory),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path = markdown or output.with_suffix(".md")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    result["json_path"] = str(output)
    result["markdown_path"] = str(md_path)
    return result


def analyze_candidate_trajectory(candidate: dict[str, Any]) -> dict[str, Any]:
    projection = candidate.get("projection") or {}
    step_summaries = [_summarize_step(step) for step in projection.get("projected_steps") or []]
    safe_flags = [step["safe_without_projection"] for step in step_summaries]
    safe_prefix_length = _safe_prefix_length(safe_flags)
    longest_safe_run = _longest_true_run(safe_flags)
    first_violation = next((step for step in step_summaries if not step["safe_without_projection"]), None)
    violation_steps = [step for step in step_summaries if not step["safe_without_projection"]]
    safe_steps = [step for step in step_summaries if step["safe_without_projection"]]
    return {
        "action_chunk_steps": len(step_summaries),
        "safe_prefix_length": safe_prefix_length,
        "safe_suffix_after_first_violations": longest_safe_run,
        "all_steps_safe_without_projection": bool(step_summaries) and safe_prefix_length == len(step_summaries),
        "first_violation_step": first_violation,
        "violation_step_count": len(violation_steps),
        "safe_step_count": len(safe_steps),
        "step_summaries": step_summaries,
        "dominant_violation_joints": _dominant_violation_joints(step_summaries),
        "source_projection_penalty_score": projection.get("projection_penalty_score"),
        "source_total_raw_distortion": projection.get("total_raw_distortion"),
        "source_max_raw_distortion": projection.get("max_raw_distortion"),
        "source_range_violation_count": projection.get("range_violation_count"),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real SO-100 Trajectory Diagnostic",
        "",
        f"- Status: `{report['status']}`",
        f"- Source projection: `{report.get('source_projection_report')}`",
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
    trajectory = report["trajectory"]
    lines.extend(
        [
            "## Summary",
            "",
            f"- Action chunk steps: `{trajectory['action_chunk_steps']}`",
            f"- Safe prefix length: `{trajectory['safe_prefix_length']}`",
            f"- Safe step count: `{trajectory['safe_step_count']}`",
            f"- Violation step count: `{trajectory['violation_step_count']}`",
            f"- Longest safe run: `{trajectory['safe_suffix_after_first_violations']}`",
            f"- Dominant violation joints: `{trajectory['dominant_violation_joints']}`",
            "",
            "## Step Summary",
            "",
        ]
    )
    for step in trajectory["step_summaries"]:
        lines.append(
            f"- Step `{step['step_index']}`: safe=`{step['safe_without_projection']}`, "
            f"violations=`{step['range_violation_count']}`, total_distortion=`{step['total_raw_distortion']}`, "
            f"joints=`{step['violation_joints']}`"
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


def _select_candidate(report: dict[str, Any], candidate_index: int | None) -> dict[str, Any] | None:
    if candidate_index is None:
        return report.get("best_candidate")
    for candidate in report.get("candidates") or report.get("ranked_candidates") or []:
        if int(candidate.get("candidate_index", -1)) == candidate_index:
            return candidate
    return None


def _summarize_step(step: dict[str, Any]) -> dict[str, Any]:
    violations = []
    total = 0.0
    max_distortion = 0.0
    for target in step.get("projected_targets") or []:
        if not target.get("finite", True):
            violations.append({"joint": target.get("joint"), "raw_distortion": None, "reason": "nonfinite"})
            continue
        raw_distortion = float(target.get("raw_distortion", 0.0) or 0.0)
        if target.get("was_out_of_range"):
            violations.append({"joint": target.get("joint"), "raw_distortion": round(raw_distortion, 4)})
            total += raw_distortion
            max_distortion = max(max_distortion, raw_distortion)
    return {
        "step_index": int(step.get("step_index", 0)),
        "safe_without_projection": not violations,
        "range_violation_count": len(violations),
        "total_raw_distortion": round(total, 4),
        "max_raw_distortion": round(max_distortion, 4),
        "violation_joints": [str(item["joint"]) for item in violations],
        "violations": violations,
    }


def _safe_prefix_length(flags: list[bool]) -> int:
    count = 0
    for flag in flags:
        if not flag:
            break
        count += 1
    return count


def _longest_true_run(flags: list[bool]) -> dict[str, int]:
    best_start = -1
    best_length = 0
    current_start = -1
    current_length = 0
    for index, flag in enumerate(flags):
        if flag:
            if current_length == 0:
                current_start = index
            current_length += 1
            if current_length > best_length:
                best_start = current_start
                best_length = current_length
        else:
            current_start = -1
            current_length = 0
    return {"start_step": best_start, "length": best_length}


def _dominant_violation_joints(step_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float | int]] = {}
    for step in step_summaries:
        for violation in step["violations"]:
            joint = str(violation["joint"])
            raw = float(violation.get("raw_distortion") or 0.0)
            entry = totals.setdefault(joint, {"count": 0, "total_raw_distortion": 0.0})
            entry["count"] = int(entry["count"]) + 1
            entry["total_raw_distortion"] = round(float(entry["total_raw_distortion"]) + raw, 4)
    return [
        {"joint": joint, "count": int(value["count"]), "total_raw_distortion": value["total_raw_distortion"]}
        for joint, value in sorted(totals.items(), key=lambda item: (int(item[1]["count"]), float(item[1]["total_raw_distortion"])), reverse=True)
    ]


def _next_step(trajectory: dict[str, Any]) -> dict[str, Any]:
    if trajectory["all_steps_safe_without_projection"]:
        return {
            "type": "hold_for_observer_camera_3_and_human_confirmation",
            "reason": "The full chunk is within calibrated range, but real execution still requires observer camera 3 and confirmation.",
        }
    if trajectory["safe_prefix_length"] > 0:
        return {
            "type": "consider_shorter_prefix_chunk_after_observer_returns",
            "reason": "The first part of the chunk is range-safe, but later steps violate calibration.",
            "safe_prefix_length": trajectory["safe_prefix_length"],
        }
    longest = trajectory["safe_suffix_after_first_violations"]
    if longest["length"] > 0:
        return {
            "type": "do_not_execute_prefix_replan_to_safe_late_pose",
            "reason": "The chunk starts with calibrated range violations, even though a later contiguous run is range-safe; the agentic layer should replan the transition before any hardware execution.",
            "safe_run_start_step": longest["start_step"],
            "safe_run_length": longest["length"],
        }
    return {
        "type": "rerun_smolvla_or_replan_from_observation",
        "reason": "No range-safe prefix or suffix exists in this chunk.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SO-100 SmolVLA projection trajectory prefix/suffix safety.")
    parser.add_argument("--projection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_projection_trajectory(
                projection_report=args.projection_report,
                output=args.output,
                candidate_index=args.candidate_index,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
