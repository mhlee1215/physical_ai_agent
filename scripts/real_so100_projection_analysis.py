#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from physical_ai_agent.safety.so100_smolvla_metadata_adapter import raw_to_lerobot_so100_position


def analyze_projection_sweep(
    *,
    sweep_report: Path,
    output: Path,
    markdown: Path | None = None,
) -> dict[str, Any]:
    sweep = json.loads(sweep_report.read_text(encoding="utf-8"))
    candidates = []
    for candidate in sweep.get("candidates", []):
        execute_gate_path = Path(candidate["execute_gate_path"])
        execute_gate = json.loads(execute_gate_path.read_text(encoding="utf-8"))
        analysis = analyze_execute_gate_projection(execute_gate)
        candidates.append(
            {
                "candidate_index": candidate["candidate_index"],
                "prompt": candidate["prompt"],
                "execute_gate_path": str(execute_gate_path),
                "action_path": candidate.get("action_path"),
                "dry_status": candidate.get("dry_status"),
                "execute_status": candidate.get("execute_status"),
                "send_action_called": False,
                "policy_actions_executed": False,
                "physical_robot_motion": False,
                "projection": analysis,
            }
        )
    ranked = sorted(candidates, key=_projection_sort_key)
    best = ranked[0] if ranked else None
    report = {
        "operation": "real_so100_projection_analysis",
        "status": "passed" if candidates else "blocked",
        "source_sweep_report": str(sweep_report),
        "policy_camera_indexes": sweep.get("policy_camera_indexes"),
        "observer_camera_indexes": sweep.get("observer_camera_indexes", []),
        "observer_camera_status": sweep.get("observer_camera_status", "unknown"),
        "camera_3_status": sweep.get("camera_3_status", "unknown"),
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "projection_semantics": (
            "Analysis only: clamp each postprocessed raw target to calibrated range, convert the clamped "
            "raw target back to LeRobot SO-100 command units, and measure distortion. Do not execute "
            "projection candidates without observer camera evidence and normal execution gates."
        ),
        "candidates": candidates,
        "ranked_candidates": ranked,
        "best_candidate": best,
        "next_agentic_layer_step": _next_step(best),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path = markdown or output.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report["json_path"] = str(output)
    report["markdown_path"] = str(md_path)
    return report


def analyze_execute_gate_projection(execute_gate: dict[str, Any]) -> dict[str, Any]:
    dry_plan = execute_gate.get("dry_plan") or {}
    step_plans = dry_plan.get("step_plans") or []
    projected_steps = []
    joint_distortion: dict[str, dict[str, float | int]] = {
        joint: {
            "violation_count": 0,
            "total_raw_distortion": 0.0,
            "max_raw_distortion": 0.0,
            "total_command_distortion": 0.0,
            "max_command_distortion": 0.0,
        }
        for joint in SO100_JOINT_ORDER
    }
    total_raw_distortion = 0.0
    max_raw_distortion = 0.0
    projected_target_count = 0
    violation_count = 0
    nonfinite_count = 0

    for step in step_plans:
        projected_targets = []
        for target in step.get("joint_targets") or []:
            projection = project_joint_target(target)
            projected_targets.append(projection)
            projected_target_count += 1
            if not projection["finite"]:
                nonfinite_count += 1
                continue
            raw_distortion = float(projection["raw_distortion"])
            command_distortion = float(projection["command_distortion"])
            joint = str(projection["joint"])
            total_raw_distortion += raw_distortion
            max_raw_distortion = max(max_raw_distortion, raw_distortion)
            if projection["was_out_of_range"]:
                violation_count += 1
                joint_distortion[joint]["violation_count"] = int(joint_distortion[joint]["violation_count"]) + 1
            joint_distortion[joint]["total_raw_distortion"] = round(float(joint_distortion[joint]["total_raw_distortion"]) + raw_distortion, 4)
            joint_distortion[joint]["max_raw_distortion"] = round(max(float(joint_distortion[joint]["max_raw_distortion"]), raw_distortion), 4)
            joint_distortion[joint]["total_command_distortion"] = round(float(joint_distortion[joint]["total_command_distortion"]) + command_distortion, 4)
            joint_distortion[joint]["max_command_distortion"] = round(max(float(joint_distortion[joint]["max_command_distortion"]), command_distortion), 4)
        projected_steps.append(
            {
                "step_index": step.get("step_index"),
                "projected_targets": projected_targets,
            }
        )

    finite_count = max(0, projected_target_count - nonfinite_count)
    mean_raw_distortion = total_raw_distortion / finite_count if finite_count else 0.0
    projected_ready = bool(step_plans) and nonfinite_count == 0
    projection_penalty = total_raw_distortion + max_raw_distortion * 3.0 + violation_count * 50.0
    return {
        "projected_ready_for_execution_shape_only": projected_ready,
        "source_ready_for_execution": bool(dry_plan.get("ready_for_execution")),
        "action_chunk_steps": len(step_plans),
        "projected_target_count": projected_target_count,
        "range_violation_count": violation_count,
        "nonfinite_target_count": nonfinite_count,
        "total_raw_distortion": round(total_raw_distortion, 4),
        "max_raw_distortion": round(max_raw_distortion, 4),
        "mean_raw_distortion": round(mean_raw_distortion, 4),
        "projection_penalty_score": round(projection_penalty, 4),
        "joint_distortion": joint_distortion,
        "top_projected_targets": _top_projected_targets(projected_steps),
        "projected_steps": projected_steps,
    }


def project_joint_target(target: dict[str, Any]) -> dict[str, Any]:
    joint = str(target.get("joint"))
    raw = _float_or_nan(target.get("target_raw"))
    range_min = _float_or_nan(target.get("range_min"))
    range_max = _float_or_nan(target.get("range_max"))
    command_value = _float_or_nan(target.get("target_command_value"))
    finite = all(math.isfinite(value) for value in [raw, range_min, range_max, command_value])
    if not finite:
        projected_raw = math.nan
        projected_command = math.nan
        raw_distortion = math.nan
        command_distortion = math.nan
        was_out = False
    else:
        projected_raw = _clip(raw, range_min, range_max)
        projected_command = raw_to_lerobot_so100_position(
            joint=joint,
            raw_value=projected_raw,
            calibration={"range_min": range_min, "range_max": range_max},
        )
        raw_distortion = abs(projected_raw - raw)
        command_distortion = abs(projected_command - command_value)
        was_out = raw_distortion > 0.0
    return {
        "joint": joint,
        "target_raw": raw,
        "projected_raw": projected_raw,
        "range_min": range_min,
        "range_max": range_max,
        "target_command_value": command_value,
        "projected_command_value": projected_command,
        "raw_distortion": round(raw_distortion, 4) if math.isfinite(raw_distortion) else math.nan,
        "command_distortion": round(command_distortion, 4) if math.isfinite(command_distortion) else math.nan,
        "was_out_of_range": was_out,
        "finite": finite,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real SO-100 Projection Analysis",
        "",
        f"- Status: `{report['status']}`",
        f"- Source sweep: `{report['source_sweep_report']}`",
        f"- Observer cameras: `{report['observer_camera_indexes']}` (`{report['observer_camera_status']}`)",
        f"- Actuation enabled: `{report['actuation_enabled']}`",
        f"- Physical robot motion: `{report['physical_robot_motion']}`",
        f"- Task success claim allowed: `{report['task_success_claim_allowed']}`",
        "",
        "## Ranking",
        "",
    ]
    for candidate in report.get("ranked_candidates") or []:
        projection = candidate["projection"]
        lines.extend(
            [
                f"### Candidate {candidate['candidate_index']:02d}",
                "",
                f"- Prompt: {candidate['prompt']}",
                f"- Shape-only projected ready: `{projection['projected_ready_for_execution_shape_only']}`",
                f"- Range violations clipped: `{projection['range_violation_count']}`",
                f"- Total raw distortion: `{projection['total_raw_distortion']}`",
                f"- Max raw distortion: `{projection['max_raw_distortion']}`",
                f"- Mean raw distortion: `{projection['mean_raw_distortion']}`",
                f"- Projection penalty score: `{projection['projection_penalty_score']}`",
                f"- Execute gate: `{candidate['execute_gate_path']}`",
                "",
            ]
        )
    best = report.get("best_candidate")
    if best:
        lines.extend(
            [
                "## Selected Feedback",
                "",
                f"- Best candidate: `{best['candidate_index']:02d}`",
                f"- Best prompt: {best['prompt']}",
                f"- Next step: `{report['next_agentic_layer_step']['type']}`",
                f"- Reason: {report['next_agentic_layer_step']['reason']}",
                "",
            ]
        )
    return "\n".join(lines)


def _projection_sort_key(candidate: dict[str, Any]) -> tuple[int, float, int, float, int]:
    projection = candidate["projection"]
    ready_rank = 0 if projection["projected_ready_for_execution_shape_only"] else 1
    return (
        ready_rank,
        float(projection["projection_penalty_score"]),
        int(projection["range_violation_count"]),
        float(projection["max_raw_distortion"]),
        int(candidate["candidate_index"]),
    )


def _next_step(best: dict[str, Any] | None) -> dict[str, Any]:
    if best is None:
        return {"type": "rerun_smolvla_proposal_sweep", "reason": "No candidates were available for projection analysis."}
    projection = best["projection"]
    if projection["range_violation_count"] == 0:
        return {
            "type": "hold_for_observer_camera_3_and_human_confirmation",
            "reason": "The source candidate already satisfies calibrated ranges; physical execution still requires observer evidence and confirmation.",
            "selected_prompt": best["prompt"],
        }
    return {
        "type": "generate_low_distortion_prompt_or_projection_candidate",
        "reason": "Projection can produce in-range shape-only commands, but distortion remains nonzero; use lowest-distortion candidate as next agentic feedback.",
        "selected_prompt": best["prompt"],
    }


def _top_projected_targets(projected_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = [
        dict(target, step_index=step["step_index"])
        for step in projected_steps
        for target in step["projected_targets"]
        if target["finite"] and target["raw_distortion"] > 0
    ]
    return sorted(targets, key=lambda item: float(item["raw_distortion"]), reverse=True)[:10]


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze calibrated projection distortion for a real SO-100 SmolVLA sweep.")
    parser.add_argument("--sweep-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_projection_sweep(
                sweep_report=args.sweep_report,
                output=args.output,
                markdown=args.markdown,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
