#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER


def gate_transition_candidate(
    *,
    transition_plan: Path,
    output: Path,
    expected_chunk_size: int = 10,
    max_abs_raw_delta_per_step: float | None = None,
    markdown: Path | None = None,
) -> dict[str, Any]:
    if expected_chunk_size < 1:
        raise ValueError(f"expected_chunk_size must be positive, got {expected_chunk_size}")
    plan = json.loads(transition_plan.read_text(encoding="utf-8"))
    limit = (
        float(max_abs_raw_delta_per_step)
        if max_abs_raw_delta_per_step is not None
        else float(plan.get("max_abs_raw_delta_per_step", 80.0))
    )
    blockers = _input_blockers(plan)
    chunks = _chunk_summaries(plan.get("transition_steps") or [], expected_chunk_size, limit)
    blockers.extend(_chunk_blockers(chunks, expected_chunk_size))
    status = "passed" if not blockers else "blocked"
    result = {
        "operation": "real_so100_transition_candidate_gate",
        "status": status,
        "source_transition_plan": str(transition_plan),
        "source_transition_status": plan.get("status"),
        "policy_camera_indexes": plan.get("policy_camera_indexes"),
        "observer_camera_indexes": plan.get("observer_camera_indexes", []),
        "observer_camera_status": plan.get("observer_camera_status", "unknown"),
        "camera_3_status": plan.get("camera_3_status", "unknown"),
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "execution_ready_with_observer": False,
        "execution_blocker": "camera 3 observer evidence is temporarily unavailable; this gate is analysis-only.",
        "expected_chunk_size": expected_chunk_size,
        "max_abs_raw_delta_per_step": limit,
        "transition_chunk_count": len(chunks),
        "transition_step_count": sum(chunk["step_count"] for chunk in chunks),
        "chunks": chunks,
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
        "# Real SO-100 Transition Candidate Gate",
        "",
        f"- Status: `{report['status']}`",
        f"- Source transition: `{report.get('source_transition_plan')}`",
        f"- Observer cameras: `{report.get('observer_camera_indexes', [])}` (`{report.get('observer_camera_status', 'unknown')}`)",
        f"- Execution ready with observer: `{report.get('execution_ready_with_observer', False)}`",
        f"- Actuation enabled: `{report.get('actuation_enabled', False)}`",
        f"- Physical robot motion: `{report.get('physical_robot_motion', False)}`",
        f"- Task success claim allowed: `{report.get('task_success_claim_allowed', False)}`",
        "",
        "## Chunks",
        "",
    ]
    for chunk in report.get("chunks", []):
        lines.append(
            f"- Chunk `{chunk['chunk_index']}`: steps=`{chunk['step_count']}`, "
            f"all_targets_in_range=`{chunk['all_targets_in_range']}`, "
            f"max_abs_raw_delta=`{chunk['max_abs_raw_delta']}`, blockers=`{len(chunk['blockers'])}`"
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


def _input_blockers(plan: dict[str, Any]) -> list[str]:
    blockers = []
    if plan.get("status") != "passed":
        blockers.append("Source transition plan did not pass.")
    if plan.get("send_action_called"):
        blockers.append("Source transition plan unexpectedly called send_action.")
    if plan.get("physical_robot_motion"):
        blockers.append("Source transition plan unexpectedly records physical robot motion.")
    if plan.get("task_success_claim_allowed"):
        blockers.append("Source transition plan unexpectedly allows task success claims.")
    if plan.get("observer_camera_indexes") not in ([], None):
        blockers.append("Observer cameras must be empty while camera 3 is off.")
    if plan.get("observer_camera_status") != "temporarily_unavailable":
        blockers.append("Observer camera status must be temporarily_unavailable in observer-off mode.")
    if not plan.get("transition_steps"):
        blockers.append("Transition plan has no steps.")
    return blockers


def _chunk_summaries(
    steps: list[dict[str, Any]],
    expected_chunk_size: int,
    max_abs_raw_delta_per_step: float,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for step in steps:
        grouped.setdefault(int(step.get("chunk_index", 0)), []).append(step)
    chunks = []
    previous_targets: dict[str, float] | None = None
    for chunk_index in sorted(grouped):
        chunk_steps = sorted(grouped[chunk_index], key=lambda step: int(step.get("step_index_in_chunk", step.get("step_index", 0))))
        chunk_blockers = []
        max_delta = 0.0
        range_violations = []
        for expected_index, step in enumerate(chunk_steps):
            if int(step.get("step_index_in_chunk", expected_index)) != expected_index:
                chunk_blockers.append(f"Chunk {chunk_index} has non-contiguous step_index_in_chunk at local step {expected_index}.")
            targets = _targets_by_joint(step)
            missing = [joint for joint in SO100_JOINT_ORDER if joint not in targets]
            if missing:
                chunk_blockers.append(f"Chunk {chunk_index} step {step.get('step_index')} is missing joints {missing}.")
                continue
            for joint, target in targets.items():
                if not target.get("raw_target_in_calibrated_range", False):
                    range_violations.append({"step_index": step.get("step_index"), "joint": joint})
                if previous_targets is not None and joint in previous_targets:
                    delta = abs(float(target["target_raw"]) - previous_targets[joint])
                    max_delta = max(max_delta, delta)
                    if delta > max_abs_raw_delta_per_step:
                        chunk_blockers.append(
                            f"Chunk {chunk_index} step {step.get('step_index')} joint {joint} delta {delta:.4f} exceeds {max_abs_raw_delta_per_step:.4f}."
                        )
            previous_targets = {joint: float(targets[joint]["target_raw"]) for joint in SO100_JOINT_ORDER if joint in targets}
        chunks.append(
            {
                "chunk_index": chunk_index,
                "step_count": len(chunk_steps),
                "expected_chunk_size": expected_chunk_size,
                "all_targets_in_range": not range_violations,
                "range_violations": range_violations,
                "max_abs_raw_delta": round(max_delta, 4),
                "blockers": chunk_blockers,
            }
        )
    return chunks


def _chunk_blockers(chunks: list[dict[str, Any]], expected_chunk_size: int) -> list[str]:
    blockers = []
    if not chunks:
        return ["No chunks were found in transition candidate."]
    expected_indexes = list(range(len(chunks)))
    actual_indexes = [int(chunk["chunk_index"]) for chunk in chunks]
    if actual_indexes != expected_indexes:
        blockers.append(f"Chunk indexes must be contiguous from 0, got {actual_indexes}.")
    for chunk in chunks:
        if int(chunk["step_count"]) != expected_chunk_size:
            blockers.append(
                f"Chunk {chunk['chunk_index']} has {chunk['step_count']} steps, expected {expected_chunk_size}."
            )
        if not chunk["all_targets_in_range"]:
            blockers.append(f"Chunk {chunk['chunk_index']} has calibrated range violations.")
        blockers.extend(chunk["blockers"])
    return blockers


def _targets_by_joint(step: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(target.get("joint")): target for target in step.get("joint_targets") or []}


def _next_step(status: str) -> dict[str, Any]:
    if status == "passed":
        return {
            "type": "wait_for_observer_camera_3_before_physical_execution_gate",
            "reason": (
                "The transition candidate is internally valid as two bounded 10-step chunks, but real execution "
                "still requires camera 3 observer evidence, live readback regeneration, and user confirmation."
            ),
        }
    return {
        "type": "regenerate_transition_candidate",
        "reason": "The transition candidate failed chunk, range, delta, or observer-off contract checks.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate an SO-100 transition candidate before any observer-backed execution.")
    parser.add_argument("--transition-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-chunk-size", type=int, default=10)
    parser.add_argument("--max-abs-raw-delta-per-step", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            gate_transition_candidate(
                transition_plan=args.transition_plan,
                output=args.output,
                expected_chunk_size=args.expected_chunk_size,
                max_abs_raw_delta_per_step=args.max_abs_raw_delta_per_step,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
