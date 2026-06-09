#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_transition_execution_feedback(
    *,
    execution_report: Path,
    output: Path,
    packet: Path | None = None,
    grasp_outcome: Path | None = None,
    relocation_outcome: Path | None = None,
    markdown: Path | None = None,
) -> dict[str, Any]:
    execution = _load_json(execution_report)
    packet_path = packet or _optional_path(execution.get("packet"))
    packet_payload = _load_json(packet_path) if packet_path is not None and packet_path.exists() else None
    grasp = _load_json(grasp_outcome) if grasp_outcome is not None else None
    relocation = _load_json(relocation_outcome) if relocation_outcome is not None else None

    failure_modes = _failure_modes(
        execution=execution,
        packet=packet_payload,
        grasp=grasp,
        relocation=relocation,
    )
    task_success_candidate = _task_success_candidate(grasp=grasp, relocation=relocation)
    prompt_mutation_allowed = bool(execution.get("policy_actions_executed")) and not _preflight_blocked(
        execution=execution,
        packet=packet_payload,
    )
    result: dict[str, Any] = {
        "operation": "real_so100_transition_execution_feedback",
        "status": "passed",
        "purpose": "normalize transition execution outcome into feedback for the SmolVLA agentic layer",
        "source_execution_report": str(execution_report),
        "source_packet": str(packet_path) if packet_path is not None else None,
        "source_grasp_outcome": str(grasp_outcome) if grasp_outcome is not None else None,
        "source_relocation_outcome": str(relocation_outcome) if relocation_outcome is not None else None,
        "camera_contract": _camera_contract(execution=execution, packet=packet_payload),
        "execution_outcome": _execution_outcome(execution),
        "packet_outcome": _packet_outcome(packet_payload),
        "verifier_outcome": _verifier_outcome(grasp=grasp, relocation=relocation),
        "failure_modes": failure_modes,
        "task_success_candidate": task_success_candidate,
        "task_success_claim_allowed": bool(task_success_candidate),
        "prompt_mutation_allowed": prompt_mutation_allowed,
        "in_loop_prompt_target": "SmolVLA",
        "does_not_prompt_operator": True,
        "next_agentic_layer_step": _next_step(
            execution=execution,
            packet=packet_payload,
            failure_modes=failure_modes,
            task_success_candidate=task_success_candidate,
            prompt_mutation_allowed=prompt_mutation_allowed,
        ),
        "notes": _notes(execution=execution, task_success_candidate=task_success_candidate),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path = markdown or output.with_suffix(".md")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    result["json_path"] = str(output)
    result["markdown_path"] = str(md_path)
    return result


def render_markdown(report: dict[str, Any]) -> str:
    execution = report["execution_outcome"]
    step = report["next_agentic_layer_step"]
    lines = [
        "# Real SO-100 Transition Execution Feedback",
        "",
        f"- Status: `{report['status']}`",
        f"- Execution status: `{execution.get('status')}`",
        f"- Send action called: `{execution.get('send_action_called', False)}`",
        f"- Physical robot motion: `{execution.get('physical_robot_motion', False)}`",
        f"- Task success candidate: `{report.get('task_success_candidate', False)}`",
        f"- Prompt mutation allowed: `{report.get('prompt_mutation_allowed', False)}`",
        f"- Prompt target: `{report.get('in_loop_prompt_target')}`",
        "",
        "## Failure Modes",
        "",
    ]
    if report.get("failure_modes"):
        lines.extend(f"- `{mode}`" for mode in report["failure_modes"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            f"- Type: `{step['type']}`",
            f"- Reason: {step['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def _execution_outcome(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": execution.get("status"),
        "execute_requested": bool(execution.get("execute_requested")),
        "packet_status": execution.get("packet_status"),
        "packet_execution_ready": bool(execution.get("packet_execution_ready")),
        "send_action_called": bool(execution.get("send_action_called")),
        "policy_actions_executed": bool(execution.get("policy_actions_executed")),
        "physical_robot_motion": bool(execution.get("physical_robot_motion")),
        "transition_chunk_count": execution.get("transition_chunk_count"),
        "transition_step_count": execution.get("transition_step_count"),
        "executed_action_steps": execution.get("executed_action_steps", 0),
        "blockers": execution.get("blockers") or [],
        "motion_video": execution.get("motion_video"),
        "visual_check": execution.get("visual_check"),
        "readback_before_raw_present": execution.get("readback_before_raw") is not None,
        "readback_after_raw_present": execution.get("readback_after_raw") is not None,
        "observed_delta_raw_present": execution.get("observed_delta_raw") is not None,
    }


def _packet_outcome(packet: dict[str, Any] | None) -> dict[str, Any]:
    if packet is None:
        return {"available": False}
    return {
        "available": True,
        "status": packet.get("status"),
        "execution_ready": bool(packet.get("execution_ready")),
        "live_readback_regenerated": bool(packet.get("live_readback_regenerated")),
        "observer_camera_status": packet.get("observer_camera_status"),
        "observer_camera_indexes": packet.get("observer_camera_indexes", []),
        "policy_camera_indexes": packet.get("policy_camera_indexes", []),
        "transition_chunk_count": packet.get("transition_chunk_count"),
        "transition_step_count": packet.get("transition_step_count"),
        "blockers": packet.get("blockers") or [],
    }


def _verifier_outcome(*, grasp: dict[str, Any] | None, relocation: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "grasp_available": grasp is not None,
        "grasp_status": grasp.get("status") if grasp else "not_run",
        "grasp_outcome": grasp.get("grasp_outcome") if grasp else None,
        "relocation_available": relocation is not None,
        "relocation_status": relocation.get("status") if relocation else "not_run",
        "relocation_outcome": relocation.get("relocation_outcome") if relocation else None,
        "relocation_task_success_candidate": bool(relocation.get("task_success_candidate")) if relocation else False,
    }


def _camera_contract(*, execution: dict[str, Any], packet: dict[str, Any] | None) -> dict[str, Any]:
    policy_indexes = (packet or {}).get("policy_camera_indexes") or ["0", "1"]
    observer_status = (packet or {}).get("observer_camera_status") or (
        "available" if execution.get("observer_camera_index") == 3 and execution.get("visual_check") else "temporarily_unavailable"
    )
    if observer_status in {"off", "temporarily_unavailable"}:
        observer_indexes: list[int] = []
    else:
        observer_indexes = [int(execution.get("observer_camera_index", 3))]
    return {
        "policy_camera_indexes": [int(index) for index in policy_indexes],
        "policy_camera_roles": {
            "0": "SmolVLA policy input",
            "1": "SmolVLA policy input wide context",
        },
        "observer_camera_indexes": observer_indexes,
        "observer_camera_status": observer_status,
        "camera_3_policy_input": False,
    }


def _failure_modes(
    *,
    execution: dict[str, Any],
    packet: dict[str, Any] | None,
    grasp: dict[str, Any] | None,
    relocation: dict[str, Any] | None,
) -> list[str]:
    modes: list[str] = []
    if packet is None:
        modes.append("execution_packet_missing")
    elif packet.get("status") != "ready_for_observer_backed_execution" or not packet.get("execution_ready"):
        modes.append("execution_packet_not_ready")
    if _preflight_blocked(execution=execution, packet=packet):
        modes.append("observer_or_live_readback_preflight_incomplete")
    if execution.get("status") in {"failed", "blocked"}:
        modes.append(f"execution_{execution.get('status')}")
    if not execution.get("policy_actions_executed"):
        modes.append("no_policy_action_executed")
    if execution.get("policy_actions_executed"):
        visual = execution.get("visual_check") or {}
        if not visual.get("before") or not visual.get("after"):
            modes.append("observer_before_after_missing")
        if not execution.get("motion_video"):
            modes.append("observer_motion_video_missing")
        if grasp is None:
            modes.append("grasp_outcome_not_verified")
        if relocation is None:
            modes.append("task_success_not_verified")
        elif not relocation.get("task_success_candidate"):
            modes.append("relocation_goal_not_met")
    else:
        modes.append("task_success_not_verified")
    return _dedupe(modes)


def _preflight_blocked(*, execution: dict[str, Any], packet: dict[str, Any] | None) -> bool:
    if execution.get("packet_status") == "blocked" or execution.get("packet_execution_ready") is False:
        return True
    if packet is None:
        return True
    if not packet.get("live_readback_regenerated"):
        return True
    if packet.get("observer_camera_status") in {"off", "temporarily_unavailable"}:
        return True
    return False


def _task_success_candidate(*, grasp: dict[str, Any] | None, relocation: dict[str, Any] | None) -> bool:
    if relocation is None or not relocation.get("task_success_candidate"):
        return False
    if grasp is None:
        return False
    return grasp.get("status") == "passed" and grasp.get("grasp_outcome") != "grasp_failed_object_stationary"


def _next_step(
    *,
    execution: dict[str, Any],
    packet: dict[str, Any] | None,
    failure_modes: list[str],
    task_success_candidate: bool,
    prompt_mutation_allowed: bool,
) -> dict[str, Any]:
    if task_success_candidate:
        return {
            "type": "record_task_success_candidate_and_prepare_repro_run",
            "reason": "Both grasp and relocation verifier evidence support the task outcome; keep final success gated by repeatable observer-backed evidence.",
        }
    if "observer_or_live_readback_preflight_incomplete" in failure_modes:
        return {
            "type": "rerun_observer_return_refresh_live_readonly_when_camera_3_available",
            "reason": "The transition candidate is agentically useful, but execution was blocked by observer/live-readback preflight rather than SmolVLA prompt quality.",
        }
    if execution.get("policy_actions_executed") and "task_success_not_verified" in failure_modes:
        return {
            "type": "run_grasp_and_relocation_verifiers_before_prompt_mutation",
            "reason": "A physical policy chunk executed, so the next feedback must come from task-level before/after verifiers before changing the SmolVLA prompt.",
        }
    if prompt_mutation_allowed:
        return {
            "type": "build_next_smolvla_prompt_from_execution_feedback",
            "reason": "Execution evidence exists and preflight was satisfied; the next agentic iteration may update the in-loop prompt or plan.",
        }
    return {
        "type": "preserve_current_transition_candidate_and_resolve_execution_blockers",
        "reason": "No policy action executed, so the agentic layer should not infer task behavior or mutate the prompt from this run.",
    }


def _notes(*, execution: dict[str, Any], task_success_candidate: bool) -> list[str]:
    notes = [
        "Camera indexes 0 and 1 remain the only SmolVLA inputs.",
        "Camera 3 observer evidence is required for physical execution and task-level success claims.",
    ]
    if not execution.get("policy_actions_executed"):
        notes.append("This feedback is about execution gating, not object manipulation performance.")
    if not task_success_candidate:
        notes.append("Do not claim grasp, relocation, or final task success from this artifact.")
    return notes


def _optional_path(value: Any) -> Path | None:
    if not value:
        return None
    return Path(str(value))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feedback from an SO-100 transition execution report.")
    parser.add_argument("--execution-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--grasp-outcome", type=Path)
    parser.add_argument("--relocation-outcome", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_transition_execution_feedback(
                execution_report=args.execution_report,
                output=args.output,
                packet=args.packet,
                grasp_outcome=args.grasp_outcome,
                relocation_outcome=args.relocation_outcome,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
