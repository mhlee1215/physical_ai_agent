#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_prompt_iteration(
    *,
    prompt: str,
    smolvla_report: Path,
    refresh_manifest: Path,
    analysis: Path,
    agentic_state: Path,
    next_plan: Path,
    output_json: Path | None = None,
    output_md: Path | None = None,
    iteration_index: int | None = None,
    vla_prompt_packet: Path | None = None,
    agentic_policy_patch: Path | None = None,
) -> dict[str, Any]:
    smolvla = _load_json(smolvla_report)
    refresh = _load_json(refresh_manifest)
    analysis_payload = _load_json(analysis)
    state = _load_json(agentic_state)
    plan = _load_json(next_plan)

    policy_camera_indexes = [str(item) for item in smolvla.get("policy_camera_indexes", [])]
    observer_camera_indexes = [str(item) for item in smolvla.get("observer_camera_indexes", [])]
    observer_camera_status = str(
        refresh.get("observer_camera_status")
        or ("available" if observer_camera_indexes else "temporarily_unavailable")
    )
    if observer_camera_status == "temporarily_unavailable":
        observer_camera_indexes = []
    failure_modes = [str(item.get("type")) for item in analysis_payload.get("failure_modes", [])]
    active_constraints = _normalized_active_constraints(state.get("active_constraints", []))
    policy_updates = _normalized_policy_updates(state.get("policy_updates", []))
    next_steps = plan.get("next_steps", [])
    physical_motion = bool(refresh.get("physical_robot_motion")) or bool(plan.get("physical_robot_motion"))

    result = {
        "status": "passed",
        "operation": "real_so100_prompt_iteration",
        "purpose": "canonical prompt-to-feedback record for improving the SmolVLA agentic layer",
        "iteration_index": iteration_index,
        "prompt": prompt,
        "task": refresh.get("task") or smolvla.get("instruction") or prompt,
        "camera_contract": {
            "smolvla_policy_inputs": policy_camera_indexes,
            "observer_inputs": observer_camera_indexes,
            "observer_camera_status": observer_camera_status,
            "camera_source_mapping": smolvla.get("camera_source_mapping"),
            "observer_camera_role": smolvla.get("observer_camera_role"),
            "policy_rule": _policy_rule(observer_camera_status=observer_camera_status),
        },
        "policy_proposal": {
            "smolvla_report": str(smolvla_report),
            "vla_prompt_packet": str(vla_prompt_packet) if vla_prompt_packet else None,
            "instruction_tokenized": bool(smolvla.get("instruction_tokenized")),
            "language_token_count": smolvla.get("language_token_count"),
            "send_action_called": bool(smolvla.get("send_action_called")),
            "policy_actions_executed": bool(smolvla.get("policy_actions_executed")),
            "action_preview": smolvla.get("action_preview"),
        },
        "execution_log": {
            "refresh_manifest": str(refresh_manifest),
            "gate_status": refresh.get("gate_status"),
            "agentic_decision": refresh.get("agentic_decision"),
            "physical_robot_motion": physical_motion,
            "send_action_called": bool(refresh.get("send_action_called")),
            "observer_camera_status": observer_camera_status,
        },
        "analysis": {
            "analysis": str(analysis),
            "failure_modes": failure_modes,
            "improvement_targets": [
                str(item.get("target")) for item in analysis_payload.get("agentic_layer_improvements", [])
            ],
            "loop_continuation": analysis_payload.get("loop_continuation", {}),
        },
        "agentic_policy_patch": str(agentic_policy_patch) if agentic_policy_patch else None,
        "agentic_state": {
            "state": str(agentic_state),
            "active_constraints": active_constraints,
            "failure_memory": state.get("failure_memory", {}),
            "policy_updates": policy_updates,
        },
        "next_iteration": {
            "next_plan": str(next_plan),
            "stage": plan.get("stage"),
            "physical_robot_motion": bool(plan.get("physical_robot_motion")),
            "next_step_type": next_steps[0].get("type") if next_steps else None,
            "next_step": next_steps[0] if next_steps else None,
            "autonomous_next_steps": plan.get("autonomous_next_steps"),
            "external_setup_blocker": plan.get("external_setup_blocker"),
            "post_external_setup_verification": plan.get("post_external_setup_verification"),
            "repair_escalation": plan.get("repair_escalation"),
            "repeat_prompt_after_repair": bool(
                analysis_payload.get("loop_continuation", {}).get("repeat_prompt_after_repair")
            ),
            "prompt_to_repeat": prompt,
        },
        "success_accounting": _success_accounting(
            refresh=refresh,
            analysis_payload=analysis_payload,
            plan=plan,
        ),
        "generalized_lessons": _generalized_lessons(
            failures=failure_modes,
            active_constraints=active_constraints,
            policy_updates=policy_updates,
        ),
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        result["manifest_path"] = str(output_json)
        output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(result), encoding="utf-8")
        result["markdown_path"] = str(output_md)
        if output_json is not None:
            output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def render_markdown(record: dict[str, Any]) -> str:
    lines = [
        "# Real SO-100 Prompt Iteration",
        "",
        f"- Prompt: `{record.get('prompt')}`",
        f"- Stage: `{record.get('next_iteration', {}).get('stage')}`",
        f"- Physical robot motion: `{record.get('execution_log', {}).get('physical_robot_motion')}`",
        f"- SmolVLA policy cameras: `{record.get('camera_contract', {}).get('smolvla_policy_inputs')}`",
        f"- Codex observer cameras: `{record.get('camera_contract', {}).get('observer_inputs')}`",
        f"- Task success claim allowed: `{record.get('success_accounting', {}).get('task_success_claim_allowed')}`",
        f"- Agentic policy patch: `{record.get('agentic_policy_patch')}`",
        "",
        "## Failure Modes",
    ]
    failures = record.get("analysis", {}).get("failure_modes", [])
    lines.extend(f"- `{item}`" for item in failures)
    lines.extend(["", "## Active Constraints"])
    constraints = record.get("agentic_state", {}).get("active_constraints", [])
    lines.extend(f"- `{item}`" for item in constraints)
    lines.extend(["", "## Next Iteration"])
    next_iteration = record.get("next_iteration", {})
    lines.extend(
        [
            f"- Next step: `{next_iteration.get('next_step_type')}`",
            f"- Repair escalation: `{next_iteration.get('repair_escalation')}`",
            f"- Repeat prompt after repair: `{next_iteration.get('repeat_prompt_after_repair')}`",
            f"- Prompt to repeat: `{next_iteration.get('prompt_to_repeat')}`",
        ]
    )
    lines.extend(["", "## Generalized Lessons"])
    lines.extend(f"- {item}" for item in record.get("generalized_lessons", []))
    lines.append("")
    return "\n".join(lines)


def _success_accounting(
    *,
    refresh: dict[str, Any],
    analysis_payload: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    failure_modes = analysis_payload.get("failure_modes", [])
    relocation_verified = refresh.get("relocation_verifier_status") == "passed" or refresh.get(
        "relocation_task_success_candidate"
    ) is True
    blocked_stages = {"observation_repair", "external_setup_blocked"}
    task_success_allowed = bool(relocation_verified and not failure_modes and plan.get("stage") not in blocked_stages)
    return {
        "task_success_claim_allowed": task_success_allowed,
        "relocation_verified": relocation_verified,
        "required_before_success": [
            "no blocking observation gate",
            "adapter semantics validated before contact",
            "video-backed physical movement evidence for executed robot actions",
            "object relocation verifier passes for transport prompts",
        ],
        "note": "Internal verifier or Codex observer evidence can drive retries, but final task success needs the task-level relocation verifier.",
    }


def _policy_rule(*, observer_camera_status: str) -> str:
    if observer_camera_status == "temporarily_unavailable":
        return (
            "Innomaker U20CAM indexes 0 and 1 are policy/context inputs; iPhone observer index 3 is temporarily "
            "unavailable, so this iteration is no-actuation agentic-layer development only."
        )
    return "Innomaker U20CAM indexes 0 and 1 are policy inputs; iPhone index 3 is Codex observer/debug only."


def _generalized_lessons(
    *,
    failures: list[str],
    active_constraints: list[str],
    policy_updates: list[dict[str, Any]],
) -> list[str]:
    lessons = []
    if "jaw_object_framing_not_ready" in failures:
        lessons.append("Block VLA prompting/contact execution whenever required policy-camera evidence is outside the robot action space.")
    if "adapter_semantics_not_executable" in failures:
        lessons.append("Treat lightweight VLA output as a proposal until robot-native sign and scale are validated.")
    if "task_success_not_verified" in failures:
        lessons.append("For transport tasks, require before/after object relocation evidence rather than grasp-only evidence.")
    for update in policy_updates:
        generalization = update.get("generalization")
        if generalization and str(generalization) not in lessons:
            lessons.append(str(generalization))
    for constraint in active_constraints:
        lesson = f"Carry constraint `{constraint}` into the next prompt iteration."
        if lesson not in lessons:
            lessons.append(lesson)
    return lessons


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_active_constraints(constraints: list[Any]) -> list[str]:
    replacements = {
        "observation_repair_before_contact": "external_setup_ready_before_contact",
    }
    return sorted({replacements.get(str(item), str(item)) for item in constraints})


def _normalized_policy_updates(updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for update in updates:
        item = dict(update)
        if item.get("target") == "observation_repair_policy":
            item["target"] = "policy_input_quality_gate"
            item["change"] = "block VLA prompting/contact execution when required policy cameras do not provide usable target evidence"
            item["generalization"] = (
                "applies to any task where required policy inputs are edge-clipped, missing, stale, "
                "or otherwise outside the agent action space"
            )
            item.pop("latest_advice", None)
        normalized.append(item)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one real SO-100 prompt-to-feedback iteration record.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--smolvla-report", type=Path, required=True)
    parser.add_argument("--refresh-manifest", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--agentic-state", type=Path, required=True)
    parser.add_argument("--next-plan", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--iteration-index", type=int)
    parser.add_argument("--vla-prompt-packet", type=Path)
    parser.add_argument("--agentic-policy-patch", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_prompt_iteration(
                prompt=args.prompt,
                smolvla_report=args.smolvla_report,
                refresh_manifest=args.refresh_manifest,
                analysis=args.analysis,
                agentic_state=args.agentic_state,
                next_plan=args.next_plan,
                output_json=args.output_json,
                output_md=args.output_md,
                iteration_index=args.iteration_index,
                vla_prompt_packet=args.vla_prompt_packet,
                agentic_policy_patch=args.agentic_policy_patch,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
