#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_agentic_layer_contract(
    *,
    smolvla_report: Path,
    smolvla_action: Path,
    safety_report: Path,
    command_plan: Path,
    next_action_gate: Path,
    grasp_outcome: Path,
    pre_stage_pack: Path,
    output: Path,
    relocation_outcome: Path | None = None,
    action_metadata_report: Path | None = None,
    execute_gate_report: Path | None = None,
    output_markdown: Path | None = None,
) -> dict[str, Any]:
    smolvla = _load_json(smolvla_report)
    action = _load_json(smolvla_action)
    safety = _load_json(safety_report)
    command = _load_json(command_plan)
    gate = _load_json(next_action_gate)
    grasp = _load_json(grasp_outcome)
    relocation = _load_json(relocation_outcome) if relocation_outcome is not None else None
    execute_gate = _load_json(execute_gate_report) if execute_gate_report is not None else None
    pack = _load_json(pre_stage_pack)
    action_metadata_path = action_metadata_report or _infer_action_metadata_path(smolvla_report)
    action_metadata = _load_json(action_metadata_path) if action_metadata_path and action_metadata_path.exists() else None

    blockers = _collect_blockers(
        safety=safety,
        command=command,
        gate=gate,
        grasp=grasp,
        action_metadata=action_metadata,
        execute_gate=execute_gate,
    )
    decision = _decide_agentic_state(smolvla=smolvla, action=action, safety=safety, command=command, gate=gate, grasp=grasp)
    observer_camera_indexes = smolvla.get("observer_camera_indexes")
    observer_camera_status = smolvla.get("observer_camera_status") or (
        "temporarily_unavailable" if observer_camera_indexes == [] else "available"
    )
    contract = {
        "status": "passed",
        "operation": "real_so100_agentic_layer_contract",
        "purpose": "define the SmolVLA-plus-agentic-layer pre-stage feedback contract; not benchmark success",
        "agentic_success_claim": False,
        "final_task_success_claim": False,
        "policy": {
            "model_id": smolvla.get("model_id"),
            "instruction": smolvla.get("instruction") or action.get("instruction"),
            "instruction_tokenized": bool(smolvla.get("instruction_tokenized") or action.get("instruction_tokenized")),
            "language_token_count": smolvla.get("language_token_count") or action.get("language_token_count"),
            "raw_action_dim": smolvla.get("raw_action_dim") or len(action.get("raw_action", [])),
            "raw_action_chunk_steps": smolvla.get("raw_action_chunk_steps")
            or action.get("raw_action_chunk_steps")
            or len(action.get("raw_action_chunk", [])),
            "predicted_chunk_size": smolvla.get("predicted_chunk_size") or action.get("predicted_chunk_size"),
            "planned_action_steps": smolvla.get("planned_action_steps") or action.get("planned_action_steps"),
            "executed_action_steps": smolvla.get("executed_action_steps") or action.get("executed_action_steps"),
            "action_chunk_semantics": action.get("action_chunk_semantics"),
            "camera_source_mapping": smolvla.get("camera_source_mapping"),
            "policy_camera_indexes": smolvla.get("policy_camera_indexes"),
            "observer_camera_indexes": observer_camera_indexes,
            "observer_camera_status": observer_camera_status,
            "observer_camera_note": smolvla.get("observer_camera_note")
            or (
                "camera 3 is temporarily off; camera 1 may support no-actuation feedback only"
                if observer_camera_status == "temporarily_unavailable"
                else None
            ),
            "observer_camera_role": smolvla.get("observer_camera_role"),
            "action_path": str(smolvla_action),
            "report_path": str(smolvla_report),
            "actuation_enabled": bool(smolvla.get("actuation_enabled")),
            "send_action_called": bool(smolvla.get("send_action_called")),
        },
        "task_goal": _task_goal_from_instruction(smolvla.get("instruction") or action.get("instruction")),
        "agentic_layer": {
            "role": "wrap lightweight SmolVLA with explicit verification, retry, reframe, and evidence selection",
            "decision": decision,
            "blockers": blockers,
            "vla_prompt_allowed": bool(gate.get("vla_prompt_allowed")),
            "vla_prompt_gate": gate.get("vla_prompt_gate"),
            "physical_execution_gate": gate.get("physical_execution_gate"),
            "next_agentic_action": _next_agentic_action(decision=decision, gate=gate, grasp=grasp),
            "retry_policy_update": _retry_policy_update(gate=gate, grasp=grasp),
            "verifier_contract": {
                "pregrasp_gate_status": _nested(gate, ["evidence", "pregrasp_status"]),
                "jaw_gate_status": _nested(gate, ["evidence", "jaw_status"]),
                "last_grasp_outcome": grasp.get("grasp_outcome"),
                "relocation_verifier_status": relocation.get("status") if relocation else "not_run",
                "relocation_outcome": relocation.get("relocation_outcome") if relocation else None,
                "relocation_task_success_candidate": relocation.get("task_success_candidate") if relocation else False,
                "final_success_source": "none_in_real_so100_prestage",
                "internal_verifier_success_is_not_task_success": True,
                "final_task_success_requires_relocation_verifier": True,
            },
        },
        "adapter_and_safety": {
            "safety_status": safety.get("status"),
            "execution_allowed": safety.get("execution_allowed"),
            "safe_to_execute": action.get("safe_to_execute"),
            "command_plan_ready_for_execution": command.get("ready_for_execution"),
            "adapter_semantics_confirmed": command.get("adapter_semantics_confirmed"),
            "human_confirmed": command.get("human_confirmed") or safety.get("human_confirmed"),
            "command_plan_path": str(command_plan),
            "safety_report_path": str(safety_report),
            "action_metadata_path": str(action_metadata_path) if action_metadata_path and action_metadata_path.exists() else None,
            "action_metadata_status": action_metadata.get("status") if action_metadata else None,
            "action_metadata": _summarize_action_metadata(action_metadata),
            "execute_gate_path": str(execute_gate_report) if execute_gate_report else None,
            "execute_gate_status": execute_gate.get("status") if execute_gate else None,
            "execute_gate_ready_for_execution": _nested(execute_gate or {}, ["dry_plan", "ready_for_execution"]),
            "execute_gate_command_units": execute_gate.get("command_units") if execute_gate else None,
        },
        "evidence": {
            "next_action_gate": str(next_action_gate),
            "grasp_outcome": str(grasp_outcome),
            "relocation_outcome": str(relocation_outcome) if relocation_outcome else None,
            "pre_stage_pack": str(pre_stage_pack),
            "movement_report_html": pack.get("movement_report_html"),
            "gate_report_html": pack.get("gate_report_html"),
            "video_evidence_role": "debug_and_human_feedback_only",
        },
        "learning_signals": _learning_signals(pack=pack, gate=gate, grasp=grasp, safety=safety, command=command),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    contract["manifest_path"] = str(output)
    if output_markdown is not None:
        contract["output_markdown"] = str(output_markdown)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(_render_markdown(contract), encoding="utf-8")
    return contract


def _collect_blockers(
    *,
    safety: dict[str, Any],
    command: dict[str, Any],
    gate: dict[str, Any],
    grasp: dict[str, Any],
    action_metadata: dict[str, Any] | None = None,
    execute_gate: dict[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    for source in (safety, command, gate):
        for item in source.get("blockers", []):
            if item not in blockers:
                blockers.append(str(item))
    metadata = action_metadata.get("metadata", {}) if action_metadata else {}
    for item in metadata.get("blockers", []):
        if item not in blockers:
            blockers.append(str(item))
    for item in (action_metadata or {}).get("required_next_steps", []):
        blocker = f"action metadata required next step: {item}"
        if blocker not in blockers:
            blockers.append(blocker)
    for item in (execute_gate or {}).get("blockers", []):
        blocker = f"execute gate: {item}"
        if blocker not in blockers:
            blockers.append(blocker)
    for item in _nested(execute_gate or {}, ["dry_plan", "blockers"]) or []:
        blocker = f"dry plan: {item}"
        if blocker not in blockers:
            blockers.append(blocker)
    if grasp.get("grasp_outcome") == "grasp_failed_object_stationary":
        blockers.append("last gripper close failed because the object stayed stationary")
    return blockers


def _decide_agentic_state(
    *,
    smolvla: dict[str, Any],
    action: dict[str, Any],
    safety: dict[str, Any],
    command: dict[str, Any],
    gate: dict[str, Any],
    grasp: dict[str, Any],
) -> str:
    if not bool(smolvla.get("instruction_tokenized") or action.get("instruction_tokenized")):
        return "blocked_policy_prompt_not_wired"
    if action.get("safe_to_execute") is not False:
        return "blocked_policy_action_not_explicitly_dry"
    if safety.get("execution_allowed") is True or command.get("ready_for_execution") is True:
        return "unsafe_contract_violation_policy_path_marked_executable"
    if gate.get("status") == "blocked" and gate.get("vla_prompt_allowed") is True:
        return "ready_for_smolvla_proposal_physical_blocked"
    if gate.get("status") == "blocked":
        return "blocked_reframe_before_retry"
    if grasp.get("grasp_outcome") == "grasp_failed_object_stationary":
        return "ready_for_reframed_contact_probe"
    return "ready_for_next_gate_checked_probe"


def _next_agentic_action(*, decision: str, gate: dict[str, Any], grasp: dict[str, Any]) -> dict[str, Any]:
    object_view_camera = str(_nested(gate, ["evidence", "object_view_camera"]) or "1")
    jaw_camera = str(_nested(gate, ["evidence", "jaw_camera"]) or "0")
    if decision == "ready_for_smolvla_proposal_physical_blocked":
        return {
            "type": "smolvla_proposal_only",
            "reason": (gate.get("vla_prompt_gate") or {}).get("reason"),
            "physical_robot_motion": False,
            "vla_prompt_allowed": True,
            "physical_execution_blocked": True,
            "required_before_execution": [
                "adapter_semantics_confirmed",
                "camera_0_jaw_object_framing_ready",
                "explicit physical micro-step confirmation",
                "observer-camera before/after evidence contract",
            ],
            "then": "run_smolvla_dry_no_actuation_and_keep_action_as_proposal",
        }
    if decision == "blocked_reframe_before_retry":
        return {
            "type": "observe_reframe",
            "reason": gate.get("recommended_action"),
            "physical_robot_motion": False,
            "required_observations": [
                f"camera_{jaw_camera}_jaw_object_framing",
                f"camera_{object_view_camera}_object_view",
            ],
            "then": "rerun_no_actuation_cp26_gate",
        }
    if decision == "ready_for_reframed_contact_probe":
        return {
            "type": "minimal_contact_probe",
            "joint": "gripper",
            "physical_robot_motion": True,
            "requires": [
                "human_confirmed",
                "contact_ok_for_gripper",
                "record_video",
                f"camera_{object_view_camera}_before_after",
                "grasp_outcome_verifier",
            ],
            "then": "classify_object_motion_and_update_retry_policy",
        }
    return {
        "type": "hold",
        "physical_robot_motion": False,
        "reason": f"decision={decision}",
        "then": "inspect_contract_blockers",
    }


def _retry_policy_update(*, gate: dict[str, Any], grasp: dict[str, Any]) -> dict[str, Any]:
    updates = []
    recommended_action = str(gate.get("recommended_action") or "")
    if recommended_action.startswith("reframe_camera_") or recommended_action == "reframe_camera_0_or_object":
        updates.append("prioritize observation-quality repair before another SmolVLA-derived contact action")
    if grasp.get("grasp_outcome") == "grasp_failed_object_stationary":
        updates.append("do not repeat gripper close from the same pose; require pose or perception change first")
    return {
        "updates": updates,
        "failure_memory_key": grasp.get("grasp_outcome") or "none",
        "retry_budget_consumed_by_failed_grasp": grasp.get("grasp_outcome") == "grasp_failed_object_stationary",
    }


def _task_goal_from_instruction(instruction: Any) -> dict[str, Any]:
    text = str(instruction or "")
    lowered = text.lower()
    direction = None
    for candidate in ("right", "left", "up", "down"):
        if candidate in lowered:
            direction = candidate
            break
    return {
        "instruction": text,
        "target_object": "green Android figure" if "green" in lowered else None,
        "transport_direction": direction,
        "requires_grasp": any(token in lowered for token in ("pick", "grasp", "grab")),
        "requires_transport": direction is not None or any(token in lowered for token in ("move", "place")),
        "final_success_verifier": "object_relocation_image_space" if direction is not None else "task_specific_verifier_required",
    }


def _learning_signals(
    *,
    pack: dict[str, Any],
    gate: dict[str, Any],
    grasp: dict[str, Any],
    safety: dict[str, Any],
    command: dict[str, Any],
) -> list[dict[str, str]]:
    signals = []
    for item in pack.get("agentic_lessons", []):
        signals.append(
            {
                "source": "pre_stage_pack",
                "observation": str(item.get("observation")),
                "agentic_update": str(item.get("agentic_update")),
            }
        )
    if safety.get("execution_allowed") is not True:
        signals.append(
            {
                "source": "safety_gate",
                "observation": "SmolVLA action is available but not executable under current safety semantics.",
                "agentic_update": "Treat raw policy output as a proposal; execution must pass adapter semantics and verifier gates.",
            }
        )
    if command.get("ready_for_execution") is not True:
        signals.append(
            {
                "source": "command_adapter",
                "observation": "Command plan is not ready for execution.",
                "agentic_update": "Keep the agentic layer in observe/repair mode rather than policy-only actuation.",
            }
        )
    if grasp.get("grasp_outcome"):
        signals.append(
            {
                "source": "grasp_verifier",
                "observation": str(grasp.get("grasp_outcome")),
                "agentic_update": "Use verifier outcome to select retry/reframe, not as final benchmark success.",
            }
        )
    return signals


def _nested(payload: dict[str, Any], keys: list[str]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_action_metadata_path(smolvla_report: Path) -> Path | None:
    candidate = smolvla_report.parent / "action_metadata" / "smolvla_action_metadata_report.json"
    return candidate if candidate.exists() else None


def _summarize_action_metadata(action_metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if action_metadata is None:
        return None
    metadata = action_metadata.get("metadata", {})
    return {
        "model_id": action_metadata.get("model_id") or metadata.get("model_id"),
        "status": action_metadata.get("status"),
        "action_normalization": metadata.get("action_normalization"),
        "output_is_normalized": metadata.get("output_is_normalized"),
        "action_stats_available": metadata.get("action_stats_available"),
        "stats_source": metadata.get("stats_source"),
        "selected_action_stats_key": metadata.get("selected_action_stats_key"),
        "available_action_stats_keys": metadata.get("available_action_stats_keys"),
        "action_semantics": metadata.get("action_semantics"),
        "joint_order": metadata.get("joint_order"),
        "gripper_semantics": metadata.get("gripper_semantics"),
        "command_units": metadata.get("command_units"),
        "required_next_steps": action_metadata.get("required_next_steps", []),
    }


def _render_markdown(contract: dict[str, Any]) -> str:
    agentic = contract["agentic_layer"]
    policy = contract["policy"]
    task_goal = contract.get("task_goal", {})
    safety = contract["adapter_and_safety"]
    evidence = contract["evidence"]
    lines = [
        "# Real SO-100 SmolVLA Agentic Layer Contract",
        "",
        "This is a pre-stage contract for improving the agentic layer around SmolVLA. It is not a benchmark success claim.",
        "",
        "## Policy Proposal",
        "",
        f"- Model: `{policy.get('model_id')}`",
        f"- Instruction: `{policy.get('instruction')}`",
        f"- Instruction tokenized: `{policy.get('instruction_tokenized')}`",
        f"- Language token count: `{policy.get('language_token_count')}`",
        f"- Raw action dimension: `{policy.get('raw_action_dim')}`",
        f"- Raw action chunk steps: `{policy.get('raw_action_chunk_steps')}`",
        f"- Predicted chunk size: `{policy.get('predicted_chunk_size')}`",
        f"- Policy cameras: `{policy.get('policy_camera_indexes')}`",
        f"- Observer cameras: `{policy.get('observer_camera_indexes')}`",
        f"- Observer status: `{policy.get('observer_camera_status')}`",
        f"- Observer role: `{policy.get('observer_camera_role')}`",
        f"- Action sent to robot: `{policy.get('send_action_called')}`",
        "",
        "## Task Goal",
        "",
        f"- Target object: `{task_goal.get('target_object')}`",
        f"- Transport direction: `{task_goal.get('transport_direction')}`",
        f"- Final success verifier: `{task_goal.get('final_success_verifier')}`",
        "",
        "## Agentic Decision",
        "",
        f"- Decision: `{agentic.get('decision')}`",
        f"- Next action type: `{agentic.get('next_agentic_action', {}).get('type')}`",
        f"- Physical robot motion: `{agentic.get('next_agentic_action', {}).get('physical_robot_motion')}`",
        f"- Then: `{agentic.get('next_agentic_action', {}).get('then')}`",
        "",
        "## Verifier Contract",
        "",
        f"- Pregrasp gate: `{agentic.get('verifier_contract', {}).get('pregrasp_gate_status')}`",
        f"- Jaw gate: `{agentic.get('verifier_contract', {}).get('jaw_gate_status')}`",
        f"- Last grasp outcome: `{agentic.get('verifier_contract', {}).get('last_grasp_outcome')}`",
        f"- Relocation verifier: `{agentic.get('verifier_contract', {}).get('relocation_verifier_status')}`",
        f"- Relocation outcome: `{agentic.get('verifier_contract', {}).get('relocation_outcome')}`",
        f"- Final success source: `{agentic.get('verifier_contract', {}).get('final_success_source')}`",
        "- Internal verifier success is a retry signal, not final task success.",
        "- Final task success requires the relocation verifier to pass after a physical attempt.",
        "",
        "## Safety And Adapter",
        "",
        f"- Safety status: `{safety.get('safety_status')}`",
        f"- Execution allowed: `{safety.get('execution_allowed')}`",
        f"- Command plan ready: `{safety.get('command_plan_ready_for_execution')}`",
        f"- Human confirmed: `{safety.get('human_confirmed')}`",
        f"- Action metadata status: `{safety.get('action_metadata_status')}`",
        f"- Action normalization: `{(safety.get('action_metadata') or {}).get('action_normalization')}`",
        f"- Action stats available: `{(safety.get('action_metadata') or {}).get('action_stats_available')}`",
        f"- Stats source: `{(safety.get('action_metadata') or {}).get('stats_source')}`",
        f"- Selected action stats key: `{(safety.get('action_metadata') or {}).get('selected_action_stats_key')}`",
        f"- Command units: `{(safety.get('action_metadata') or {}).get('command_units')}`",
        f"- Execute gate status: `{safety.get('execute_gate_status')}`",
        f"- Execute gate ready: `{safety.get('execute_gate_ready_for_execution')}`",
        f"- Execute gate command units: `{safety.get('execute_gate_command_units')}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in agentic.get("blockers", []))
    lines.extend(["", "## Learning Signals", ""])
    lines.extend(
        f"- `{item.get('source')}`: {item.get('observation')} -> {item.get('agentic_update')}"
        for item in contract.get("learning_signals", [])
    )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Movement report: `{evidence.get('movement_report_html')}`",
            f"- Gate report: `{evidence.get('gate_report_html')}`",
            f"- Grasp outcome: `{evidence.get('grasp_outcome')}`",
            f"- Video evidence role: `{evidence.get('video_evidence_role')}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the real SO-100 SmolVLA agentic-layer pre-stage contract.")
    parser.add_argument("--smolvla-report", type=Path, required=True)
    parser.add_argument("--smolvla-action", type=Path, required=True)
    parser.add_argument("--safety-report", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--next-action-gate", type=Path, required=True)
    parser.add_argument("--grasp-outcome", type=Path, required=True)
    parser.add_argument("--pre-stage-pack", type=Path, required=True)
    parser.add_argument("--relocation-outcome", type=Path)
    parser.add_argument("--action-metadata-report", type=Path)
    parser.add_argument("--execute-gate-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_agentic_layer_contract(
                smolvla_report=args.smolvla_report,
                smolvla_action=args.smolvla_action,
                safety_report=args.safety_report,
                command_plan=args.command_plan,
                next_action_gate=args.next_action_gate,
                grasp_outcome=args.grasp_outcome,
                pre_stage_pack=args.pre_stage_pack,
                relocation_outcome=args.relocation_outcome,
                action_metadata_report=args.action_metadata_report,
                execute_gate_report=args.execute_gate_report,
                output=args.output,
                output_markdown=args.output_markdown,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
