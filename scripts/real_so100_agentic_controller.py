#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = "_workspace/real_so100/calibration/so100_local.json"


def build_agentic_next_plan(
    *,
    contract: Path,
    output: Path | None = None,
    reframe_advice: Path | None = None,
    agentic_state: Path | None = None,
    port: str = DEFAULT_PORT,
    calibration_file: str = DEFAULT_CALIBRATION,
    next_output_dir: str = "_workspace/real_so100/checkpoint_26_gate_next",
    contact_output_dir: str = "_workspace/real_so100/contact_probe_next",
    vla_prompt_packet: Path | None = None,
) -> dict[str, Any]:
    payload = _load_json(contract)
    advice = _load_json(reframe_advice) if reframe_advice is not None else None
    state = _load_json(agentic_state) if agentic_state is not None else None
    policy = payload.get("policy", {})
    task_goal = payload.get("task_goal", {})
    agentic = payload.get("agentic_layer", {})
    safety = payload.get("adapter_and_safety", {})
    verifier = agentic.get("verifier_contract", {})
    next_action = agentic.get("next_agentic_action", {})
    plan = {
        "status": "passed",
        "operation": "real_so100_agentic_next_plan",
        "contract": str(contract),
        "policy_instruction": policy.get("instruction"),
        "task_goal": task_goal,
        "stage": None,
        "physical_robot_motion": False,
        "smolvla_role": "proposal_generator",
        "codex_observer_role": policy.get("observer_camera_role"),
        "agentic_state": str(agentic_state) if agentic_state else None,
        "vla_prompt_packet": str(vla_prompt_packet) if vla_prompt_packet else None,
        "active_constraints": _normalized_active_constraints((state or {}).get("active_constraints", [])),
        "failure_memory": (state or {}).get("failure_memory", {}),
        "repair_escalation": _repair_escalation(state or {}),
        "required_evidence_before_success_claim": _required_success_evidence(task_goal=task_goal),
        "next_steps": [],
        "guardrails": [
            "Do not execute raw SmolVLA actions directly.",
            "Do not claim task success from internal verifier success alone.",
            "Record observer/debug video for every physical movement.",
        ],
    }

    if not policy.get("instruction_tokenized"):
        _policy_prompt_step(plan=plan, payload=payload)
    elif next_action.get("type") == "observe_reframe":
        _external_setup_blocked_step(
            plan=plan,
            next_action=next_action,
            policy=policy,
            payload=payload,
            reframe_advice=advice,
            port=port,
            calibration_file=calibration_file,
            next_output_dir=next_output_dir,
        )
    elif next_action.get("type") == "smolvla_proposal_only":
        _smolvla_proposal_only_step(plan=plan, next_action=next_action, payload=payload)
    elif safety.get("command_plan_ready_for_execution") is not True:
        _adapter_calibration_step(plan=plan, safety=safety)
    elif next_action.get("type") == "minimal_contact_probe":
        _contact_probe_step(
            plan=plan,
            next_action=next_action,
            task_goal=task_goal,
            payload=payload,
            port=port,
            calibration_file=calibration_file,
            contact_output_dir=contact_output_dir,
            vla_prompt_packet=str(vla_prompt_packet) if vla_prompt_packet else None,
        )
    elif _relocation_required_but_not_passed(task_goal=task_goal, verifier=verifier):
        _relocation_verification_step(plan=plan, task_goal=task_goal, contact_output_dir=contact_output_dir)
    else:
        plan["stage"] = "hold_for_human_review"
        plan["next_steps"].append(
            {
                "type": "inspect_contract",
                "physical_robot_motion": False,
                "reason": agentic.get("decision"),
            }
        )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return plan


def _policy_prompt_step(*, plan: dict[str, Any], payload: dict[str, Any]) -> None:
    plan["stage"] = "policy_prompt_repair"
    plan["next_steps"].append(
        {
            "type": "rerun_smolvla_dry",
            "physical_robot_motion": False,
            "reason": payload.get("agentic_layer", {}).get("decision"),
        }
    )


def _external_setup_blocked_step(
    *,
    plan: dict[str, Any],
    next_action: dict[str, Any],
    policy: dict[str, Any],
    payload: dict[str, Any],
    reframe_advice: dict[str, Any] | None,
    port: str,
    calibration_file: str,
    next_output_dir: str,
) -> None:
    task_goal = payload.get("task_goal", {})
    policy_indexes = [str(index) for index in policy.get("policy_camera_indexes") or ["0", "1"]]
    observer_indexes = [str(index) for index in policy.get("observer_camera_indexes") or []]
    wrist_index = _camera_index_for_role(policy.get("camera_source_mapping"), "wrist_cam", fallback="0")
    egocentric_index = _camera_index_for_role(policy.get("camera_source_mapping"), "egocentric_cam", fallback="1")
    plan["stage"] = "external_setup_blocked"
    plan["physical_robot_motion"] = False
    setup_blocker = {
        "type": "external_setup_blocker",
        "physical_robot_motion": False,
        "reason": next_action.get("reason"),
        "required_observations": next_action.get("required_observations", []),
        "agent_actionable": False,
        "vla_prompt_allowed": False,
        "why_not_agent_action": "camera/object framing is outside the robot policy action space; do not count external setup changes as agentic actions",
    }
    escalation = plan.get("repair_escalation")
    if escalation:
        setup_blocker["requires_external_setup_change_before_rerun"] = True
        setup_blocker["escalation_reason"] = escalation.get("reason")
    if reframe_advice:
        setup_blocker["diagnostics"] = reframe_advice.get("actions", [])
        setup_blocker["diagnostics_path"] = reframe_advice.get("manifest_path") or reframe_advice.get("output")
    plan["external_setup_blocker"] = setup_blocker
    if not observer_indexes:
        plan["observer_camera_status"] = "temporarily_unavailable"
        plan["guardrails"].append(
            "Observer camera is temporarily unavailable; keep physical execution disabled and use camera 1 only as policy-context feedback."
        )
    plan["autonomous_next_steps"] = []
    plan["post_external_setup_verification"] = (
        [
            {
                "type": "rerun_no_actuation_gate_after_external_setup_change",
                "physical_robot_motion": False,
                "command": _checkpoint_gate_command(
                    port=port,
                    calibration_file=calibration_file,
                    output_dir=next_output_dir,
                    policy_indexes=policy_indexes,
                    observer_indexes=observer_indexes,
                    wrist_index=wrist_index,
                    egocentric_index=egocentric_index,
                    task=str(task_goal.get("instruction") or "real_so100_task"),
                    grasp_outcome=str(payload.get("evidence", {}).get("grasp_outcome")),
                ),
            },
        ]
    )


def _smolvla_proposal_only_step(*, plan: dict[str, Any], next_action: dict[str, Any], payload: dict[str, Any]) -> None:
    policy = payload.get("policy", {})
    observer_indexes = policy.get("observer_camera_indexes") or []
    plan["stage"] = "smolvla_proposal_only"
    plan["physical_robot_motion"] = False
    plan["vla_prompt_allowed"] = True
    plan["physical_execution_blocked"] = True
    if not observer_indexes:
        plan["observer_camera_status"] = "temporarily_unavailable"
        plan["guardrails"].append(
            "Observer camera is temporarily unavailable; do not claim physical task success from policy-camera-only evidence."
        )
    plan["autonomous_next_steps"] = [
        {
            "type": "rerun_smolvla_dry",
            "physical_robot_motion": False,
            "reason": next_action.get("reason"),
            "policy_camera_indexes": policy.get("policy_camera_indexes"),
            "observer_camera_indexes_excluded_from_policy": observer_indexes,
            "required_before_execution": next_action.get("required_before_execution", []),
        }
    ]
    plan["next_steps"].extend(plan["autonomous_next_steps"])


def _adapter_calibration_step(*, plan: dict[str, Any], safety: dict[str, Any]) -> None:
    plan["stage"] = "adapter_semantics_repair"
    plan["physical_robot_motion"] = False
    plan["next_steps"].append(
        {
            "type": "adapter_semantics_calibration",
            "physical_robot_motion": False,
            "reason": safety.get("command_plan_path"),
            "required_before_contact": [
                "validate sign and scale on non-contact micro-steps",
                "keep command_plan_ready_for_execution=false until semantics are explicit",
            ],
        }
    )


def _contact_probe_step(
    *,
    plan: dict[str, Any],
    next_action: dict[str, Any],
    task_goal: dict[str, Any],
    payload: dict[str, Any],
    port: str,
    calibration_file: str,
    contact_output_dir: str,
    vla_prompt_packet: str | None,
) -> None:
    observer_indexes = payload.get("policy", {}).get("observer_camera_indexes") or []
    if not observer_indexes:
        plan["stage"] = "observer_camera_unavailable_physical_execution_blocked"
        plan["physical_robot_motion"] = False
        plan["physical_execution_blocked"] = True
        plan["observer_camera_status"] = "temporarily_unavailable"
        plan["next_steps"].append(
            {
                "type": "wait_for_observer_camera_or_continue_no_actuation_agentic_layer",
                "physical_robot_motion": False,
                "reason": "camera 3 observer/debug evidence is unavailable; contact probes require video-backed before/after evidence",
            }
        )
        return
    joint = next_action.get("joint") or "gripper"
    direction = task_goal.get("transport_direction") or "right"
    plan["stage"] = "minimal_contact_probe"
    plan["physical_robot_motion"] = True
    plan["next_steps"].extend(
        [
            {
                "type": "execute_video_backed_contact_probe",
                "physical_robot_motion": True,
                "command": _contact_probe_command(
                    port=port,
                    joint=str(joint),
                    output_dir=contact_output_dir,
                ),
            },
            {
                "type": "run_relocation_verifier",
                "physical_robot_motion": False,
                "command": _relocation_command(
                    contact_output_dir=contact_output_dir,
                    direction=str(direction),
                ),
            },
            {
                "type": "rebuild_agentic_contract",
                "physical_robot_motion": False,
                "reason": "include grasp outcome and relocation outcome before any success claim",
                "source_contract": payload.get("manifest_path"),
            },
        ]
    )
    if vla_prompt_packet:
        plan["next_steps"].insert(
            1,
            {
                "type": "materialize_relocation_verifier_packet",
                "physical_robot_motion": False,
                "command": _relocation_verifier_packet_command(
                    vla_prompt_packet=vla_prompt_packet,
                    contact_output_dir=contact_output_dir,
                    direction=str(direction),
                ),
                "reason": "bind observer-camera before/after evidence to the SmolVLA prompt semantics before success accounting",
            },
        )


def _relocation_verification_step(*, plan: dict[str, Any], task_goal: dict[str, Any], contact_output_dir: str) -> None:
    direction = task_goal.get("transport_direction") or "right"
    plan["stage"] = "relocation_verification"
    plan["physical_robot_motion"] = False
    plan["next_steps"].append(
        {
            "type": "run_relocation_verifier",
            "physical_robot_motion": False,
            "command": _relocation_command(contact_output_dir=contact_output_dir, direction=str(direction)),
        }
    )


def _repair_escalation(state: dict[str, Any]) -> dict[str, Any] | None:
    failures = state.get("failure_memory", {})
    jaw_count = int(failures.get("jaw_object_framing_not_ready", {}).get("count", 0))
    observation_count = int(failures.get("observation_gate_blocked", {}).get("count", 0))
    repeated_count = max(jaw_count, observation_count)
    if repeated_count < 3:
        return None
    return {
        "type": "repeated_observation_repair_blocker",
        "count": repeated_count,
        "reason": "same observation blocker repeated; external camera/object reframe is required before another contact attempt",
        "required_before_contact": [
            "camera_0 target object no longer touches image boundary",
            "camera_0 jaw marker remains visible",
            "camera_1 object view remains usable",
        ],
    }


def _normalized_active_constraints(constraints: list[Any]) -> list[str]:
    replacements = {
        "observation_repair_before_contact": "external_setup_ready_before_contact",
    }
    return sorted({replacements.get(str(item), str(item)) for item in constraints})


def _checkpoint_gate_command(
    *,
    port: str,
    calibration_file: str,
    output_dir: str,
    policy_indexes: list[str],
    observer_indexes: list[str],
    wrist_index: str,
    egocentric_index: str,
    task: str,
    grasp_outcome: str,
) -> list[str]:
    command = [
        "PYTHONPATH=src:.",
        ".venv/bin/python",
        "-B",
        "scripts/real_so100_checkpoint_26_gate.py",
        "--port",
        port,
        "--output-dir",
        output_dir,
        "--duration-seconds",
        "1.0",
        "--fps",
        "2",
        "--wrist-camera-index",
        wrist_index,
        "--egocentric-camera-index",
        egocentric_index,
        "--task",
        task,
        "--calibration-file",
        calibration_file,
        "--grasp-outcome",
        grasp_outcome,
    ]
    for index in policy_indexes:
        command.extend(["--policy-camera-index", index])
    for index in observer_indexes:
        command.extend(["--observer-camera-index", index])
    return command


def _contact_probe_command(*, port: str, joint: str, output_dir: str) -> list[str]:
    return [
        "PYTHONPATH=src:.",
        ".venv/bin/python",
        "-B",
        "scripts/real_so100_micro_step.py",
        "--port",
        port,
        "--joint",
        joint,
        "--manual-delta-raw",
        "-30",
        "--output",
        f"{output_dir}/report.json",
        "--execute",
        "--human-confirmed",
        "--contact-ok-for-gripper",
        "--max-abs-delta-raw",
        "30",
        "--settle-seconds",
        "1.5",
        "--camera-index",
        "3",
        "--visual-output-dir",
        f"{output_dir}/visual",
        "--record-video",
        "--video-fps",
        "12",
    ]


def _relocation_command(*, contact_output_dir: str, direction: str) -> list[str]:
    return [
        "PYTHONPATH=src:.",
        ".venv/bin/python",
        "-B",
        "scripts/real_so100_object_relocation.py",
        "--before",
        f"{contact_output_dir}/visual/before.jpg",
        "--after",
        f"{contact_output_dir}/visual/after.jpg",
        "--target-direction",
        direction,
        "--min-delta-px",
        "40",
        "--output",
        f"{contact_output_dir}/object_relocation_{direction}.json",
    ]


def _relocation_verifier_packet_command(
    *,
    vla_prompt_packet: str,
    contact_output_dir: str,
    direction: str,
) -> list[str]:
    return [
        "PYTHONPATH=src:.",
        ".venv/bin/python",
        "-B",
        "scripts/build_real_so100_relocation_verifier_packet.py",
        "--vla-prompt-packet",
        vla_prompt_packet,
        "--execution-report",
        f"{contact_output_dir}/report.json",
        "--output",
        f"{contact_output_dir}/relocation_verifier_packet_{direction}.json",
        "--relocation-output",
        f"{contact_output_dir}/object_relocation_{direction}.json",
    ]


def _required_success_evidence(*, task_goal: dict[str, Any]) -> list[str]:
    evidence = ["policy_proposal_tokenized", "safety_and_adapter_gates_passed"]
    if task_goal.get("requires_grasp"):
        evidence.append("grasp_outcome_verifier")
    if task_goal.get("requires_transport"):
        evidence.append("object_relocation_verifier")
    evidence.append("observer_video_or_before_after_frames")
    return evidence


def _relocation_required_but_not_passed(*, task_goal: dict[str, Any], verifier: dict[str, Any]) -> bool:
    return bool(task_goal.get("requires_transport")) and verifier.get("relocation_task_success_candidate") is not True


def _camera_index_for_role(mapping: Any, role: str, *, fallback: str) -> str:
    if isinstance(mapping, dict):
        for index, mapped_role in mapping.items():
            if mapped_role == role:
                return str(index)
    return fallback


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the next agentic plan for real SO-100 SmolVLA tasks.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reframe-advice", type=Path)
    parser.add_argument("--agentic-state", type=Path)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration-file", default=DEFAULT_CALIBRATION)
    parser.add_argument("--next-output-dir", default="_workspace/real_so100/checkpoint_26_gate_next")
    parser.add_argument("--contact-output-dir", default="_workspace/real_so100/contact_probe_next")
    parser.add_argument("--vla-prompt-packet", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_agentic_next_plan(
                contract=args.contract,
                output=args.output,
                reframe_advice=args.reframe_advice,
                agentic_state=args.agentic_state,
                port=args.port,
                calibration_file=args.calibration_file,
                next_output_dir=args.next_output_dir,
                contact_output_dir=args.contact_output_dir,
                vla_prompt_packet=args.vla_prompt_packet,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
