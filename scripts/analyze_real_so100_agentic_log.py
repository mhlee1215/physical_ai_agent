#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def analyze_agentic_log(
    *,
    refresh_manifest: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    refresh = _load_json(refresh_manifest)
    contract = _load_json(Path(refresh["contract"]))
    next_plan = _load_json(Path(refresh["next_plan"]))
    advice = _load_json(Path(refresh["reframe_advice"]))
    failure_modes = _failure_modes(refresh=refresh, contract=contract, next_plan=next_plan, advice=advice)
    improvements = _improvements(failure_modes=failure_modes, next_plan=next_plan, advice=advice)
    result = {
        "status": "passed",
        "operation": "real_so100_agentic_log_analysis",
        "refresh_manifest": str(refresh_manifest),
        "task": refresh.get("task"),
        "stage": next_plan.get("stage") or refresh.get("next_stage"),
        "gate_status": refresh.get("gate_status"),
        "physical_robot_motion": refresh.get("physical_robot_motion"),
        "agentic_decision": refresh.get("agentic_decision"),
        "failure_modes": failure_modes,
        "agentic_layer_improvements": improvements,
        "loop_continuation": _loop_continuation(next_plan=next_plan, failure_modes=failure_modes),
        "notes": [
            "This analyzes logs from one loop iteration and proposes agentic-layer updates.",
            "It does not execute robot actions and does not claim task success.",
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result["manifest_path"] = str(output)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _failure_modes(
    *,
    refresh: dict[str, Any],
    contract: dict[str, Any],
    next_plan: dict[str, Any],
    advice: dict[str, Any],
) -> list[dict[str, Any]]:
    modes = []
    verifier = contract.get("agentic_layer", {}).get("verifier_contract", {})
    if refresh.get("gate_status") == "blocked":
        modes.append(
            {
                "type": "observation_gate_blocked",
                "stage": next_plan.get("stage"),
                "evidence": refresh.get("gate_manifest"),
                "reason": [item.get("reason") for item in advice.get("actions", [])],
            }
        )
    if verifier.get("jaw_gate_status") == "blocked":
        modes.append(
            {
                "type": "jaw_object_framing_not_ready",
                "camera": advice.get("jaw_camera"),
                "clipped_sides": advice.get("jaw_object_clipped_sides", []),
                "candidate": advice.get("jaw_object_candidate"),
            }
        )
    if contract.get("adapter_and_safety", {}).get("command_plan_ready_for_execution") is not True:
        modes.append(
            {
                "type": "adapter_semantics_not_executable",
                "command_plan": contract.get("adapter_and_safety", {}).get("command_plan_path"),
            }
        )
    if verifier.get("last_grasp_outcome") == "grasp_failed_object_stationary":
        modes.append(
            {
                "type": "previous_contact_failed_stationary_object",
                "evidence": contract.get("evidence", {}).get("grasp_outcome"),
            }
        )
    if verifier.get("relocation_task_success_candidate") is not True:
        modes.append(
            {
                "type": "task_success_not_verified",
                "required_verifier": contract.get("task_goal", {}).get("final_success_verifier"),
            }
        )
    return modes


def _improvements(*, failure_modes: list[dict[str, Any]], next_plan: dict[str, Any], advice: dict[str, Any]) -> list[dict[str, Any]]:
    improvements = []
    mode_types = {item.get("type") for item in failure_modes}
    if "jaw_object_framing_not_ready" in mode_types:
        improvements.append(
            {
                "target": "policy_input_quality_gate",
                "change": "block VLA prompting/contact execution when required policy cameras do not provide usable target evidence",
                "current_advice": advice.get("actions", []),
                "generalization": "applies to any task where required policy inputs are edge-clipped, missing, stale, or otherwise outside the agent action space",
            }
        )
    if "adapter_semantics_not_executable" in mode_types:
        improvements.append(
            {
                "target": "action_adapter_gate",
                "change": "keep SmolVLA outputs as proposals until sign and scale are validated on non-contact movements",
                "generalization": "prevents raw lightweight-policy actions from being treated as robot-native commands",
            }
        )
    if "previous_contact_failed_stationary_object" in mode_types:
        improvements.append(
            {
                "target": "retry_policy",
                "change": "forbid repeated gripper closes from the same pose after stationary-object failure",
                "generalization": "turns verifier failures into retry-state memory rather than repeating policy-only actions",
            }
        )
    if "task_success_not_verified" in mode_types:
        improvements.append(
            {
                "target": "success_criteria",
                "change": "require object relocation verifier after grasp/contact attempts for transport tasks",
                "generalization": "separates grasp success from task-level transport success",
            }
        )
    if not improvements:
        improvements.append(
            {
                "target": "loop",
                "change": "continue with next planned step",
                "next_stage": next_plan.get("stage"),
            }
        )
    return improvements


def _loop_continuation(*, next_plan: dict[str, Any], failure_modes: list[dict[str, Any]]) -> dict[str, Any]:
    next_steps = next_plan.get("next_steps", [])
    external_setup_blocked = next_plan.get("stage") == "external_setup_blocked"
    return {
        "next_stage": next_plan.get("stage"),
        "physical_robot_motion_next": bool(next_plan.get("physical_robot_motion")),
        "next_step_type": next_steps[0].get("type") if next_steps else None,
        "blocked_by_external_reframe": any(item.get("type") == "jaw_object_framing_not_ready" for item in failure_modes),
        "repeat_prompt_after_repair": not external_setup_blocked,
        "external_setup_blocked": external_setup_blocked,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one real SO-100 agentic loop log.")
    parser.add_argument("--refresh-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_agentic_log(refresh_manifest=args.refresh_manifest, output=args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
