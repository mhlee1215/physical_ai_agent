#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_vla_prompt_packet(
    *,
    contract: Path,
    prompt_iteration: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    payload = _load_json(contract)
    iteration = _load_json(prompt_iteration) if prompt_iteration else {}
    policy = payload.get("policy", {})
    policy = _policy_with_iteration_camera_overrides(policy=policy, iteration=iteration)
    task_goal = payload.get("task_goal", {})
    prompt = str(policy.get("instruction") or task_goal.get("instruction") or "")
    target_direction = str(task_goal.get("transport_direction") or "right")
    verifier_frame = _verifier_frame(policy=policy)
    packet = {
        "status": "passed",
        "operation": "real_so100_vla_prompt_packet",
        "purpose": "separate the SmolVLA prompt target from verifier semantics and external setup diagnostics",
        "contract": str(contract),
        "prompt_iteration": str(prompt_iteration) if prompt_iteration else None,
        "vla_prompt": {
            "target": "SmolVLA",
            "text": prompt,
            "model_id": policy.get("model_id"),
            "policy_camera_indexes": policy.get("policy_camera_indexes"),
            "camera_source_mapping": policy.get("camera_source_mapping"),
            "observer_camera_indexes_excluded_from_policy": policy.get("observer_camera_indexes"),
            "send_action_called": bool(policy.get("send_action_called")),
        },
        "agentic_layer_contract": {
            "role": "verify inputs, map proposals through safety/adapter gates, select retries, and verify task outcome",
            "does_not_prompt_operator": True,
            "external_setup_blocked": bool(
                (iteration.get("next_iteration", {}) or {}).get("external_setup_blocker")
            ),
            "autonomous_next_steps": (iteration.get("next_iteration", {}) or {}).get("autonomous_next_steps"),
        },
        "success_verifier": {
            "type": task_goal.get("final_success_verifier") or "object_relocation_image_space",
            "frame": verifier_frame,
            "target_object": task_goal.get("target_object"),
            "target_direction": target_direction,
            "success_predicate": _success_predicate(target_direction=target_direction, frame=verifier_frame),
            "min_delta_px_default": 40.0,
            "do_not_translate_goal_to_fixed_robot_direction": True,
        },
        "coordinate_semantics": {
            "natural_language_goal": f"object moves {target_direction}",
            "robot_motion_is_policy_dependent": True,
            "not_equivalent_to": [
                "robot arm moves left",
                "robot arm moves right",
                "fixed joint sign",
            ],
            "why": (
                "Object displacement, camera image axes, robot base axes, and joint signs can be inverted by mounting "
                "and viewpoint. The agentic layer verifies object displacement instead of hard-coding an arm direction."
            ),
        },
        "external_setup_diagnostics": (iteration.get("next_iteration", {}) or {}).get("external_setup_blocker"),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        packet["manifest_path"] = str(output)
        output.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    return packet


def _verifier_frame(*, policy: dict[str, Any]) -> dict[str, Any]:
    observer_indexes = policy.get("observer_camera_indexes") or []
    if not observer_indexes:
        policy_indexes = [str(item) for item in policy.get("policy_camera_indexes") or ["0", "1"]]
        context_index = "1" if "1" in policy_indexes else policy_indexes[-1]
        return {
            "name": "policy_context_image_frame",
            "camera_indexes": [context_index],
            "primary_camera_index": context_index,
            "axis": "image_x",
            "positive_direction": "right",
            "role": (
                "Temporary observer-off mode: use the wide policy context camera for no-actuation feedback only; "
                "do not claim physical task success without observer/debug evidence."
            ),
            "task_success_claim_allowed": False,
        }
    return {
        "name": "observer_image_frame",
        "camera_indexes": observer_indexes,
        "primary_camera_index": str(observer_indexes[0]) if observer_indexes else "3",
        "axis": "image_x",
        "positive_direction": "right",
        "role": "Codex/debug observer verifies task outcome; not a SmolVLA policy input",
    }


def _policy_with_iteration_camera_overrides(*, policy: dict[str, Any], iteration: dict[str, Any]) -> dict[str, Any]:
    camera_contract = iteration.get("camera_contract") or {}
    if camera_contract.get("observer_camera_status") != "temporarily_unavailable":
        return policy
    updated = dict(policy)
    updated["observer_camera_indexes"] = []
    updated["observer_camera_status"] = "temporarily_unavailable"
    if camera_contract.get("smolvla_policy_inputs"):
        updated["policy_camera_indexes"] = camera_contract.get("smolvla_policy_inputs")
    if camera_contract.get("camera_source_mapping"):
        updated["camera_source_mapping"] = camera_contract.get("camera_source_mapping")
    return updated


def _success_predicate(*, target_direction: str, frame: dict[str, Any]) -> str:
    axis = str(frame.get("axis") or "image_x")
    if target_direction == "right":
        return f"after_object_center.{axis} - before_object_center.{axis} >= min_delta_px"
    if target_direction == "left":
        return f"before_object_center.{axis} - after_object_center.{axis} >= min_delta_px"
    if target_direction == "down":
        return "after_object_center.image_y - before_object_center.image_y >= min_delta_px"
    if target_direction == "up":
        return "before_object_center.image_y - after_object_center.image_y >= min_delta_px"
    return "task_specific_success_predicate_required"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real SO-100 SmolVLA prompt packet with verifier semantics.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--prompt-iteration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_vla_prompt_packet(
                contract=args.contract,
                prompt_iteration=args.prompt_iteration,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
