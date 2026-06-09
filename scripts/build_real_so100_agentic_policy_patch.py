#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LEGACY_TARGETS = {
    "observation_repair_policy": "policy_input_quality_gate",
}

LEGACY_CONSTRAINTS = {
    "observation_repair_before_contact": "external_setup_ready_before_contact",
}


def build_agentic_policy_patch(
    *,
    analysis: Path,
    agentic_state: Path,
    prompt_iteration: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    analysis_payload = _load_json(analysis)
    state_payload = _load_json(agentic_state)
    iteration_payload = _load_json(prompt_iteration) if prompt_iteration else {}
    failure_modes = [str(item.get("type")) for item in analysis_payload.get("failure_modes", [])]
    normalized_updates, normalizations = _normalized_policy_updates(state_payload, analysis_payload)
    constraints = _normalized_constraints(state_payload.get("active_constraints", []))
    external_setup_blocked = bool(
        analysis_payload.get("loop_continuation", {}).get("external_setup_blocked")
        or (iteration_payload.get("next_iteration", {}) or {}).get("stage") == "external_setup_blocked"
    )
    patch = {
        "status": "passed",
        "operation": "real_so100_agentic_policy_patch",
        "purpose": "convert loop failures into reusable, normalized agentic-layer rules",
        "analysis": str(analysis),
        "agentic_state": str(agentic_state),
        "prompt_iteration": str(prompt_iteration) if prompt_iteration else None,
        "task": analysis_payload.get("task") or iteration_payload.get("task"),
        "stage": "external_setup_blocked" if external_setup_blocked else analysis_payload.get("stage"),
        "failure_modes": failure_modes,
        "normalized_active_constraints": constraints,
        "normalized_policy_updates": normalized_updates,
        "legacy_normalizations": normalizations,
        "prompt_contract": {
            "vla_prompt_target": "SmolVLA",
            "does_not_prompt_operator": True,
            "operator_setup_changes_are_not_agent_actions": True,
            "vla_prompt_allowed": not external_setup_blocked,
        },
        "rules": _rules(
            updates=normalized_updates,
            constraints=constraints,
            failure_modes=failure_modes,
            external_setup_blocked=external_setup_blocked,
        ),
        "success_accounting": {
            "task_success_claim_allowed": False,
            "required_before_success": [
                "policy input quality gate passes",
                "safety and action adapter gates pass",
                "grasp outcome verifier passes for grasp tasks",
                "observer-frame object relocation verifier passes for transport tasks",
            ],
            "note": "Verifier success can drive retry decisions; final task success still requires task-level evidence.",
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        patch["manifest_path"] = str(output)
        output.write_text(json.dumps(patch, indent=2, sort_keys=True), encoding="utf-8")
    return patch


def _normalized_policy_updates(
    state_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[str, dict[str, Any]] = {}
    normalizations: list[dict[str, Any]] = []
    for item in list(state_payload.get("policy_updates", [])) + list(analysis_payload.get("agentic_layer_improvements", [])):
        target = str(item.get("target"))
        normalized_target = LEGACY_TARGETS.get(target, target)
        row = dict(item)
        row["target"] = normalized_target
        row.pop("latest_advice", None)
        row.pop("current_advice", None)
        if target != normalized_target:
            normalizations.append(
                {
                    "field": "policy_updates.target",
                    "from": target,
                    "to": normalized_target,
                    "reason": "camera/object setup diagnostics are not autonomous operator prompts",
                }
            )
        if _contains_operator_instruction(item):
            normalizations.append(
                {
                    "field": "policy_updates.advice",
                    "from": "operator_instruction",
                    "to": "external_setup_diagnostic",
                    "reason": "agentic loop prompts SmolVLA, not the human operator",
                }
            )
        rows[normalized_target] = _merge_update(rows.get(normalized_target), row)
    return sorted(rows.values(), key=lambda row: str(row.get("target"))), _dedupe_normalizations(normalizations)


def _merge_update(existing: dict[str, Any] | None, update: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        merged = dict(update)
        merged["count"] = int(merged.get("count", 1) or 1)
        return merged
    merged = dict(existing)
    merged["count"] = max(int(existing.get("count", 0)), int(update.get("count", 1) or 1))
    merged["change"] = update.get("change") or existing.get("change")
    merged["generalization"] = update.get("generalization") or existing.get("generalization")
    return merged


def _normalized_constraints(constraints: list[Any]) -> list[str]:
    return sorted({LEGACY_CONSTRAINTS.get(str(item), str(item)) for item in constraints})


def _rules(
    *,
    updates: list[dict[str, Any]],
    constraints: list[str],
    failure_modes: list[str],
    external_setup_blocked: bool,
) -> list[dict[str, Any]]:
    targets = {str(update.get("target")) for update in updates}
    rules = []
    if "policy_input_quality_gate" in targets or "external_setup_ready_before_contact" in constraints:
        rules.append(
            {
                "id": "policy_input_quality_gate",
                "trigger": "required policy input is missing, stale, edge-clipped, or outside robot action space",
                "action": "block SmolVLA prompting and contact execution until a no-actuation gate passes",
                "agent_actionable": False if external_setup_blocked else None,
                "generalizes_to": "any task with camera-role-specific policy inputs",
            }
        )
    if "action_adapter_gate" in targets:
        rules.append(
            {
                "id": "action_adapter_gate",
                "trigger": "raw lightweight VLA action units are not validated against robot-native signs and scales",
                "action": "treat SmolVLA output as a proposal, not an executable command",
                "generalizes_to": "any lightweight policy adapter",
            }
        )
    if "retry_policy" in targets:
        rules.append(
            {
                "id": "retry_policy",
                "trigger": "object remains stationary after a gripper close/contact attempt",
                "action": "forbid repeating the same close from the same pose; require new observation or pregrasp evidence",
                "generalizes_to": "grasp and contact-probe tasks",
            }
        )
    if "success_criteria" in targets or "task_success_not_verified" in failure_modes:
        rules.append(
            {
                "id": "success_criteria",
                "trigger": "task includes transport or spatial relocation",
                "action": "require observer-frame before/after relocation verifier before any success claim",
                "generalizes_to": "directional object movement tasks",
            }
        )
    return rules


def _contains_operator_instruction(item: dict[str, Any]) -> bool:
    text = json.dumps(item, sort_keys=True)
    return "operator_instruction" in text


def _dedupe_normalizations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = (item.get("field"), item.get("from"), item.get("to"))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized real SO-100 agentic policy patch.")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--agentic-state", type=Path, required=True)
    parser.add_argument("--prompt-iteration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_agentic_policy_patch(
                analysis=args.analysis,
                agentic_state=args.agentic_state,
                prompt_iteration=args.prompt_iteration,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
