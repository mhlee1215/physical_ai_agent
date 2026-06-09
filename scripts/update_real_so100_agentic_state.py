#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POLICY_TARGET_REPLACEMENTS = {
    "observation_repair_policy": "policy_input_quality_gate",
}

CONSTRAINT_REPLACEMENTS = {
    "observation_repair_before_contact": "external_setup_ready_before_contact",
}

LOOP_STAGE_REPLACEMENTS = {
    "observation_repair": "external_setup_blocked",
}


def update_agentic_state(
    *,
    analysis: Path,
    state: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    analysis_payload = _load_json(analysis)
    current = _normalize_existing_state(_load_json(state)) if state.exists() else _initial_state()
    if any(item.get("analysis") == str(analysis) for item in current.get("iterations", [])):
        current["next_loop_hint"] = _normalize_loop_hint(analysis_payload.get("loop_continuation", {}))
        current["status"] = "passed"
        destination = output or state
        destination.parent.mkdir(parents=True, exist_ok=True)
        current["manifest_path"] = str(destination)
        destination.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        return current
    iteration = {
        "analysis": str(analysis),
        "task": analysis_payload.get("task"),
        "stage": _normalize_stage(analysis_payload.get("stage")),
        "gate_status": analysis_payload.get("gate_status"),
        "physical_robot_motion": analysis_payload.get("physical_robot_motion"),
        "failure_modes": [item.get("type") for item in analysis_payload.get("failure_modes", [])],
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    current["iterations"].append(iteration)
    current["failure_memory"] = _merge_failure_memory(
        current.get("failure_memory", {}),
        analysis_payload.get("failure_modes", []),
    )
    current["policy_updates"] = _merge_policy_updates(
        current.get("policy_updates", []),
        analysis_payload.get("agentic_layer_improvements", []),
    )
    current["active_constraints"] = _active_constraints(current)
    current["next_loop_hint"] = _normalize_loop_hint(analysis_payload.get("loop_continuation", {}))
    current["updated_at"] = iteration["created_at"]
    current["status"] = "passed"
    destination = output or state
    destination.parent.mkdir(parents=True, exist_ok=True)
    current["manifest_path"] = str(destination)
    destination.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    return current


def normalize_agentic_state(
    *,
    state: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    current = _normalize_existing_state(_load_json(state) if state.exists() else _initial_state())
    current = _dedupe_and_rebuild_from_iterations(current)
    destination = output or state
    current["status"] = "passed"
    current["manifest_path"] = str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    return current


def _initial_state() -> dict[str, Any]:
    return {
        "status": "passed",
        "operation": "real_so100_agentic_state",
        "iterations": [],
        "failure_memory": {},
        "policy_updates": [],
        "active_constraints": [],
        "next_loop_hint": {},
    }


def _merge_failure_memory(existing: dict[str, Any], modes: list[dict[str, Any]]) -> dict[str, Any]:
    memory = dict(existing)
    for mode in modes:
        mode_type = str(mode.get("type"))
        row = dict(memory.get(mode_type, {"count": 0, "examples": []}))
        row["count"] = int(row.get("count", 0)) + 1
        examples = list(row.get("examples", []))
        examples.append({key: mode.get(key) for key in ("camera", "clipped_sides", "evidence", "required_verifier") if key in mode})
        row["examples"] = examples[-5:]
        memory[mode_type] = row
    return memory


def _merge_policy_updates(existing: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {}
    for item in existing:
        normalized = _normalize_policy_update(item)
        merged[str(normalized.get("target"))] = normalized
    for update in updates:
        update = _normalize_policy_update(update)
        target = str(update.get("target"))
        row = dict(merged.get(target, {"target": target, "count": 0}))
        row["count"] = int(row.get("count", 0)) + 1
        row["change"] = update.get("change")
        row["generalization"] = update.get("generalization")
        if update.get("current_advice") is not None:
            row["latest_advice"] = update.get("current_advice")
        merged[target] = row
    return sorted(merged.values(), key=lambda item: str(item.get("target")))


def _active_constraints(state: dict[str, Any]) -> list[str]:
    constraints = set()
    failures = state.get("failure_memory", {})
    if failures.get("jaw_object_framing_not_ready", {}).get("count", 0):
        constraints.add("external_setup_ready_before_contact")
    if failures.get("adapter_semantics_not_executable", {}).get("count", 0):
        constraints.add("smolvla_proposal_only_until_adapter_validated")
    if failures.get("previous_contact_failed_stationary_object", {}).get("count", 0):
        constraints.add("no_repeat_gripper_close_same_pose")
    if failures.get("task_success_not_verified", {}).get("count", 0):
        constraints.add("relocation_verifier_required_for_transport_success")
    return sorted(constraints)


def _normalize_existing_state(state: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(state)
    normalized["active_constraints"] = _normalize_constraints(normalized.get("active_constraints", []))
    normalized["policy_updates"] = _merge_policy_updates(normalized.get("policy_updates", []), [])
    normalized["next_loop_hint"] = _normalize_loop_hint(normalized.get("next_loop_hint", {}))
    normalized["iterations"] = [_normalize_iteration(item) for item in normalized.get("iterations", [])]
    return normalized


def _dedupe_and_rebuild_from_iterations(state: dict[str, Any]) -> dict[str, Any]:
    deduped = []
    seen = set()
    for iteration in state.get("iterations", []):
        key = iteration.get("analysis")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(iteration)
    rebuilt = dict(state)
    rebuilt["iterations"] = deduped
    failure_memory: dict[str, Any] = {}
    policy_updates: list[dict[str, Any]] = []
    for iteration in deduped:
        analysis_path = iteration.get("analysis")
        if not analysis_path:
            continue
        path = Path(str(analysis_path))
        if not path.exists():
            continue
        payload = _load_json(path)
        failure_memory = _merge_failure_memory(failure_memory, payload.get("failure_modes", []))
        policy_updates = _merge_policy_updates(policy_updates, payload.get("agentic_layer_improvements", []))
    if deduped and failure_memory:
        rebuilt["failure_memory"] = failure_memory
        rebuilt["policy_updates"] = policy_updates
        rebuilt["active_constraints"] = _active_constraints(rebuilt)
    return rebuilt


def _normalize_policy_update(update: dict[str, Any]) -> dict[str, Any]:
    row = dict(update)
    original_target = str(row.get("target"))
    row["target"] = POLICY_TARGET_REPLACEMENTS.get(original_target, original_target)
    row.pop("latest_advice", None)
    row.pop("current_advice", None)
    if row["target"] == "policy_input_quality_gate" and (
        original_target == "observation_repair_policy" or "reframe advice" in str(row.get("change", ""))
    ):
        row["change"] = "block VLA prompting/contact execution when required policy cameras do not provide usable target evidence"
        row["generalization"] = (
            "applies to any task where required policy inputs are edge-clipped, missing, stale, "
            "or otherwise outside the agent action space"
        )
    return row


def _normalize_constraints(constraints: list[Any]) -> list[str]:
    return sorted({CONSTRAINT_REPLACEMENTS.get(str(item), str(item)) for item in constraints})


def _normalize_loop_hint(hint: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(hint)
    normalized["next_stage"] = _normalize_stage(normalized.get("next_stage"))
    if normalized.get("next_stage") == "external_setup_blocked":
        normalized["external_setup_blocked"] = True
        normalized["repeat_prompt_after_repair"] = False
        if normalized.get("next_step_type") == "manual_or_fixture_reframe":
            normalized["next_step_type"] = None
    return normalized


def _normalize_iteration(iteration: dict[str, Any]) -> dict[str, Any]:
    row = dict(iteration)
    row["stage"] = _normalize_stage(row.get("stage"))
    return row


def _normalize_stage(stage: Any) -> Any:
    return LOOP_STAGE_REPLACEMENTS.get(str(stage), stage) if stage is not None else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Update persistent real SO-100 agentic state from log analysis.")
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--normalize-only", action="store_true")
    args = parser.parse_args()
    if args.normalize_only:
        print(
            json.dumps(
                normalize_agentic_state(state=args.state, output=args.output),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.analysis is None:
        raise SystemExit("--analysis is required unless --normalize-only is set")
    print(
        json.dumps(
            update_agentic_state(analysis=args.analysis, state=args.state, output=args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
