#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def route_agentic_feedback(
    *,
    feedback_reports: list[Path],
    output: Path,
    markdown: Path | None = None,
) -> dict[str, Any]:
    feedback_items = [_feedback_item(path) for path in feedback_reports]
    route = _select_route(feedback_items)
    result = {
        "operation": "real_so100_agentic_feedback_router",
        "status": "passed" if feedback_items else "blocked",
        "purpose": "route agentic-layer feedback into prompt, execution, verifier, or success-accounting work",
        "source_feedback_reports": [str(path) for path in feedback_reports],
        "policy_camera_indexes": _first_non_null([item.get("policy_camera_indexes") for item in feedback_items]) or [0, 1],
        "observer_camera_indexes": _first_non_null([item.get("observer_camera_indexes") for item in feedback_items]) or [],
        "observer_camera_status": _first_non_null([item.get("observer_camera_status") for item in feedback_items])
        or "temporarily_unavailable",
        "send_action_called": any(bool(item.get("send_action_called")) for item in feedback_items),
        "policy_actions_executed": any(bool(item.get("policy_actions_executed")) for item in feedback_items),
        "physical_robot_motion": any(bool(item.get("physical_robot_motion")) for item in feedback_items),
        "task_success_claim_allowed": any(bool(item.get("task_success_claim_allowed")) for item in feedback_items),
        "feedback_items": feedback_items,
        "selected_route": route,
        "next_agentic_layer_step": _next_step(route),
        "guardrails": [
            "camera indexes 0 and 1 are policy inputs",
            "camera 3 is observer/debug only and must not be fed to SmolVLA",
            "do not mutate SmolVLA prompts from execution-preflight blockers",
            "do not claim task success without task-level verifier evidence",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_path = markdown or output.with_suffix(".md")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    result["json_path"] = str(output)
    result["markdown_path"] = str(md_path)
    return result


def render_markdown(report: dict[str, Any]) -> str:
    route = report.get("selected_route", {})
    lines = [
        "# Real SO-100 Agentic Feedback Router",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Route: `{route.get('type')}`",
        f"- Reason: {route.get('reason')}",
        f"- Policy cameras: `{report.get('policy_camera_indexes')}`",
        f"- Observer cameras: `{report.get('observer_camera_indexes')}` (`{report.get('observer_camera_status')}`)",
        f"- Physical robot motion: `{report.get('physical_robot_motion', False)}`",
        f"- Task success claim allowed: `{report.get('task_success_claim_allowed', False)}`",
        "",
        "## Feedback Items",
        "",
    ]
    for item in report.get("feedback_items", []):
        lines.append(
            f"- `{item.get('operation')}` from `{item.get('path')}` -> `{item.get('recommended_route')}`"
        )
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


def _feedback_item(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    operation = payload.get("operation")
    item = {
        "path": str(path),
        "operation": operation,
        "status": payload.get("status"),
        "failure_modes": payload.get("failure_modes") or _source_failure_modes(payload),
        "next_step_type": (payload.get("next_agentic_layer_step") or {}).get("type"),
        "policy_camera_indexes": _policy_camera_indexes(payload),
        "observer_camera_indexes": _observer_camera_indexes(payload),
        "observer_camera_status": _observer_camera_status(payload),
        "send_action_called": bool(_nested_get(payload, ["execution_outcome", "send_action_called"], payload.get("send_action_called"))),
        "policy_actions_executed": bool(
            _nested_get(payload, ["execution_outcome", "policy_actions_executed"], payload.get("policy_actions_executed"))
        ),
        "physical_robot_motion": bool(
            _nested_get(payload, ["execution_outcome", "physical_robot_motion"], payload.get("physical_robot_motion"))
        ),
        "task_success_claim_allowed": bool(payload.get("task_success_claim_allowed")),
        "prompt_mutation_allowed": payload.get("prompt_mutation_allowed"),
    }
    item["recommended_route"] = _recommended_route(item, payload)
    return item


def _recommended_route(item: dict[str, Any], payload: dict[str, Any]) -> str:
    if item.get("task_success_claim_allowed"):
        return "success_accounting"
    if payload.get("operation") == "real_so100_agentic_feedback_router":
        selected = payload.get("selected_route") or {}
        if selected.get("type") == "await_observer_camera_3":
            return "await_observer_camera_3"
    if payload.get("operation") == "real_so100_execution_preflight_runbook_audit":
        if payload.get("status") == "passed" and payload.get("failed_check_count") == 0:
            return "await_observer_camera_3"
        return "fix_execution_preflight_runbook"
    if payload.get("operation") == "real_so100_agentic_candidate_memory":
        regression = payload.get("regression_from_best") or {}
        if regression.get("is_regression") is True:
            return "preserve_best_historical_candidate"
    if payload.get("operation") == "real_so100_agentic_proposal_sweep":
        gate = payload.get("feedback_gate") or {}
        if gate.get("prompt_mutation_allowed") is False:
            return "execution_preflight"
    modes = set(item.get("failure_modes") or [])
    if "observer_or_live_readback_preflight_incomplete" in modes or "execution_packet_not_ready" in modes:
        return "execution_preflight"
    if item.get("policy_actions_executed") and (
        "task_success_not_verified" in modes or "grasp_outcome_not_verified" in modes
    ):
        return "task_verifier"
    if item.get("prompt_mutation_allowed") is True:
        return "prompt_mutation"
    if item.get("next_step_type") in {
        "rerun_smolvla_proposal_with_best_prompt_family",
        "reuse_best_historical_prompt_family",
        "continue_from_latest_best_prompt_family",
    }:
        return "prompt_mutation"
    return "hold"


def _select_route(items: list[dict[str, Any]]) -> dict[str, Any]:
    routes = [item.get("recommended_route") for item in items]
    if not items:
        return {
            "type": "collect_feedback",
            "reason": "No feedback reports were supplied.",
            "prompt_mutation_allowed": False,
        }
    if "success_accounting" in routes:
        return {
            "type": "success_accounting",
            "reason": "At least one feedback report has task-success evidence; prepare repeatable observer-backed accounting.",
            "prompt_mutation_allowed": False,
        }
    if "task_verifier" in routes:
        return {
            "type": "run_task_verifiers",
            "reason": "A policy action executed, but grasp or relocation evidence is missing or incomplete.",
            "prompt_mutation_allowed": False,
        }
    if "fix_execution_preflight_runbook" in routes:
        return {
            "type": "fix_execution_preflight_runbook",
            "reason": "The no-actuation execution preflight runbook audit failed; fix the runbook before any prompt mutation.",
            "prompt_mutation_allowed": False,
        }
    if "await_observer_camera_3" in routes:
        return {
            "type": "await_observer_camera_3",
            "reason": "The no-actuation execution preflight runbook passed audit; preserve the candidate and wait for observer camera 3 evidence.",
            "prompt_mutation_allowed": False,
        }
    if "execution_preflight" in routes:
        return {
            "type": "resolve_execution_preflight",
            "reason": "The latest blocker is observer/live-readback execution preflight, not SmolVLA prompt quality.",
            "prompt_mutation_allowed": False,
        }
    if "preserve_best_historical_candidate" in routes:
        return {
            "type": "preserve_best_historical_candidate",
            "reason": "The latest policy-camera prompt regressed against candidate memory; preserve the best historical candidate instead of mutating prompts.",
            "prompt_mutation_allowed": False,
        }
    if "prompt_mutation" in routes:
        return {
            "type": "mutate_smolvla_prompt_or_plan",
            "reason": "Feedback points to policy/prompt quality rather than execution preflight or verifier gaps.",
            "prompt_mutation_allowed": True,
        }
    return {
        "type": "hold_current_candidate",
        "reason": "Feedback does not justify prompt mutation or physical execution.",
        "prompt_mutation_allowed": False,
    }


def _next_step(route: dict[str, Any]) -> dict[str, Any]:
    route_type = route.get("type")
    if route_type == "resolve_execution_preflight":
        return {
            "type": "rerun_observer_return_refresh_live_readonly_when_camera_3_available",
            "reason": "Preserve the current candidate and reopen only the observer/live-readback gate.",
        }
    if route_type == "fix_execution_preflight_runbook":
        return {
            "type": "repair_and_reaudit_execution_preflight_runbook",
            "reason": "Do not change the SmolVLA prompt while the no-actuation execution preflight runbook is failing audit.",
        }
    if route_type == "await_observer_camera_3":
        return {
            "type": "wait_for_camera_3_then_run_live_readonly_refresh",
            "reason": "Camera 3 is observer-only and currently required before physical execution or task-success evidence.",
        }
    if route_type == "preserve_best_historical_candidate":
        return {
            "type": "preserve_best_transition_candidate_until_observer_gate",
            "reason": "Do not spend the next loop on the regressed prompt; keep the best historical candidate and reopen only observer/live-readback gates.",
        }
    if route_type == "run_task_verifiers":
        return {
            "type": "run_grasp_and_relocation_verifiers",
            "reason": "Task feedback must come from before/after verifier evidence before prompt mutation.",
        }
    if route_type == "mutate_smolvla_prompt_or_plan":
        return {
            "type": "run_no_actuation_proposal_sweep",
            "reason": "Prompt mutation is allowed; generate and gate the next SmolVLA candidate without physical execution first.",
        }
    if route_type == "success_accounting":
        return {
            "type": "prepare_repeatability_check",
            "reason": "A task-success candidate needs repeatable observer-backed evidence before final reporting.",
        }
    return {
        "type": "collect_more_feedback",
        "reason": "Current feedback is insufficient for action.",
    }


def _source_failure_modes(payload: dict[str, Any]) -> list[str]:
    gate = payload.get("feedback_gate") or {}
    return gate.get("source_failure_modes") or []


def _policy_camera_indexes(payload: dict[str, Any]) -> list[int] | None:
    camera_contract = payload.get("camera_contract") or {}
    value = camera_contract["policy_camera_indexes"] if "policy_camera_indexes" in camera_contract else payload.get("policy_camera_indexes")
    if value is None:
        return None
    return [int(index) for index in value]


def _observer_camera_indexes(payload: dict[str, Any]) -> list[int] | None:
    camera_contract = payload.get("camera_contract") or {}
    value = camera_contract["observer_camera_indexes"] if "observer_camera_indexes" in camera_contract else payload.get("observer_camera_indexes")
    if value is None:
        return None
    return [int(index) for index in value]


def _observer_camera_status(payload: dict[str, Any]) -> str | None:
    return (payload.get("camera_contract") or {}).get("observer_camera_status") or payload.get("observer_camera_status")


def _nested_get(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _first_non_null(values: list[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Route real SO-100 agentic feedback into the next loop step.")
    parser.add_argument("--feedback-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            route_agentic_feedback(
                feedback_reports=args.feedback_report,
                output=args.output,
                markdown=args.markdown,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
