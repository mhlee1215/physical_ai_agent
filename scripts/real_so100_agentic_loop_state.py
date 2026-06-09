#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_agentic_loop_state(
    *,
    router_report: Path,
    candidate_memory: Path | None,
    observation_manifest: Path | None,
    output: Path,
    markdown: Path | None = None,
) -> dict[str, Any]:
    router = _load_json(router_report)
    memory = _load_json(candidate_memory) if candidate_memory else {}
    observation = _load_json(observation_manifest) if observation_manifest else {}
    route_type = (router.get("selected_route") or {}).get("type")
    best = memory.get("best_candidate") or {}
    state = {
        "operation": "real_so100_agentic_loop_state",
        "status": "passed" if router.get("status") == "passed" else "blocked",
        "purpose": "single machine-readable state for the real SO-100 SmolVLA agentic loop",
        "source_reports": {
            "router_report": str(router_report),
            "candidate_memory": str(candidate_memory) if candidate_memory else None,
            "observation_manifest": str(observation_manifest) if observation_manifest else None,
        },
        "task": observation.get("task") or "Pick up the green Android figure and move it to the right.",
        "camera_contract": {
            "policy_camera_indexes": router.get("policy_camera_indexes") or [0, 1],
            "observer_camera_indexes": router.get("observer_camera_indexes") or [],
            "observer_camera_status": router.get("observer_camera_status") or "temporarily_unavailable",
            "camera_3_is_policy_input": False,
        },
        "execution_flags": {
            "send_action_called": bool(router.get("send_action_called")),
            "policy_actions_executed": bool(router.get("policy_actions_executed")),
            "physical_robot_motion": bool(router.get("physical_robot_motion")),
            "task_success_claim_allowed": bool(router.get("task_success_claim_allowed")),
        },
        "selected_route": router.get("selected_route") or {},
        "best_historical_candidate": _best_candidate_summary(best),
        "latest_regression": memory.get("regression_from_best") or {},
        "allowed_next_actions": _allowed_next_actions(route_type=route_type, router=router),
        "blocked_actions": _blocked_actions(route_type=route_type, memory=memory),
        "success_requirements": [
            "observer camera 3 before/during/after evidence for any physical movement",
            "live readback-regenerated transition gate passes",
            "human and workspace-clear confirmation before motor writes",
            "grasp outcome verifier after contact",
            "object relocation verifier shows the object moved right in image space",
        ],
        "generalization_notes": [
            "The loop state is task-parameterized by object, transport direction, camera contract, and verifier requirements.",
            "Policy-camera-only feedback may create candidate prompts, but regression memory can suppress further prompt mutation.",
            "Physical task success is separate from internal route or proposal success.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    md_path = markdown or output.with_suffix(".md")
    md_path.write_text(render_markdown(state), encoding="utf-8")
    state["manifest_path"] = str(output)
    state["markdown_path"] = str(md_path)
    return state


def render_markdown(state: dict[str, Any]) -> str:
    route = state.get("selected_route") or {}
    best = state.get("best_historical_candidate") or {}
    lines = [
        "# Real SO-100 Agentic Loop State",
        "",
        f"- Status: `{state.get('status')}`",
        f"- Route: `{route.get('type')}`",
        f"- Next allowed action: `{(state.get('allowed_next_actions') or [{}])[0].get('type')}`",
        f"- Physical robot motion: `{state.get('execution_flags', {}).get('physical_robot_motion')}`",
        f"- Task success claim allowed: `{state.get('execution_flags', {}).get('task_success_claim_allowed')}`",
        "",
        "## Best Historical Candidate",
        "",
        f"- Source: `{best.get('source_report')}`",
        f"- Candidate: `{best.get('candidate_index')}`",
        f"- Penalty: `{best.get('penalty_score')}`",
        f"- Prompt: {best.get('prompt')}",
        "",
        "## Blocked Actions",
        "",
    ]
    lines.extend(f"- `{item.get('type')}`: {item.get('reason')}" for item in state.get("blocked_actions", []))
    lines.extend(["", "## Success Requirements", ""])
    lines.extend(f"- {item}" for item in state.get("success_requirements", []))
    lines.append("")
    return "\n".join(lines)


def _best_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    score = candidate.get("score") or {}
    return {
        "source_report": candidate.get("source_report"),
        "candidate_index": candidate.get("candidate_index"),
        "prompt": candidate.get("prompt"),
        "action_path": candidate.get("action_path"),
        "execute_gate_path": candidate.get("execute_gate_path"),
        "penalty_score": score.get("penalty_score"),
        "range_violation_count": score.get("range_violation_count"),
        "ready_for_execution": bool(score.get("ready_for_execution")),
    }


def _allowed_next_actions(*, route_type: str | None, router: dict[str, Any]) -> list[dict[str, Any]]:
    next_step = router.get("next_agentic_layer_step") or {}
    if route_type == "await_observer_camera_3":
        return [
            {
                "type": "wait_for_camera_3_then_run_live_readonly_refresh",
                "physical_robot_motion": False,
                "reason": next_step.get("reason"),
                "requires_observer_camera_3": True,
                "preserves_best_candidate": True,
            }
        ]
    return [
        {
            "type": next_step.get("type") or "collect_more_feedback",
            "physical_robot_motion": False,
            "reason": next_step.get("reason", "Follow router next step."),
        }
    ]


def _blocked_actions(*, route_type: str | None, memory: dict[str, Any]) -> list[dict[str, Any]]:
    blocked = [
        {
            "type": "physical_execution",
            "reason": "observer camera 3 evidence is unavailable or the observer/live-readback gate has not been reopened",
        },
        {
            "type": "task_success_claim",
            "reason": "no observer-backed grasp and relocation verifier evidence exists for the green-object transport task",
        },
    ]
    regression = memory.get("regression_from_best") or {}
    if regression.get("is_regression") is True:
        blocked.append(
            {
                "type": "rerun_regressed_policy_camera_prompt",
                "reason": f"latest prompt regressed from best by penalty_delta={regression.get('penalty_delta')}",
            }
        )
    if route_type == "await_observer_camera_3":
        blocked.append(
            {
                "type": "prompt_mutation_before_observer_refresh",
                "reason": "the current router preserves the best transition candidate and waits for camera 3",
            }
        )
    return blocked


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the current real SO-100 SmolVLA agentic loop state.")
    parser.add_argument("--router-report", type=Path, required=True)
    parser.add_argument("--candidate-memory", type=Path)
    parser.add_argument("--observation-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_agentic_loop_state(
                router_report=args.router_report,
                candidate_memory=args.candidate_memory,
                observation_manifest=args.observation_manifest,
                output=args.output,
                markdown=args.markdown,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
