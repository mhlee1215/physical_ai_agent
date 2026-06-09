#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_development_lane(
    *,
    loop_state: Path,
    launch_packet: Path,
    policy_feedback: Path | None,
    candidate_memory: Path | None,
    output: Path,
    markdown: Path | None = None,
) -> dict[str, Any]:
    state = _load_json(loop_state)
    launch = _load_json(launch_packet)
    feedback = _load_json(policy_feedback) if policy_feedback else {}
    memory = _load_json(candidate_memory) if candidate_memory else {}
    blockers = _blockers(state=state, launch=launch)
    execution_lane = _execution_lane(state=state, launch=launch, blocked=bool(blockers))
    policy_lane = _policy_lane(feedback=feedback, memory=memory)
    report = {
        "operation": "real_so100_agentic_development_lane",
        "status": "passed" if not blockers else "blocked",
        "purpose": "separate camera-3-gated physical execution from camera-0/1-only agentic-layer development",
        "source_loop_state": str(loop_state),
        "source_launch_packet": str(launch_packet),
        "source_policy_feedback": str(policy_feedback) if policy_feedback else None,
        "source_candidate_memory": str(candidate_memory) if candidate_memory else None,
        "camera_contract": {
            "policy_camera_indexes": [0, 1],
            "policy_camera_roles": _policy_roles(feedback),
            "observer_camera_indexes": [],
            "observer_camera_status": "temporarily_unavailable",
            "camera_3_is_policy_input": False,
        },
        "execution_lane": execution_lane,
        "policy_camera_development_lane": policy_lane,
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "prompt_mutation_allowed": bool(policy_lane.get("prompt_mutation_allowed")),
        "blockers": blockers,
        "next_agentic_layer_step": _next_step(blocked=bool(blockers), policy_lane=policy_lane),
        "generalization_contract": [
            "represent task goals as object, relation, and success verifier rather than robot-frame joint directions",
            "keep VLA policy inputs separate from observer/debug evidence",
            "allow replaceable LLM/VLM policy-camera reasoning while deterministic code only gates safety and evidence",
            "use candidate memory to avoid repeatedly evaluating regressed prompts",
            "require task-level grasp and relocation verifiers before success accounting",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path = markdown or output.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report["json_path"] = str(output)
    report["markdown_path"] = str(md_path)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    execution = report.get("execution_lane") or {}
    policy = report.get("policy_camera_development_lane") or {}
    lines = [
        "# Real SO-100 Agentic Development Lane",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Execution lane: `{execution.get('state')}`",
        f"- Policy-camera development lane: `{policy.get('state')}`",
        f"- Prompt mutation allowed: `{report.get('prompt_mutation_allowed')}`",
        f"- Physical robot motion: `{report.get('physical_robot_motion')}`",
        f"- Task success claim allowed: `{report.get('task_success_claim_allowed')}`",
        "",
        "## Execution Lane",
        "",
        f"- Next command: `{execution.get('next_command_name')}`",
        f"- Allowed when: `{execution.get('allowed_when')}`",
        f"- Reason: {execution.get('reason')}",
        "",
        "## Policy-Camera Development Lane",
        "",
    ]
    for action in policy.get("allowed_no_actuation_actions", []):
        lines.append(f"- `{action.get('type')}`: {action.get('reason')}")
    if policy.get("blocked_no_actuation_actions"):
        lines.extend(["", "## Blocked No-Actuation Actions", ""])
        for action in policy["blocked_no_actuation_actions"]:
            lines.append(f"- `{action.get('type')}`: {action.get('reason')}")
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in report["blockers"])
    lines.append("")
    return "\n".join(lines)


def _execution_lane(*, state: dict[str, Any], launch: dict[str, Any], blocked: bool) -> dict[str, Any]:
    allowed = state.get("allowed_next_actions") or []
    first_allowed = allowed[0] if allowed else {}
    if blocked:
        return {
            "state": "blocked",
            "next_command_name": None,
            "allowed_when": None,
            "reason": "Loop state or launch packet is not ready for even the live-readonly observer refresh.",
        }
    return {
        "state": "waiting_for_observer_camera_3",
        "next_command_name": launch.get("launch_command_name"),
        "launch_command": launch.get("launch_command"),
        "allowed_when": launch.get("launch_command_allowed_when"),
        "physical_robot_motion": False,
        "reason": first_allowed.get("reason") or "Observer camera 3 is required before reopening the execution gate.",
    }


def _policy_lane(*, feedback: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    regression = memory.get("regression_from_best") or {}
    best = memory.get("best_candidate") or {}
    feedback_ok = feedback.get("status") == "passed"
    prompt_mutation_allowed = False
    allowed = [
        {
            "type": "capture_or_refresh_policy_cameras_0_1_only",
            "reason": "Policy-camera evidence can improve the replaceable VLM reasoning layer without physical movement.",
        },
        {
            "type": "build_policy_camera_task_state_packet",
            "reason": "Convert camera 0/1 frames into object visibility, gripper/jaw context, and task-relation fields for the in-loop agent.",
        },
        {
            "type": "evaluate_best_historical_prompt_against_current_policy_state",
            "reason": "Reuse the best historical prompt family and evaluate it no-actuation before considering any new prompt.",
        },
    ]
    blocked = [
        {
            "type": "physical_execution_from_policy_cameras_only",
            "reason": "Camera 3 observer evidence is required for any physical movement.",
        },
        {
            "type": "task_success_claim_from_policy_cameras_only",
            "reason": "Task success requires observer-backed grasp and relocation verifier evidence.",
        },
    ]
    if regression.get("is_regression") is True:
        blocked.append(
            {
                "type": "mutate_prompt_from_latest_policy_camera_feedback",
                "reason": f"Latest prompt regressed from best by penalty_delta={regression.get('penalty_delta')}.",
            }
        )
    elif feedback_ok:
        prompt_mutation_allowed = True
        allowed.append(
            {
                "type": "run_no_actuation_prompt_variant_sweep",
                "reason": "Policy-camera feedback is valid and candidate memory does not show a latest regression.",
            }
        )
    return {
        "state": "active_no_actuation",
        "feedback_available": feedback_ok,
        "prompt_mutation_allowed": prompt_mutation_allowed,
        "best_candidate": {
            "source_report": best.get("source_report"),
            "candidate_index": best.get("candidate_index"),
            "prompt": best.get("prompt"),
            "penalty_score": (best.get("score") or {}).get("penalty_score"),
            "ready_for_execution": bool((best.get("score") or {}).get("ready_for_execution")),
        },
        "regression_from_best": regression,
        "allowed_no_actuation_actions": allowed,
        "blocked_no_actuation_actions": blocked,
    }


def _blockers(*, state: dict[str, Any], launch: dict[str, Any]) -> list[str]:
    blockers = []
    if state.get("operation") != "real_so100_agentic_loop_state":
        blockers.append(f"Loop state operation is {state.get('operation')!r}.")
    if state.get("status") != "passed":
        blockers.append(f"Loop state status is {state.get('status')!r}.")
    if launch.get("operation") != "real_so100_first_command_launch_packet":
        blockers.append(f"Launch packet operation is {launch.get('operation')!r}.")
    if launch.get("status") != "passed":
        blockers.append(f"Launch packet status is {launch.get('status')!r}.")
    if launch.get("launch_command_name") != "observer_return_refresh_live_readonly":
        blockers.append("Launch packet does not contain the live-readonly observer refresh command.")
    if launch.get("not_a_physical_execution_authorization") is not True:
        blockers.append("Launch packet does not explicitly reject physical-execution authorization.")
    if bool(launch.get("physical_robot_motion")):
        blockers.append("Launch packet records physical robot motion.")
    return blockers


def _next_step(*, blocked: bool, policy_lane: dict[str, Any]) -> dict[str, str]:
    if blocked:
        return {
            "type": "repair_loop_state_or_launch_packet",
            "reason": "The agentic development lane could not trust the current execution state.",
        }
    return {
        "type": "build_policy_camera_task_state_packet",
        "reason": "Continue agentic-layer development with cameras 0 and 1 while the execution lane waits for camera 3.",
    }


def _policy_roles(feedback: dict[str, Any]) -> dict[str, Any]:
    return (feedback.get("camera_contract") or {}).get("policy_camera_roles") or {"0": "wrist_cam", "1": "egocentric_cam"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the real SO-100 agentic development lane state.")
    parser.add_argument("--loop-state", type=Path, required=True)
    parser.add_argument("--launch-packet", type=Path, required=True)
    parser.add_argument("--policy-feedback", type=Path)
    parser.add_argument("--candidate-memory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_development_lane(
                loop_state=args.loop_state,
                launch_packet=args.launch_packet,
                policy_feedback=args.policy_feedback,
                candidate_memory=args.candidate_memory,
                output=args.output,
                markdown=args.markdown,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
