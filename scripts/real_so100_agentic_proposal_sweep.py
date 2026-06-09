#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from physical_ai_agent.policies.smolvla_adapter import DEFAULT_SMOLVLA_MODEL_ID
from scripts.inspect_smolvla_action_metadata import DEFAULT_LOCAL_CONFIG
from scripts.real_so100_execute_chunk import execute_action_chunk
from scripts.real_so100_smolvla_dry import run_dry_inference


DEFAULT_PROMPTS = [
    "Pick up the green Android figure and move it to the right.",
    "Move the gripper to a comfortable pre-grasp pose above the green Android figure. Keep the green figure visible in both cameras.",
    "Center the green Android figure in the wrist camera without grasping it.",
    "Approach the green Android figure slowly from the current pose and stop before closing the gripper.",
]

LOWER_PREGRASP_PROMPTS = [
    "Move the gripper toward a low, relaxed pre-grasp pose near the green Android figure. Keep the elbow bent and avoid lifting the arm high.",
    "Approach the green Android figure from the side at table height. Stop in a nearby pre-grasp pose before closing the gripper.",
    "Lower the gripper slightly and align it with the green Android figure without reaching upward or extending the elbow far.",
    "Make a small conservative approach toward the green Android figure, keeping the shoulder low and the wrist within a comfortable range.",
]

PROJECTION_AWARE_PROMPTS = [
    "Move only a short distance toward the green Android figure while staying inside a compact reachable posture: shoulder low, elbow tucked, wrist neutral, gripper open.",
    "Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, extending the elbow, or flexing the wrist upward.",
    "Keep the arm close to its current reachable workspace and make a small table-height alignment toward the green Android figure before any grasp.",
    "Approach the green Android figure with a low side pose, neutral wrist, and open gripper; stop early if the arm would need to stretch high or far.",
]

RESIDUAL_DISTORTION_PROMPTS = [
    "Hold the current table-height approach and make only a tiny side alignment toward the green Android figure; keep the wrist straight, shoulder from rising, elbow midrange, and gripper open.",
    "Do not lift or reach upward. Keep the wrist neutral and the shoulder low while preparing a nearby side pre-grasp for the green Android figure.",
    "Use a very short in-place pre-grasp adjustment near the green Android figure: preserve current height, avoid wrist flexion, avoid shoulder lift, and keep the gripper open.",
    "Stabilize in a reachable pre-grasp staging pose beside the green Android figure with neutral wrist, low shoulder, and no grasp closure.",
]

MEMORY_REFINE_FALLBACK_PROMPTS = [
    "Set up a conservative side pre-grasp near the green Android figure without raising the shoulder, extending the elbow, or flexing the wrist upward.",
    "Set up the same conservative side pre-grasp near the green Android figure, but stop earlier and keep the gripper open.",
    "Prepare a conservative side pre-grasp near the green Android figure using a shorter reach and no grasp closure.",
    "Move toward the same conservative side pre-grasp near the green Android figure while keeping the wrist neutral and stopping before the arm stretches.",
]

MEMORY_SAMPLE_COUNT = 5


def run_proposal_sweep(
    *,
    episode: Path,
    frame_index: int,
    output_dir: Path,
    prompts: list[str],
    model_id: str,
    local_files_only: bool,
    wrist_camera_index: str,
    egocentric_camera_index: str,
    action_steps: int,
    metadata_config: Path,
    action_stats: Path,
    calibration: Path,
    port: str,
    action_semantics: str,
    gripper_semantics: str,
    command_units: str,
    feedback_report: Path | None = None,
    prompt_profile: str = "default",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feedback = load_feedback_report(feedback_report)
    if _feedback_blocks_prompt_mutation(feedback) and not prompts:
        return _write_prompt_mutation_blocked_report(
            output_dir=output_dir,
            episode=episode,
            frame_index=frame_index,
            model_id=model_id,
            wrist_camera_index=wrist_camera_index,
            egocentric_camera_index=egocentric_camera_index,
            action_steps=action_steps,
            metadata_config=metadata_config,
            action_stats=action_stats,
            calibration=calibration,
            action_semantics=action_semantics,
            gripper_semantics=gripper_semantics,
            command_units=command_units,
            feedback_report=feedback_report,
            feedback=feedback,
            prompt_profile=prompt_profile,
        )
    prompts = resolve_prompts(
        explicit_prompts=prompts,
        prompt_profile=prompt_profile,
        feedback=feedback,
    )
    candidates = []
    for index, prompt in enumerate(prompts, start=1):
        candidate_dir = output_dir / f"candidate_{index:02d}"
        smolvla_dir = candidate_dir / "smolvla"
        execute_report_path = candidate_dir / "execute_dry_gate.json"
        dry_report = run_dry_inference(
            episode=episode,
            frame_index=frame_index,
            output_dir=smolvla_dir,
            instruction=prompt,
            model_id=model_id,
            local_files_only=local_files_only,
            wrist_camera_index=wrist_camera_index,
            egocentric_camera_index=egocentric_camera_index,
            observer_camera_indexes=[],
            action_steps=action_steps,
            calibration=calibration,
            state_units="raw_ticks",
        )
        action_path = Path(dry_report["action_path"])
        execute_report: dict[str, Any] | None = None
        if dry_report.get("status") == "passed" and action_path.exists():
            execute_report = execute_action_chunk(
                port=port,
                action=action_path,
                output=execute_report_path,
                calibration=calibration,
                execute=False,
                human_confirmed=False,
                experimental_adapter_confirmed=False,
                action_steps=action_steps,
                delta_scale_raw_ticks=2.0,
                max_abs_delta_raw=4.0,
                step_settle_seconds=0.0,
                camera_index=None,
                visual_output_dir=None,
                record_video=False,
                video_fps=12.0,
                metadata_config=metadata_config,
                action_stats=action_stats,
                action_semantics=action_semantics,
                gripper_semantics=gripper_semantics,
                command_units=command_units,
                confirm_so100_joint_order=True,
            )
        else:
            execute_report = {
                "status": "blocked",
                "send_action_called": False,
                "policy_actions_executed": False,
                "dry_plan": {"ready_for_execution": False, "blockers": [dry_report.get("blocker", "SmolVLA dry run did not pass.")]},
                "blockers": [dry_report.get("blocker", "SmolVLA dry run did not pass.")],
            }
            execute_report_path.parent.mkdir(parents=True, exist_ok=True)
            execute_report_path.write_text(json.dumps(execute_report, indent=2, sort_keys=True), encoding="utf-8")

        score = score_execute_dry_gate(execute_report)
        candidates.append(
            {
                "candidate_index": index,
                "prompt": prompt,
                "candidate_dir": str(candidate_dir),
                "smolvla_dry_report_path": str(smolvla_dir / "smolvla_dry_report.json"),
                "action_path": str(action_path),
                "execute_gate_path": str(execute_report_path),
                "dry_status": dry_report.get("status"),
                "execute_status": execute_report.get("status"),
                "send_action_called": bool(execute_report.get("send_action_called")),
                "policy_actions_executed": bool(execute_report.get("policy_actions_executed")),
                "physical_robot_motion": False,
                "score": score,
            }
        )

    ranked = sorted(candidates, key=_candidate_sort_key)
    best = ranked[0] if ranked else None
    report = {
        "operation": "real_so100_agentic_proposal_sweep",
        "status": "passed" if candidates else "blocked",
        "episode": str(episode),
        "frame_index": frame_index,
        "model_id": model_id,
        "policy_camera_indexes": [wrist_camera_index, egocentric_camera_index],
        "observer_camera_indexes": [],
        "observer_camera_status": "temporarily_unavailable",
        "camera_3_status": "off",
        "camera_3_role": "codex_observer_only_not_available_for_this_sweep",
        "action_steps": action_steps,
        "metadata_config": str(metadata_config),
        "action_stats": str(action_stats),
        "calibration": str(calibration),
        "action_semantics": action_semantics,
        "gripper_semantics": gripper_semantics,
        "command_units": command_units,
        "prompt_profile": prompt_profile,
        "feedback_report": str(feedback_report) if feedback_report else None,
        "feedback_summary": summarize_feedback(feedback),
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "task_success_claim_blocker": "camera 3 observer evidence is temporarily unavailable and all candidates are no-actuation dry gates.",
        "candidates": candidates,
        "ranked_candidates": ranked,
        "best_candidate": best,
        "next_agentic_layer_step": _next_step(best),
    }
    json_path = output_dir / "proposal_sweep_report.json"
    md_path = output_dir / "proposal_sweep_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(md_path)
    return report


def _write_prompt_mutation_blocked_report(
    *,
    output_dir: Path,
    episode: Path,
    frame_index: int,
    model_id: str,
    wrist_camera_index: str,
    egocentric_camera_index: str,
    action_steps: int,
    metadata_config: Path,
    action_stats: Path,
    calibration: Path,
    action_semantics: str,
    gripper_semantics: str,
    command_units: str,
    feedback_report: Path | None,
    feedback: dict[str, Any] | None,
    prompt_profile: str,
) -> dict[str, Any]:
    next_step = (feedback or {}).get("next_agentic_layer_step") or {}
    report = {
        "operation": "real_so100_agentic_proposal_sweep",
        "status": "passed",
        "episode": str(episode),
        "frame_index": frame_index,
        "model_id": model_id,
        "policy_camera_indexes": [wrist_camera_index, egocentric_camera_index],
        "observer_camera_indexes": [],
        "observer_camera_status": "temporarily_unavailable",
        "camera_3_status": "off",
        "camera_3_role": "codex_observer_only_not_available_for_this_sweep",
        "action_steps": action_steps,
        "metadata_config": str(metadata_config),
        "action_stats": str(action_stats),
        "calibration": str(calibration),
        "action_semantics": action_semantics,
        "gripper_semantics": gripper_semantics,
        "command_units": command_units,
        "prompt_profile": prompt_profile,
        "feedback_report": str(feedback_report) if feedback_report else None,
        "feedback_summary": summarize_feedback(feedback),
        "feedback_gate": {
            "prompt_mutation_allowed": False,
            "source_operation": (feedback or {}).get("operation"),
            "source_failure_modes": (feedback or {}).get("failure_modes") or [],
            "source_next_step_type": next_step.get("type"),
        },
        "actuation_enabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "task_success_claim_allowed": False,
        "task_success_claim_blocker": "camera 3 observer evidence is temporarily unavailable and no policy action executed.",
        "candidates": [],
        "ranked_candidates": [],
        "best_candidate": None,
        "next_agentic_layer_step": {
            "type": "preserve_transition_candidate_until_observer_live_readback_gate",
            "reason": next_step.get("reason")
            or "Execution feedback explicitly blocks prompt mutation; preserve the current transition candidate and resolve observer/live-readback gates.",
            "source_feedback_next_step": next_step.get("type"),
        },
    }
    json_path = output_dir / "proposal_sweep_report.json"
    md_path = output_dir / "proposal_sweep_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(md_path)
    return report


def score_execute_dry_gate(report: dict[str, Any]) -> dict[str, Any]:
    dry_plan = report.get("dry_plan") or {}
    step_plans = dry_plan.get("step_plans") or []
    violations = []
    for step in step_plans:
        step_index = int(step.get("step_index", 0))
        for target in step.get("joint_targets") or []:
            raw = _optional_float(target.get("target_raw"))
            range_min = _optional_float(target.get("range_min"))
            range_max = _optional_float(target.get("range_max"))
            if raw is None or range_min is None or range_max is None:
                continue
            excess = _range_excess(raw, range_min, range_max)
            if excess > 0:
                violations.append(
                    {
                        "step_index": step_index,
                        "joint": target.get("joint"),
                        "target_raw": raw,
                        "range_min": range_min,
                        "range_max": range_max,
                        "excess_raw_ticks": round(excess, 4),
                    }
                )

    if not step_plans:
        violations.extend(_violations_from_blocker_text(dry_plan.get("blockers") or report.get("blockers") or []))

    total_excess = sum(float(item["excess_raw_ticks"]) for item in violations)
    max_excess = max([float(item["excess_raw_ticks"]) for item in violations] or [0.0])
    ready = bool(dry_plan.get("ready_for_execution")) and not violations
    joint_counts: dict[str, int] = {}
    joint_excess: dict[str, float] = {}
    for violation in violations:
        joint = str(violation.get("joint"))
        joint_counts[joint] = joint_counts.get(joint, 0) + 1
        joint_excess[joint] = round(joint_excess.get(joint, 0.0) + float(violation["excess_raw_ticks"]), 4)
    penalty_score = total_excess + max_excess * 3.0 + len(violations) * 50.0
    return {
        "ready_for_execution": ready,
        "range_violation_count": len(violations),
        "total_range_excess_raw_ticks": round(total_excess, 4),
        "max_range_excess_raw_ticks": round(max_excess, 4),
        "range_penalty_score": round(penalty_score, 4),
        "violation_joint_counts": joint_counts,
        "violation_joint_excess_raw_ticks": joint_excess,
        "top_range_violations": sorted(violations, key=lambda item: float(item["excess_raw_ticks"]), reverse=True)[:8],
        "blocker_count": len(dry_plan.get("blockers") or report.get("blockers") or []),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real SO-100 Agentic SmolVLA Proposal Sweep",
        "",
        f"- Status: `{report['status']}`",
        f"- Policy cameras: `{report['policy_camera_indexes']}`",
        f"- Observer cameras: `{report['observer_camera_indexes']}` (`{report['observer_camera_status']}`)",
        f"- Action steps per candidate: `{report['action_steps']}`",
        f"- Actuation enabled: `{report['actuation_enabled']}`",
        f"- Physical robot motion: `{report['physical_robot_motion']}`",
        f"- Task success claim allowed: `{report['task_success_claim_allowed']}`",
        "",
        "## Ranking",
        "",
    ]
    for candidate in report.get("ranked_candidates") or []:
        score = candidate["score"]
        lines.extend(
            [
                f"### Candidate {candidate['candidate_index']:02d}",
                "",
                f"- Prompt: {candidate['prompt']}",
                f"- Ready for execution: `{score['ready_for_execution']}`",
                f"- Range violations: `{score['range_violation_count']}`",
                f"- Total excess raw ticks: `{score['total_range_excess_raw_ticks']}`",
                f"- Max excess raw ticks: `{score['max_range_excess_raw_ticks']}`",
                f"- Range penalty score: `{score['range_penalty_score']}`",
                f"- Joint violation counts: `{score['violation_joint_counts']}`",
                f"- Execute gate: `{candidate['execute_gate_path']}`",
                "",
            ]
        )
    best = report.get("best_candidate")
    if best:
        lines.extend(
            [
                "## Selected Feedback",
                "",
                f"- Best candidate: `{best['candidate_index']:02d}`",
                f"- Best prompt: {best['prompt']}",
                f"- Next step: `{report['next_agentic_layer_step']['type']}`",
                f"- Reason: {report['next_agentic_layer_step']['reason']}",
                "",
            ]
        )
    return "\n".join(lines)


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float, int, float, float, int]:
    score = candidate["score"]
    ready_rank = 0 if score["ready_for_execution"] else 1
    return (
        ready_rank,
        float(score["range_penalty_score"]),
        int(score["range_violation_count"]),
        float(score["total_range_excess_raw_ticks"]),
        float(score["max_range_excess_raw_ticks"]),
        int(candidate["candidate_index"]),
    )


def _next_step(best: dict[str, Any] | None) -> dict[str, Any]:
    if best is None:
        return {
            "type": "collect_policy_observation",
            "reason": "No SmolVLA proposal candidates were generated.",
        }
    score = best["score"]
    if score["ready_for_execution"]:
        return {
            "type": "hold_for_observer_camera_3_and_human_confirmation",
            "reason": "A candidate passed the no-actuation execution gate, but physical execution remains blocked while camera 3 is off.",
            "selected_prompt": best["prompt"],
        }
    return {
        "type": "rerun_smolvla_proposal_with_best_prompt_family",
        "reason": "No candidate passed the calibrated range gate; use the lowest-penalty prompt family and dominant joint blockers as Pseudo-LLM feedback for the next agentic layer version.",
        "selected_prompt": best["prompt"],
    }


def resolve_prompts(
    *,
    explicit_prompts: list[str],
    prompt_profile: str,
    feedback: dict[str, Any] | None,
) -> list[str]:
    if explicit_prompts:
        return explicit_prompts
    if prompt_profile == "projection_aware":
        return feedback_driven_prompts(feedback, projection_aware=True) or PROJECTION_AWARE_PROMPTS
    if prompt_profile == "residual_distortion":
        return feedback_driven_prompts(feedback, residual_distortion=True) or RESIDUAL_DISTORTION_PROMPTS
    if prompt_profile == "memory_refine":
        return memory_refine_prompts(feedback) or MEMORY_REFINE_FALLBACK_PROMPTS
    if prompt_profile == "memory_sample":
        return memory_sample_prompts(feedback) or MEMORY_REFINE_FALLBACK_PROMPTS
    if prompt_profile == "memory_residual":
        return memory_residual_prompts(feedback) or MEMORY_REFINE_FALLBACK_PROMPTS
    if prompt_profile == "memory_structured":
        return memory_structured_prompts(feedback) or MEMORY_REFINE_FALLBACK_PROMPTS
    if prompt_profile == "policy_camera_feedback":
        return policy_camera_feedback_prompts(feedback) or DEFAULT_PROMPTS[:1]
    if prompt_profile == "lower_pregrasp":
        return feedback_driven_prompts(feedback) or LOWER_PREGRASP_PROMPTS
    if feedback:
        return feedback_driven_prompts(feedback) + DEFAULT_PROMPTS[:1]
    return DEFAULT_PROMPTS


def policy_camera_feedback_prompts(feedback: dict[str, Any] | None) -> list[str]:
    if not feedback:
        return []
    if feedback.get("operation") != "real_so100_policy_camera_pseudo_llm_feedback":
        return []
    pseudo_llm = feedback.get("pseudo_llm_feedback") or {}
    if pseudo_llm.get("does_not_prompt_operator") is not True:
        return []
    prompt = str(pseudo_llm.get("next_smolvla_prompt") or "").strip()
    if not prompt:
        return []
    return [prompt]


def memory_refine_prompts(feedback: dict[str, Any] | None) -> list[str]:
    if not feedback:
        return []
    selected = (
        (feedback.get("next_agentic_layer_step") or {}).get("selected_prompt")
        or (feedback.get("best_candidate") or {}).get("prompt")
    )
    if not selected:
        return []
    return _dedupe_prompts(
        [
            selected,
            f"{selected} Stop earlier than the previous attempt and keep the gripper open.",
            f"{selected} Use the shortest useful reach and avoid any upward stretch.",
            f"{selected} Keep the same side approach, but make the pre-grasp staging motion smaller.",
            "Set up the same conservative side pre-grasp near the green Android figure, but stop earlier and keep the gripper open.",
        ]
    )


def memory_sample_prompts(feedback: dict[str, Any] | None, *, sample_count: int = MEMORY_SAMPLE_COUNT) -> list[str]:
    if not feedback:
        return []
    selected = (
        (feedback.get("next_agentic_layer_step") or {}).get("selected_prompt")
        or (feedback.get("best_candidate") or {}).get("prompt")
    )
    if not selected:
        return []
    return [selected] * max(1, int(sample_count))


def memory_residual_prompts(feedback: dict[str, Any] | None) -> list[str]:
    if not feedback:
        return []
    selected = (
        (feedback.get("next_agentic_layer_step") or {}).get("selected_prompt")
        or (feedback.get("best_candidate") or {}).get("prompt")
    )
    if not selected:
        return []
    summary = summarize_feedback(feedback) or {}
    dominant = summary.get("dominant_blocker_joints") or []
    prompts = [selected]
    if "wrist_flex" in dominant and "shoulder_lift" in dominant:
        prompts.extend(_append_prompt_clause(selected, clause) for clause in [
            "Preserve the same side pre-grasp, but keep the wrist neutral and keep the upper arm low.",
            "Make the next staging chunk smaller; avoid wrist flexion and avoid shoulder lift.",
            "Hold table height with a neutral wrist, no upward reach, and no gripper closure.",
        ])
    if "elbow_flex" in dominant:
        prompts.extend(_append_prompt_clause(selected, clause) for clause in [
            "Keep the elbow comfortably bent near midrange and avoid extending past the reachable side pre-grasp.",
            "Use a shorter reach with elbow midrange, shoulder low, wrist neutral, and gripper open.",
        ])
    if "gripper" in dominant:
        prompts.append(_append_prompt_clause(selected, "Keep the gripper open and unchanged while improving the pre-grasp pose."))
    prompts.append(
        "Use the same conservative side pre-grasp target, prioritizing a low shoulder, neutral wrist, open gripper, and a short reachable motion."
    )
    return _dedupe_prompts(prompts)


def memory_structured_prompts(feedback: dict[str, Any] | None) -> list[str]:
    if not feedback:
        return []
    selected = (
        (feedback.get("next_agentic_layer_step") or {}).get("selected_prompt")
        or (feedback.get("best_candidate") or {}).get("prompt")
    )
    if not selected:
        return []
    summary = summarize_feedback(feedback) or {}
    dominant = summary.get("dominant_blocker_joints") or []
    hints = _structured_joint_hints(dominant)
    anchor = _compact_prompt_anchor(selected)
    prompts = [
        selected,
        f"Goal: {anchor} Constraints: {hints}. Move as one short 10-step pre-grasp chunk.",
        f"Goal: {anchor} Constraints: neutral wrist; low upper arm; elbow midrange; gripper open. Stop at a nearby pre-grasp pose.",
        f"Goal: {anchor} Constraints: keep all joints near the current reachable posture; avoid high reach; keep the gripper open.",
        f"Goal: {anchor} Constraints: smallest useful side alignment; no grasp closure; no upward stretch.",
    ]
    return _dedupe_prompts(prompts)


def _structured_joint_hints(dominant: list[str]) -> str:
    hint_by_joint = {
        "elbow_flex": "elbow midrange",
        "shoulder_lift": "low upper arm",
        "wrist_flex": "neutral wrist",
        "gripper": "gripper open",
        "shoulder_pan": "short side alignment",
        "wrist_roll": "stable wrist roll",
    }
    hints = [hint_by_joint[joint] for joint in dominant if joint in hint_by_joint]
    for fallback in ["neutral wrist", "low upper arm", "elbow midrange", "gripper open"]:
        if fallback not in hints:
            hints.append(fallback)
    return "; ".join(hints[:5])


def _compact_prompt_anchor(prompt: str) -> str:
    first_sentence = prompt.strip().split(".")[0].strip()
    if not first_sentence:
        return prompt.strip()
    return first_sentence


def feedback_driven_prompts(
    feedback: dict[str, Any] | None,
    *,
    projection_aware: bool = False,
    residual_distortion: bool = False,
) -> list[str]:
    if not feedback:
        return []
    summary = summarize_feedback(feedback)
    dominant = summary.get("dominant_blocker_joints") or []
    prompts = _base_prompts_for_profile(
        projection_aware=projection_aware,
        residual_distortion=residual_distortion,
    )
    if residual_distortion and "wrist_flex" in dominant and "shoulder_lift" in dominant:
        prompts.insert(
            0,
            "Make a minimal reachable staging adjustment toward the green Android figure: wrist straight, shoulder not rising, elbow near midrange, gripper open, no lifting.",
        )
    elif "shoulder_lift" in dominant and "elbow_flex" in dominant and "wrist_flex" in dominant:
        prompts.insert(
            0,
            "Make the smallest reachable side pre-grasp adjustment toward the green Android figure: keep shoulder low, elbow tucked, wrist neutral, and gripper open.",
        )
    if not projection_aware and not residual_distortion:
        if "shoulder_lift" in dominant and "elbow_flex" in dominant:
            prompts.insert(
                0,
                "Move toward the green Android figure using a low compact posture: keep the shoulder low, keep the elbow bent, and stop before grasping.",
            )
        elif "elbow_flex" in dominant:
            prompts.insert(
                0,
                "Move the gripper near the green Android figure with the elbow kept comfortably bent, using a small side approach before grasping.",
            )
        elif "shoulder_lift" in dominant:
            prompts.insert(
                0,
                "Approach the green Android figure without lifting the shoulder high; keep the gripper low and stop at a pre-grasp pose.",
            )
    return _dedupe_prompts(prompts)


def _append_prompt_clause(prompt: str, clause: str) -> str:
    normalized_prompt = prompt.rstrip()
    normalized_clause = clause.strip()
    if normalized_clause.rstrip(".") in normalized_prompt.rstrip("."):
        return normalized_prompt
    return f"{normalized_prompt} {normalized_clause}"


def _base_prompts_for_profile(*, projection_aware: bool, residual_distortion: bool) -> list[str]:
    if residual_distortion:
        return list(RESIDUAL_DISTORTION_PROMPTS)
    if projection_aware:
        return list(PROJECTION_AWARE_PROMPTS)
    return list(LOWER_PREGRASP_PROMPTS)


def load_feedback_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _feedback_blocks_prompt_mutation(feedback: dict[str, Any] | None) -> bool:
    if not feedback:
        return False
    if feedback.get("operation") != "real_so100_transition_execution_feedback":
        return False
    return feedback.get("prompt_mutation_allowed") is False


def summarize_feedback(feedback: dict[str, Any] | None) -> dict[str, Any] | None:
    if not feedback:
        return None
    joint_counts: dict[str, int] = {}
    joint_excess: dict[str, float] = {}
    for candidate in feedback.get("ranked_candidates") or feedback.get("candidates") or []:
        score = candidate.get("score") or {}
        for joint, count in (score.get("violation_joint_counts") or score.get("joint_violation_counts") or {}).items():
            joint_counts[str(joint)] = joint_counts.get(str(joint), 0) + int(count)
        for violation in score.get("top_range_violations") or []:
            joint = str(violation.get("joint"))
            joint_counts[joint] = joint_counts.get(joint, 0) + 1
            joint_excess[joint] = round(joint_excess.get(joint, 0.0) + float(violation.get("excess_raw_ticks", 0.0)), 4)
        for joint, excess in (score.get("violation_joint_excess_raw_ticks") or score.get("joint_excess_raw_ticks") or {}).items():
            joint_excess[str(joint)] = round(joint_excess.get(str(joint), 0.0) + float(excess), 4)
        projection = candidate.get("projection") or {}
        for joint, distortion in (projection.get("joint_distortion") or {}).items():
            count = int(distortion.get("violation_count", 0) or 0)
            excess = float(distortion.get("total_raw_distortion", 0.0) or 0.0)
            if count:
                joint_counts[str(joint)] = joint_counts.get(str(joint), 0) + count
            if excess:
                joint_excess[str(joint)] = round(joint_excess.get(str(joint), 0.0) + excess, 4)
    joint_names = set(joint_counts) | set(joint_excess)
    dominant = sorted(joint_names, key=lambda joint: (joint_counts.get(joint, 0), joint_excess.get(joint, 0.0)), reverse=True)
    return {
        "source_status": feedback.get("status"),
        "source_path": feedback.get("json_path"),
        "dominant_blocker_joints": dominant[:4],
        "joint_violation_counts": joint_counts,
        "joint_excess_raw_ticks": joint_excess,
    }


def _dedupe_prompts(prompts: list[str]) -> list[str]:
    seen = set()
    result = []
    for prompt in prompts:
        if prompt in seen:
            continue
        seen.add(prompt)
        result.append(prompt)
    return result


def _range_excess(value: float, range_min: float, range_max: float) -> float:
    if value < range_min:
        return range_min - value
    if value > range_max:
        return value - range_max
    return 0.0


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _violations_from_blocker_text(blockers: list[Any]) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"Step (?P<step>\d+) joint (?P<joint>[a-z_]+) maps to raw target (?P<raw>-?\d+(?:\.\d+)?) "
        r"outside calibrated range \[(?P<min>-?\d+(?:\.\d+)?), (?P<max>-?\d+(?:\.\d+)?)\]"
    )
    violations = []
    for blocker in blockers:
        match = pattern.search(str(blocker))
        if not match:
            continue
        raw = float(match.group("raw"))
        range_min = float(match.group("min"))
        range_max = float(match.group("max"))
        violations.append(
            {
                "step_index": int(match.group("step")),
                "joint": match.group("joint"),
                "target_raw": raw,
                "range_min": range_min,
                "range_max": range_max,
                "excess_raw_ticks": round(_range_excess(raw, range_min, range_max), 4),
            }
        )
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-actuation SmolVLA prompt proposal sweep for real SO-100.")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--model-id", default=DEFAULT_SMOLVLA_MODEL_ID)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--wrist-camera-index", default="0")
    parser.add_argument("--egocentric-camera-index", default="1")
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--metadata-config", type=Path, default=DEFAULT_LOCAL_CONFIG)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("_workspace/real_so100/calibration/so100_local.json"))
    parser.add_argument("--port", default="/dev/cu.usbmodem5AE60824791")
    parser.add_argument("--action-semantics", default="absolute_joint_position", choices=["absolute_joint_position", "joint_delta"])
    parser.add_argument("--gripper-semantics", default="higher_raw_opens", choices=["higher_raw_opens", "higher_raw_closes"])
    parser.add_argument("--command-units", default="lerobot_so100_position", choices=["feetech_raw_ticks", "lerobot_so100_position"])
    parser.add_argument("--feedback-report", type=Path)
    parser.add_argument(
        "--prompt-profile",
        default="default",
        choices=[
            "default",
            "lower_pregrasp",
            "projection_aware",
            "residual_distortion",
            "memory_refine",
            "memory_sample",
            "memory_residual",
            "memory_structured",
            "policy_camera_feedback",
        ],
    )
    args = parser.parse_args()

    print(
        json.dumps(
            run_proposal_sweep(
                episode=args.episode,
                frame_index=args.frame_index,
                output_dir=args.output_dir,
                prompts=args.prompt,
                model_id=args.model_id,
                local_files_only=not args.allow_download,
                wrist_camera_index=args.wrist_camera_index,
                egocentric_camera_index=args.egocentric_camera_index,
                action_steps=args.action_steps,
                metadata_config=args.metadata_config,
                action_stats=args.action_stats,
                calibration=args.calibration,
                port=args.port,
                action_semantics=args.action_semantics,
                gripper_semantics=args.gripper_semantics,
                command_units=args.command_units,
                feedback_report=args.feedback_report,
                prompt_profile=args.prompt_profile,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
