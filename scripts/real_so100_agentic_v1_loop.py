#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.real_so100_micro_step import run_micro_step
from scripts.real_so100_observe import record_observation


REQUIRED_DECISION_FIELDS = [
    "backend",
    "agentic_layer_version",
    "task",
    "scene_interpretation",
    "selected_subgoal",
    "smolvla_prompt",
    "limited_step",
    "expected_observer_evidence",
]


def run_agentic_v1_loop(
    *,
    decision: Path,
    output_dir: Path,
    port: str,
    calibration_file: Path | None,
    execute: bool,
    human_confirmed: bool,
    policy_camera_indexes: list[int],
    observer_camera_index: int,
    prepost_duration_seconds: float = 0.6,
    prepost_fps: float = 2.0,
    settle_seconds: float = 0.75,
    video_fps: float = 12.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_payload = _load_json(decision)
    validation = validate_pseudo_llm_decision(decision_payload)
    manifest: dict[str, Any] = {
        "status": "blocked" if validation["blockers"] else "ready",
        "operation": "real_so100_agentic_v1_loop",
        "agentic_layer_version": decision_payload.get("agentic_layer_version"),
        "decision": str(decision),
        "decision_backend": decision_payload.get("backend"),
        "task": decision_payload.get("task"),
        "selected_subgoal": decision_payload.get("selected_subgoal"),
        "smolvla_prompt": decision_payload.get("smolvla_prompt"),
        "semantic_decision_source": "pseudo_llm_or_replaceable_on_device_llm",
        "runner_semantics": "records and gates the LLM decision; does not choose semantic subgoals by rule",
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "policy_camera_indexes": policy_camera_indexes,
        "observer_camera_index": observer_camera_index,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "validation": validation,
        "output_dir": str(output_dir),
    }
    if validation["blockers"]:
        return _write_manifest(output_dir, manifest)
    if execute and not human_confirmed:
        manifest["validation"]["blockers"].append("human_confirmed is required for physical execution")
        manifest["status"] = "blocked"
        return _write_manifest(output_dir, manifest)

    camera_indexes = _dedupe(policy_camera_indexes + [observer_camera_index])
    roles = {str(index): f"policy_camera_{index}" for index in policy_camera_indexes}
    roles[str(observer_camera_index)] = "codex_observer"
    task = str(decision_payload["task"])

    pre_observe = record_observation(
        port=port,
        camera_indexes=camera_indexes,
        output_dir=output_dir / "pre_observe",
        duration_seconds=prepost_duration_seconds,
        fps=prepost_fps,
        task=task,
        calibration_file=calibration_file,
        camera_roles=roles,
        policy_camera_indexes=policy_camera_indexes,
        observer_camera_indexes=[observer_camera_index],
    )
    manifest["pre_observe"] = pre_observe
    if not pre_observe.get("ok"):
        manifest["status"] = "failed"
        manifest["error"] = "pre_observe failed"
        return _write_manifest(output_dir, manifest)

    step = decision_payload["limited_step"]
    micro_report = run_micro_step(
        port=port,
        command_plan=None,
        joint=str(step["joint"]),
        output=output_dir / "micro_step" / "report.json",
        execute=execute,
        human_confirmed=human_confirmed,
        non_contact_confirmed=bool(step.get("non_contact_confirmed")),
        contact_ok_for_gripper=bool(step.get("contact_ok_for_gripper")),
        max_abs_delta_raw=float(step["max_abs_delta_raw"]),
        manual_delta_raw=float(step["manual_delta_raw"]),
        settle_seconds=settle_seconds,
        camera_index=observer_camera_index,
        visual_output_dir=output_dir / "observer_camera_3",
        record_video=True,
        video_fps=video_fps,
    )
    manifest["micro_step"] = micro_report
    manifest["send_action_called"] = bool(micro_report.get("send_action_called"))
    manifest["policy_actions_executed"] = bool(micro_report.get("policy_actions_executed"))
    manifest["physical_robot_motion"] = bool(micro_report.get("send_action_called"))

    post_observe = record_observation(
        port=port,
        camera_indexes=camera_indexes,
        output_dir=output_dir / "post_observe",
        duration_seconds=prepost_duration_seconds,
        fps=prepost_fps,
        task=task,
        calibration_file=calibration_file,
        camera_roles=roles,
        policy_camera_indexes=policy_camera_indexes,
        observer_camera_indexes=[observer_camera_index],
    )
    manifest["post_observe"] = post_observe
    feedback = build_feedback_packet(
        decision=decision_payload,
        micro_report=micro_report,
        pre_observe=pre_observe,
        post_observe=post_observe,
        output=output_dir / "feedback_packet.json",
    )
    manifest["feedback_packet"] = str(output_dir / "feedback_packet.json")
    manifest["feedback_summary"] = feedback["summary"]
    manifest["status"] = _loop_status(micro_report=micro_report, post_observe=post_observe)
    return _write_manifest(output_dir, manifest)


def validate_pseudo_llm_decision(decision: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    for field in REQUIRED_DECISION_FIELDS:
        if field not in decision:
            blockers.append(f"missing decision field: {field}")
    step = decision.get("limited_step")
    if not isinstance(step, dict):
        blockers.append("limited_step must be an object")
    else:
        for field in ["joint", "manual_delta_raw", "max_abs_delta_raw"]:
            if field not in step:
                blockers.append(f"missing limited_step field: {field}")
        try:
            delta = abs(float(step.get("manual_delta_raw")))
            limit = abs(float(step.get("max_abs_delta_raw")))
            if delta > limit:
                blockers.append("manual_delta_raw exceeds max_abs_delta_raw")
            if limit > 50.0:
                blockers.append("max_abs_delta_raw must be <= 50 for agentic v1 limited steps")
        except (TypeError, ValueError):
            blockers.append("manual_delta_raw and max_abs_delta_raw must be numeric")
        if not step.get("non_contact_confirmed") and not step.get("contact_ok_for_gripper"):
            blockers.append("limited_step must declare non_contact_confirmed or contact_ok_for_gripper")
    if decision.get("target_runtime") not in {None, "on_device_llm_or_vlm"}:
        blockers.append("target_runtime must be on_device_llm_or_vlm when provided")
    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
    }


def build_feedback_packet(
    *,
    decision: dict[str, Any],
    micro_report: dict[str, Any],
    pre_observe: dict[str, Any],
    post_observe: dict[str, Any],
    output: Path | None = None,
) -> dict[str, Any]:
    after_visual = ((micro_report.get("visual_check") or {}).get("after") or {})
    feedback = {
        "status": "passed",
        "operation": "real_so100_agentic_v1_feedback_packet",
        "task": decision.get("task"),
        "agentic_layer_version": decision.get("agentic_layer_version"),
        "selected_subgoal": decision.get("selected_subgoal"),
        "smolvla_prompt": decision.get("smolvla_prompt"),
        "expected_observer_evidence": decision.get("expected_observer_evidence"),
        "evidence": {
            "send_action_called": bool(micro_report.get("send_action_called")),
            "micro_step_status": micro_report.get("status"),
            "joint": micro_report.get("joint"),
            "planned_delta_raw": micro_report.get("planned_delta_raw"),
            "observed_delta_raw": micro_report.get("observed_delta_raw"),
            "target_error_raw": micro_report.get("target_error_raw"),
            "observer_camera_index": micro_report.get("camera_index"),
            "observer_before": ((micro_report.get("visual_check") or {}).get("before") or {}).get("image_path"),
            "observer_after": after_visual.get("image_path"),
            "observer_motion_video": (micro_report.get("motion_video") or {}).get("path"),
            "observer_mean_absdiff": after_visual.get("mean_absdiff"),
            "observer_visual_motion_detected": after_visual.get("visual_motion_detected"),
            "pre_observe_episode": pre_observe.get("episode_jsonl"),
            "post_observe_episode": post_observe.get("episode_jsonl"),
            "policy_camera_frames_recorded": {
                "pre": pre_observe.get("frames_recorded"),
                "post": post_observe.get("frames_recorded"),
            },
        },
        "summary": {
            "robot_write_sent": bool(micro_report.get("send_action_called")),
            "joint_readback_changed": _abs_or_none(micro_report.get("observed_delta_raw")),
            "observer_visual_motion_detected": after_visual.get("visual_motion_detected"),
            "needs_pseudo_llm_or_on_device_llm_feedback": True,
        },
        "next_version_input": {
            "instruction": (
                "Use the observer camera evidence plus policy camera before/after frames to decide whether the "
                "subgoal moved toward task completion. If not, produce the next agentic-layer decision JSON."
            ),
            "semantic_judgment_owner": "pseudo_llm_during_development_on_device_llm_at_runtime",
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(feedback, indent=2, sort_keys=True), encoding="utf-8")
    return feedback


def _loop_status(*, micro_report: dict[str, Any], post_observe: dict[str, Any]) -> str:
    if micro_report.get("status") not in {"passed", "dry_run"}:
        return "failed"
    if not post_observe.get("ok"):
        return "failed"
    return "passed"


def _abs_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return None


def _dedupe(items: list[int]) -> list[int]:
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output_dir / "agentic_v1_loop_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Pseudo-LLM Agentic SmolVLA v1 limited-step loop.")
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--policy-camera-index", type=int, action="append", default=[])
    parser.add_argument("--observer-camera-index", type=int, default=3)
    parser.add_argument("--prepost-duration-seconds", type=float, default=0.6)
    parser.add_argument("--prepost-fps", type=float, default=2.0)
    parser.add_argument("--settle-seconds", type=float, default=0.75)
    parser.add_argument("--video-fps", type=float, default=12.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run_agentic_v1_loop(
                decision=args.decision,
                output_dir=args.output_dir,
                port=args.port,
                calibration_file=args.calibration_file,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                policy_camera_indexes=args.policy_camera_index or [0, 1],
                observer_camera_index=args.observer_camera_index,
                prepost_duration_seconds=args.prepost_duration_seconds,
                prepost_fps=args.prepost_fps,
                settle_seconds=args.settle_seconds,
                video_fps=args.video_fps,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
