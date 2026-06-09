#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from scripts.real_so100_micro_step import (
    _capture_visual,
    _make_so100_bus,
    _probe_motion_video,
    _record_motion_video,
    _start_motion_video,
)


DEFAULT_NATURAL_FRACTIONS = {
    "shoulder_pan": 0.50,
    "shoulder_lift": 0.50,
    "elbow_flex": 0.50,
    "wrist_flex": 0.50,
    "wrist_roll": 0.50,
    "gripper": 0.67,
}


def move_to_natural_pose(
    *,
    port: str,
    calibration: Path,
    output: Path,
    execute: bool,
    human_confirmed: bool,
    workspace_clear_confirmed: bool,
    max_abs_delta_raw: float,
    step_settle_seconds: float,
    camera_index: int | None,
    visual_output_dir: Path | None,
    record_video: bool,
    video_fps: float,
    target_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    if max_abs_delta_raw <= 0:
        raise ValueError(f"max_abs_delta_raw must be positive, got {max_abs_delta_raw}")
    if step_settle_seconds < 0:
        raise ValueError(f"step_settle_seconds must be non-negative, got {step_settle_seconds}")
    calibration_payload = _load_calibration(calibration)
    natural_target = _natural_target_from_calibration(calibration_payload)
    if target_overrides:
        for joint, value in target_overrides.items():
            if joint not in SO100_JOINT_ORDER:
                raise ValueError(f"unknown target override joint: {joint}")
            natural_target[joint] = _clip_to_calibration(joint, float(value), calibration_payload)

    blockers: list[str] = []
    if not human_confirmed:
        blockers.append("Human confirmation flag is required.")
    if not workspace_clear_confirmed:
        blockers.append("Workspace clear confirmation flag is required.")
    if execute:
        if not record_video:
            blockers.append("Executed real robot movements must pass --record-video.")
        if camera_index is None:
            blockers.append("Executed real robot movements must pass --camera-index.")
        if visual_output_dir is None:
            blockers.append("Executed real robot movements must pass --visual-output-dir.")

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_move_to_natural_pose",
        "port": port,
        "calibration": str(calibration),
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "workspace_clear_confirmed": workspace_clear_confirmed,
        "max_abs_delta_raw": max_abs_delta_raw,
        "step_settle_seconds": step_settle_seconds,
        "camera_index": camera_index,
        "observer_camera_status": "temporarily_unavailable",
        "observer_camera_indexes": [],
        "visual_evidence_role": "policy_context_camera_recovery_evidence",
        "record_video_requested": record_video,
        "target_natural_raw": natural_target,
        "target_overrides_raw": target_overrides or {},
        "send_action_called": False,
        "initial_hold_sent_before_visual_capture": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "disconnect_disable_torque": True,
        "blockers": blockers,
        "status": "blocked" if blockers else "ready",
    }
    if not execute:
        report["status"] = "dry_run"
        _write_json(output, report)
        return report
    if blockers:
        _write_json(output, report)
        return report

    bus, _motors = _make_so100_bus(port)
    video_capture = None
    video_writer = None
    video_result: dict[str, Any] | None = None
    try:
        bus.connect(handshake=True)
        before_state = {
            joint: float(value)
            for joint, value in bus.sync_read("Present_Position", normalize=False).items()
            if joint in SO100_JOINT_ORDER
        }
        missing = [joint for joint in SO100_JOINT_ORDER if joint not in before_state]
        if missing:
            raise RuntimeError(f"missing readback joints: {missing}")

        # Hold the current pose before any camera work. If a previous command
        # left torque disabled, this prevents the arm from sagging while we
        # capture before-evidence or build the natural-pose trajectory.
        current_hold = {joint: int(round(before_state[joint])) for joint in SO100_JOINT_ORDER}
        bus.sync_write("Goal_Position", current_hold, normalize=False, num_retry=3)
        report["initial_hold_sent_before_visual_capture"] = True
        report["readback_before_raw"] = before_state

        before_image_path = None
        if camera_index is not None and visual_output_dir is not None:
            before_visual = _capture_visual(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                label="before",
                before_path=None,
            )
            before_image_path = Path(before_visual["image_path"])
            report["visual_check"] = {"before": before_visual}

        step_targets = _build_interpolated_targets(
            current=before_state,
            target=natural_target,
            calibration=calibration_payload,
            max_abs_delta_raw=max_abs_delta_raw,
        )
        report["planned_action_steps"] = len(step_targets)
        report["planned_total_delta_raw"] = {
            joint: round(natural_target[joint] - before_state[joint], 4)
            for joint in SO100_JOINT_ORDER
        }

        if record_video and camera_index is not None and visual_output_dir is not None:
            video_capture, video_writer, video_result = _start_motion_video(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                fps=video_fps,
            )
            report["motion_video"] = video_result

        executed_steps: list[dict[str, Any]] = []
        for step_index, target in enumerate(step_targets):
            bus.sync_write("Goal_Position", target, normalize=False, num_retry=3)
            report["send_action_called"] = True
            report["physical_robot_motion"] = True
            executed_steps.append({"step_index": step_index, "target_raw": target})
            if video_capture is not None and video_writer is not None and video_result is not None:
                _record_motion_video(
                    capture=video_capture,
                    writer=video_writer,
                    result=video_result,
                    duration_seconds=step_settle_seconds,
                    fps=video_fps,
                )
            else:
                time.sleep(step_settle_seconds)

        after_state = {
            joint: float(value)
            for joint, value in bus.sync_read("Present_Position", normalize=False).items()
            if joint in SO100_JOINT_ORDER
        }
        report["executed_steps"] = executed_steps
        report["executed_action_steps"] = len(executed_steps)
        report["readback_after_raw"] = after_state
        report["observed_delta_raw"] = {
            joint: round(after_state[joint] - before_state[joint], 4)
            for joint in SO100_JOINT_ORDER
        }
        report["final_target_error_raw"] = {
            joint: round(natural_target[joint] - after_state[joint], 4)
            for joint in SO100_JOINT_ORDER
        }
        if camera_index is not None and visual_output_dir is not None:
            after_visual = _capture_visual(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                label="after",
                before_path=before_image_path,
            )
            report.setdefault("visual_check", {})["after"] = after_visual
        report["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - hardware report should preserve exact failure.
        report["status"] = "failed"
        report["error"] = repr(exc)
    finally:
        if video_writer is not None:
            video_writer.release()
        if video_capture is not None:
            video_capture.release()
        if isinstance(report.get("motion_video"), dict):
            report["motion_video"].update(_probe_motion_video(Path(report["motion_video"]["path"])))
        try:
            if bus.is_connected:
                reached_natural_pose = report.get("status") == "passed"
                bus.disconnect(disable_torque=reached_natural_pose)
                report["disconnect_disable_torque"] = bool(reached_natural_pose)
                report["post_task_torque_disabled"] = bool(reached_natural_pose)
                if not reached_natural_pose:
                    report["torque_kept_on_reason"] = "natural_pose_not_confirmed"
        except Exception as exc:  # noqa: BLE001
            report["disconnect_error"] = repr(exc)
            report["post_task_torque_disabled"] = False

    _write_json(output, report)
    return report


def _load_calibration(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"calibration must be an object: {path}")
    result: dict[str, dict[str, float]] = {}
    for joint in SO100_JOINT_ORDER:
        item = payload.get(joint)
        if not isinstance(item, dict):
            raise ValueError(f"calibration missing joint: {joint}")
        result[joint] = {
            "range_min": float(item["range_min"]),
            "range_max": float(item["range_max"]),
        }
    return result


def _natural_target_from_calibration(calibration: dict[str, dict[str, float]]) -> dict[str, float]:
    target: dict[str, float] = {}
    for joint in SO100_JOINT_ORDER:
        low = calibration[joint]["range_min"]
        high = calibration[joint]["range_max"]
        fraction = DEFAULT_NATURAL_FRACTIONS[joint]
        target[joint] = round(low + ((high - low) * fraction), 4)
    return target


def _build_interpolated_targets(
    *,
    current: dict[str, float],
    target: dict[str, float],
    calibration: dict[str, dict[str, float]],
    max_abs_delta_raw: float,
) -> list[dict[str, int]]:
    largest_delta = max(abs(target[joint] - current[joint]) for joint in SO100_JOINT_ORDER)
    steps = max(1, int(math.ceil(largest_delta / max_abs_delta_raw)))
    targets: list[dict[str, int]] = []
    for step in range(1, steps + 1):
        alpha = step / steps
        step_target: dict[str, int] = {}
        for joint in SO100_JOINT_ORDER:
            raw_value = current[joint] + ((target[joint] - current[joint]) * alpha)
            raw_value = _clip_to_calibration(joint, raw_value, calibration)
            step_target[joint] = int(round(raw_value))
        targets.append(step_target)
    return targets


def _clip_to_calibration(joint: str, value: float, calibration: dict[str, dict[str, float]]) -> float:
    low = calibration[joint]["range_min"]
    high = calibration[joint]["range_max"]
    return min(max(value, low), high)


def _parse_target_overrides(items: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--target-raw must use joint=value, got: {item}")
        joint, raw_value = item.split("=", 1)
        overrides[joint.strip()] = float(raw_value)
    return overrides


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move the real SO-100 follower to a conservative natural pose.")
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    parser.add_argument("--max-abs-delta-raw", type=float, default=80.0)
    parser.add_argument("--step-settle-seconds", type=float, default=0.12)
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--visual-output-dir", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument(
        "--target-raw",
        action="append",
        default=[],
        help="Optional joint=value raw target override, clipped to calibration.",
    )
    args = parser.parse_args()

    print(
        json.dumps(
            move_to_natural_pose(
                port=args.port,
                calibration=args.calibration,
                output=args.output,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                workspace_clear_confirmed=args.workspace_clear_confirmed,
                max_abs_delta_raw=args.max_abs_delta_raw,
                step_settle_seconds=args.step_settle_seconds,
                camera_index=args.camera_index,
                visual_output_dir=args.visual_output_dir,
                record_video=args.record_video,
                video_fps=args.video_fps,
                target_overrides=_parse_target_overrides(args.target_raw),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
