#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER, load_calibration
from scripts.real_so100_micro_step import (
    _capture_visual,
    _make_so100_bus,
    _probe_motion_video,
    _record_motion_video,
    _start_motion_video,
)


def execute_legalaspro_chunk(
    *,
    action: Path,
    output: Path,
    port: str,
    calibration: Path,
    execute: bool,
    human_confirmed: bool,
    action_steps: int,
    max_abs_delta_raw: float,
    step_settle_seconds: float,
    camera_index: int | None,
    visual_output_dir: Path | None,
    record_video: bool,
    video_fps: float,
    keep_torque_on_after_run: bool,
) -> dict[str, Any]:
    if action_steps < 1:
        raise ValueError(f"action_steps must be positive, got {action_steps}")
    if max_abs_delta_raw <= 0:
        raise ValueError(f"max_abs_delta_raw must be positive, got {max_abs_delta_raw}")
    payload = json.loads(action.read_text(encoding="utf-8"))
    chunk = payload.get("raw_action_chunk")
    if not isinstance(chunk, list) or not chunk:
        raise ValueError(f"action chunk missing raw_action_chunk: {action}")
    chunk = [[float(value) for value in row] for row in chunk[:action_steps]]
    calibration_payload = load_calibration(calibration) or {}
    blockers: list[str] = []
    if not human_confirmed:
        blockers.append("Human confirmation flag is required.")
    if execute:
        if camera_index is None:
            blockers.append("Execution requires --camera-index.")
        if visual_output_dir is None:
            blockers.append("Execution requires --visual-output-dir.")
        if not record_video:
            blockers.append("Execution requires --record-video.")

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_execute_legalaspro_smolvla_chunk",
        "source_action": str(action),
        "port": port,
        "calibration": str(calibration),
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "action_steps": len(chunk),
        "max_abs_delta_raw": max_abs_delta_raw,
        "step_settle_seconds": step_settle_seconds,
        "camera_index": camera_index,
        "record_video_requested": record_video,
        "keep_torque_on_after_run": keep_torque_on_after_run,
        "send_action_called": False,
        "policy_actions_executed": False,
        "post_task_torque_disabled": False,
        "home_return_required_before_torque_off": keep_torque_on_after_run,
        "blockers": blockers,
        "status": "blocked" if blockers else ("ready" if execute else "dry_run"),
        "adapter": {
            "name": "legalaspro_so101_ros_radian_absolute_to_so100_raw_v0",
            "arm_units": "radians_absolute_joint_position",
            "arm_conversion": "raw = calibration_mid + radians * 4095 / (2*pi)",
            "gripper_conversion": "raw = calibration_min + clamp(action,0,1) * calibrated_span",
            "safety": "calibration clipping plus per-step raw delta limiting",
        },
    }
    if blockers or not execute:
        report["desired_plan_preview"] = _desired_raw_plan(chunk=chunk, calibration=calibration_payload)
        _write_json(output, report)
        return report

    bus, _motors = _make_so100_bus(port)
    video_capture = None
    video_writer = None
    video_result = None
    before_image_path = None
    try:
        if camera_index is not None and visual_output_dir is not None:
            before_visual = _capture_visual(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                label="before",
                before_path=None,
            )
            before_image_path = Path(before_visual["image_path"])
            report["visual_check"] = {"before": before_visual}
        bus.connect(handshake=True)
        current = {joint: float(value) for joint, value in bus.sync_read("Present_Position", normalize=False).items()}
        report["readback_before_raw"] = current
        desired = _desired_raw_plan(chunk=chunk, calibration=calibration_payload)
        clipped_plan = _delta_limited_plan(
            desired_plan=desired,
            current=current,
            max_abs_delta_raw=max_abs_delta_raw,
            calibration=calibration_payload,
        )
        report["desired_plan"] = desired
        report["clipped_plan"] = clipped_plan
        if record_video and camera_index is not None and visual_output_dir is not None:
            video_capture, video_writer, video_result = _start_motion_video(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                fps=video_fps,
            )
            report["motion_video"] = video_result

        executed_steps = []
        for step in clipped_plan:
            target = {joint: int(round(value)) for joint, value in step["target_raw"].items()}
            bus.sync_write("Goal_Position", target, normalize=False, num_retry=3)
            report["send_action_called"] = True
            executed_steps.append(step)
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

        after = {joint: float(value) for joint, value in bus.sync_read("Present_Position", normalize=False).items()}
        report["executed_steps"] = executed_steps
        report["executed_action_steps"] = len(executed_steps)
        report["readback_after_raw"] = after
        report["observed_delta_raw"] = {joint: round(after[joint] - current[joint], 4) for joint in SO100_JOINT_ORDER}
        report["policy_actions_executed"] = bool(executed_steps)
        if camera_index is not None and visual_output_dir is not None:
            report.setdefault("visual_check", {})["after"] = _capture_visual(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                label="after",
                before_path=before_image_path,
            )
        report["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - preserve hardware failure detail.
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
                bus.disconnect(disable_torque=not keep_torque_on_after_run)
                report["post_task_torque_disabled"] = not keep_torque_on_after_run
                report["torque_kept_on_for_home_return"] = keep_torque_on_after_run
        except Exception as exc:  # noqa: BLE001
            report["disconnect_error"] = repr(exc)
            report["post_task_torque_disabled"] = False
    _write_json(output, report)
    return report


def _desired_raw_plan(*, chunk: list[list[float]], calibration: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for step_index, action in enumerate(chunk):
        if len(action) != len(SO100_JOINT_ORDER):
            raise ValueError(f"action step {step_index} has dim {len(action)}; expected {len(SO100_JOINT_ORDER)}")
        target_raw: dict[str, float] = {}
        details: dict[str, Any] = {}
        for joint, value in zip(SO100_JOINT_ORDER, action, strict=True):
            item = calibration[joint]
            low = float(item["range_min"])
            high = float(item["range_max"])
            mid = (low + high) / 2.0
            if joint == "gripper":
                fraction = _clip(float(value), 0.0, 1.0)
                raw = low + fraction * (high - low)
                units = "fraction_0_1"
            else:
                raw = mid + float(value) * 4095.0 / (2.0 * math.pi)
                units = "radians"
            clipped = _clip(raw, low, high)
            target_raw[joint] = clipped
            details[joint] = {
                "action_value": float(value),
                "action_units": units,
                "raw_before_calibration_clip": raw,
                "raw_after_calibration_clip": clipped,
                "range_min": low,
                "range_max": high,
            }
        out.append({"step_index": step_index, "target_raw": target_raw, "details": details})
    return out


def _delta_limited_plan(
    *,
    desired_plan: list[dict[str, Any]],
    current: dict[str, float],
    max_abs_delta_raw: float,
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    simulated = {joint: float(current[joint]) for joint in SO100_JOINT_ORDER}
    out = []
    for step in desired_plan:
        target_raw = {}
        crop_details = {}
        for joint in SO100_JOINT_ORDER:
            desired = float(step["target_raw"][joint])
            low = float(calibration[joint]["range_min"])
            high = float(calibration[joint]["range_max"])
            limited = _clip(desired, simulated[joint] - max_abs_delta_raw, simulated[joint] + max_abs_delta_raw)
            final = _clip(limited, low, high)
            target_raw[joint] = final
            crop_details[joint] = {
                "desired_raw": desired,
                "delta_from_previous_raw": round(final - simulated[joint], 4),
                "delta_limited_raw": final,
                "range_min": low,
                "range_max": high,
            }
            simulated[joint] = final
        out.append({"step_index": step["step_index"], "target_raw": target_raw, "crop_details": crop_details})
    return out


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute legalaspro/SO101-style SmolVLA action chunk on SO-100 with a conservative raw adapter.")
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--action-steps", type=int, default=15)
    parser.add_argument("--max-abs-delta-raw", type=float, default=40.0)
    parser.add_argument("--step-settle-seconds", type=float, default=0.12)
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--visual-output-dir", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument(
        "--keep-torque-on-after-run",
        action="store_true",
        help=(
            "Do not disable torque after the policy chunk. Use this only when a "
            "home-return step will run immediately and perform the final torque-off."
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            execute_legalaspro_chunk(
                action=args.action,
                output=args.output,
                port=args.port,
                calibration=args.calibration,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                action_steps=args.action_steps,
                max_abs_delta_raw=args.max_abs_delta_raw,
                step_settle_seconds=args.step_settle_seconds,
                camera_index=args.camera_index,
                visual_output_dir=args.visual_output_dir,
                record_video=args.record_video,
                video_fps=args.video_fps,
                keep_torque_on_after_run=args.keep_torque_on_after_run,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
