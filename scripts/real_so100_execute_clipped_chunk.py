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


def execute_clipped_chunk(
    *,
    dry_run_result: Path,
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
) -> dict[str, Any]:
    source = _load_json(dry_run_result)
    calibration_payload = load_calibration(calibration)
    blockers = []
    if source.get("operation") != "real_so100_execute_smolvla_action_chunk":
        blockers.append(f"Unexpected source operation: {source.get('operation')!r}.")
    if not (source.get("dry_plan") or {}).get("step_plans"):
        blockers.append("Source dry plan has no step plans.")
    if not human_confirmed:
        blockers.append("Human confirmation flag is required.")
    if execute and (camera_index is None or visual_output_dir is None or not record_video):
        blockers.append("Execution requires --camera-index, --visual-output-dir, and --record-video.")
    if max_abs_delta_raw <= 0:
        blockers.append("max_abs_delta_raw must be positive.")

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_execute_clipped_smolvla_chunk",
        "source_dry_run_result": str(dry_run_result),
        "port": port,
        "calibration": str(calibration),
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "action_steps": action_steps,
        "max_abs_delta_raw": max_abs_delta_raw,
        "step_settle_seconds": step_settle_seconds,
        "camera_index": camera_index,
        "record_video_requested": record_video,
        "disconnect_disable_torque": bool(execute),
        "post_task_torque_disabled": False,
        "send_action_called": False,
        "policy_actions_executed": False,
        "writes_intended": execute,
        "blockers": blockers,
        "status": "blocked" if blockers else ("ready" if execute else "dry_run"),
        "notes": [
            "Targets are first clipped to calibrated raw ranges, then rate-limited per step from the simulated current state.",
            "This is an execution adapter for a user-confirmed clipped baseline attempt, not proof of task success.",
        ],
    }
    if blockers or not execute:
        if not execute and not blockers:
            report["clipped_plan_preview"] = _build_preview_plan(
                source=source,
                calibration=calibration_payload,
                current_state={joint: 0.0 for joint in SO100_JOINT_ORDER},
                action_steps=action_steps,
                max_abs_delta_raw=max_abs_delta_raw,
            )
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
        clipped_plan = _build_preview_plan(
            source=source,
            calibration=calibration_payload,
            current_state=current,
            action_steps=action_steps,
            max_abs_delta_raw=max_abs_delta_raw,
        )
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
        report["observed_delta_raw"] = {
            joint: round(after[joint] - report["readback_before_raw"][joint], 4)
            for joint in SO100_JOINT_ORDER
        }
        report["policy_actions_executed"] = bool(executed_steps)
        if camera_index is not None and visual_output_dir is not None:
            report.setdefault("visual_check", {})["after"] = _capture_visual(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                label="after",
                before_path=before_image_path,
            )
        report["status"] = "passed"
    except Exception as exc:  # noqa: BLE001
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
                bus.disconnect(disable_torque=True)
                report["disconnect_disable_torque"] = True
                report["post_task_torque_disabled"] = True
        except Exception as exc:  # noqa: BLE001
            report["disconnect_error"] = repr(exc)
            report["post_task_torque_disabled"] = False

    _write_json(output, report)
    return report


def _build_preview_plan(
    *,
    source: dict[str, Any],
    calibration: dict[str, Any],
    current_state: dict[str, float],
    action_steps: int,
    max_abs_delta_raw: float,
) -> list[dict[str, Any]]:
    step_plans = (source.get("dry_plan") or {}).get("step_plans") or []
    simulated = {joint: float(current_state[joint]) for joint in SO100_JOINT_ORDER}
    output_steps = []
    for step in step_plans[:action_steps]:
        raw_targets = {}
        crop_details = {}
        for target in step.get("joint_targets", []):
            joint = target.get("joint")
            if joint not in SO100_JOINT_ORDER:
                continue
            desired = _finite_or_current(target.get("target_raw"), simulated[joint])
            joint_cal = calibration.get(joint, {})
            range_min = float(joint_cal.get("range_min"))
            range_max = float(joint_cal.get("range_max"))
            calibration_clipped = _clip(desired, range_min, range_max)
            delta_limited = _clip(
                calibration_clipped,
                simulated[joint] - max_abs_delta_raw,
                simulated[joint] + max_abs_delta_raw,
            )
            final = _clip(delta_limited, range_min, range_max)
            raw_targets[joint] = final
            crop_details[joint] = {
                "source_target_raw": desired,
                "calibration_clipped_raw": calibration_clipped,
                "delta_limited_raw": final,
                "delta_from_previous_raw": round(final - simulated[joint], 4),
                "range_min": range_min,
                "range_max": range_max,
            }
            simulated[joint] = final
        output_steps.append(
            {
                "step_index": int(step.get("step_index", len(output_steps))),
                "target_raw": raw_targets,
                "crop_details": crop_details,
            }
        )
    return output_steps


def _finite_or_current(value: Any, current: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return current
    return number if math.isfinite(number) else current


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a clipped SO-100 SmolVLA chunk from a metadata dry-run plan.")
    parser.add_argument("--dry-run-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--max-abs-delta-raw", type=float, default=40.0)
    parser.add_argument("--step-settle-seconds", type=float, default=0.15)
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--visual-output-dir", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    args = parser.parse_args()
    print(
        json.dumps(
            execute_clipped_chunk(
                dry_run_result=args.dry_run_result,
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
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
