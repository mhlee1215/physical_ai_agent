#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _make_so100_bus(port: str):
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    return FeetechMotorsBus(port=port, motors=motors), motors


def _load_command_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan.get("joint_plans"), list):
        raise ValueError(f"joint_plans list not found in {path}")
    return plan


def _single_joint_target(plan: dict[str, Any], joint: str) -> tuple[float, float]:
    matches = [item for item in plan["joint_plans"] if item.get("joint") == joint]
    if not matches:
        raise ValueError(f"joint={joint!r} not found in command plan")
    item = matches[0]
    return float(item["current_raw"]), float(item["target_raw"])


def run_micro_step(
    *,
    port: str,
    command_plan: Path | None,
    joint: str,
    output: Path,
    execute: bool,
    human_confirmed: bool,
    non_contact_confirmed: bool,
    contact_ok_for_gripper: bool,
    max_abs_delta_raw: float,
    settle_seconds: float,
    manual_delta_raw: float | None = None,
    camera_index: int | None = None,
    visual_output_dir: Path | None = None,
    record_video: bool = False,
    video_fps: float = 12.0,
) -> dict[str, Any]:
    plan = _load_command_plan(command_plan) if command_plan else None
    if manual_delta_raw is None:
        if plan is None:
            raise ValueError("--command-plan is required unless --manual-delta-raw is set")
        planned_current, planned_target = _single_joint_target(plan, joint)
        planned_delta = planned_target - planned_current
    else:
        planned_current = 0.0
        planned_target = manual_delta_raw
        planned_delta = manual_delta_raw
    blockers: list[str] = []

    if abs(planned_delta) > max_abs_delta_raw:
        blockers.append(f"Planned delta {planned_delta:.4f} exceeds max_abs_delta_raw={max_abs_delta_raw}.")
    if not human_confirmed:
        blockers.append("Human confirmation flag is required.")
    contact_probe_allowed = bool(joint == "gripper" and contact_ok_for_gripper)
    if not non_contact_confirmed and not contact_probe_allowed:
        blockers.append("Non-contact workspace confirmation flag is required.")
    if manual_delta_raw is None and plan and plan.get("ready_for_execution") is not True:
        blockers.append("Command plan is not marked ready_for_execution=true.")
    if execute:
        if not record_video:
            blockers.append("Executed real robot movements must pass --record-video.")
        if camera_index is None:
            blockers.append("Executed real robot movements must pass --camera-index for visual/video evidence.")
        if visual_output_dir is None:
            blockers.append("Executed real robot movements must pass --visual-output-dir for visual/video evidence.")

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_micro_step_validation",
        "port": port,
        "joint": joint,
        "command_plan": str(command_plan) if command_plan else None,
        "manual_delta_raw": manual_delta_raw,
        "planned_current_raw": planned_current,
        "planned_target_raw": planned_target,
        "planned_delta_raw": planned_delta,
        "max_abs_delta_raw": max_abs_delta_raw,
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "non_contact_confirmed": non_contact_confirmed,
        "contact_ok_for_gripper": contact_ok_for_gripper,
        "contact_probe_allowed": contact_probe_allowed,
        "send_action_called": False,
        "policy_actions_executed": False,
        "writes_intended": execute,
        "disconnect_disable_torque": True,
        "blockers": blockers,
        "status": "blocked" if blockers else "ready",
        "visual_check_required": bool(execute),
        "camera_index": camera_index,
        "record_video_requested": record_video,
        "motion_video_required": bool(execute),
    }

    if not execute:
        if camera_index is not None and visual_output_dir is not None:
            report["visual_check"] = _capture_visual(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                label="dry_run",
                before_path=None,
            )
        report["status"] = "dry_run"
        report["notes"] = ["No serial write was attempted. Re-run with --execute plus confirmations to actuate."]
        _write_json(output, report)
        return report

    if blockers:
        report["notes"] = ["Execution was requested but blocked before connecting to the robot."]
        _write_json(output, report)
        return report

    bus, _motors = _make_so100_bus(port)
    video_capture = None
    video_writer = None
    video_result: dict[str, Any] | None = None
    try:
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
        bus.connect(handshake=True)
        before = bus.sync_read("Present_Position", normalize=False)
        if manual_delta_raw is not None:
            planned_current = float(before[joint])
            planned_target = planned_current + manual_delta_raw
            report["planned_current_raw"] = planned_current
            report["planned_target_raw"] = planned_target
            report["planned_delta_raw"] = manual_delta_raw
        target = int(round(planned_target))
        if record_video and camera_index is not None and visual_output_dir is not None:
            video_capture, video_writer, video_result = _start_motion_video(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                fps=video_fps,
            )
            report["motion_video"] = video_result
        bus.sync_write("Goal_Position", {joint: target}, normalize=False, num_retry=3)
        report["send_action_called"] = True
        if video_capture is not None and video_writer is not None and video_result is not None:
            _record_motion_video(
                capture=video_capture,
                writer=video_writer,
                result=video_result,
                duration_seconds=settle_seconds,
                fps=video_fps,
            )
            report["motion_video"] = video_result
        else:
            time.sleep(settle_seconds)
        after = bus.sync_read("Present_Position", normalize=False)
        report["readback_before_raw"] = before
        report["readback_after_raw"] = after
        report["commanded_target_raw"] = {joint: target}
        before_joint = float(before[joint])
        after_joint = float(after[joint])
        report["observed_delta_raw"] = after_joint - before_joint
        report["target_error_raw"] = float(target) - after_joint
        if camera_index is not None and visual_output_dir is not None:
            after_visual = _capture_visual(
                camera_index=camera_index,
                output_dir=visual_output_dir,
                label="after",
                before_path=before_image_path,
            )
            report["visual_check"]["after"] = after_visual
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
                bus.disconnect(disable_torque=True)
                report["disconnect_disable_torque"] = True
                report["post_task_torque_disabled"] = True
        except Exception as exc:  # noqa: BLE001
            report["disconnect_error"] = repr(exc)
            report["post_task_torque_disabled"] = False

    _write_json(output, report)
    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _capture_visual(
    *,
    camera_index: int,
    output_dir: Path,
    label: str,
    before_path: Path | None,
) -> dict[str, Any]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _capture_frame(camera_index)
    image_path = output_dir / f"{label}.jpg"
    cv2.imwrite(str(image_path), frame)
    result: dict[str, Any] = {
        "camera_index": camera_index,
        "image_path": str(image_path),
        "shape": list(frame.shape),
    }
    if before_path is not None:
        before = cv2.imread(str(before_path))
        if before is not None and before.shape == frame.shape:
            diff = cv2.absdiff(before, frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            result["before_path"] = str(before_path)
            result["mean_absdiff"] = round(float(gray.mean()), 4)
            result["changed_pixel_ratio_gt_10"] = round(float((gray > 10).mean()), 6)
            result["visual_motion_detected"] = bool(result["mean_absdiff"] >= 2.0)
        else:
            result["visual_diff_error"] = "before image missing or shape mismatch"
    return result


def _capture_frame(camera_index: int):
    import cv2

    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"camera index {camera_index} did not open")
    try:
        frame = None
        ok = False
        for _ in range(5):
            ok, frame = cap.read()
            time.sleep(0.05)
        if not ok or frame is None:
            raise RuntimeError(f"camera index {camera_index} did not return a frame")
        return frame
    finally:
        cap.release()


def _start_motion_video(*, camera_index: int, output_dir: Path, fps: float):
    import cv2

    if fps <= 0:
        raise ValueError("--video-fps must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"camera index {camera_index} did not open for motion video")
    frame = None
    ok = False
    for _ in range(5):
        ok, frame = cap.read()
        time.sleep(0.03)
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"camera index {camera_index} did not return a video frame")
    video_path = output_dir / "motion.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (int(frame.shape[1]), int(frame.shape[0])))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open motion video writer: {video_path}")
    writer.write(frame)
    result = {
        "camera_index": camera_index,
        "path": str(video_path),
        "fps": fps,
        "frames_recorded": 1,
        "shape": list(frame.shape),
        "records_goal_write_and_settle": True,
    }
    return cap, writer, result


def _record_motion_video(*, capture: Any, writer: Any, result: dict[str, Any], duration_seconds: float, fps: float) -> None:
    deadline = time.monotonic() + max(duration_seconds, 0.0)
    period = 1.0 / fps
    next_frame = time.monotonic()
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now < next_frame:
            time.sleep(min(next_frame - now, 0.01))
            continue
        ok, frame = capture.read()
        if ok and frame is not None:
            writer.write(frame)
            result["frames_recorded"] += 1
        next_frame += period


def _probe_motion_video(path: Path) -> dict[str, Any]:
    import cv2

    result: dict[str, Any] = {
        "probe_path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        result["probe_error"] = "motion video file does not exist"
        result["browser_preview_recommended"] = True
        return result
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            result["probe_error"] = "opencv failed to open motion video"
            result["browser_preview_recommended"] = True
            return result
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip()
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        ok, frame = capture.read()
        result.update(
            {
                "actual_codec": codec,
                "actual_frame_count": frame_count,
                "actual_fps": round(fps, 4),
                "first_frame_readable": bool(ok and frame is not None),
                "browser_preview_recommended": codec.upper() not in {"AVC1", "H264", "MP4V"},
            }
        )
        if frame is not None:
            result["first_frame_shape"] = list(frame.shape)
    finally:
        capture.release()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a single SO-100 non-contact micro-step.")
    parser.add_argument("--port", required=True)
    parser.add_argument("--command-plan", type=Path)
    parser.add_argument("--joint", default="wrist_roll")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--non-contact-confirmed", action="store_true")
    parser.add_argument("--contact-ok-for-gripper", action="store_true")
    parser.add_argument("--max-abs-delta-raw", type=float, default=2.0)
    parser.add_argument("--manual-delta-raw", type=float)
    parser.add_argument("--settle-seconds", type=float, default=0.75)
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--visual-output-dir", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    args = parser.parse_args()

    print(
        json.dumps(
            run_micro_step(
                port=args.port,
                command_plan=args.command_plan,
                joint=args.joint,
                output=args.output,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                non_contact_confirmed=args.non_contact_confirmed,
                contact_ok_for_gripper=args.contact_ok_for_gripper,
                max_abs_delta_raw=args.max_abs_delta_raw,
                settle_seconds=args.settle_seconds,
                manual_delta_raw=args.manual_delta_raw,
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
