#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from scripts.real_so100_apriltag_detection_check import MARKER_DICTIONARIES
from scripts.real_so100_body_map_micro_step import _clip, _load_calibration, _select_transition
from scripts.real_so100_micro_step import _make_so100_bus, _probe_motion_video
from scripts.real_so100_move_to_home_pose import DEFAULT_HOME_POSE, move_to_home_pose


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = Path("_workspace/real_so100/calibration/so100_local.json")
DEFAULT_MODEL = Path(
    "_workspace/real_so100/body_map_collection_20260611/session_20260611_184136/body_map/body_map_knn_model.json"
)
DEFAULT_SAFE_POSE_MANIFEST = Path(
    "_workspace/real_so100/pose_calibration/"
    "good_pose_shoulder_pan_lifted_20260611_214650/manifest.json"
)
SAFE_MANIFOLD_JOINTS = ["shoulder_lift", "elbow_flex", "wrist_flex"]


def run_approach_green(
    *,
    model_path: Path,
    output_dir: Path,
    port: str,
    calibration: Path,
    servo_camera_index: int,
    observer_camera_index: int,
    marker_id: int,
    green_selection: str,
    max_iterations: int,
    target_distance_px: float,
    desired_step_px: float,
    max_abs_delta_raw: float,
    scale: float,
    manual_delta_raw: dict[str, float] | None,
    freeze_joints: list[str],
    safe_pose_manifest: Path | None,
    max_safe_manifold_distance_raw: float,
    step_settle_seconds: float,
    video_fps: float,
    execute: bool,
    human_confirmed: bool,
    workspace_clear_confirmed: bool,
) -> dict[str, Any]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visual"
    visual_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "approach_green_report.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    calibration_payload = _load_calibration(calibration)
    safe_poses = _load_safe_poses(safe_pose_manifest) if safe_pose_manifest else []
    blockers: list[str] = []
    if not human_confirmed:
        blockers.append("Human confirmation flag is required.")
    if not workspace_clear_confirmed:
        blockers.append("Workspace clear confirmation flag is required.")
    if max_iterations <= 0:
        blockers.append("max_iterations must be positive.")
    if desired_step_px <= 0:
        blockers.append("desired_step_px must be positive.")
    if execute and not safe_poses:
        blockers.append("Executed approach requires a safe pose manifest unless explicitly disabled.")

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_body_map_approach_green",
        "model": str(model_path),
        "port": port,
        "calibration": str(calibration),
        "servo_camera_index": servo_camera_index,
        "observer_camera_index": observer_camera_index,
        "marker_id": marker_id,
        "green_selection": green_selection,
        "max_iterations": max_iterations,
        "target_distance_px": target_distance_px,
        "desired_step_px": desired_step_px,
        "max_abs_delta_raw": max_abs_delta_raw,
        "scale": scale,
        "manual_delta_raw": manual_delta_raw,
        "freeze_joints": freeze_joints,
        "safe_pose_manifest": str(safe_pose_manifest) if safe_pose_manifest else None,
        "safe_manifold_joints": SAFE_MANIFOLD_JOINTS,
        "max_safe_manifold_distance_raw": max_safe_manifold_distance_raw,
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "workspace_clear_confirmed": workspace_clear_confirmed,
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "action_chunk_steps": max_iterations,
        "executed_action_steps": 0,
        "steps": [],
        "blockers": blockers,
        "status": "blocked" if blockers else ("ready" if execute else "dry_run"),
        "post_task_torque_disabled": False,
        "home_return_status": None,
    }
    if blockers or not execute:
        _write_json(report_path, report)
        return report

    servo_cap = cv2.VideoCapture(servo_camera_index, cv2.CAP_AVFOUNDATION)
    observer_cap = cv2.VideoCapture(observer_camera_index, cv2.CAP_AVFOUNDATION)
    video_writer = None
    video_result: dict[str, Any] | None = None
    bus = None
    try:
        if not servo_cap.isOpened():
            raise RuntimeError(f"servo camera index {servo_camera_index} did not open")
        if not observer_cap.isOpened():
            raise RuntimeError(f"observer camera index {observer_camera_index} did not open")
        _warmup(servo_cap, frames=8)
        observer_frame = _warmup(observer_cap, frames=8)
        video_path = visual_dir / "observer_motion.mp4"
        video_writer = _open_video_writer(cv2, video_path, observer_frame, video_fps)
        video_result = {
            "camera_index": observer_camera_index,
            "path": str(video_path),
            "fps": video_fps,
            "frames_recorded": 0,
            "shape": list(observer_frame.shape),
        }
        _write_video_frame(observer_cap, video_writer, video_result)

        bus, _motors = _make_so100_bus(port)
        bus.connect(handshake=True)
        before_state = _read_state(bus)
        report["readback_before_raw"] = before_state
        bus.sync_write("Goal_Position", {joint: int(round(before_state[joint])) for joint in SO100_JOINT_ORDER}, normalize=False, num_retry=3)
        report["initial_hold_sent_before_visual_capture"] = True

        for step_index in range(max_iterations):
            servo_frame = _read_frame(servo_cap)
            observer_frame = _read_frame(observer_cap)
            servo_image = visual_dir / f"step_{step_index:02d}_servo.jpg"
            observer_image = visual_dir / f"step_{step_index:02d}_observer.jpg"
            cv2.imwrite(str(servo_image), servo_frame)
            cv2.imwrite(str(observer_image), observer_frame)

            marker = _detect_marker(servo_frame, marker_id=marker_id)
            green = _detect_green_object(servo_frame, selection=green_selection)
            step: dict[str, Any] = {
                "step_index": step_index,
                "servo_image": str(servo_image),
                "observer_image": str(observer_image),
                "marker": marker,
                "green_object": green,
                "executed": False,
            }
            if marker is None:
                step["stop_reason"] = f"marker id {marker_id} not detected on camera {servo_camera_index}"
                report["steps"].append(step)
                break
            if green is None:
                step["stop_reason"] = f"green object not detected on camera {servo_camera_index}"
                report["steps"].append(step)
                break

            dx = float(green["center_px"][0] - marker["center_px"][0])
            dy = float(green["center_px"][1] - marker["center_px"][1])
            distance = math.sqrt(dx * dx + dy * dy)
            step["marker_to_green_delta_px"] = [round(dx, 4), round(dy, 4)]
            step["marker_to_green_distance_px"] = round(distance, 4)
            if distance <= target_distance_px:
                step["stop_reason"] = "target_distance_reached"
                report["steps"].append(step)
                report["touch_proximity_reached"] = True
                break

            step_fraction = min(1.0, desired_step_px / max(distance, 1e-6))
            desired_delta_px = (dx * step_fraction, dy * step_fraction)
            selected: dict[str, Any] | None = None
            if manual_delta_raw is None:
                selected = _select_transition(
                    model=model,
                    desired_camera=servo_camera_index,
                    desired_marker_id=marker_id,
                    desired_delta_px=desired_delta_px,
                )
                proposed_delta = {
                    joint: float(value) * scale
                    for joint, value in selected["joint_delta_raw"].items()
                    if joint in SO100_JOINT_ORDER
                }
            else:
                proposed_delta = {
                    joint: float(manual_delta_raw.get(joint, 0.0)) * scale
                    for joint in SO100_JOINT_ORDER
                }
            limited_delta = {
                joint: (0.0 if joint in freeze_joints else _clip(value, -max_abs_delta_raw, max_abs_delta_raw))
                for joint, value in proposed_delta.items()
            }
            current_state = _read_state(bus)
            target = {
                joint: int(
                    round(
                        _clip(
                            current_state[joint] + limited_delta.get(joint, 0.0),
                            calibration_payload[joint]["range_min"],
                            calibration_payload[joint]["range_max"],
                        )
                    )
                )
                for joint in SO100_JOINT_ORDER
            }
            safe_check = _safe_manifold_check(
                target=target,
                safe_poses=safe_poses,
                max_distance=max_safe_manifold_distance_raw,
            )
            step["safe_manifold_check"] = safe_check
            if not safe_check["ok"]:
                step["stop_reason"] = "target_outside_safe_pose_manifold"
                report["blockers"].append(
                    "KNN target left the good-pose height/orientation manifold; refusing physical step."
                )
                report["steps"].append(step)
                report["status"] = "blocked"
                break
            bus.sync_write("Goal_Position", target, normalize=False, num_retry=3)
            report["send_action_called"] = True
            report["policy_actions_executed"] = True
            report["physical_robot_motion"] = True
            step["executed"] = True
            step["desired_delta_px"] = [round(desired_delta_px[0], 4), round(desired_delta_px[1], 4)]
            if selected is not None:
                step["selected_transition_index"] = selected["transition_index"]
                step["selected_transition_marker_delta"] = selected.get("selected_marker_delta")
            else:
                step["selected_transition_index"] = None
                step["selected_transition_marker_delta"] = None
            step["limited_delta_raw"] = limited_delta
            step["target_raw"] = target
            _record_video_for(observer_cap, video_writer, video_result, step_settle_seconds, video_fps)
            step["readback_after_raw"] = _read_state(bus)
            report["executed_action_steps"] += 1
            report["steps"].append(step)

        report["readback_after_raw"] = _read_state(bus)
        report["observed_delta_raw"] = {
            joint: round(report["readback_after_raw"][joint] - report["readback_before_raw"][joint], 4)
            for joint in SO100_JOINT_ORDER
        }
        final_servo = _read_frame(servo_cap)
        final_observer = _read_frame(observer_cap)
        final_servo_path = visual_dir / "final_servo.jpg"
        final_observer_path = visual_dir / "final_observer.jpg"
        cv2.imwrite(str(final_servo_path), final_servo)
        cv2.imwrite(str(final_observer_path), final_observer)
        report["visual_check"] = {
            "before": {
                "camera_index": observer_camera_index,
                "image_path": report["steps"][0]["observer_image"] if report["steps"] else None,
            },
            "after": {
                "camera_index": observer_camera_index,
                "image_path": str(final_observer_path),
            },
            "final_servo_image": str(final_servo_path),
        }
        report["status"] = "passed"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["error"] = repr(exc)
    finally:
        if video_writer is not None:
            video_writer.release()
        servo_cap.release()
        observer_cap.release()
        if isinstance(video_result, dict):
            video_result.update(_probe_motion_video(Path(video_result["path"])))
            report["motion_video"] = video_result
        try:
            if bus is not None and bus.is_connected:
                bus.disconnect(disable_torque=False)
        except Exception as exc:  # noqa: BLE001
            report["disconnect_error"] = repr(exc)

    home_report = move_to_home_pose(
        port=port,
        calibration=calibration,
        home_pose=DEFAULT_HOME_POSE,
        output=output_dir / "home_return_after_approach_green.json",
        execute=True,
        human_confirmed=human_confirmed,
        workspace_clear_confirmed=workspace_clear_confirmed,
        max_abs_delta_raw=120.0,
        step_settle_seconds=0.10,
        camera_index=observer_camera_index,
        visual_output_dir=output_dir / "home_return_visual",
        record_video=True,
        video_fps=video_fps,
    )
    report["home_return_report"] = str(output_dir / "home_return_after_approach_green.json")
    report["home_return_status"] = home_report.get("status")
    report["post_task_torque_disabled"] = bool(home_report.get("post_task_torque_disabled"))
    _write_json(report_path, report)
    return report


def _warmup(capture: Any, *, frames: int) -> Any:
    frame = None
    ok = False
    for _ in range(max(1, frames)):
        ok, frame = capture.read()
        time.sleep(0.03)
    if not ok or frame is None:
        raise RuntimeError("camera did not return a warmup frame")
    return frame


def _read_frame(capture: Any) -> Any:
    frame = None
    ok = False
    for _ in range(3):
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.03)
    raise RuntimeError("camera frame read failed")


def _detect_marker(frame: Any, *, marker_id: int) -> dict[str, Any] | None:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICTIONARIES["tag36h11"]))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or not len(ids):
        return None
    matches = []
    for detected_id, corner in zip(ids, corners):
        if int(detected_id[0]) != marker_id:
            continue
        points = corner.reshape(-1, 2)
        center = points.mean(axis=0)
        side_lengths = [
            float(((points[(index + 1) % 4] - points[index]) ** 2).sum() ** 0.5)
            for index in range(4)
        ]
        matches.append(
            {
                "id": marker_id,
                "center_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
                "mean_side_px": round(sum(side_lengths) / len(side_lengths), 3),
                "min_side_px": round(min(side_lengths), 3),
            }
        )
    if not matches:
        return None
    return max(matches, key=lambda item: item["min_side_px"])


def _detect_green_object(frame: Any, *, selection: str = "largest") -> dict[str, Any] | None:
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 35, 35), (95, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    image_area = float(frame.shape[0] * frame.shape[1])
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 250.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if area > image_area * 0.40:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        candidates.append(
            {
                "area_px": round(area, 2),
                "bbox_xywh": [int(x), int(y), int(w), int(h)],
                "center_px": [round(float(moments["m10"] / moments["m00"]), 3), round(float(moments["m01"] / moments["m00"]), 3)],
            }
        )
    if not candidates:
        return None
    if selection == "android_doll":
        height, width = frame.shape[:2]
        doll_like = [
            item
            for item in candidates
            if item["area_px"] >= 1500.0
            and item["center_px"][0] >= width * 0.45
            and item["center_px"][1] <= height * 0.65
            and 0.45 <= (item["bbox_xywh"][2] / max(item["bbox_xywh"][3], 1)) <= 1.7
        ]
        if doll_like:
            return max(doll_like, key=lambda item: item["area_px"])
        return None
    if selection == "rightmost":
        return max(candidates, key=lambda item: (item["center_px"][0], item["area_px"]))
    if selection == "largest":
        return max(candidates, key=lambda item: item["area_px"])
    raise ValueError(f"unknown green selection: {selection}")


def _load_safe_poses(path: Path | None) -> list[dict[str, float]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for sample in payload.get("samples", []):
        state = sample.get("readback_raw") or sample.get("commanded_target_raw")
        if not isinstance(state, dict):
            continue
        if all(joint in state for joint in SAFE_MANIFOLD_JOINTS):
            rows.append({joint: float(state[joint]) for joint in SAFE_MANIFOLD_JOINTS})
    return rows


def _safe_manifold_check(
    *,
    target: dict[str, int],
    safe_poses: list[dict[str, float]],
    max_distance: float,
) -> dict[str, Any]:
    if not safe_poses:
        return {"ok": False, "reason": "no_safe_poses"}
    best = min(
        safe_poses,
        key=lambda pose: math.sqrt(sum((float(target[joint]) - pose[joint]) ** 2 for joint in SAFE_MANIFOLD_JOINTS)),
    )
    distance = math.sqrt(sum((float(target[joint]) - best[joint]) ** 2 for joint in SAFE_MANIFOLD_JOINTS))
    return {
        "ok": distance <= max_distance,
        "distance_raw": round(distance, 4),
        "max_distance_raw": max_distance,
        "nearest_safe_pose_raw": {joint: round(best[joint], 4) for joint in SAFE_MANIFOLD_JOINTS},
        "target_manifold_joints_raw": {joint: target[joint] for joint in SAFE_MANIFOLD_JOINTS},
    }


def _read_state(bus: Any) -> dict[str, float]:
    return {
        joint: float(value)
        for joint, value in bus.sync_read("Present_Position", normalize=False).items()
        if joint in SO100_JOINT_ORDER
    }


def _open_video_writer(cv2: Any, path: Path, frame: Any, fps: float) -> Any:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (int(frame.shape[1]), int(frame.shape[0])))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def _write_video_frame(capture: Any, writer: Any, result: dict[str, Any]) -> None:
    ok, frame = capture.read()
    if ok and frame is not None:
        writer.write(frame)
        result["frames_recorded"] += 1


def _record_video_for(capture: Any, writer: Any, result: dict[str, Any], duration_seconds: float, fps: float) -> None:
    deadline = time.monotonic() + max(duration_seconds, 0.0)
    period = 1.0 / fps
    next_frame = time.monotonic()
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now < next_frame:
            time.sleep(min(next_frame - now, 0.01))
            continue
        _write_video_frame(capture, writer, result)
        next_frame += period


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Approach the visible green object with the SO-100 gripper marker using the body-map model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--safe-pose-manifest", type=Path, default=DEFAULT_SAFE_POSE_MANIFEST)
    parser.add_argument("--disable-safe-manifold-guard", action="store_true")
    parser.add_argument("--max-safe-manifold-distance-raw", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--servo-camera-index", type=int, default=2)
    parser.add_argument("--observer-camera-index", type=int, default=3)
    parser.add_argument("--marker-id", type=int, default=8)
    parser.add_argument("--green-selection", choices=["android_doll", "largest", "rightmost"], default="android_doll")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--target-distance-px", type=float, default=45.0)
    parser.add_argument("--desired-step-px", type=float, default=14.0)
    parser.add_argument("--max-abs-delta-raw", type=float, default=45.0)
    parser.add_argument("--scale", type=float, default=0.85)
    parser.add_argument(
        "--manual-delta-json",
        default=None,
        help="Optional per-step raw joint delta JSON object. When set, bypasses body-map transition selection.",
    )
    parser.add_argument(
        "--freeze-joint",
        action="append",
        default=["gripper", "wrist_roll"],
        choices=SO100_JOINT_ORDER,
        help="Joint to keep fixed during approach. Defaults freeze gripper and wrist_roll.",
    )
    parser.add_argument("--step-settle-seconds", type=float, default=0.75)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_approach_green(
                model_path=args.model,
                output_dir=args.output_dir,
                port=args.port,
                calibration=args.calibration,
                servo_camera_index=args.servo_camera_index,
                observer_camera_index=args.observer_camera_index,
                marker_id=args.marker_id,
                green_selection=args.green_selection,
                max_iterations=args.max_iterations,
                target_distance_px=args.target_distance_px,
                desired_step_px=args.desired_step_px,
                max_abs_delta_raw=args.max_abs_delta_raw,
                scale=args.scale,
                manual_delta_raw=json.loads(args.manual_delta_json) if args.manual_delta_json else None,
                freeze_joints=args.freeze_joint,
                safe_pose_manifest=None if args.disable_safe_manifold_guard else args.safe_pose_manifest,
                max_safe_manifold_distance_raw=args.max_safe_manifold_distance_raw,
                step_settle_seconds=args.step_settle_seconds,
                video_fps=args.video_fps,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                workspace_clear_confirmed=args.workspace_clear_confirmed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
