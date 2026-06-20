#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from physical_ai_agent.safety.so100_action_gate import SO100_JOINT_ORDER
from scripts.real_so100_apriltag_detection_check import MARKER_DICTIONARIES
from scripts.real_so100_build_visual_ik_forward_model import IK_JOINTS, _features
from scripts.real_so100_micro_step import _make_so100_bus, _probe_motion_video
from scripts.real_so100_move_to_home_pose import DEFAULT_HOME_POSE, move_to_home_pose


DEFAULT_PORT = "/dev/cu.usbmodem5AE60824791"
DEFAULT_CALIBRATION = Path("_workspace/real_so100/calibration/so100_local.json")
DEFAULT_FORWARD_MODEL = Path(
    "_workspace/real_so100/pose_calibration/good_pose_dense_20260611_220240/visual_ik_forward_model.json"
)
DEFAULT_GOOD_POSE_MANIFEST = Path(
    "_workspace/real_so100/pose_calibration/good_pose_dense_20260611_220240/manifest.json"
)
USER_REPORTED_FLOOR_CONTACT_GUARD = {
    "source": "user_observed_floor_contact_after_run_006_run_007",
    "raw_threshold_guard": "deprecated_for_default_execution",
    "notes": [
        "Camera-only clearance judgment was insufficient; the operator's physical observation is authoritative.",
        "Default execution must stay on the user-approved good-pose manifold, not on independent raw joint thresholds.",
    ],
}


def run_visual_ik_approach_green(
    *,
    output_dir: Path,
    forward_model: Path,
    model_key: str,
    port: str,
    calibration: Path,
    servo_camera_index: int,
    observer_camera_index: int,
    marker_id: int,
    execute: bool,
    human_confirmed: bool,
    workspace_clear_confirmed: bool,
    target_offset_px: tuple[float, float],
    target_fraction: float,
    ik_control_mode: str,
    jacobian_damping: float,
    jacobian_step_px_fraction: float,
    max_iterations: int,
    max_abs_step_delta_raw: float,
    residual_x_gain_raw_per_px: float,
    residual_y_gain_raw_per_px: float,
    min_wrist_flex_raw: float | None,
    max_shoulder_lift_raw: float | None,
    allow_floor_contact_risk: bool,
    good_pose_manifest: Path | None,
    use_good_pose_manifold: bool,
    project_to_good_pose_manifold: bool,
    settle_seconds: float,
    video_fps: float,
) -> dict[str, Any]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visual"
    visual_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "visual_ik_approach_green_report.json"
    model_payload = json.loads(forward_model.read_text(encoding="utf-8"))
    group = model_payload["models"][model_key]
    calibration_payload = _load_calibration(calibration)
    good_pose_manifold = _load_good_pose_manifold(good_pose_manifest) if use_good_pose_manifold and good_pose_manifest else []
    effective_min_wrist_flex_raw, effective_max_shoulder_lift_raw, floor_guard_notes = _resolve_floor_contact_guard(
        min_wrist_flex_raw=min_wrist_flex_raw,
        max_shoulder_lift_raw=max_shoulder_lift_raw,
        allow_floor_contact_risk=allow_floor_contact_risk,
        use_good_pose_manifold=use_good_pose_manifold,
    )

    blockers = []
    if not human_confirmed:
        blockers.append("Human confirmation flag is required.")
    if not workspace_clear_confirmed:
        blockers.append("Workspace clear confirmation flag is required.")
    if max_iterations <= 0:
        blockers.append("max_iterations must be positive.")
    if execute and use_good_pose_manifold and not good_pose_manifold:
        blockers.append(f"good-pose manifold is required but empty or missing: {good_pose_manifest}")
    if execute and project_to_good_pose_manifold and not good_pose_manifold:
        blockers.append("good-pose projection requested but the manifold is empty.")

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "operation": "real_so100_visual_ik_approach_green",
        "forward_model": str(forward_model),
        "model_key": model_key,
        "servo_camera_index": servo_camera_index,
        "observer_camera_index": observer_camera_index,
        "same_camera_frame_reuse": servo_camera_index == observer_camera_index,
        "marker_id": marker_id,
        "target_offset_px": list(target_offset_px),
        "target_fraction": target_fraction,
        "ik_control_mode": ik_control_mode,
        "jacobian_damping": jacobian_damping,
        "jacobian_step_px_fraction": jacobian_step_px_fraction,
        "residual_x_gain_raw_per_px": residual_x_gain_raw_per_px,
        "residual_y_gain_raw_per_px": residual_y_gain_raw_per_px,
        "min_wrist_flex_raw": min_wrist_flex_raw,
        "max_shoulder_lift_raw": max_shoulder_lift_raw,
        "effective_min_wrist_flex_raw": effective_min_wrist_flex_raw,
        "effective_max_shoulder_lift_raw": effective_max_shoulder_lift_raw,
        "allow_floor_contact_risk": allow_floor_contact_risk,
        "floor_contact_guard": USER_REPORTED_FLOOR_CONTACT_GUARD,
        "floor_contact_guard_notes": floor_guard_notes,
        "good_pose_manifest": str(good_pose_manifest) if good_pose_manifest else None,
        "use_good_pose_manifold": use_good_pose_manifold,
        "project_to_good_pose_manifold": project_to_good_pose_manifold,
        "good_pose_manifold_size": len(good_pose_manifold),
        "max_iterations": max_iterations,
        "max_abs_step_delta_raw": max_abs_step_delta_raw,
        "execute_requested": execute,
        "human_confirmed": human_confirmed,
        "workspace_clear_confirmed": workspace_clear_confirmed,
        "blockers": blockers,
        "steps": [],
        "send_action_called": False,
        "policy_actions_executed": False,
        "physical_robot_motion": False,
        "post_task_torque_disabled": False,
        "status": "blocked" if blockers else ("ready" if execute else "dry_run"),
    }
    if blockers or not execute:
        _write_json(report_path, report)
        return report

    servo_cap = cv2.VideoCapture(servo_camera_index, cv2.CAP_AVFOUNDATION)
    observer_cap = servo_cap if observer_camera_index == servo_camera_index else cv2.VideoCapture(observer_camera_index, cv2.CAP_AVFOUNDATION)
    video_writer = None
    video_result: dict[str, Any] | None = None
    bus = None
    try:
        if not servo_cap.isOpened():
            raise RuntimeError(f"servo camera {servo_camera_index} did not open")
        if not observer_cap.isOpened():
            raise RuntimeError(f"observer camera {observer_camera_index} did not open")
        _warmup(servo_cap, frames=8)
        observer_frame = _warmup(observer_cap, frames=8)
        video_path = visual_dir / "observer_motion.mp4"
        video_writer = _open_video_writer(cv2, video_path, observer_frame, video_fps)
        video_result = {"camera_index": observer_camera_index, "path": str(video_path), "fps": video_fps, "frames_recorded": 0}

        bus, _motors = _make_so100_bus(port)
        bus.connect(handshake=True)
        before_state = _read_state(bus)
        report["readback_before_raw"] = before_state
        bus.sync_write("Goal_Position", {joint: int(round(before_state[joint])) for joint in SO100_JOINT_ORDER}, normalize=False, num_retry=3)

        for step_index in range(max_iterations):
            servo_frame = _read_frame(servo_cap)
            observer_frame = servo_frame.copy() if observer_cap is servo_cap else _read_frame(observer_cap)
            servo_path = visual_dir / f"step_{step_index:02d}_servo.jpg"
            observer_path = visual_dir / f"step_{step_index:02d}_observer.jpg"
            cv2.imwrite(str(servo_path), servo_frame)
            cv2.imwrite(str(observer_path), observer_frame)
            marker = _detect_marker(servo_frame, marker_id=marker_id)
            marker_retry_count = 0
            while marker is None and marker_retry_count < 5:
                time.sleep(0.06)
                servo_frame = _read_frame(servo_cap)
                observer_frame = servo_frame.copy() if observer_cap is servo_cap else _read_frame(observer_cap)
                cv2.imwrite(str(servo_path), servo_frame)
                cv2.imwrite(str(observer_path), observer_frame)
                marker = _detect_marker(servo_frame, marker_id=marker_id)
                marker_retry_count += 1
            green = _detect_green_object(servo_frame)
            step: dict[str, Any] = {
                "step_index": step_index,
                "servo_image": str(servo_path),
                "observer_image": str(observer_path),
                "marker": marker,
                "marker_retry_count": marker_retry_count,
                "green_object": green,
                "executed": False,
            }
            if marker is None:
                step["stop_reason"] = f"marker id {marker_id} not detected"
                report["steps"].append(step)
                break
            if green is None:
                step["stop_reason"] = "green object not detected"
                report["steps"].append(step)
                break

            current_state = _read_state(bus)
            current_q = np.asarray([current_state[joint] for joint in IK_JOINTS], dtype=np.float64)
            green_target = np.asarray(green["approach_target_px"], dtype=np.float64) + np.asarray(target_offset_px, dtype=np.float64)
            marker_px = np.asarray(marker["center_px"], dtype=np.float64)
            desired_px = marker_px + ((green_target - marker_px) * target_fraction)
            if ik_control_mode == "differential":
                q_solution, solve = _solve_differential_ik(
                    group=group,
                    marker_px=marker_px,
                    desired_px=desired_px,
                    current_q=current_q,
                    damping=jacobian_damping,
                    step_px_fraction=jacobian_step_px_fraction,
                )
                residual = None
            elif ik_control_mode == "global_search":
                q_solution, solve = _solve_inverse(group=group, target_px=desired_px, current_q=current_q)
                residual = _residual_servo_q(
                    q_solution=q_solution,
                    marker_px=marker_px,
                    desired_px=desired_px,
                    x_gain=residual_x_gain_raw_per_px,
                    y_gain=residual_y_gain_raw_per_px,
                    calibration=calibration_payload,
                )
                q_solution = residual["q_solution_with_residual_raw"]
            else:
                raise ValueError(f"unknown ik_control_mode: {ik_control_mode}")
            local_manifold = _nearest_good_pose_manifold(
                current_q=current_q,
                good_pose_manifold=good_pose_manifold,
            )
            manifold_projection = None
            if project_to_good_pose_manifold and good_pose_manifold:
                q_solution, manifold_projection = _project_to_good_pose_manifold(
                    q_solution=q_solution,
                    current_q=current_q,
                    good_pose_manifold=good_pose_manifold,
                )
            target = _bounded_target(
                current_state=current_state,
                q_solution=q_solution,
                calibration=calibration_payload,
                max_abs_delta=max_abs_step_delta_raw,
                min_wrist_flex_raw=effective_min_wrist_flex_raw,
                max_shoulder_lift_raw=effective_max_shoulder_lift_raw,
            )
            step.update(
                {
                    "green_target_px": _round_list(green_target),
                    "desired_marker_px": _round_list(desired_px),
                    "inverse_solve": solve,
                    "residual_servo": residual["report"] if residual else None,
                    "nearest_good_pose": local_manifold,
                    "good_pose_projection": manifold_projection,
                    "target_raw": target,
                    "limited_delta_raw": {
                        joint: round(float(target[joint] - current_state[joint]), 4)
                        for joint in SO100_JOINT_ORDER
                    },
                }
            )
            bus.sync_write("Goal_Position", target, normalize=False, num_retry=3)
            report["send_action_called"] = True
            report["policy_actions_executed"] = True
            report["physical_robot_motion"] = True
            step["executed"] = True
            _record_video_for(observer_cap, video_writer, video_result, settle_seconds, video_fps)
            step["readback_after_raw"] = _read_state(bus)
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
            "before": {"camera_index": observer_camera_index, "image_path": report["steps"][0]["observer_image"] if report["steps"] else None},
            "after": {"camera_index": observer_camera_index, "image_path": str(final_observer_path)},
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
        if observer_cap is not servo_cap:
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
        output=output_dir / "home_return_after_visual_ik_approach.json",
        execute=True,
        human_confirmed=human_confirmed,
        workspace_clear_confirmed=workspace_clear_confirmed,
        max_abs_delta_raw=90.0,
        step_settle_seconds=0.10,
        camera_index=observer_camera_index,
        visual_output_dir=output_dir / "home_return_visual",
        record_video=True,
        video_fps=video_fps,
    )
    report["home_return_report"] = str(output_dir / "home_return_after_visual_ik_approach.json")
    report["home_return_status"] = home_report.get("status")
    report["post_task_torque_disabled"] = bool(home_report.get("post_task_torque_disabled"))
    _write_json(report_path, report)
    return report


def _solve_inverse(*, group: dict[str, Any], target_px: np.ndarray, current_q: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    bounds = np.asarray([group["q_bounds_raw"][joint] for joint in IK_JOINTS], dtype=np.float64)
    low = bounds[:, 0]
    high = bounds[:, 1]
    best_q = np.clip(current_q, low, high)
    best_score = _score(group, best_q, target_px, current_q)
    steps = np.asarray([80.0, 120.0, 120.0, 90.0], dtype=np.float64)
    evals = 1
    while float(steps.max()) >= 4.0:
        improved = False
        for joint_index in range(len(IK_JOINTS)):
            for direction in (-1.0, 1.0):
                candidate = best_q.copy()
                candidate[joint_index] = np.clip(candidate[joint_index] + (steps[joint_index] * direction), low[joint_index], high[joint_index])
                score = _score(group, candidate, target_px, current_q)
                evals += 1
                if score < best_score:
                    best_q = candidate
                    best_score = score
                    improved = True
        if not improved:
            steps *= 0.5
    pred = _predict(group, best_q)
    return best_q, {
        "q_solution_raw": {joint: round(float(best_q[index]), 4) for index, joint in enumerate(IK_JOINTS)},
        "predicted_marker_px": _round_list(pred),
        "target_px": _round_list(target_px),
        "pixel_error_px": _round_list(pred - target_px),
        "pixel_distance_px": round(float(np.linalg.norm(pred - target_px)), 4),
        "evaluations": evals,
    }


def _solve_differential_ik(
    *,
    group: dict[str, Any],
    marker_px: np.ndarray,
    desired_px: np.ndarray,
    current_q: np.ndarray,
    damping: float,
    step_px_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pixel_error = (desired_px - marker_px) * float(step_px_fraction)
    predicted_px = _predict(group, current_q)
    jacobian = _finite_difference_jacobian(group=group, q=current_q)
    jj_t = jacobian @ jacobian.T
    damped = jj_t + (float(damping) ** 2 * np.eye(jj_t.shape[0], dtype=np.float64))
    dq = jacobian.T @ np.linalg.solve(damped, pixel_error)
    q_solution = current_q + dq
    predicted_next_px = _predict(group, q_solution)
    return q_solution, {
        "mode": "differential_visual_servo_ik",
        "marker_px": _round_list(marker_px),
        "desired_px": _round_list(desired_px),
        "pixel_error_px": _round_list(desired_px - marker_px),
        "used_pixel_step_px": _round_list(pixel_error),
        "predicted_current_marker_px_from_model": _round_list(predicted_px),
        "local_jacobian_px_per_raw_tick": [
            _round_list(row)
            for row in jacobian
        ],
        "joint_delta_raw": {
            joint: round(float(dq[index]), 4)
            for index, joint in enumerate(IK_JOINTS)
        },
        "q_solution_raw": {
            joint: round(float(q_solution[index]), 4)
            for index, joint in enumerate(IK_JOINTS)
        },
        "predicted_next_marker_px": _round_list(predicted_next_px),
        "jacobian_damping": damping,
        "step_px_fraction": step_px_fraction,
    }


def _finite_difference_jacobian(*, group: dict[str, Any], q: np.ndarray) -> np.ndarray:
    eps = np.asarray([8.0, 10.0, 10.0, 8.0], dtype=np.float64)
    columns = []
    for index in range(len(IK_JOINTS)):
        left = q.copy()
        right = q.copy()
        left[index] -= eps[index]
        right[index] += eps[index]
        columns.append((_predict(group, right) - _predict(group, left)) / (2.0 * eps[index]))
    return np.stack(columns, axis=1)


def _score(group: dict[str, Any], q: np.ndarray, target_px: np.ndarray, current_q: np.ndarray) -> float:
    pred = _predict(group, q)
    pixel_cost = float(np.sum((pred - target_px) ** 2))
    move_cost = 0.002 * float(np.sum(((q - current_q) / np.asarray([80.0, 150.0, 150.0, 120.0])) ** 2))
    return pixel_cost + move_cost


def _predict(group: dict[str, Any], q: np.ndarray) -> np.ndarray:
    mean = np.asarray(group["q_mean_raw"], dtype=np.float64)
    std = np.asarray(group["q_std_raw"], dtype=np.float64)
    coef = np.asarray(group["coef_px"], dtype=np.float64)
    x = _features(((q - mean) / std)[None, :])
    return (x @ coef)[0]


def _bounded_target(
    *,
    current_state: dict[str, float],
    q_solution: np.ndarray,
    calibration: dict[str, dict[str, float]],
    max_abs_delta: float,
    min_wrist_flex_raw: float | None,
    max_shoulder_lift_raw: float | None,
) -> dict[str, int]:
    target = {joint: int(round(current_state[joint])) for joint in SO100_JOINT_ORDER}
    for index, joint in enumerate(IK_JOINTS):
        desired = float(q_solution[index])
        if joint == "wrist_flex" and min_wrist_flex_raw is not None:
            desired = max(desired, min_wrist_flex_raw)
        if joint == "shoulder_lift" and max_shoulder_lift_raw is not None:
            desired = min(desired, max_shoulder_lift_raw)
        limited = current_state[joint] + max(min(desired - current_state[joint], max_abs_delta), -max_abs_delta)
        target[joint] = int(round(_clip(limited, calibration[joint]["range_min"], calibration[joint]["range_max"])))
    return target


def _load_good_pose_manifold(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifold: list[dict[str, Any]] = []
    for sample in payload.get("samples", []):
        state = sample.get("readback_raw") or sample.get("commanded_target_raw")
        if not isinstance(state, dict):
            continue
        if not all(joint in state for joint in IK_JOINTS):
            continue
        manifold.append(
            {
                "sample_index": sample.get("sample_index"),
                "pose_index": sample.get("pose_index"),
                "offset_index": sample.get("offset_index"),
                "pose_label": sample.get("pose_label"),
                "q_raw": np.asarray([float(state[joint]) for joint in IK_JOINTS], dtype=np.float64),
                "state_raw": {
                    joint: float(state[joint])
                    for joint in SO100_JOINT_ORDER
                    if joint in state
                },
            }
        )
    return manifold


def _project_to_good_pose_manifold(
    *,
    q_solution: np.ndarray,
    current_q: np.ndarray,
    good_pose_manifold: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    scale = np.asarray([120.0, 220.0, 220.0, 180.0], dtype=np.float64)
    best: dict[str, Any] | None = None
    best_score = float("inf")
    for candidate in good_pose_manifold:
        q = candidate["q_raw"]
        ik_cost = float(np.sum(((q - q_solution) / scale) ** 2))
        move_cost = 0.08 * float(np.sum(((q - current_q) / scale) ** 2))
        score = ik_cost + move_cost
        if score < best_score:
            best = candidate
            best_score = score
    if best is None:
        return q_solution, {"status": "empty"}
    projected_q = best["q_raw"].copy()
    return projected_q, {
        "status": "projected_to_user_approved_good_pose_manifold",
        "sample_index": best.get("sample_index"),
        "pose_index": best.get("pose_index"),
        "offset_index": best.get("offset_index"),
        "pose_label": best.get("pose_label"),
        "q_solution_before_projection_raw": {
            joint: round(float(q_solution[index]), 4)
            for index, joint in enumerate(IK_JOINTS)
        },
        "projected_q_raw": {
            joint: round(float(projected_q[index]), 4)
            for index, joint in enumerate(IK_JOINTS)
        },
        "projection_score": round(best_score, 6),
    }


def _nearest_good_pose_manifold(
    *,
    current_q: np.ndarray,
    good_pose_manifold: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not good_pose_manifold:
        return None
    scale = np.asarray([120.0, 220.0, 220.0, 180.0], dtype=np.float64)
    nearest: dict[str, Any] | None = None
    nearest_score = float("inf")
    for candidate in good_pose_manifold:
        q = candidate["q_raw"]
        score = float(np.sum(((q - current_q) / scale) ** 2))
        if score < nearest_score:
            nearest = candidate
            nearest_score = score
    if nearest is None:
        return None
    return {
        "sample_index": nearest.get("sample_index"),
        "pose_index": nearest.get("pose_index"),
        "offset_index": nearest.get("offset_index"),
        "pose_label": nearest.get("pose_label"),
        "q_raw": {
            joint: round(float(nearest["q_raw"][index]), 4)
            for index, joint in enumerate(IK_JOINTS)
        },
        "normalized_distance": round(nearest_score, 6),
    }


def _resolve_floor_contact_guard(
    *,
    min_wrist_flex_raw: float | None,
    max_shoulder_lift_raw: float | None,
    allow_floor_contact_risk: bool,
    use_good_pose_manifold: bool,
) -> tuple[float | None, float | None, list[str]]:
    notes: list[str] = []
    if use_good_pose_manifold:
        notes.append("raw floor-contact thresholds are not applied by default; execution is projected to user-approved good poses")
        return min_wrist_flex_raw, max_shoulder_lift_raw, notes
    if allow_floor_contact_risk:
        notes.append("floor contact guard override was explicitly enabled")
        return min_wrist_flex_raw, max_shoulder_lift_raw, notes
    notes.append("no raw threshold floor guard is active; use the good-pose manifold for default safety")
    return min_wrist_flex_raw, max_shoulder_lift_raw, notes


def _residual_servo_q(
    *,
    q_solution: np.ndarray,
    marker_px: np.ndarray,
    desired_px: np.ndarray,
    x_gain: float,
    y_gain: float,
    calibration: dict[str, dict[str, float]],
) -> dict[str, Any]:
    adjusted = q_solution.copy()
    error = desired_px - marker_px
    if x_gain:
        adjusted[IK_JOINTS.index("shoulder_pan")] += float(error[0]) * x_gain
    if y_gain:
        y_delta = float(error[1]) * y_gain
        adjusted[IK_JOINTS.index("shoulder_lift")] += y_delta
        adjusted[IK_JOINTS.index("wrist_flex")] -= y_delta
    for index, joint in enumerate(IK_JOINTS):
        adjusted[index] = _clip(
            float(adjusted[index]),
            calibration[joint]["range_min"],
            calibration[joint]["range_max"],
        )
    return {
        "q_solution_with_residual_raw": adjusted,
        "report": {
            "pixel_error_marker_to_desired_px": _round_list(error),
            "x_gain_raw_per_px": x_gain,
            "y_gain_raw_per_px": y_gain,
            "q_solution_with_residual_raw": {
                joint: round(float(adjusted[index]), 4)
                for index, joint in enumerate(IK_JOINTS)
            },
        },
    }


def _detect_marker(frame: Any, *, marker_id: int) -> dict[str, Any] | None:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICTIONARIES["tag36h11"]))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return None
    matches = []
    for detected_id, corner in zip(ids, corners):
        if int(detected_id[0]) != marker_id:
            continue
        points = corner.reshape(-1, 2)
        center = points.mean(axis=0)
        matches.append({"id": marker_id, "center_px": _round_list(center)})
    return matches[0] if matches else None


def _detect_green_object(frame: Any) -> dict[str, Any] | None:
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 35, 35), (95, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 1200.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center = [float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])]
        candidates.append({"area_px": area, "bbox_xywh": [int(x), int(y), int(w), int(h)], "center_px": _round_list(center)})
    if not candidates:
        return None
    # Pick the main Android doll: large green blob in the lower/middle observer view.
    selected = max(candidates, key=lambda item: item["area_px"])
    x, y, w, h = selected["bbox_xywh"]
    selected["approach_target_px"] = [round(float(x + (w * 0.5)), 4), round(float(y + (h * 0.20)), 4)]
    return selected


def _load_calibration(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        joint: {"range_min": float(payload[joint]["range_min"]), "range_max": float(payload[joint]["range_max"])}
        for joint in SO100_JOINT_ORDER
    }


def _read_state(bus: Any) -> dict[str, float]:
    return {
        joint: float(value)
        for joint, value in bus.sync_read("Present_Position", normalize=False).items()
        if joint in SO100_JOINT_ORDER
    }


def _warmup(capture: Any, *, frames: int) -> Any:
    frame = None
    ok = False
    for _ in range(max(1, frames)):
        ok, frame = capture.read()
        time.sleep(0.03)
    if not ok or frame is None:
        raise RuntimeError("camera did not return warmup frame")
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


def _open_video_writer(cv2: Any, path: Path, frame: Any, fps: float) -> Any:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(frame.shape[1]), int(frame.shape[0])))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def _record_video_for(capture: Any, writer: Any, result: dict[str, Any], duration_seconds: float, fps: float) -> None:
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


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _round_list(values: Any) -> list[float]:
    return [round(float(value), 4) for value in values]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Approach the green Android doll using a visual IK forward model.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forward-model", type=Path, default=DEFAULT_FORWARD_MODEL)
    parser.add_argument("--good-pose-manifest", type=Path, default=DEFAULT_GOOD_POSE_MANIFEST)
    parser.add_argument("--model-key", default="camera_3/tag36h11/id_8")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--calibration", type=Path, default=Path("_workspace/real_so100/calibration/so100_local.json"))
    parser.add_argument("--servo-camera-index", type=int, default=3)
    parser.add_argument("--observer-camera-index", type=int, default=3)
    parser.add_argument("--marker-id", type=int, default=8)
    parser.add_argument("--target-offset-x-px", type=float, default=0.0)
    parser.add_argument("--target-offset-y-px", type=float, default=-90.0)
    parser.add_argument("--target-fraction", type=float, default=0.55)
    parser.add_argument("--ik-control-mode", choices=["differential", "global_search"], default="differential")
    parser.add_argument("--jacobian-damping", type=float, default=0.08)
    parser.add_argument("--jacobian-step-px-fraction", type=float, default=0.35)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--max-abs-step-delta-raw", type=float, default=90.0)
    parser.add_argument("--residual-x-gain-raw-per-px", type=float, default=0.0)
    parser.add_argument("--residual-y-gain-raw-per-px", type=float, default=0.0)
    parser.add_argument("--min-wrist-flex-raw", type=float, default=None)
    parser.add_argument("--max-shoulder-lift-raw", type=float, default=None)
    parser.add_argument(
        "--allow-floor-contact-risk",
        action="store_true",
        help="Override the user-reported floor-contact guard. Do not use for normal autonomous approach runs.",
    )
    parser.add_argument(
        "--project-to-good-pose-manifold",
        action="store_true",
        help="Project IK output to a saved good-pose sample. This is a safety fallback, not the default IK controller.",
    )
    parser.add_argument("--settle-seconds", type=float, default=0.75)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-confirmed", action="store_true")
    parser.add_argument("--workspace-clear-confirmed", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_visual_ik_approach_green(
                output_dir=args.output_dir,
                forward_model=args.forward_model,
                model_key=args.model_key,
                port=args.port,
                calibration=args.calibration,
                servo_camera_index=args.servo_camera_index,
                observer_camera_index=args.observer_camera_index,
                marker_id=args.marker_id,
                execute=args.execute,
                human_confirmed=args.human_confirmed,
                workspace_clear_confirmed=args.workspace_clear_confirmed,
                target_offset_px=(args.target_offset_x_px, args.target_offset_y_px),
                target_fraction=args.target_fraction,
                ik_control_mode=args.ik_control_mode,
                jacobian_damping=args.jacobian_damping,
                jacobian_step_px_fraction=args.jacobian_step_px_fraction,
                max_iterations=args.max_iterations,
                max_abs_step_delta_raw=args.max_abs_step_delta_raw,
                residual_x_gain_raw_per_px=args.residual_x_gain_raw_per_px,
                residual_y_gain_raw_per_px=args.residual_y_gain_raw_per_px,
                min_wrist_flex_raw=args.min_wrist_flex_raw,
                max_shoulder_lift_raw=args.max_shoulder_lift_raw,
                allow_floor_contact_risk=args.allow_floor_contact_risk,
                good_pose_manifest=args.good_pose_manifest,
                use_good_pose_manifold=True,
                project_to_good_pose_manifold=args.project_to_good_pose_manifold,
                settle_seconds=args.settle_seconds,
                video_fps=args.video_fps,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
